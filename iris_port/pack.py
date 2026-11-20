# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""Whole-pack conversion: walks a pack, converts programs and includes, maps the properties files, writes a report."""
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path
import re

from . import api
from .blocks import map_block_properties
from .geometry import fold_geometry
from .convert import BLOCKING, FileConverter, PackContext, host_macros, strip_empty_calls
from .samplers import specialize
from .types import GLSL_TYPES, sampler

PROGRAM_SUFFIXES = {'.vsh': 'vertex', '.fsh': 'fragment', '.csh': 'compute', '.gsh': 'geometry',
                    '.tcs': 'tess', '.tes': 'tess'}
INCLUDE_SUFFIXES = {'.glsl', '.inc'}
DEFAULT_DROP = ['physics_*', 'voxy_*']
# NWR runs no shader pack while Distant Horizons is on, so its programs are never converted.
DH_DROP = 'dh_*'
DH_DROP_NOTE = 'Distant Horizons program; NWR runs no shader pack while Distant Horizons is on; not converted'
DIMENSION_IDS = {'minecraft:the_nether': 'minecraft:nether', 'minecraft:the_end': 'minecraft:the_end',
                 'minecraft:overworld': 'minecraft:overworld'}
FOG_COORD = re.compile(r'\bgl_FogFragCoord\b')
FOG_COORD_DECLARED = re.compile(r'\b(?:varying|in|out)\s+float\s+gl_FogFragCoord\b')
FOG_VARYING = 'NWR_VARYING(float, gl_FogFragCoord);\n'
COMMENTS = re.compile(r'//[^\n]*|/\*.*?\*/', re.S)
CATEGORY_ORDER = ['untyped-product', 'untyped-constructor', 'parse', 'api-gap', 'hand', 'block', 'dropped',
                  'untyped-vector-product', 'parse-note', 'note']


def find_shaders(root):
    root = Path(root)
    if (root / 'shaders').is_dir():
        return root, root / 'shaders'
    if root.name == 'shaders':
        return root.parent, root
    raise FileNotFoundError(f'no shaders/ folder under {root}')


def read_text(path):
    return path.read_bytes().decode('utf-8', errors='surrogateescape')


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode('utf-8', errors='surrogateescape'))


def custom_names(properties_text):
    uniforms = {}
    textures = set()
    images = {}
    for line in properties_text.splitlines():
        m = re.match(r'\s*image\.(\w+)\s*=\s*(\w+)', line)
        if m:
            images[m.group(1)] = m.group(2)
        m = re.match(r'\s*uniform\.(\w+)\.(\w+)\s*=', line)
        if m and m.group(1) in GLSL_TYPES:
            uniforms[m.group(2)] = GLSL_TYPES[m.group(1)]
        m = re.match(r'\s*texture\.\w+\.(\w+)(?:\.\w+)?\s*=', line)
        if m:
            textures.add(m.group(1))
        m = re.match(r'\s*customTexture\.(\w+)\s*=', line)
        if m:
            textures.add(m.group(1))
    return uniforms, textures, images


def volume_textures(properties_text):
    return {m.group(1) for m in re.finditer(r'(?m)^\s*texture\.\w+\.(\w+)(?:\.\w+)?\s*=\s*\S+\s+TEXTURE_3D\b',
                                            properties_text)}


VOLUME_SUFFIX = '_nwrVolume'
VOLUME_LINE = re.compile(r'(?m)^(\s*texture\.\w+\.)(\w+)((?:\.\w+)?\s*=\s*\S+\s+TEXTURE_3D\b)')
FLAT_SAMPLER = re.compile(r'\buniform\s+(?:(?:lowp|mediump|highp)\s+)?[iu]?sampler2D\s+(\w+)\s*;')
VOLUME_SAMPLER = re.compile(r'\buniform\s+(?:(?:lowp|mediump|highp)\s+)?[iu]?sampler3D\s+(\w+)\s*;')
INCLUDE_LINE = re.compile(r'(?m)^\s*#\s*include\s+"([^"]+)"')


SPLIT_SAMPLER_CALL = re.compile(r'\b(texture\w*|texelFetch\w*|shadow2D\w*|image(?:Load|Store|Size|Atomic\w+))\(([ \t]*\r?\n\s*)(\w+)[ \t]*,')


def join_split_sampler_calls(texts):
    """The loader rewrites `texture(name,` and `imageStore(name,` within one line, so a sampler or image argument that
    starts on the next line moves up;
    the line break goes after its comma and line numbers stay the same."""
    return {path: SPLIT_SAMPLER_CALL.sub(lambda m: f'{m.group(1)}({m.group(3)},{m.group(2)}', text)
            for path, text in texts.items()}


def volume_renames(properties_text):
    """Built-in sampler names that a stage overrides with a 3D texture, and the name each override takes instead."""
    return {name: name + VOLUME_SUFFIX for name in sorted(volume_textures(properties_text)) if name in api.TEXTURES}


def rename_volume_lines(properties_text, renames):
    return VOLUME_LINE.sub(lambda m: m.group(1) + renames.get(m.group(2), m.group(2)) + m.group(3), properties_text)


def volume_reads(texts, renames, resolve):
    """Iris binds a stage's 3D texture.<stage>.<name> only to samplers declared 3D; a program that declares the name
    as a 2D sampler reads the render target (or other built-in) of that name. The 3D declarers, and the includes that
    only they reach, read the override under its own name, which its properties line takes too, so every other
    program of the stage keeps the built-in."""
    includes = {path: [resolve(path, name) for name in INCLUDE_LINE.findall(text)] for path, text in texts.items()}

    def reach(starts):
        seen = set()
        pending = list(starts)
        while pending:
            for target in includes.get(pending.pop(), ()):
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
        return seen

    renamed = {}
    for name in renames:
        volume = {path for path, text in texts.items() if name in VOLUME_SAMPLER.findall(text)}
        if not volume:
            continue
        flat = {path for path, text in texts.items() if name in FLAT_SAMPLER.findall(text)}
        for path in (volume - flat) | (reach(volume) - reach(flat) - flat):
            renamed.setdefault(path, []).append(name)
    out = {}
    for path, text in texts.items():
        for name in renamed.get(path, ()):
            text = re.sub(r'\b' + name + r'\b', renames[name], text)
        out[path] = text
    return out


def output_name(relative):
    suffix = relative.suffix
    if suffix in PROGRAM_SUFFIXES:
        return relative.with_name(relative.name + '.hlsl')
    if suffix in INCLUDE_SUFFIXES:
        return relative.with_suffix('.hlsli')
    return relative


def drop_unreachable(by_path, includes, drop):
    """Include files named like a drop pattern that no kept file includes."""
    candidates = {path for path in by_path if any(fnmatch(Path(path).stem, pattern) for pattern in drop)}
    while True:
        used = {target for path in by_path if path not in candidates for target, _ in includes[path]}
        still = {path for path in candidates if path not in used}
        if still == candidates:
            return sorted(candidates)
        candidates = still


def drop_orphaned(by_path, includes, unused):
    """Include files that only dropped files include, so no kept program can reach them."""
    gone = set(unused)
    while True:
        includers = {}
        for path in by_path:
            for target, _ in includes[path]:
                includers.setdefault(target, set()).add(path)
        more = {path for path, c in by_path.items() if path not in gone and c.file_stage is None
                and includers.get(path) and includers[path] <= gone}
        if not more:
            return sorted(gone - set(unused))
        gone |= more


def body_fragments(by_path, includes):
    """Files #included inside a function body, and everything they include: they hold statements, not globals."""
    fragments = {target for path in by_path for target, body in includes[path] if body and target in by_path}
    pending = sorted(fragments)
    while pending:
        path = pending.pop()
        for target, _ in includes[path]:
            if target in by_path and target not in fragments:
                fragments.add(target)
                pending.append(target)
    return sorted(fragments)


def fragment_order(fragments, includes):
    """Fragments after every fragment that includes them, so each one sees all of its include sites."""
    remaining = sorted(fragments)
    parents = {path: {p for p in fragments if p != path and any(t == path for t, _ in includes[p])}
               for path in fragments}
    done = set()
    order = []
    while remaining:
        ready = [p for p in remaining if parents[p] <= done] or remaining[:1]
        for path in ready:
            order.append(path)
            done.add(path)
            remaining.remove(path)
    return order


def fog_programs(by_path, includes, texts):
    """Program files that need the legacy gl_FogFragCoord varying declared: both stages of a pair get it when either
    stage (or anything it includes) uses it, so the vertex write reaches the fragment read."""
    code = {path: COMMENTS.sub(' ', texts[path]) for path in by_path}

    def uses(path, seen):
        if path in seen or path not in by_path:
            return False
        seen.add(path)
        return bool(FOG_COORD.search(code[path])) or any(uses(target, seen) for target, _ in includes[path])

    programs = [path for path, c in by_path.items() if c.file_stage in ('vertex', 'fragment')]
    if any(FOG_COORD_DECLARED.search(text) for text in code.values()):
        return set()
    pairs = {path.rsplit('.', 1)[0] for path in programs if uses(path, set())}
    return {path for path in programs if path.rsplit('.', 1)[0] in pairs}


LINE_COMMENTS = re.compile(r'//[^\n]*')
PACKER_DEF = re.compile(r'\bfloat\s+(\w+)\s*\([^)]*\)\s*\{')
PACKER_SCALE = re.compile(r'\b(?:255|256|65535)(?:\.0*)?\b')
TARGET_LIST = re.compile(r'/\*\s*(?:RENDERTARGETS\s*:\s*([\d,\s]+)|DRAWBUFFERS\s*:\s*(\d+))')
FRAG_WRITE = re.compile(r'\bgl_FragData\s*\[\s*(\d+)\s*\][^=;]*=(?!=)([^;]*);')
LAYOUT_OUT = re.compile(r'\blayout\s*\(\s*location\s*=\s*(\d+)\s*\)\s*out\s+\w+\s+(\w+)\s*;')


def packed_targets(by_path, includes, texts):
    """Colour targets that gbuffers programs fill with bit-packed values (a float function that floors and scales by
    255/256/65535). Averaging their MSAA samples mixes the packed fields, so they resolve from sample 0."""
    code = {path: LINE_COMMENTS.sub(' ', texts[path]) for path in by_path}
    packers = set()
    for text in code.values():
        for match in PACKER_DEF.finditer(text):
            depth, end = 1, match.end()
            while end < len(text) and depth:
                depth += {'{': 1, '}': -1}.get(text[end], 0)
                end += 1
            body = text[match.end():end]
            if 'floor' in body and PACKER_SCALE.search(body):
                packers.add(match.group(1))
    if not packers:
        return set()
    calls = re.compile(r'\b(?:' + '|'.join(sorted(packers)) + r')\s*\(')
    reach = set()

    def walk(path):
        if path in reach or path not in by_path:
            return
        reach.add(path)
        for target, _ in includes[path]:
            walk(target)

    for path in by_path:
        if Path(path).name.startswith('gbuffers_'):
            walk(path)
    found = set()
    for path in reach:
        text = code[path]
        listed = TARGET_LIST.search(text)
        order = None
        if listed:
            order = [int(t) for t in re.findall(r'\d+', listed.group(1))] if listed.group(1) else \
                [int(c) for c in listed.group(2)]
        for match in FRAG_WRITE.finditer(text):
            if calls.search(match.group(2)):
                index = int(match.group(1))
                found.add(order[index] if order and index < len(order) else index)
        found |= packed_layout_outputs(text, calls)
    return found


def packed_layout_outputs(text, calls):
    """Targets of `layout(location = N) out` variables assigned a packer call, whole or per component. N indexes
    every RENDERTARGETS list of the file, since packs pick one list per #if branch."""
    lists = [[int(t) for t in re.findall(r'\d+', m.group(1))] if m.group(1) else [int(c) for c in m.group(2)]
             for m in TARGET_LIST.finditer(text)]
    found = set()
    for declared in LAYOUT_OUT.finditer(text):
        location, name = int(declared.group(1)), declared.group(2)
        writes = re.finditer(rf'\b{re.escape(name)}\s*(?:\.\s*[xyzwrgba]+\s*)?=(?!=)([^;]*);', text)
        if not any(calls.search(write.group(1)) for write in writes):
            continue
        targets = {order[location] for order in lists if location < len(order)}
        found |= targets or {location}
    return found


def voxelizing(geometry):
    """Iris assumes a pack voxelizes when its shadow program has a geometry stage, and then turns shadow culling off
    unless the pack sets shadow.culling itself. The header line comes first, so any shadow.culling line the pack writes
    (even under a condition) overrides it."""
    return any(vertex_path.stem == 'shadow' for vertex_path, _ in geometry)


def convert_pack(source, out, drop=None, standins=True):
    """Converts the pack at source into out. Returns (summary dict, report entries)."""
    base, shaders = find_shaders(source)
    out = Path(out)
    drop = [DH_DROP] + [pattern for pattern in (DEFAULT_DROP if drop is None else drop) if pattern != DH_DROP]
    properties = shaders / 'shaders.properties'
    properties_text = read_text(properties) if properties.exists() else ''
    renames = volume_renames(properties_text)
    properties_text = rename_volume_lines(properties_text, renames)
    uniforms, textures, images = custom_names(properties_text)
    ctx = PackContext(uniforms, textures, standins, images, volume_textures(properties_text))
    files = sorted(p for p in base.rglob('*') if p.is_file() and '.git' not in p.relative_to(base).parts)
    sources = []
    copies = []
    dropped = []
    geometry = []
    for path in files:
        relative = path.relative_to(base)
        suffix = path.suffix
        if suffix in PROGRAM_SUFFIXES and any(fnmatch(path.stem, pattern) for pattern in drop):
            dropped.append(relative.as_posix())
            continue
        if suffix in PROGRAM_SUFFIXES and PROGRAM_SUFFIXES[suffix] == 'geometry' and path.with_suffix('.vsh').is_file():
            geometry.append((relative.with_suffix('.vsh'), read_text(path)))
            continue
        if suffix in PROGRAM_SUFFIXES and PROGRAM_SUFFIXES[suffix] in ('geometry', 'tess'):
            ctx.report(relative.as_posix(), 1, 'api-gap', f'{PROGRAM_SUFFIXES[suffix]} programs are not supported')
            continue
        if suffix in PROGRAM_SUFFIXES or suffix in INCLUDE_SUFFIXES:
            stage = PROGRAM_SUFFIXES.get(suffix)
            stage = stage if stage in ('vertex', 'fragment', 'compute') else None
            sources.append((relative, read_text(path), stage))
        else:
            copies.append((relative, path))
    for vertex_path, geometry_text in geometry:
        for index, (relative, text, stage) in enumerate(sources):
            if relative == vertex_path:
                folded = fold_geometry(text, geometry_text, vertex_path.with_suffix('.gsh').as_posix(), ctx.report)
                if folded is not None:
                    sources[index] = (relative, folded, stage)
    texts = join_split_sampler_calls({relative.as_posix(): text for relative, text, _ in sources})
    shaders_prefix = shaders.relative_to(base).as_posix()
    shaders_prefix = '' if shaders_prefix == '.' else shaders_prefix + '/'

    def resolve(from_path, name):
        if name.startswith('/'):
            return shaders_prefix + name.lstrip('/')
        return (Path(from_path).parent / name).as_posix()

    texts = volume_reads(texts, renames, resolve)
    texts = specialize(texts, textures | set(images.values()) | set(images), ctx.report)
    converters = [(relative, FileConverter(ctx, relative.as_posix(), texts[relative.as_posix()], stage))
                  for relative, _, stage in sources]

    ctx.resolve_include = resolve
    by_path = {relative.as_posix(): converter for relative, converter in converters}
    includes = {path: [(resolve(path, name), body) for name, body in c.include_targets()] for path, c in by_path.items()}
    unused = drop_unreachable(by_path, includes, drop)
    orphaned = drop_orphaned(by_path, includes, unused)
    unused = sorted(set(unused) | set(orphaned))
    converters = [(relative, c) for relative, c in converters if relative.as_posix() not in unused]
    by_path = {path: c for path, c in by_path.items() if path not in unused}
    for path in body_fragments(by_path, includes):
        by_path[path].fragment = True
        ctx.fragments[path] = by_path[path]
    for relative in dropped:
        note = DH_DROP_NOTE if fnmatch(Path(relative).stem, DH_DROP) else 'program matches a drop pattern; not converted'
        ctx.report(relative, 1, 'dropped', note)
    for relative in unused:
        if relative in orphaned:
            ctx.report(relative, 1, 'dropped', 'include used only by dropped files')
        else:
            ctx.report(relative, 1, 'dropped', 'include matches a drop pattern and only dropped programs use it')
    dropped += unused
    for _, converter in converters:
        if converter.fragment:
            converter.collect_fragment()
        else:
            converter.walk_global('collect')
    ctx.find_empty_macros()
    ctx.find_deferred_inits()
    for _, converter in converters:
        strip_empty_calls(converter.tl, ctx.empty_macros)
    for _, converter in converters:
        if not converter.fragment:
            converter.walk_global('convert')
    for path in fragment_order(ctx.fragments, includes):
        by_path[path].walk_fragment(ctx.fragment_scope(path))
    for _, converter in converters:
        converter.convert_macros()
    fog = fog_programs(by_path, includes, texts)
    packed = packed_targets(by_path, includes, texts)
    for relative, converter in converters:
        prefix = FOG_VARYING if relative.as_posix() in fog else ''
        write_text(out / output_name(relative), prefix + converter.output())
    for relative, path in copies:
        text_kinds = {'.properties'}
        if relative.name == 'shaders.properties' and relative.parent.name == 'shaders':
            text = rename_volume_lines(read_text(path), renames)
            header = ''
            if not re.search(r'^\s*nwr\.format\s*=', text, re.M):
                header += 'nwr.format = 2\n'
            if not re.search(r'^\s*nwr\.weather\s*=', text, re.M):
                header += 'nwr.weather = columns\n'
            for target in sorted(packed):
                if not re.search(rf'^\s*nwr\.colortex{target}\.resolve\s*=', text, re.M):
                    header += f'nwr.colortex{target}.resolve = sample0\n'
            if voxelizing(geometry):
                header += 'shadow.culling = false\n'
            text = header + host_macros(text)
            write_text(out / relative, text)
            ctx.report(relative.as_posix(), 1, 'hand',
                       'set nwr.merge.target (and nwr.msaa if the pack has its own AA) by hand')
            continue
        if relative.name == 'dimension.properties':
            text = read_text(path)
            for java, bedrock in DIMENSION_IDS.items():
                text = text.replace(java, bedrock)
            write_text(out / relative, text)
            continue
        if relative.name == 'block.properties':
            text, warnings = map_block_properties(read_text(path))
            write_text(out / relative, text)
            for line, warning in warnings:
                ctx.report(relative.as_posix(), line, 'block', warning)
            continue
        if relative.suffix in text_kinds:
            write_text(out / relative, read_text(path))
            continue
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    entries = sorted(ctx.entries, key=lambda e: (CATEGORY_ORDER.index(e.category)
                                                 if e.category in CATEGORY_ORDER else 99, e.file, e.line, e.message))
    counts = Counter(e.category for e in entries)
    summary = {
        'files': len(converters),
        'copied': len(copies),
        'dropped': len(dropped),
        'counts': dict(sorted(counts.items())),
        'blocking': sum(counts[c] for c in BLOCKING),
    }
    return summary, entries


def format_report(summary, entries):
    lines = [f"iris-port report: {summary['files']} GLSL files converted, {summary['copied']} copied, "
             f"{summary['dropped']} dropped",
             'counts: ' + ', '.join(f'{k}={v}' for k, v in summary['counts'].items()),
             'status: ' + ('COMPLETE' if summary['blocking'] == 0 else
                           f"INCOMPLETE ({summary['blocking']} blocking entries: fix these before using the output)"),
             '']
    category = None
    for e in entries:
        if e.category != category:
            category = e.category
            lines.append(f'[{category}]')
        lines.append(f'{e.file}:{e.line}: {e.message}')
    gaps = Counter(re.sub(r'\s+\(.*', '', e.message) for e in entries if e.category == 'api-gap')
    if gaps:
        lines.append('')
        lines.append('[api-gap summary]')
        for message, count in sorted(gaps.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f'{count:5d} x {message}')
    return '\n'.join(lines) + '\n'
