# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""Functions that take samplers or images as parameters, specialised per call-site texture or image.

Format 2 reads a texture through a per-name accessor (`texture(t, p)` pastes `t` into `nwrTexture_t`), so a texture
cannot travel through a function parameter. Before conversion, every function with sampler parameters is replaced by
one copy per distinct tuple of textures its callers pass (`blur(colortex2, uv)` calls `blur_nwr_colortex2(uv)`), with
the sampler parameters removed and their uses renamed to the texture. A sampler parameter passed on to another such
function is followed through, so nested calls get their own copies.

Each copy stays where the original was, inside `#ifdef NWR_CLONE_<copy>`, and each program file (or a file no other file
includes) defines the flags of the copies its own includes call. A copy that names `colortex0` is then compiled only
into programs that call it, so a program that includes the library without calling it does not read the texture (the
loader binds every texture a program names, and refuses a gbuffers program that names a target it writes). A copy of a
texture declared only under `#if` conditions also carries those conditions, so it is not compiled where the texture is
not declared and the call that needs it is inactive.
"""
import posixpath
from collections import defaultdict
from dataclasses import dataclass, field
import re

from . import api
from .lexer import COMMENT, ID, NL, OP, PP, branch_depth, is_trivia, tokenize
from .tables import GLSL_QUALIFIERS
from .types import SAMPLER_TYPES

CLONE_INFIX = '_nwr_'
CLONE_FLAG = 'NWR_CLONE_'
PROGRAM_SUFFIXES = ('.vsh', '.fsh', '.csh', '.gsh', '.tcs', '.tes')
INCLUDE = re.compile(r'#\s*include\s+"([^"]+)"')
UNIFORM_SAMPLER = re.compile(r'\buniform\s+(?:(?:lowp|mediump|highp)\s+)?([iu]?sampler\w*)\s+([\w\s,\[\]]+);')
OBJECT_MACRO = re.compile(r'#\s*define\s+(\w+)\s+(\w+)\s*$')
CONDITIONAL = re.compile(r'#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)$', re.S)


def is_sampler_type(name):
    return name in SAMPLER_TYPES


@dataclass
class Param:
    name: str
    type_name: str
    start: int
    end: int
    comma: int = None
    array: bool = False

    @property
    def is_sampler(self):
        return is_sampler_type(self.type_name)


@dataclass
class Definition:
    file: str
    name: str
    name_index: int
    start: int
    end: int
    open_paren: int
    close_paren: int
    params: list
    body: bool

    @property
    def positions(self):
        return tuple(k for k, p in enumerate(self.params) if p.is_sampler)

    @property
    def sampler_types(self):
        return tuple(self.params[p].type_name for p in self.positions)

    @property
    def key(self):
        return (self.name, len(self.params), self.positions, self.sampler_types)


@dataclass
class Call:
    file: str
    name_index: int
    key: tuple
    bindings: list
    drop: set = field(default_factory=set)
    enclosing: Definition = None


class SourceFile:
    def __init__(self, path, text):
        self.path = path
        self.tokens = tokenize(text)
        self.sig = [i for i, t in enumerate(self.tokens) if not is_trivia(t) and t.kind != PP]
        self.next_sig = {a: b for a, b in zip(self.sig, self.sig[1:])}
        self.prev_sig = {b: a for a, b in zip(self.sig, self.sig[1:])}
        self.position = {i: k for k, i in enumerate(self.sig)}
        self.definitions = []
        self.functions = []
        self.calls = {}
        self.balanced = True

    def text(self, i):
        return self.tokens[i].text

    def line(self, i):
        return self.tokens[i].line

    def match_paren(self, i):
        depth = 0
        for j in self.sig[self.position[i]:]:
            t = self.text(j)
            if t in '([':
                depth += 1
            elif t in ')]':
                depth -= 1
                if depth == 0:
                    return j
        return None

    def split_args(self, open_paren, close_paren):
        args = []
        current = []
        depth = 0
        for j in self.sig[self.position[open_paren] + 1:]:
            if j >= close_paren:
                break
            t = self.text(j)
            if t in '([{':
                depth += 1
            elif t in ')]}':
                depth -= 1
            if t == ',' and depth == 0:
                args.append((current, j))
                current = []
                continue
            current.append(j)
        if current or args:
            args.append((current, None))
        return args


def brace_depths(src):
    """Brace depth after each significant token; blocks opened in each branch of an #if chain count once."""
    depths = {}
    depth = 0
    branches = []
    for i, t in enumerate(src.tokens):
        if t.kind == PP:
            depth = branch_depth(branches, t.text, depth)
            continue
        if t.kind == OP and t.text == '{':
            depth += 1
        elif t.kind == OP and t.text == '}':
            depth -= 1
        depths[i] = depth
    return depths


def scan_definitions(src):
    depths = brace_depths(src)
    depth = 0
    for k, i in enumerate(src.sig):
        t = src.tokens[i]
        depth = depths.get(i, depth)
        if depth < 0:
            src.balanced = False
            return
        if depth != 0 or t.kind != ID or k < 1 or src.next_sig.get(i) is None or src.text(src.next_sig[i]) != '(':
            continue
        ret = src.sig[k - 1]
        if src.tokens[ret].kind != ID or src.text(ret) in ('return', 'else', 'layout'):
            continue
        open_paren = src.next_sig[i]
        close_paren = src.match_paren(open_paren)
        if close_paren is None:
            continue
        after = src.next_sig.get(close_paren)
        if after is None or src.text(after) not in ('{', ';'):
            continue
        params = []
        for tokens, comma in src.split_args(open_paren, close_paren):
            words = [j for j in tokens if src.text(j) not in GLSL_QUALIFIERS]
            if not words or (len(words) == 1 and src.text(words[0]) == 'void'):
                continue
            name = src.text(words[1]) if len(words) > 1 and src.tokens[words[1]].kind == ID else None
            params.append(Param(name, src.text(words[0]), tokens[0], tokens[-1], comma,
                                any(src.text(j) == '[' for j in words)))
        start = ret
        before = src.prev_sig.get(start)
        while before is not None and src.text(before) in GLSL_QUALIFIERS:
            start = before
            before = src.prev_sig.get(start)
        body = src.text(after) == '{'
        end = after
        if body:
            for j in src.sig[src.position[after]:]:
                if depths.get(j, 0) == 0:
                    end = j
                    break
        definition = Definition(src.path, t.text, i, start, end, open_paren, close_paren, params, body)
        src.functions.append(definition)
        if definition.positions:
            src.definitions.append(definition)
    if depth != 0:
        src.balanced = False


def directive_condition(text):
    """(kind, condition) of a conditional directive; condition is None when it cannot stand alone (#else, #elif)."""
    m = CONDITIONAL.match(text.replace('\\\n', ' ').strip())
    if m is None:
        return None, None
    kind, rest = m.group(1), re.sub(r'/\*.*?\*/|//.*', ' ', m.group(2), flags=re.S).strip()
    if kind == 'ifdef':
        return 'push', f'defined({rest})'
    if kind == 'ifndef':
        return 'push', f'!defined({rest})'
    if kind == 'if':
        return 'push', f'({rest})'
    if kind in ('elif', 'else'):
        return 'flip', None
    return 'pop', None


def declaration_conditions(sources):
    """Texture name -> set of guards its uniform declarations sit under; '' marks one that is always declared."""
    found = defaultdict(set)
    for src in sources:
        stack = []
        segment = []

        def flush():
            text = ''.join(segment)
            segment.clear()
            if None in stack:
                guard = ''
            else:
                guard = ' && '.join(stack)
            for m in UNIFORM_SAMPLER.finditer(text):
                for declarator in m.group(2).split(','):
                    word = re.match(r'\s*(\w+)', declarator)
                    if word:
                        found[word.group(1)].add(guard)
        for t in src.tokens:
            if t.kind == COMMENT:
                continue
            if t.kind != PP:
                segment.append(t.text)
                continue
            flush()
            kind, condition = directive_condition(t.text)
            if kind == 'push':
                stack.append(condition)
            elif kind == 'flip' and stack:
                stack[-1] = None
            elif kind == 'pop' and stack:
                stack.pop()
        flush()
    return found


def known_samplers(sources, extra):
    names = set(api.TEXTURES) | set(extra)
    aliases = {}
    for src in sources:
        text = ''.join(t.text for t in src.tokens if t.kind != COMMENT)
        for m in UNIFORM_SAMPLER.finditer(text):
            for declarator in m.group(2).split(','):
                word = re.match(r'\s*(\w+)', declarator)
                if word:
                    names.add(word.group(1))
        for t in src.tokens:
            if t.kind == PP:
                m = OBJECT_MACRO.match(t.text.strip())
                if m:
                    aliases[m.group(1)] = m.group(2)
    for alias, target in aliases.items():
        seen = set()
        while target in aliases and target not in seen:
            seen.add(target)
            target = aliases[target]
        if target in names:
            names.add(alias)
    return names


def declared_types(sources):
    """Texture name -> sampler types of its uniform declarations."""
    found = defaultdict(set)
    for src in sources:
        text = ''.join(t.text for t in src.tokens if t.kind != COMMENT)
        for m in UNIFORM_SAMPLER.finditer(text):
            for declarator in m.group(2).split(','):
                word = re.match(r'\s*(\w+)', declarator)
                if word:
                    found[word.group(1)].add(m.group(1))
    return found


def drop_arguments(args, positions):
    """Token indices to leave out so the arguments (start, end, comma after) at positions go with one comma each."""
    drop = set()
    kept = [k for k in range(len(args)) if k not in positions]
    for p in positions:
        start, end, comma = args[p]
        if any(k > p for k in kept):
            drop.update(range(start, args[p + 1][0]))
        elif p > 0:
            drop.update(range(args[p - 1][2], end + 1))
        else:
            drop.update(range(start, end + 1))
    return drop


def enclosing(src, index):
    for definition in src.functions:
        if definition.body and definition.close_paren < index < definition.end:
            return definition
    return None


class Specializer:
    def __init__(self, files, samplers=(), report=None):
        self.sources = {path: SourceFile(path, text) for path, text in files.items()}
        self.report = report or (lambda file, line, category, message: None)
        for src in self.sources.values():
            scan_definitions(src)
            if not src.balanced:
                self.report(src.path, 1, 'hand', 'unbalanced braces at file level; functions taking samplers in this '
                            'file were not specialised')
                src.definitions = []
                src.functions = []
        self.groups = defaultdict(list)
        self.overloads = defaultdict(list)
        for src in self.sources.values():
            for definition in src.functions:
                self.overloads[(definition.name, len(definition.params))].append(definition)
            for definition in src.definitions:
                self.groups[definition.key].append(definition)
        self.names = {key[0] for key in self.groups}
        self.samplers = known_samplers(self.sources.values(), samplers)
        self.declared = declaration_conditions(self.sources.values())
        self.types = declared_types(self.sources.values())
        shapes = defaultdict(set)
        for key in self.groups:
            shapes[key[:3]].add(key[3])
        self.typed = {shape for shape, types in shapes.items() if len(types) > 1}
        self.instances = defaultdict(set)
        self.roots = defaultdict(set)
        self.children = defaultdict(set)
        self.tops = {}

    def run(self):
        if not self.groups:
            return {path: None for path in self.sources}
        for src in self.sources.values():
            self.scan_calls(src)
        self.solve()
        self.tops = self.flags()
        return {path: self.render_file(src) if src.definitions or src.calls or self.tops.get(path) else None
                for path, src in self.sources.items()}

    def includes(self, src):
        found = []
        root = src.path[:src.path.index('shaders/') + len('shaders/')] if 'shaders/' in src.path else ''
        for t in src.tokens:
            m = INCLUDE.match(t.text) if t.kind == PP else None
            if m is None:
                continue
            target = m.group(1)
            if target.startswith('/'):
                path = posixpath.normpath(root + target[1:])
            else:
                path = posixpath.normpath(posixpath.join(posixpath.dirname(src.path), target))
            if path in self.sources:
                found.append(path)
        return found

    def flags(self):
        needed = defaultdict(list)
        for inst, files in self.roots.items():
            order = []
            seen = set()

            def visit(node):
                if node in seen:
                    return
                seen.add(node)
                for child in sorted(self.children.get(node, ())):
                    visit(child)
                order.append(node)
            visit(inst)
            for path in files:
                for node in order:
                    if node not in needed[path]:
                        needed[path].append(node)
        graph = {path: self.includes(src) for path, src in self.sources.items()}
        included = {path for targets in graph.values() for path in targets}
        tops = {}
        for path in self.sources:
            if not path.endswith(PROGRAM_SUFFIXES) and path in included:
                continue
            closure = []
            stack = [path]
            while stack:
                current = stack.pop()
                if current in closure:
                    continue
                closure.append(current)
                stack.extend(reversed(graph.get(current, [])))
            names = []
            for current in closure:
                for key, clone in needed.get(current, ()):
                    name = self.clone_name(key, clone)
                    if name not in names:
                        names.append(name)
            if names:
                tops[path] = names
        return tops

    def scan_calls(self, src):
        headers = {d.name_index for d in src.functions}
        for i in src.sig:
            t = src.tokens[i]
            if t.kind != ID or t.text not in self.names or i in headers:
                continue
            open_paren = src.next_sig.get(i)
            if open_paren is None or src.text(open_paren) != '(':
                continue
            prev = src.prev_sig.get(i)
            if prev is not None and src.text(prev) == '.':
                continue
            close_paren = src.match_paren(open_paren)
            if close_paren is None:
                continue
            args = src.split_args(open_paren, close_paren)
            inside = enclosing(src, i)
            call = self.resolve(src, i, t.text, args, inside)
            if call is not None:
                src.calls[i] = call
        for t in src.tokens:
            if t.kind == PP and re.match(r'#\s*define\b', t.text):
                for name in self.names:
                    if re.search(rf'\b{name}\s*\(', t.text):
                        self.report(src.path, t.line, 'hand', f'macro calls {name}(), which takes a sampler; '
                                    'expand the macro by hand')

    def classify(self, src, tokens, inside):
        if len(tokens) != 1 or src.tokens[tokens[0]].kind != ID:
            return None
        name = src.text(tokens[0])
        if inside is not None and any(p.name == name and p.is_sampler for p in inside.params):
            return ('param', name)
        if name in self.samplers:
            return ('tex', name)
        return ('maybe', name)

    def resolve(self, src, i, name, args, inside):
        arity = len(args)
        candidates = [key for key in self.groups if key[0] == name and key[1] == arity]
        if not candidates:
            return None
        plain = any(not d.positions for d in self.overloads.get((name, arity), []))
        fits = []
        for key in candidates:
            bindings = [self.classify(src, args[p][0], inside) for p in key[2]]
            if any(b is None for b in bindings):
                if not plain:
                    self.report(src.path, src.line(i), 'hand', f'{name}() gets a sampler argument that is not a '
                                'texture name; pass the texture itself')
                continue
            if plain and any(b[0] == 'maybe' for b in bindings):
                continue
            fits.append((key, [(k if k != 'maybe' else 'tex', n) for k, n in bindings]))
        if not fits:
            return None
        if len(fits) > 1:
            fits = self.by_sampler_type(fits, inside) or fits
        if len(fits) > 1:
            self.report(src.path, src.line(i), 'hand', f'{name}() matches several overloads taking samplers; '
                        'the first was used')
        key, bindings = fits[0]
        call = Call(src.path, i, key, bindings, enclosing=inside)
        call.drop = drop_arguments([(tokens[0], tokens[-1], comma) for tokens, comma in args], key[2])
        return call

    def by_sampler_type(self, fits, inside):
        """Overloads that differ only in sampler type (sampler2D against sampler2DShadow) are told apart by how each
        texture argument is declared, as GLSL overload resolution does."""
        def binding_types(kind, name):
            if kind == 'param':
                return {p.type_name for p in inside.params if p.name == name}
            return self.types.get(name, set())
        return [(key, bindings) for key, bindings in fits
                if all(t in binding_types(kind, name) for t, (kind, name) in zip(key[3], bindings))]

    def concrete(self, call, binding):
        names = []
        for kind, name in call.bindings:
            if kind == 'param':
                if name not in binding:
                    return None
                names.append(binding[name])
            else:
                names.append(name)
        return tuple(names)

    def solve(self):
        pending = []
        for src in self.sources.values():
            for call in src.calls.values():
                if call.enclosing is None or not call.enclosing.positions:
                    pending.append((call, {}, None))
        by_definition = defaultdict(list)
        for src in self.sources.values():
            for call in src.calls.values():
                if call.enclosing is not None and call.enclosing.positions:
                    by_definition[call.enclosing.key].append(call)
        while pending:
            call, binding, parent = pending.pop()
            names = self.concrete(call, binding)
            if names is None:
                continue
            if parent is None:
                self.roots[(call.key, names)].add(call.file)
            else:
                self.children[parent].add((call.key, names))
            if names in self.instances[call.key]:
                continue
            self.instances[call.key].add(names)
            params = None
            for definition in self.groups[call.key]:
                params = [definition.params[p].name for p in call.key[2]]
                inner = dict(zip(params, names))
                for nested in by_definition.get(call.key, []):
                    if nested.enclosing is definition:
                        pending.append((nested, inner, (call.key, names)))

    def clone_name(self, key, names):
        name = key[0] + CLONE_INFIX + '_'.join(names)
        if key[:3] in self.typed:
            name += '_' + '_'.join(key[3])
        return name

    def render(self, src, a, b, binding, skip=(), rename=None):
        out = []
        drop = set(skip)
        for i in range(a, b + 1):
            call = src.calls.get(i)
            if call is not None:
                drop |= call.drop
        for i in range(a, b + 1):
            if i in drop:
                if src.tokens[i].kind in (NL, PP):
                    out.append(src.text(i))
                elif src.tokens[i].kind == COMMENT:
                    out.append('\n' * src.text(i).count('\n'))
                continue
            t = src.tokens[i]
            if rename is not None and i == rename[0]:
                out.append(rename[1])
                continue
            call = src.calls.get(i)
            if call is not None:
                names = self.concrete(call, binding)
                out.append(self.clone_name(call.key, names) if names is not None else t.text)
                continue
            if t.kind == ID and t.text in binding and (src.prev_sig.get(i) is None
                                                       or src.text(src.prev_sig[i]) != '.'):
                out.append(binding[t.text])
                continue
            out.append(t.text)
        return ''.join(out)

    def guard(self, name, textures):
        terms = [f'defined({CLONE_FLAG}{name})']
        for texture in textures:
            guards = self.declared.get(texture)
            if not guards or '' in guards:
                continue
            ordered = sorted(guards)
            terms.append(ordered[0] if len(ordered) == 1 else '(' + ' || '.join(f'({g})' for g in ordered) + ')')
        if len(terms) == 1:
            return f'#ifdef {CLONE_FLAG}{name}\n'
        return '#if ' + ' && '.join(terms) + '\n'

    def header_drop(self, definition):
        return drop_arguments([(p.start, p.end, p.comma) for p in definition.params], definition.positions)

    def render_file(self, src):
        out = []
        starts = {d.start: d for d in src.definitions}
        i = 0
        n = len(src.tokens)
        while i < n:
            definition = starts.get(i)
            if definition is None:
                j = i
                while j < n and j not in starts:
                    j += 1
                out.append(self.render(src, i, j - 1, {}))
                i = j
                continue
            for p in definition.params:
                if p.is_sampler and p.array:
                    self.report(src.path, src.line(definition.name_index), 'hand',
                                f'{definition.name}() takes an array of samplers; split it by hand')
            names_set = sorted(self.instances.get(definition.key, ()))
            if not names_set:
                text = ''.join(src.text(k) for k in range(definition.start, definition.end + 1))
                out.append('\n' * text.count('\n'))
                self.report(src.path, src.line(definition.name_index), 'note',
                            f'{definition.name}() takes a sampler and no call passes one; left out')
            else:
                drop = self.header_drop(definition)
                clones = []
                for names in names_set:
                    binding = {definition.params[p].name: tex for p, tex in zip(definition.positions, names)
                               if definition.params[p].name}
                    name = self.clone_name(definition.key, names)
                    clones.append(self.guard(name, names)
                                  + self.render(src, definition.start, definition.end, binding, drop,
                                                (definition.name_index, name))
                                  + '\n#endif')
                out.append('\n'.join(clones))
            i = definition.end + 1
        flags = ''.join(f'#define {CLONE_FLAG}{name}\n' for name in self.tops.get(src.path, ()))
        return flags + ''.join(out)


def specialize(files, samplers=(), report=None):
    """files: {path: GLSL text}. Returns {path: new text}; files without sampler functions or calls are unchanged."""
    results = Specializer(files, samplers, report).run()
    return {path: (text if results.get(path) is None else results[path]) for path, text in files.items()}
