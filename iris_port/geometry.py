# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""Folds a pass-through geometry program into its vertex program.

Format 2 has no geometry stage. A geometry program that takes triangles and emits K separate triangles per input
triangle (each EndPrimitive closes one three-vertex strip) is run inside the vertex program instead: the vertex program
keeps its own outputs as globals, then runs the geometry body with every input array reading the vertex's own value.
The loader draws the program K times with NWR_EMISSION = 0..K-1 (the /* NWR_EMISSIONS: K */ directive); draw k keeps
the outputs of the k-th primitive the body emits. A vertex that does not emit primitive k gets a NaN position, which
discards its triangle, so a per-primitive test that rejects the triangle when any vertex fails it keeps its meaning.
Tests that compare different vertices of the triangle see three equal values and pass.
"""
import re

QUALIFIERS = r'(?:(?:flat|smooth|noperspective|centroid|invariant)\s+)*'
INPUT_LAYOUT = re.compile(r'\blayout\s*\(\s*triangles\s*\)\s*in\s*;')
OUTPUT_LAYOUT = re.compile(r'\blayout\s*\(\s*triangle_strip\s*,\s*max_vertices\s*=\s*(\d+)\s*\)\s*out\s*;')
ANY_LAYOUT = re.compile(r'^\s*layout\s*\([^)]*\)\s*(?:in|out)\s*;\s*$', re.M)
INPUT_DECL = re.compile(r'^\s*' + QUALIFIERS + r'(?:in|varying\s+in)\s+(\w+)\s+(\w+)\s*\[\s*\d*\s*\]\s*;\s*$', re.M)
OUTPUT_DECL = re.compile(r'^\s*' + QUALIFIERS + r'(?:out|varying\s+out)\s+(\w+)\s+(\w+)\s*;\s*$', re.M)
VERTEX_OUTPUT = re.compile(r'^(\s*)' + QUALIFIERS + r'(?:out|varying)\s+(?=\w+\s+\w+(?:\s*\[[^\]]*\])?\s*(?:=[^;]*)?;)', re.M)
UNSUPPORTED = ('gl_PrimitiveIDIn', 'gl_InvocationID', 'EmitStreamVertex', 'EndStreamPrimitive', 'gl_PrimitiveID')


def strip_comments(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*', lambda m: '\n' * m.group(0).count('\n'), text, flags=re.S)


def drop_subscripts(text, names):
    if not names:
        return text
    pattern = re.compile(r'\b(' + '|'.join(sorted(names, key=len, reverse=True)) + r')\s*\[')
    out = []
    at = 0
    while True:
        m = pattern.search(text, at)
        if not m:
            out.append(text[at:])
            return ''.join(out)
        out.append(text[at:m.start()] + m.group(1))
        depth = 1
        i = m.end()
        while i < len(text) and depth:
            depth += {'[': 1, ']': -1}.get(text[i], 0)
            i += 1
        at = i


def uniform_names(text):
    return set(re.findall(r'\buniform\s+\w+\s+(\w+)', text))


def fold_geometry(vertex, geometry, path, report):
    """Returns the vertex program with the geometry program folded in, or None (reported) when it has another shape."""
    code = strip_comments(geometry)
    limit = OUTPUT_LAYOUT.search(code)
    primitives = len(re.findall(r'\bEndPrimitive\s*\(\s*\)', code))
    problem = None
    if not INPUT_LAYOUT.search(code) or not limit:
        problem = 'only triangles in, triangle_strip out can be folded'
    elif not primitives or int(limit.group(1)) != 3 * primitives:
        problem = f'max_vertices {limit.group(1)} is not three per EndPrimitive ({primitives})'
    elif any(re.search(r'\b' + name + r'\b', code) for name in UNSUPPORTED):
        problem = 'uses a per-primitive built-in or streams'
    elif re.search(r'\b(?:in|out)\s+\w+\s*\{', code):
        problem = 'interface blocks cannot be folded'
    elif not re.search(r'\bvoid\s+main\s*\(', vertex):
        problem = 'the vertex program has no main'
    if problem:
        report(path, 1, 'api-gap', f'geometry program not folded into the vertex program: {problem}')
        return None
    inputs = [m.group(2) for m in INPUT_DECL.finditer(code)]
    outputs = [(m.group(1), m.group(2)) for m in OUTPUT_DECL.finditer(code)]
    if re.search(r'\b(?:' + '|'.join(inputs + ['gl_in']) + r')\s*\[\s*\d+\s*\]', code):
        report(path, 1, 'note', 'geometry program folded: reads of other vertices see the vertex\'s own value')
    known_uniforms = uniform_names(strip_comments(vertex))
    includes = set(re.findall(r'^\s*#\s*include\s*(\S+)', vertex, re.M))

    def keep_line(line):
        if re.match(r'^\s*#\s*(?:version|extension)\b', line):
            return False
        include = re.match(r'^\s*#\s*include\s*(\S+)', line)
        if include and include.group(1) in includes:
            return False
        declared = re.match(r'^\s*uniform\s+\w+\s+(\w+)\s*;', line)
        return not (declared and declared.group(1) in known_uniforms)

    body = ANY_LAYOUT.sub('', code)
    body = INPUT_DECL.sub('', body)
    body = '\n'.join(line for line in body.split('\n') if keep_line(line))
    body = re.sub(r'\bgl_in\s*\[[^\]]*\]\s*\.\s*gl_Position\b', 'nwrVertexPosition', body)
    body = drop_subscripts(body, inputs)
    body = re.sub(r'\bEmitVertex\s*\(\s*\)', 'nwr_emit_vertex()', body)
    body = re.sub(r'\bEndPrimitive\s*\(\s*\)', 'nwr_end_primitive()', body)
    body = re.sub(r'\bvoid\s+main\s*\(', 'void nwr_geometry_main(', body)

    vertex = re.sub(r'\bvoid\s+main\s*\(', 'void nwr_vertex_main(', vertex)
    vertex = VERTEX_OUTPUT.sub(lambda m: m.group(1), vertex)
    keep = ''.join(f'{t} nwrKeep_{n};\n' for t, n in outputs)
    save = ''.join(f'        nwrKeep_{n} = {n};\n' for _, n in outputs)
    restore = ''.join(f'    {n} = nwrKeep_{n};\n' for _, n in outputs)
    # The loader resolves pack #if/#ifndef before it prepends NWR_EMISSION, so a guarded default would override it.
    default = '' if primitives > 1 else '#define NWR_EMISSION 0\n'
    glue = (
        f'\n/* NWR_EMISSIONS: {primitives} */\n'
        f'{default}'
        'vec4 nwrVertexPosition;\nvec4 nwrKeepPosition;\nint nwrPrimitive = 0;\nbool nwrEmitted = false;\n'
        f'{keep}'
        'void nwr_emit_vertex();\nvoid nwr_end_primitive();\n'
    )
    functions = (
        '\nvoid nwr_emit_vertex() {\n'
        '    if (nwrPrimitive == NWR_EMISSION && !nwrEmitted) {\n'
        '        nwrEmitted = true;\n'
        f'{save}'
        '        nwrKeepPosition = gl_Position;\n'
        '    }\n'
        '}\n'
        'void nwr_end_primitive() {\n    nwrPrimitive++;\n}\n'
    )
    main = (
        '\nvoid main() {\n'
        '    nwr_vertex_main();\n'
        '    nwrVertexPosition = gl_Position;\n'
        '    nwr_geometry_main();\n'
        f'{restore}'
        '    gl_Position = nwrEmitted ? nwrKeepPosition : vec4(uintBitsToFloat(0x7fc00000u));\n'
        '}\n'
    )
    return vertex.rstrip('\n') + '\n' + glue + body.rstrip('\n') + '\n' + functions + main
