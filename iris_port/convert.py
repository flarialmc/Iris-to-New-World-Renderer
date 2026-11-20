# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""GLSL program and include conversion: declarations, statements, macros."""
from collections import defaultdict
from dataclasses import dataclass, field
import re

from . import api
from .expr import (ASSIGN, ParseError, Parser, TokenList, constructor_rewrite, initializer_braces, render,
                   significant_range)
from .lexer import COMMENT, ID, NUM, OP, PP, branch_depth, directive_word, is_trivia, tokenize
from .ppexpr import signed_condition
from .tables import (ATOMICS, CALL_RENAMES, GLSL_QUALIFIERS, HLSL_RESERVED, PRECISION, STATEMENT_KEYWORDS,
                     UNSUPPORTED_FUNCTIONS, builtin_result)
from .types import (GLSL_TO_HLSL_TYPE, GLSL_TYPES, NONMAT, UNKNOWN, Ty, array, parse_type, sampler, struct)

BLOCKING = {'untyped-product', 'untyped-constructor', 'parse'}

VERTEX_MACROS = {'VERTEX', 'VERTEX_SHADER', 'VSH', 'STAGE_VERTEX', 'NWR_VERTEX'}
FRAGMENT_MACROS = {'FRAGMENT', 'FRAGMENT_SHADER', 'FSH', 'STAGE_FRAGMENT', 'NWR_FRAGMENT'}
HOST_MACROS = {'IS_IRIS': 'IS_NWR'}
HOST_MACRO = re.compile(r'\b(' + '|'.join(HOST_MACROS) + r')\b')

STORAGE = {'in', 'out', 'varying', 'attribute', 'uniform', 'buffer', 'inout'}
MACRO_BOUND_DEPTH = 4
UNROLL_TRIP_LIMIT = 16
TRIP_NUMBER = re.compile(r'((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[uUfF]?$')
COMPARE_OPERATORS = {'lessThan': '<', 'lessThanEqual': '<=', 'greaterThan': '>', 'greaterThanEqual': '>=',
                     'equal': '==', 'notEqual': '!='}
MACRO_DEPTH = 8
LOCAL_SUFFIX = '_nwrLocal'
OPENERS = {'(', '[', '{'}
CLOSERS = {')', ']', '}'}
INTERPOLATION = {'flat': '_FLAT', 'noperspective': '_NOPERSPECTIVE', 'smooth': '', 'centroid': ''}


def host_macros(text):
    return HOST_MACRO.sub(lambda m: HOST_MACROS[m.group(1)], text)


def converted_include(text):
    """The #include line naming the file the pack converter writes for its target (pack.output_name)."""
    text = re.sub(r'\.(?:glsl|inc)([">])', r'.hlsli\1', text)
    return re.sub(r'\.(vsh|fsh|csh)([">])', r'.\1.hlsl\2', text)


@dataclass
class Entry:
    file: str
    line: int
    category: str
    message: str


@dataclass
class Function:
    name: str
    ret: Ty
    params: list


@dataclass
class Macro:
    name: str
    params: list
    tl: TokenList
    key: tuple
    file: str
    line: int
    header: str


@dataclass
class Declaration:
    qualifiers: list
    layout: dict
    type_name: str
    type_index: int
    type_array: int
    declarators: list = field(default_factory=list)
    inner_dims: dict = field(default_factory=dict)


def blank_directive_lines(comment):
    lines = comment.split('\n')
    for i in range(1, len(lines)):
        if re.match(r'\s*#', lines[i]):
            lines[i] = '*/' if i == len(lines) - 1 else ''
    return '\n'.join(lines)


def blank_directive_comments(text):
    """Block comments drop their preprocessor lines: fxc reads a directive line inside a comment, so `#endif*/` would
    swallow the comment's end. The rest of the comment stays, since Iris reads `const int colortexNFormat` and the other
    const directives from comments too (a format block guarded by an `#ifdef` inside the comment keeps its formats)."""
    tokens = tokenize(text)
    if not any(t.kind == COMMENT and re.search(r'(?m)^\s*#', t.text) for t in tokens):
        return text
    return ''.join(blank_directive_lines(t.text) if t.kind == COMMENT and t.text.startswith('/*')
                   and re.search(r'(?m)^\s*#', t.text) else t.text for t in tokens)


def combine(types):
    types = [t for t in types if t is not None]
    if not types:
        return UNKNOWN
    first = types[0]
    if all(t == first for t in types):
        return first
    if all(t.kind in ('scalar', 'vec', 'nonmat') for t in types):
        return NONMAT
    if all(t.is_matrix and t.cols == first.cols and t.rows == first.rows for t in types):
        return first
    if all(t.kind == 'array' and t.element == first.element for t in types):
        return array(first.element)
    return UNKNOWN


def matching(param, arg):
    if not arg.known or param.kind != arg.kind:
        return False
    if param.is_vector:
        return param.size == arg.size
    if param.is_matrix:
        return param.cols == arg.cols and param.rows == arg.rows
    if param.kind == 'struct':
        return param.name == arg.name
    return True


def macro_body(tl):
    """Significant tokens of a macro body without trailing semicolons, which some packs put in value macros."""
    sig, _ = significant_range(tl, 0, len(tl.tokens) - 1)
    while sig and tl.tokens[sig[-1]].text == ';':
        sig.pop()
    return sig


def list_call_arities(tl):
    """(name, count) for every `name((a, b, ...))`: a call whose only argument is a parenthesised list, which a macro
    can hand to a GLSL array constructor (`T[] x = T[] param;`)."""
    sig = tl.sig
    out = []
    for k in range(len(sig) - 2):
        t = tl.tokens[sig[k]]
        if t.kind != ID or tl.tokens[sig[k + 1]].text != '(' or tl.tokens[sig[k + 2]].text != '(':
            continue
        depth = 0
        count = 1
        for j in sig[k + 2:]:
            text = tl.tokens[j].text
            if text in OPENERS:
                depth += 1
            elif text in CLOSERS:
                depth -= 1
                if depth == 0:
                    break
            elif text == ',' and depth == 1:
                count += 1
        out.append((t.text, count))
    return out


def strip_empty_calls(tl, names):
    sig = tl.sig
    for k, i in enumerate(sig[:-2]):
        t = tl.tokens[i]
        if t.kind == ID and t.text in names and tl.tokens[sig[k + 1]].text == '(' and tl.tokens[sig[k + 2]].text == ')':
            tl.overrides[sig[k + 1]] = ''
            tl.overrides[sig[k + 2]] = ''


class PackContext:
    def __init__(self, custom_uniforms=None, custom_textures=None, standins=True, custom_images=None,
                 volume_textures=None):
        self.custom_images = dict(custom_images or {})
        self.buffer_views = {}
        self.struct_fields = {}
        self.called = set()
        self.functions = defaultdict(list)
        self.structs = {}
        self.globals = defaultdict(set)
        self.const_globals = set()
        self.const_values = {}
        self.stage_inputs = set()
        self.global_inits = []
        self.deferred_inits = []
        self.macros = defaultdict(list)
        self.decisions = defaultdict(set)
        self.entries = []
        self.custom_uniforms = dict(custom_uniforms or {})
        self.custom_textures = set(custom_textures or ()) | set(self.custom_images.values())
        self.volume_textures = set(volume_textures or ())
        self.standins = standins
        self.scopes = []
        self.file_globals = None
        self.eval_depth = 0
        self.negative_macros = {}
        self.seen = set()
        self.resolve_include = None
        self.fragments = {}
        self.empty_macros = set()
        self.fragment_sites = defaultdict(list)
        self.legacy_textures = set()
        self.list_calls = defaultdict(set)

    def record_list_calls(self, tl):
        for name, count in list_call_arities(tl):
            self.list_calls[name].add(count)

    def find_empty_macros(self):
        """Function-like macros without parameters; FXC's preprocessor rejects `#define F()`, so they become object-like
        macros and every `F()` loses its parentheses."""
        self.empty_macros = {name for name, macros in self.macros.items() if macros and all(m.params == [] for m in macros)}
        for macros in self.macros.values():
            for macro in macros:
                strip_empty_calls(macro.tl, self.empty_macros)

    def find_deferred_inits(self):
        """Globals whose initialiser reads a stage input. HLSL runs static initialisers before the entry wrapper copies
        the inputs into their statics, so these are assigned at the start of main(), as GLSL evaluates them."""
        inputs = self.input_names()
        reads = defaultdict(set)
        for name, names in self.global_inits:
            reads[name] |= names
        deferred = set()
        changed = True
        while changed:
            changed = False
            for name, names in reads.items():
                if name not in deferred and names & (inputs | deferred):
                    deferred.add(name)
                    changed = True
        order = []

        def visit(name, path):
            if name in order or name in path:
                return
            for dep in sorted(reads[name] & deferred, key=list(reads).index):
                visit(dep, path | {name})
            order.append(name)
        for name in reads:
            if name in deferred:
                visit(name, frozenset())
        self.deferred_inits = order

    def input_names(self):
        return self.stage_inputs | set(api.VERTEX_STATICS) | set(api.FRAGMENT_STATICS) | api.ATTRIBUTES

    def include_in_body(self, from_path, name):
        """Records the scope at a statement-level #include of a body fragment and declares the fragment's own
        top-level locals in the including block, so the code after the include sees them."""
        target = self.resolve_include(from_path, name) if self.resolve_include else None
        if target is None or target not in self.fragments or not self.scopes:
            return
        snapshot = {}
        for scope in self.scopes:
            for key, value in scope.items():
                snapshot[key] = set(value) if isinstance(value, set) else value
        self.fragment_sites[target].append(snapshot)
        for key, value in self.fragments[target].top_declarations(set()).items():
            self.scopes[-1][key] = value

    def fragment_scope(self, target):
        merged = defaultdict(set)
        for site in self.fragment_sites.get(target, []):
            for key, value in site.items():
                if isinstance(value, set):
                    merged[key] |= value
                else:
                    merged[key] = value
        return merged

    def report(self, file, line, category, message):
        key = (file, line, category, message)
        if key in self.seen:
            return
        self.seen.add(key)
        self.entries.append(Entry(file, line, category, message))

    def report_at(self, tl, index, category, message):
        self.report(tl.file, tl.line_offset + tl.tokens[index].line, category, message)

    def buffer_view(self, tl, i):
        t = tl.tokens[i].text
        if t == '.':
            j = tl.prev_sig.get(i)
            view = self.buffer_views.get(tl.tokens[j].text) if j is not None else None
            return view is not None and tl.next_text(i) == view[0]
        view = self.buffer_views.get(t)
        if view is not None:
            j = tl.next_sig.get(i)
            return j is not None and tl.tokens[j].text == '.' and tl.next_text(j) == view[0]
        j = tl.prev_sig.get(i)
        return j is not None and tl.tokens[j].text == '.' and self.buffer_view(tl, j)

    def renamer(self, tl, i):
        t = tl.tokens[i]
        if self.buffer_views and t.text in self.buffer_views and self.buffer_view(tl, i):
            return self.buffer_views[t.text][1]
        if self.buffer_views and (t.text == '.' or t.kind == ID) and self.buffer_view(tl, i):
            return ''
        if t.kind != ID:
            return t.text
        name = t.text
        if name in GLSL_TO_HLSL_TYPE:
            return GLSL_TO_HLSL_TYPE[name]
        if name in PRECISION:
            return ''
        if name in CALL_RENAMES and name not in self.functions and tl.next_text(i) == '(':
            return CALL_RENAMES[name]
        if name in self.legacy_textures and tl.next_text(i) != '(' and tl.prev_text(i) != '.':
            return api.LEGACY_TEXTURES[name]
        if name in HLSL_RESERVED:
            return name + '_'
        return name

    def shadows_uniform(self, name):
        return name in api.UNIFORM_TYPES or name in self.custom_uniforms

    def type_name(self, name):
        return parse_type(name, self.structs)

    def struct_member(self, name, member):
        members = self.structs.get(name)
        if not members or member not in members:
            return UNKNOWN
        return members[member]

    @staticmethod
    def narrow_attribute(parser, node, declared):
        """A pack may declare an API attribute narrower than the API's static (attribute vec2 mc_midTexCoord against
        float4): reads take the declared components, as GL does."""
        name = node.name
        if name not in api.ATTRIBUTES or parser.mode != 'file':
            return declared
        provided = api.builtin_type(name)
        if declared.kind in ('vec', 'scalar') and provided.kind == 'vec' and declared.size < provided.size:
            parser.tl.overrides[node.start] = f'{name}.{"xyzw"[:declared.size]}'
        return declared

    def lookup_type(self, parser, node):
        name = node.name
        for scope in reversed(self.scopes):
            if ('rename', name) in scope and parser.mode == 'file':
                parser.tl.overrides[node.start] = scope[('rename', name)]
            if name in scope:
                return combine(list(scope[name]))
        if self.file_globals is not None and name in self.file_globals:
            return self.narrow_attribute(parser, node, combine(list(self.file_globals[name])))
        if name in self.globals:
            return self.narrow_attribute(parser, node, combine(list(self.globals[name])))
        builtin = api.builtin_type(name)
        if builtin is not None:
            if name in api.UNSUPPORTED_BUILTINS and parser.mode == 'file':
                parser.report(node.start, 'api-gap', api.UNSUPPORTED_BUILTINS[name])
            return builtin
        if name == api.FOG_STRUCT:
            return struct(api.FOG_STRUCT)
        if name in api.UNSUPPORTED_BUILTINS:
            if parser.mode == 'file':
                parser.report(node.start, 'api-gap', api.UNSUPPORTED_BUILTINS[name])
            return api.UNSUPPORTED_BUILTIN_TYPES.get(name, UNKNOWN)
        if name in self.custom_uniforms:
            return self.custom_uniforms[name]
        if name in self.macros:
            return self.eval_macro(parser, name, None)
        if name in ('true', 'false'):
            return GLSL_TYPES['bool']
        return UNKNOWN

    def may_be_negative(self, name, depth=0):
        if name in self.negative_macros:
            return self.negative_macros[name]
        if depth > MACRO_DEPTH:
            return True
        self.negative_macros[name] = False
        found = False
        for macro in self.macros.get(name, ()):
            body = macro.tl.raw(0, len(macro.tl.tokens) - 1) if macro.tl.tokens else ''
            if re.search(r'-\s*\d', body) or any(t.kind == ID and self.may_be_negative(t.text, depth + 1)
                                                for t in macro.tl.tokens):
                found = True
                break
        self.negative_macros[name] = found
        return found

    def eval_macro(self, parser, name, arg_types):
        if self.eval_depth > 8:
            return UNKNOWN
        types = []
        self.eval_depth += 1
        try:
            for macro in self.macros[name]:
                if (macro.params is None) != (arg_types is None):
                    continue
                if arg_types is not None and len(macro.params) != len(arg_types):
                    continue
                bindings = dict(zip(macro.params, arg_types)) if arg_types is not None else {}
                sig = macro_body(macro.tl)
                if not sig:
                    types.append(UNKNOWN)
                    continue
                try:
                    node = Parser(macro.tl, sig, self, mode='eval', macro=macro.key, bindings=bindings).parse_full()
                    types.append(node.ty)
                except ParseError:
                    types.append(UNKNOWN)
        finally:
            self.eval_depth -= 1
        return combine(types)

    def atomic_call(self, parser, node):
        tokens = parser.tl.tokens
        before = next((tokens[k].text for k in range(node.start - 1, -1, -1) if not is_trivia(tokens[k])), ';')
        after = next((tokens[k].text for k in range(node.end + 1, len(tokens)) if not is_trivia(tokens[k])), ';')
        expected = 3 if node.name == 'atomicCompSwap' else 2
        if before not in (';', '{', '}', ')') or after != ';' or len(node.args) != expected:
            if parser.mode == 'file':
                parser.report(node.start, 'hand', f'{node.name}() result is used; HLSL atomics return it through an out '
                              'argument, so write the Interlocked call by hand')
            return
        intrinsic = ATOMICS[node.name]
        node.custom = lambda r, args=node.args: f"{intrinsic}({', '.join(r(a) for a in args)})"

    def record_decision(self, key, op_index, decision):
        if key is not None:
            self.decisions[(key, op_index)].add(decision)

    def sampler_of(self, name):
        for scope in reversed(self.scopes):
            if ('sampler', name) in scope:
                return scope[('sampler', name)]
        if name in api.TEXTURES or name in self.custom_textures:
            return api.SAMPLER_MACRO + name
        return None

    def sampler_arguments(self, parser, node):
        """A texture keeps its sampler when it is passed to a function, as a GL texture unit does: sampler parameters
        get a SamplerState companion, filled from the texture's own sampler at each call."""
        positions = set()
        if node.name in api.SAMPLE_FUNCTIONS:
            positions.add(0)
        for f in self.functions.get(node.name, ()):
            if len(f.params) == len(node.args):
                positions |= {i for i, p in enumerate(f.params) if p.kind == 'sampler'}
        for i in sorted(positions):
            arg = node.args[i]
            if arg.kind != 'ident' or arg.start != arg.end:
                continue
            sampler = self.sampler_of(arg.name)
            if node.name in api.SAMPLE_FUNCTIONS and not any(('sampler', arg.name) in scope for scope in self.scopes):
                continue
            if sampler is None:
                sampler = api.SAMPLER_FALLBACK
            current = parser.tl.overrides.get(arg.start, parser.tl.tokens[arg.start].text)
            if not current.endswith(', ' + sampler):
                parser.tl.overrides[arg.start] = f'{current}, {sampler}'

    def call_type(self, parser, node):
        name = node.name
        arg_types = [a.ty for a in node.args]
        if name in GLSL_TYPES:
            target = GLSL_TYPES[name]
            if parser.mode != 'eval':
                constructor_rewrite(parser, node, target)
            return target
        if name in self.structs:
            node.kind = 'structctor'
            if any(len(node.args) == len(f) for f in self.struct_fields.get(name, ())) and parser.mode != 'eval':
                node.custom = lambda r, args=node.args: f"nwrMake_{name}({', '.join(r(a) for a in args)})"
            return struct(name)
        if name in self.macros and any(m.params is not None for m in self.macros[name]):
            return self.eval_macro(parser, name, arg_types)
        if parser.mode == 'file':
            self.sampler_arguments(parser, node)
        if name in self.functions:
            candidates = self.functions[name]
            same = [f for f in candidates if len(f.params) == len(arg_types)]
            if same:
                exact = [f for f in same if all(matching(p, a) for p, a in zip(f.params, arg_types))]
                return combine([f.ret for f in (exact or same)])
        if name in UNSUPPORTED_FUNCTIONS and name not in self.functions:
            if parser.mode == 'file':
                parser.report(node.start, 'api-gap', f'{name}(): {UNSUPPORTED_FUNCTIONS[name]}')
            return UNKNOWN
        result = builtin_result(name, node.args, arg_types)
        if name in ATOMICS and parser.mode != 'eval' and name not in self.functions:
            self.atomic_call(parser, node)
        compare = COMPARE_OPERATORS.get(name)
        if compare and len(node.args) == 2 and parser.mode != 'eval' and name not in self.functions:
            node.custom = lambda r, a=node.args[0], b=node.args[1], o=compare: f'(({r(a)}) {o} ({r(b)}))'
        if name == 'atan' and len(node.args) == 2 and parser.mode != 'eval' and name not in self.functions:
            parser.tl.overrides[node.start] = 'atan2'
        if name == 'not' and parser.mode != 'eval':
            node.custom = lambda r, a=node.args[0]: f'(!({r(a)}))'
        if name == 'matrixCompMult' and parser.mode != 'eval' and len(node.args) == 2:
            node.custom = lambda r, a=node.args[0], b=node.args[1]: f'(({r(a)}) * ({r(b)}))'
        if name == 'mix' and len(node.args) == 3 and node.args[2].ty.base == 'bool' and node.args[2].ty.known \
                and parser.mode != 'eval':
            node.custom = lambda r, a=node.args[0], b=node.args[1], c=node.args[2]: \
                f'(({r(c)}) ? ({r(b)}) : ({r(a)}))'
        return result if result is not None else UNKNOWN


class FileConverter:
    def __init__(self, ctx, path, text, stage=None):
        self.ctx = ctx
        self.path = path
        self.tl = TokenList(tokenize(blank_directive_comments(text)), renamer=ctx.renamer)
        self.tl.file = path
        self.tl.line_offset = 0
        self.edits = {}
        self.globals = defaultdict(set)
        self.file_stage = stage
        self.pp_stack = []
        self.outputs = []
        self.varyings = []
        self.explicit_locations = set()
        self.implicit_locations = {}
        self.macro_defs = []
        self.fragment = False
        self.variant_edits = None
        self.variant_mode = False
        self.pp_before = [0]
        for t in self.tl.tokens:
            self.pp_before.append(self.pp_before[-1] + (t.kind == PP))
        ctx.record_list_calls(self.tl)

    # ---- shared helpers

    def tok(self, i):
        return self.tl.tokens[i]

    def match(self, i, open_text, close_text):
        depth = 0
        branches = []
        tokens = self.tl.tokens
        j = i - 1
        while j + 1 < len(tokens):
            j += 1
            t = tokens[j]
            if t.kind == PP:
                if not branches and directive_word(t.text) in ('elif', 'else'):
                    # The match started inside an #if branch: its siblings are alternatives, so skip to the #endif.
                    j = self.chain_end(j)
                    continue
                depth = branch_depth(branches, t.text, depth)
                continue
            if t.kind != OP:
                continue
            if t.text == open_text:
                depth += 1
            elif t.text == close_text:
                depth -= 1
                if depth == 0:
                    return j
        raise ParseError(f"unbalanced '{open_text}'")

    def chain_end(self, j):
        """Index of the #endif that closes the #if chain holding the #elif/#else at j."""
        nesting = 0
        tokens = self.tl.tokens
        for k in range(j + 1, len(tokens)):
            if tokens[k].kind != PP:
                continue
            word = directive_word(tokens[k].text)
            if word in ('if', 'ifdef', 'ifndef'):
                nesting += 1
            elif word == 'endif':
                if nesting == 0:
                    return k
                nesting -= 1
        raise ParseError("#elif/#else without #endif")

    def find_at_depth(self, i, end, texts):
        depth = 0
        for j in range(i, end + 1):
            t = self.tl.tokens[j]
            if t.kind != OP:
                continue
            if t.text in '([{' and t.text not in texts:
                depth += 1
            elif t.text in ')]}':
                if depth == 0 and t.text in texts:
                    return j
                depth -= 1
            elif depth == 0 and t.text in texts:
                return j
        return None

    def newlines(self, a, b):
        return ''.join('\n' * self.tok(i).text.count('\n') for i in range(a, b + 1))

    def edit(self, a, b, text):
        if self.variant_edits is not None:
            self.variant_edits[(a, b)] = text
            return
        self.edits[a] = (b, text)

    def spans_pp(self, a, b):
        return self.pp_before[b + 1] > self.pp_before[a]

    def report(self, index, category, message):
        self.ctx.report_at(self.tl, index, category, message)

    def stage(self):
        if self.file_stage:
            return self.file_stage
        found = None
        for frame in self.pp_stack:
            names = set(re.findall(r'\w+', frame['cond']))
            positive = frame['kind'] in ('ifdef', 'if', 'elif') and not frame['negated']
            if names & VERTEX_MACROS:
                found = 'vertex' if positive != frame['else'] else 'fragment'
            elif names & FRAGMENT_MACROS:
                found = 'fragment' if positive != frame['else'] else 'vertex'
        return found

    # ---- preprocessor

    def directive(self, i, mode):
        text = self.tok(i).text
        m = re.match(r'#\s*(\w+)\s*(.*)', text, re.S)
        if not m:
            return
        word, rest = m.group(1), m.group(2)
        if word in ('ifdef', 'ifndef', 'if'):
            negated = word == 'ifndef' or bool(re.match(r'!\s*defined', rest))
            self.pp_stack.append({'kind': word if word != 'ifndef' else 'ifdef', 'cond': rest, 'else': False,
                                  'negated': negated})
        elif word == 'elif' and self.pp_stack:
            self.pp_stack[-1].update({'kind': 'elif', 'cond': rest, 'else': False,
                                      'negated': bool(re.match(r'!\s*defined', rest))})
        elif word == 'else' and self.pp_stack:
            self.pp_stack[-1]['else'] = True
        elif word == 'endif' and self.pp_stack:
            self.pp_stack.pop()
        if mode == 'collect':
            if word == 'define':
                self.collect_macro(i, text)
            return
        if word in ('version', 'extension'):
            self.edit(i, i, '\n' * text.count('\n'))
        elif word == 'include':
            new = converted_include(text)
            if new != text:
                self.edit(i, i, new)
            target = re.match(r'\s*["<]([^">]+)[">]', rest)
            if target and self.ctx.scopes:
                self.ctx.include_in_body(self.path, target.group(1))
        elif word == 'define':
            self.macro_defs.append(i)
        elif word in ('if', 'elif'):
            condition = signed_condition(rest, self.ctx.may_be_negative)
            if condition is not None:
                self.edit(i, i, text[:m.start(2)] + condition)

    def collect_macro(self, i, text):
        m = re.match(r'(#\s*define\s+)(\w+)(\([^)]*\))?', text)
        if not m:
            return
        name = m.group(2)
        params = None
        if m.group(3) is not None:
            params = [p.strip() for p in m.group(3)[1:-1].split(',') if p.strip()]
        header = text[:m.end()]
        body_text = text[m.end():]
        tl = TokenList(tokenize(body_text), renamer=self.ctx.renamer)
        tl.file = self.path
        tl.line_offset = self.tok(i).line - 1
        macro = Macro(name, params, tl, (self.path, i), self.path, self.tok(i).line, header)
        self.ctx.macros[name].append(macro)
        self.ctx.record_list_calls(tl)

    # ---- global level

    def walk_global(self, mode):
        self.pp_stack = []
        self.ctx.scopes = []
        self.ctx.file_globals = self.globals if mode == 'convert' else None
        tokens = self.tl.tokens
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if is_trivia(t):
                i += 1
                continue
            if t.kind == PP:
                self.directive(i, mode)
                i += 1
                continue
            try:
                end = self.global_extent(i)
            except ParseError as e:
                if mode == 'convert':
                    self.report(i, 'parse', str(e))
                return
            try:
                self.global_statement(i, end, mode)
            except ParseError as e:
                if mode == 'convert':
                    self.report(i, 'parse', f'global statement: {e}')
            if mode == 'collect':
                for j in range(i, end + 1):
                    if tokens[j].kind == PP:
                        self.directive(j, mode)
            i = end + 1

    # ---- body fragments: files #included inside a function body

    def include_targets(self):
        """(include name, inside a block) for every #include of the file."""
        depth = 0
        branches = []
        out = []
        for t in self.tl.tokens:
            if t.kind == OP and t.text == '{':
                depth += 1
            elif t.kind == OP and t.text == '}':
                depth = max(0, depth - 1)
            elif t.kind == PP:
                m = re.match(r'#\s*include\s*["<]([^">]+)[">]', t.text)
                if m:
                    out.append((m.group(1), depth > 0))
                else:
                    depth = branch_depth(branches, t.text, depth)
        return out

    def collect_fragment(self):
        self.pp_stack = []
        for i, t in enumerate(self.tl.tokens):
            if t.kind == PP:
                self.directive(i, 'collect')

    def top_declarations(self, seen):
        """Names the fragment declares outside any block of its own, with their types."""
        found = {}
        if self.path in seen:
            return found
        seen.add(self.path)
        tokens = self.tl.tokens
        start = 0
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.kind == PP:
                m = re.match(r'#\s*include\s*["<]([^">]+)[">]', t.text)
                target = self.ctx.resolve_include(self.path, m.group(1)) if m and self.ctx.resolve_include else None
                if target in self.ctx.fragments:
                    found.update(self.ctx.fragments[target].top_declarations(seen))
            elif t.kind == OP and t.text == '{':
                try:
                    i = self.match(i, '{', '}')
                except ParseError:
                    return found
                start = i + 1
            elif t.kind == OP and t.text == ';':
                sig, _ = significant_range(self.tl, start, i)
                if sig and self.tok(sig[0]).text not in STATEMENT_KEYWORDS and self.is_declaration(sig):
                    decl = self.parse_declaration(sig)
                    for name_k, array_len, _ in decl.declarators if decl else ():
                        found[self.tok(sig[name_k]).text] = {self.declared_type(decl, sig, array_len)}
                start = i + 1
            i += 1
        return found

    def walk_fragment(self, scope):
        self.pp_stack = []
        self.ctx.file_globals = self.globals
        self.ctx.scopes = [scope]
        try:
            self.walk_block(0, len(self.tl.tokens) - 1)
        except ParseError as e:
            self.report(0, 'parse', f'body fragment: {e}')
        finally:
            self.ctx.scopes = []

    def global_extent(self, i):
        tokens = self.tl.tokens
        depth = 0
        saw_assign = False
        previous = None
        j = i
        while j < len(tokens):
            t = tokens[j]
            if t.kind == OP:
                if t.text in '([':
                    depth += 1
                elif t.text in ')]':
                    depth -= 1
                elif t.text == '=' and depth == 0:
                    saw_assign = True
                elif t.text == ';' and depth == 0:
                    return j
                elif t.text == '{' and depth == 0:
                    close = self.match(j, '{', '}')
                    if previous is not None and previous.text == ')' and not saw_assign:
                        return close
                    j = close
                    previous = tokens[j]
                    j += 1
                    continue
            if not is_trivia(t) and t.kind != PP:
                previous = t
            j += 1
        raise ParseError('statement without an end')

    def global_statement(self, a, b, mode):
        sig, has_pp = significant_range(self.tl, a, b)
        texts = [self.tok(j).text for j in sig]
        if not texts:
            return
        if texts[0] == 'precision':
            if mode == 'convert':
                self.edit(a, b, self.newlines(a, b))
            return
        if texts[0] == 'struct':
            if mode == 'collect':
                self.collect_struct(sig)
            else:
                self.struct_array_members(sig)
                self.struct_maker(sig)
            self.struct_instances(sig, mode)
            return
        paren = next((k for k, j in enumerate(sig) if self.tok(j).text == '('), None)
        assign = next((k for k, j in enumerate(sig) if self.tok(j).text == '='), None)
        if paren is not None and (assign is None or paren < assign) and paren >= 2 \
                and self.tok(sig[paren - 1]).kind == ID and self.tok(sig[paren - 2]).kind == ID:
            self.function(sig, paren, mode)
            return
        if texts[0] == 'layout' and any(t.startswith('local_size_') for t in texts):
            return
        if texts[0] == 'layout' and len(texts) > 4 and texts[-2] == ')' and texts[-1] == ';' and 'in' in texts:
            if mode == 'convert':
                self.report(sig[0], 'hand', f"input layout '{' '.join(texts)}' is not supported")
            return
        if 'buffer' in texts and '{' in texts and texts.index('buffer') < texts.index('{'):
            self.buffer_block(a, b, sig, mode)
            return
        decl = self.parse_declaration(sig)
        if decl is None:
            if mode == 'convert':
                self.report(sig[0], 'hand', f"unrecognised global statement starting '{' '.join(texts[:4])}'")
            return
        self.global_declaration(a, b, sig, decl, mode, has_pp)

    def collect_struct(self, sig):
        name = self.tok(sig[1]).text
        members, fields = self.struct_layout(sig)
        self.ctx.structs[name] = members
        if fields is not None and fields not in self.ctx.struct_fields.setdefault(name, []):
            self.ctx.struct_fields[name].append(fields)
        for j in self.tl.sig:
            if self.tl.tokens[j].kind == ID and self.tl.next_text(j) == '(':
                self.ctx.called.add(self.tl.tokens[j].text)

    def struct_layout(self, sig):
        """Member types and constructor fields of one struct definition; fields is None when a member is an array."""
        members = {}
        fields = []
        k = 3
        while k < len(sig):
            t = self.tok(sig[k]).text
            if t == '}':
                break
            ty = self.ctx.type_name(t) or UNKNOWN
            k += 1
            last = None
            while k < len(sig) and self.tok(sig[k]).text != ';':
                tt = self.tok(sig[k])
                if tt.text == '[':
                    fields = None
                    if last is None:
                        ty = array(ty)
                    else:
                        members[last] = array(ty)
                    while k < len(sig) and self.tok(sig[k]).text != ']':
                        k += 1
                elif tt.kind == ID:
                    last = tt.text
                    members[tt.text] = ty
                    if fields is not None:
                        fields.append((GLSL_TO_HLSL_TYPE.get(t, t), tt.text))
                k += 1
            k += 1
        return members, fields

    def struct_array_members(self, sig):
        """A GLSL member `float[n] a;` becomes `float a[n];`."""
        for k in range(3, len(sig) - 1):
            if self.tok(sig[k]).text != '[' or self.tok(sig[k - 1]).kind != ID or self.tok(sig[k - 2]).text not in ('{', ';'):
                continue
            end = k
            while end < len(sig) and self.tok(sig[end]).text != ']':
                end += 1
            if end + 1 >= len(sig) or self.tok(sig[end + 1]).kind != ID:
                continue
            size = ''.join(self.tl.text(sig[j]) for j in range(k + 1, end))
            self.edit(sig[k], sig[end], '')
            name = sig[end + 1]
            self.edit(name, name, f'{self.tl.text(name)}[{size}]')

    def struct_maker(self, sig):
        name = self.tok(sig[1]).text
        _, fields = self.struct_layout(sig)
        if not fields or name not in self.ctx.called or [self.tok(j).text for j in sig[-2:]] != ['}', ';']:
            return
        params = ', '.join(f'{t} {n}' for t, n in fields)
        body = ' '.join(f'nwrValue.{n} = {n};' for _, n in fields)
        self.edit(sig[-1], sig[-1], f'; {name} nwrMake_{name}({params}) {{ {name} nwrValue; {body} return nwrValue; }}')

    def struct_instances(self, sig, mode):
        """`struct S { ... } a, b;` declares globals; HLSL needs them static and outside the struct statement."""
        close = next((k for k in range(len(sig) - 1, -1, -1) if self.tok(sig[k]).text == '}'), None)
        if close is None or len(sig) < 2:
            return
        name = self.tok(sig[1]).text
        names = [self.tok(sig[k]).text for k in range(close + 1, len(sig)) if self.tok(sig[k]).kind == ID]
        if not names:
            return
        for instance in names:
            if mode == 'collect':
                self.ctx.globals[instance].add(struct(name))
            else:
                self.globals[instance].add(struct(name))
        if mode == 'convert':
            self.edit(sig[close], sig[close], '}; static ' + name)

    def buffer_block(self, a, b, sig, mode):
        texts = [self.tok(j).text for j in sig]
        open_k = texts.index('{')
        close_k = len(texts) - 1 - texts[::-1].index('}')
        block = texts[open_k - 1]
        layout = {}
        if texts[0] == 'layout' and ')' in texts[:open_k]:
            for part in ''.join(texts[2:texts.index(')')]).split(','):
                key, _, value = part.partition('=')
                layout[key] = value
        members = []
        current = []
        for k in range(open_k + 1, close_k):
            if texts[k] == ';':
                words = [w for w in current if w not in GLSL_QUALIFIERS and w not in PRECISION]
                if words:
                    members.append(words)
                current = []
            else:
                current.append(texts[k])
        parsed = []
        for words in members:
            if len(words) < 2 or self.ctx.type_name(words[0]) is None and words[0] not in self.ctx.structs:
                parsed = None
                break
            declarators = [[]]
            depth = 0
            for w in words[1:]:
                depth += (w == '[') - (w == ']')
                if w == ',' and depth == 0:
                    declarators.append([])
                else:
                    declarators[-1].append(w)
            for declarator in declarators:
                length = None
                if len(declarator) > 1:
                    if declarator[1] != '[' or declarator[-1] != ']' or declarator.count('[') != 1:
                        parsed = None
                        break
                    length = ''.join(declarator[2:-1])
                if not declarator or not re.fullmatch(r'[A-Za-z_]\w*', declarator[0]):
                    parsed = None
                    break
                parsed.append((words[0], declarator[0], length))
            if parsed is None:
                break
        instance = [t for t in texts[close_k + 1:] if t != ';']
        binding = layout.get('binding', '').strip()
        buffer = f'nwrBuffer{binding}'
        unsized = [m for m in parsed or [] if m[2] == '']
        if mode == 'collect':
            types = {}
            for type_name, name, length in parsed or []:
                ty = self.ctx.type_name(type_name) or UNKNOWN
                types[name] = array(ty, int(length) if length and length.isdigit() else -1) if length is not None else ty
            if instance:
                self.ctx.structs[block] = types
                self.ctx.globals[instance[0]].add(struct(block))
                if len(instance) == 1 and unsized and len(parsed) == 1 and binding.isdigit():
                    self.ctx.buffer_views[instance[0]] = (unsized[0][1], buffer)
                elif len(instance) == 1 and unsized and parsed[-1][2] == '' and len(unsized) == 1 and binding.isdigit():
                    self.ctx.buffer_views[instance[0]] = (unsized[0][1], f'{buffer}_{unsized[0][1]}')
            else:
                for name, ty in types.items():
                    self.ctx.globals[name].add(ty)
            return
        problem = None
        if parsed is None or not parsed:
            problem = 'members the converter cannot read'
        elif not binding.isdigit():
            problem = 'no literal binding'
        elif len(instance) > 1:
            problem = 'an instance array'
        elif unsized and (len(unsized) > 1 or parsed[-1][2] != ''):
            problem = 'an unsized array that is not its last member'
        if problem is not None:
            self.report(sig[0], 'hand', f'buffer block {block} has {problem}; declare it with NWR_BUFFER by hand')
            return
        if unsized and len(parsed) == 1:
            type_name, name, _ = parsed[0]
            lines = [f'NWR_BUFFER({binding}, {GLSL_TO_HLSL_TYPE.get(type_name, type_name)}, {buffer if instance else name});']
        else:
            head = parsed[:-1] if unsized else parsed
            fields = ' '.join(f"{GLSL_TO_HLSL_TYPE.get(t, t)} {n}{'' if length is None else f'[{length}]'};"
                              for t, n, length in head)
            lines = [f'struct {block} {{ {fields} }}; NWR_BUFFER({binding}, {block}, {buffer});']
            if unsized:
                type_name, name, _ = parsed[-1]
                tail = f'{buffer}_{name}' if instance else name
                lines[0] += f' NWR_BUFFER_TAIL({binding}, {GLSL_TO_HLSL_TYPE.get(type_name, type_name)}, {tail}, {block});'
            if instance:
                lines.append(f'#define {instance[0]} {buffer}[0]')
            else:
                lines += [f'#define {n} {buffer}[0].{n}' for _, n, _ in head]
        padding = max(0, self.newlines(a, b).count('\n') - (len(lines) - 1))
        self.edit(a, b, '\n'.join(lines) + '\n' * padding)

    def parse_type_at(self, sig, k):
        name = self.tok(sig[k]).text
        ty = self.ctx.type_name(name)
        return ty

    def parse_params(self, sig, open_k, close_k):
        params = []
        current = []
        depth = 0
        for k in range(open_k + 1, close_k):
            t = self.tok(sig[k])
            if t.text in '([':
                depth += 1
            elif t.text in ')]':
                depth -= 1
            if t.text == ',' and depth == 0:
                params.append(current)
                current = []
                continue
            current.append(k)
        if current:
            params.append(current)
        out = []
        for p in params:
            words = [self.tok(sig[k]) for k in p]
            words_nq = [w for w in words if w.text not in GLSL_QUALIFIERS]
            if not words_nq or (len(words_nq) == 1 and words_nq[0].text == 'void'):
                continue
            ty = self.ctx.type_name(words_nq[0].text) or UNKNOWN
            rest = words_nq[1:]
            if rest and rest[0].text == '[':
                close = next((j for j, w in enumerate(rest) if w.text == ']'), len(rest) - 1)
                rest = rest[close + 1:]
            name = rest[0].text if rest and rest[0].kind == ID else None
            if any(w.text == '[' for w in words_nq):
                ty = array(ty)
            out.append((name, ty))
        return out

    def function(self, sig, paren, mode):
        name = self.tok(sig[paren - 1]).text
        ret_name = self.tok(sig[paren - 2]).text
        ret = self.ctx.type_name(ret_name) or UNKNOWN
        close = None
        depth = 0
        for k in range(paren, len(sig)):
            t = self.tok(sig[k]).text
            if t == '(':
                depth += 1
            elif t == ')':
                depth -= 1
                if depth == 0:
                    close = k
                    break
        if close is None:
            raise ParseError('unbalanced parameter list')
        params = self.parse_params(sig, paren, close)
        if mode == 'collect':
            sig_types = [p[1] for p in params]
            existing = self.ctx.functions[name]
            if not any(f.params == sig_types and f.ret == ret for f in existing):
                existing.append(Function(name, ret, sig_types))
            return
        self.array_type_params(sig, paren, close)
        if self.ctx.shadows_uniform(name):
            # GLSL has the uniform only where a program declares it; the format 2 library declares it everywhere.
            first = sig[0]
            define = f'#define {name} {name}{api.SHADOWED_UNIFORM_SUFFIX}'
            self.tl.overrides[first] = f'\n{define}\n' + self.tl.overrides.get(first, self.tok(first).text)
        shadowing = {pname for pname, pty in params
                     if (pname in api.TEXTURES or pname in self.ctx.custom_textures) and pty.kind == 'sampler'}
        companions = {}
        for k in range(paren + 1, close):
            t = self.tok(sig[k])
            if t.kind == ID and t.text in api.SAMPLER_PARAMETERS:
                self.tl.overrides[sig[k]] = api.SAMPLER_PARAMETER_TYPE
                after = k + 1
                if after < close and self.tok(sig[after]).kind == ID and self.tok(sig[min(after + 1, close)]).text != '[':
                    pname = self.tok(sig[after]).text
                    final = pname + '_' if pname in shadowing else pname
                    companions[pname] = final + api.SAMPLER_COMPANION
                    self.tl.overrides[sig[after]] = f'{final}, SamplerState {companions[pname]}'
            elif t.kind == ID and t.text in shadowing and t.text not in companions:
                self.tl.overrides[sig[k]] = t.text + '_'
        if close + 1 < len(sig) and self.tok(sig[close + 1]).text == '{':
            body_open = sig[close + 1]
            body_close = sig[-1]
            scope = defaultdict(set)
            for pname, pty in params:
                if pname:
                    scope[pname].add(pty)
                if pname in shadowing:
                    scope[('rename', pname)] = pname + '_'
                if pname in companions:
                    scope[('sampler', pname)] = companions[pname]
            if name == 'main':
                self.main_inits(body_open)
            else:
                self.clear_out_params(sig, paren, close, body_open)
            self.ctx.scopes = [scope]
            try:
                self.walk_block(body_open + 1, body_close - 1)
            finally:
                self.ctx.scopes = []
            if ret_name != 'void':
                self.return_after_switch(ret_name, body_close)

    def ends_with_switch(self, body_close):
        close = self.prev_sig(body_close)
        if close < 0 or self.tok(close).text != '}':
            return False
        depth = 0
        k = close
        while k >= 0:
            t = self.tok(k)
            if t.kind == OP and t.text == '}':
                depth += 1
            elif t.kind == OP and t.text == '{':
                depth -= 1
                if depth == 0:
                    break
            k -= 1
        paren = self.prev_sig(k) if k >= 0 else -1
        if paren < 0 or self.tok(paren).text != ')':
            return False
        depth = 0
        while paren >= 0:
            t = self.tok(paren)
            if t.kind == OP and t.text == ')':
                depth += 1
            elif t.kind == OP and t.text == '(':
                depth -= 1
                if depth == 0:
                    break
            paren -= 1
        word = self.prev_sig(paren) if paren >= 0 else -1
        return word >= 0 and self.tok(word).text == 'switch'

    def return_after_switch(self, ret_name, body_close):
        """FXC cannot prove that a switch returns on every path (X3507), not even one whose only label left after the
        preprocessor is a returning `default:`; a function that ends in a switch gets a zero return after it."""
        if not self.ends_with_switch(body_close):
            return
        hlsl_type = GLSL_TO_HLSL_TYPE.get(ret_name, ret_name)
        self.tl.overrides[body_close] = f'return ({hlsl_type})0; ' + self.tl.overrides.get(body_close, self.tok(body_close).text)

    def parse_declaration(self, sig):
        k = 0
        qualifiers = []
        layout = {}
        while k < len(sig):
            t = self.tok(sig[k])
            if t.text == 'layout' and k + 1 < len(sig) and self.tok(sig[k + 1]).text == '(':
                close = k + 1
                depth = 0
                while close < len(sig):
                    tt = self.tok(sig[close]).text
                    depth += tt == '('
                    depth -= tt == ')'
                    if depth == 0:
                        break
                    close += 1
                inner = ''.join(self.tok(sig[j]).text + ' ' for j in range(k + 2, close))
                for part in inner.split(','):
                    kv = [x.strip() for x in part.split('=')]
                    if kv[0]:
                        layout[kv[0]] = kv[1] if len(kv) > 1 else ''
                k = close + 1
                continue
            if t.text in GLSL_QUALIFIERS:
                qualifiers.append(t.text)
                k += 1
                continue
            break
        if k >= len(sig) or self.tok(sig[k]).kind != ID:
            return None
        type_name = self.tok(sig[k]).text
        type_index = k
        if self.ctx.type_name(type_name) is None and not (k + 1 < len(sig) and self.tok(sig[k + 1]).kind == ID):
            return None
        k += 1
        type_array = 0
        if k < len(sig) and self.tok(sig[k]).text == '[':
            while k < len(sig) and self.tok(sig[k]).text != ']':
                k += 1
            k += 1
            type_array = 1
        if k >= len(sig) or self.tok(sig[k]).kind != ID:
            return None
        decl = Declaration(qualifiers, layout, type_name, type_index, type_array)
        end = len(sig) - 1 if self.tok(sig[-1]).text == ';' else len(sig)
        while k < end:
            name_k = k
            if self.tok(sig[k]).kind != ID:
                return None
            k += 1
            array_len = None
            dims = []
            while k < end and self.tok(sig[k]).text == '[':
                a = k
                while k < end and self.tok(sig[k]).text != ']':
                    k += 1
                dims.append(sig[a + 1:k])
                k += 1
            if dims:
                array_len = dims[0]
                decl.inner_dims[name_k] = dims[1:]
            init = None
            if k < end and self.tok(sig[k]).text == '=':
                k += 1
                start = k
                depth = 0
                while k < end:
                    tt = self.tok(sig[k]).text
                    if tt in '([{':
                        depth += 1
                    elif tt in ')]}':
                        depth -= 1
                    elif tt == ',' and depth == 0:
                        break
                    k += 1
                init = sig[start:k]
            decl.declarators.append((name_k, array_len, init))
            if k < end and self.tok(sig[k]).text == ',':
                k += 1
                continue
            if k < end:
                return None
        return decl

    def declared_type(self, decl, sig, array_len, name_k=None, init=None):
        ty = self.ctx.type_name(decl.type_name) or UNKNOWN
        for dim in reversed(decl.inner_dims.get(name_k, [])):
            ty = array(ty, self.literal_length(dim))
        if decl.type_array or array_len is not None:
            length = self.literal_length(array_len)
            if length < 0 and init:
                length = self.initializer_count(init)
            ty = array(ty, length)
        return ty

    def literal_length(self, tokens):
        if tokens and len(tokens) == 1 and self.tok(tokens[0]).kind == NUM:
            return int(self.tok(tokens[0]).text.rstrip('uU'))
        return -1

    def initializer_count(self, init):
        texts = [self.tok(j).text for j in init]
        if '(' not in texts or texts[-1] != ')' or '[' not in texts[:texts.index('(')]:
            return -1
        depth = 0
        count = 0
        for t in texts[texts.index('(') + 1:-1]:
            depth += (t in '([{') - (t in ')]}')
            count += t == ',' and depth == 0
        return count + 1 if len(texts) > texts.index('(') + 2 else 0

    def global_declaration(self, a, b, sig, decl, mode, has_pp):
        storage = next((q for q in decl.qualifiers if q in STORAGE), None)
        if mode == 'collect':
            for name_k, array_len, init in decl.declarators:
                ty = self.declared_type(decl, sig, array_len, name_k, init)
                self.ctx.globals[self.tok(sig[name_k]).text].add(ty)
                if 'const' in decl.qualifiers:
                    self.ctx.const_globals.add(self.tok(sig[name_k]).text)
                    if init:
                        self.ctx.const_values[self.tok(sig[name_k]).text] = self.init_texts(init)
                elif storage in ('in', 'varying', 'attribute'):
                    self.ctx.stage_inputs.add(self.tok(sig[name_k]).text)
                elif storage is None and init:
                    self.ctx.global_inits.append((self.tok(sig[name_k]).text, self.init_names(init)))
                elif storage == 'uniform' and self.tok(sig[name_k]).text in api.LEGACY_TEXTURES:
                    self.ctx.legacy_textures.add(self.tok(sig[name_k]).text)
            if storage == 'out' and 'location' in decl.layout and self.stage() in ('fragment', None):
                first = int(decl.layout['location'])
                self.explicit_locations.update(range(first, first + len(decl.declarators)))
            return
        for name_k, array_len, init in decl.declarators:
            self.globals[self.tok(sig[name_k]).text].add(self.declared_type(decl, sig, array_len, name_k, init))
        hlsl_type = GLSL_TO_HLSL_TYPE.get(decl.type_name, decl.type_name)
        names = [self.tok(sig[name_k]).text for name_k, _, _ in decl.declarators]
        if storage == 'uniform':
            self.uniform(a, b, sig, decl, names, hlsl_type)
            return
        stage = self.stage()
        if storage == 'attribute' or (storage == 'in' and stage == 'vertex'):
            self.attribute(a, b, sig, decl, names, hlsl_type)
            return
        if storage in ('varying', 'in', 'out'):
            output = storage == 'out' and (stage == 'fragment' or (stage is None and 'location' in decl.layout))
            if storage == 'in' and stage is None:
                self.report(sig[0], 'hand', f"'in {decl.type_name} {names[0]}' outside a known stage; taken as a varying")
            if storage == 'out' and stage is None and not output:
                self.report(sig[0], 'hand', f"'out {decl.type_name} {names[0]}' outside a known stage; taken as a varying")
            if output:
                lines = []
                for offset, name in enumerate(names):
                    if 'location' in decl.layout:
                        location = int(decl.layout['location']) + offset
                    else:
                        location = self.implicit_location(name)
                    lines.append(f'NWR_OUTPUT({location}, {hlsl_type}, {name});')
                    self.outputs.append(name)
                self.edit(a, b, ' '.join(lines) + self.newlines(a, b))
                return
            if any(array_len is not None for _, array_len, _ in decl.declarators) or decl.type_array:
                self.report(sig[0], 'hand', 'array varying: split it by hand')
            mode_suffix = ''
            for q in decl.qualifiers:
                mode_suffix = INTERPOLATION.get(q, mode_suffix) or mode_suffix
            text = ' '.join(f'NWR_VARYING{mode_suffix}({hlsl_type}, {self.unshadowed_varying(name)});' for name in names)
            text += ''.join(f'\n#define {name} {name}{api.SHADOWED_UNIFORM_SUFFIX}'
                            for name in names if self.ctx.shadows_uniform(name))
            self.varyings.extend(names)
            self.edit(a, b, text + self.newlines(a, b))
            return
        if storage == 'buffer' or storage == 'inout':
            self.report(sig[0], 'hand', f"'{storage}' global: storage buffers need RW resources by hand")
            return
        self.rename_uniform_globals(sig, decl)
        self.defer_initializer(sig, decl)
        inits = [init for _, _, init in decl.declarators if init]
        if 'shared' in decl.qualifiers:
            for j in sig:
                if self.tok(j).text == 'shared':
                    self.edit(j, j, 'groupshared')
            return
        if has_pp:
            def initializers(variant):
                active = set(variant)
                for init in inits:
                    self.initializer([j for j in init if j in active])
            self.pp_statement(a, b, sig, initializers)
        else:
            for init in inits:
                self.initializer(init)
        if decl.type_array:
            self.array_type_syntax(sig, decl)
        first = sig[0]
        self.edit(first, first, 'static ' + self.tl.text(first))

    def init_names(self, init):
        return {self.tok(j).text for j in init if self.tok(j).kind == ID and not self.member_access(j)}

    def defer_initializer(self, sig, decl):
        """`T g = <reads a stage input>;` becomes `static T g; T nwrInit_g() { return <init>; }` and a NWR_INIT_g
        macro that main() expands."""
        if 'const' in decl.qualifiers:
            return
        for name_k, array_len, init in decl.declarators:
            name = self.tok(sig[name_k]).text
            if not init or name not in self.ctx.deferred_inits:
                continue
            if not self.init_names(init) & (self.ctx.input_names() | set(self.ctx.deferred_inits)):
                continue
            end = self.tl.next_sig.get(init[-1])
            if len(decl.declarators) != 1 or array_len is not None or decl.type_array or end is None or self.tok(end).text != ';':
                self.report(sig[0], 'hand', f'global {name} is initialised from a stage input; assign it at the start of main()')
                continue
            hlsl_type = GLSL_TO_HLSL_TYPE.get(decl.type_name, decl.type_name)
            equals = self.prev_sig(init[0])
            self.edit(equals, equals, f'; {hlsl_type} nwrInit_{name}() {{ return')
            shadow = f'\n#define {name} {name}{api.SHADOWED_UNIFORM_SUFFIX}' if self.ctx.shadows_uniform(name) else ''
            self.edit(end, end, f'; }}{shadow}\n#define NWR_INIT_{name} {name} = nwrInit_{name}();')

    def clear_out_params(self, sig, open_k, close_k, body_open):
        """GLSL leaves an `out` parameter undefined on paths that never write it; FXC refuses such a function (X3508),
        so each one starts at zero."""
        clears = []
        param = []
        for k in range(open_k + 1, close_k + 1):
            t = self.tok(sig[k]).text
            if k < close_k and t != ',':
                param.append(t)
                continue
            words = [w for w in param if w not in GLSL_QUALIFIERS or w == 'out']
            param = []
            if len(words) == 3 and words[0] == 'out':
                hlsl_type = GLSL_TO_HLSL_TYPE.get(words[1], words[1])
                clears.append(f' {words[2]} = ({hlsl_type})0;')
        if clears:
            self.edit(body_open, body_open, '{' + ''.join(clears))

    def main_inits(self, body_open):
        if not self.ctx.deferred_inits:
            return
        lines = ''.join(f'\n#ifdef NWR_INIT_{name}\nNWR_INIT_{name}\n#endif' for name in self.ctx.deferred_inits)
        self.edit(body_open, body_open, '{' + lines + '\n')

    def implicit_location(self, name):
        # GLSL's linker gives an output without a location the lowest location no explicit output uses, so
        # `layout(location = 0) out a; out b;` puts b at 1 (a TAA history output can rely on it).
        if name not in self.implicit_locations:
            taken = self.explicit_locations | set(self.implicit_locations.values())
            location = 0
            while location in taken:
                location += 1
            self.implicit_locations[name] = location
        return self.implicit_locations[name]

    def unshadowed_varying(self, name):
        """A varying named like a custom or API uniform (a prepare program's varying sunDir beside
        uniform.vec3.sunDir) would collide with the uniform the format 2 library declares everywhere."""
        return name + api.SHADOWED_UNIFORM_SUFFIX if self.ctx.shadows_uniform(name) else name

    def rename_uniform_globals(self, sig, decl):
        """A pack global named like an API or custom uniform: GLSL has that uniform only where a program declares it,
        but the format 2 library declares it everywhere. The global takes its own name, and a macro from its declaration
        on sends later uses to it, so programs that declare the uniform instead still read the uniform."""
        defines = []
        for name_k, _, _ in decl.declarators:
            name = self.tok(sig[name_k]).text
            if self.ctx.shadows_uniform(name):
                self.tl.overrides[sig[name_k]] = name + api.SHADOWED_UNIFORM_SUFFIX
                defines.append(f'\n#define {name} {name}{api.SHADOWED_UNIFORM_SUFFIX}')
        if defines:
            end = sig[-1]
            self.tl.overrides[end] = self.tl.overrides.get(end, self.tok(end).text) + ''.join(defines)

    def uniform(self, a, b, sig, decl, names, hlsl_type):
        ty = self.ctx.type_name(decl.type_name)
        is_sampler = ty is not None and ty.kind == 'sampler'
        standins = []
        for name in names:
            if is_sampler and 'image' in decl.type_name:
                if name not in self.ctx.custom_images:
                    self.report(sig[0], 'api-gap', f'image {name} has no image.* line in shaders.properties')
                elif decl.type_name.endswith('Array'):
                    self.report(sig[0], 'api-gap', f'image {name} declared as {decl.type_name}; images are 1D, 2D or 3D')
                continue
            if is_sampler:
                if name in self.ctx.custom_images.values():
                    if decl.type_name not in ('sampler1D', 'usampler1D', 'isampler1D', 'sampler2D', 'usampler2D',
                                              'isampler2D', 'sampler3D', 'usampler3D', 'isampler3D'):
                        self.report(sig[0], 'api-gap', f'image sampler {name} declared as {decl.type_name}')
                elif name not in api.TEXTURES and name not in api.LEGACY_TEXTURES and name not in self.ctx.custom_textures:
                    self.report(sig[0], 'api-gap', f'texture {name} ({decl.type_name}) is not provided by the API')
                elif name in self.ctx.volume_textures:
                    if decl.type_name != 'sampler3D':
                        self.report(sig[0], 'api-gap', f'3D texture {name} declared as {decl.type_name}')
                elif 'Shadow' not in decl.type_name and decl.type_name not in ('sampler2D', 'usampler2D', 'isampler2D'):
                    self.report(sig[0], 'api-gap', f'texture {name} declared as {decl.type_name}; the API binds 2D textures')
                continue
            if name in api.UNIFORM_TYPES or name in self.ctx.custom_uniforms:
                expected = api.UNIFORM_TYPES.get(name) or self.ctx.custom_uniforms.get(name)
                if ty is not None and expected is not None and expected.known and ty != expected:
                    self.report(sig[0], 'note', f'uniform {name} declared as {decl.type_name}; the API provides {expected.hlsl()}')
                continue
            if name in api.PROVIDED_MATRIX_BUILTINS:
                continue
            self.report(sig[0], 'api-gap', f'uniform {decl.type_name} {name} is not provided by the API')
            if self.ctx.standins and ty is not None and ty.kind in ('scalar', 'vec', 'mat') and not decl.type_array:
                standins.append(f'static const {hlsl_type} {name} = ({hlsl_type})0;')
        self.edit(a, b, ' '.join(standins) + self.newlines(a, b))

    def attribute(self, a, b, sig, decl, names, hlsl_type):
        standins = []
        for name in names:
            if name in api.ATTRIBUTES:
                continue
            self.report(sig[0], 'api-gap', f'vertex attribute {decl.type_name} {name} is not provided by the API')
            if self.ctx.standins:
                standins.append(f'static {hlsl_type} {name} = ({hlsl_type})0;')
        self.edit(a, b, ' '.join(standins) + self.newlines(a, b))

    def initializer(self, init):
        parser = Parser(self.tl, init, self.ctx)
        node = parser.parse_full(allow_comma=False)
        if node.kind in ('arrayctor', 'structctor'):
            initializer_braces(node)
        self.apply(node)

    def apply(self, node):
        if self.variant_mode and self.spans_pp(node.start, node.end):
            if node.custom is not None:
                raise ParseError('a rewrite spans a preprocessor line')
            for child in node.children:
                self.apply(child)
            return
        text = render(self.tl, node)
        if text != self.tl.raw(node.start, node.end):
            self.edit(node.start, node.end, text)

    # ---- function bodies

    def walk_block(self, a, b):
        tokens = self.tl.tokens
        i = a
        while i <= b:
            t = tokens[i]
            if is_trivia(t):
                i += 1
                continue
            if t.kind == PP:
                self.directive(i, 'convert')
                i += 1
                continue
            if t.kind == OP and t.text == '{':
                close = self.match(i, '{', '}')
                self.ctx.scopes.append(defaultdict(set))
                try:
                    self.walk_block(i + 1, close - 1)
                finally:
                    self.ctx.scopes.pop()
                i = close + 1
                continue
            if t.kind == OP and t.text in ('}', ';'):
                i += 1
                continue
            if t.kind == ID and t.text in STATEMENT_KEYWORDS:
                i = self.keyword(i, b)
                continue
            call_end = self.statement_macro_end(i, b)
            if call_end is not None:
                i = call_end + 1
                continue
            end = self.find_at_depth(i, b, {';'})
            if end is None:
                self.report(i, 'parse', 'statement without a semicolon')
                return
            split_end = self.split_statement_end(i, end, b)
            if split_end is not None:
                self.statement(i, split_end)
                i = split_end + 1
                continue
            self.statement(i, end - 1)
            i = end + 1

    def statement_macro_end(self, i, b):
        """`MACRO(args)` on its own line, where every definition of MACRO ends in `;` or `}`: the call is a whole
        statement without a semicolon after it. Returns the call's last token, or None for anything else."""
        definitions = self.ctx.macros.get(self.tok(i).text)
        if self.tok(i).kind != ID or not definitions:
            return None
        for macro in definitions:
            body, _ = significant_range(macro.tl, 0, len(macro.tl.tokens) - 1)
            if not body or macro.tl.tokens[body[-1]].text not in (';', '}'):
                return None
        end = i
        if definitions[0].params is not None:
            j = self.peek_trivia(i + 1, b)
            if j is None or self.tok(j).text != '(':
                return None
            end = self.match(j, '(', ')')
        after = self.peek_trivia(end + 1, b)
        if after is not None and self.tok(after).text == ';':
            return None
        return end

    def peek_trivia(self, i, b):
        while i <= b and is_trivia(self.tok(i)):
            i += 1
        return i if i <= b and self.tok(i).kind != PP else None

    def split_statement_end(self, i, end, b):
        """A statement whose semicolon sits inside a conditional that gives each branch its own ending (`x = a` then
        `#if A` / `* b;` / `#else` / `* c;` / `#endif`): it runs to the #endif when every branch combination ends in
        exactly one semicolon there. Returns that #endif, or None."""
        depth = 0
        for k in range(i, end + 1):
            if self.tok(k).kind == PP:
                word = directive_word(self.tok(k).text)
                depth += word in ('if', 'ifdef', 'ifndef')
                depth -= word == 'endif'
                if depth < 0:
                    return None
        if depth == 0:
            return None
        stop = end + 1
        while stop <= b and depth:
            if self.tok(stop).kind == PP:
                word = directive_word(self.tok(stop).text)
                depth += word in ('if', 'ifdef', 'ifndef')
                depth -= word == 'endif'
            stop += 1
        if depth:
            return None
        stop -= 1
        groups = self.pp_groups(i, stop)
        variants = self.pp_expand(groups) if groups is not None else None
        if not variants:
            return None
        sig, _ = significant_range(self.tl, i, stop)
        for excluded in variants:
            variant = [j for j in sig if j not in excluded]
            ends = [j for j in variant if self.tok(j).text == ';']
            if not variant or ends != [variant[-1]]:
                return None
        return stop

    def next_sig(self, i, b):
        j = i
        while j <= b and (is_trivia(self.tok(j)) or self.tok(j).kind == PP):
            if self.tok(j).kind == PP:
                self.directive(j, 'convert')
            j += 1
        return j

    def keyword(self, i, b):
        word = self.tok(i).text
        if word in ('if', 'while', 'switch'):
            j = self.next_sig(i + 1, b)
            close = self.match(j, '(', ')')
            if word == 'while' and not self.after_do(i) and self.varying_bound(j + 1, close - 1, set()):
                self.edit(i, i, '[loop] while')
            self.expression(j + 1, close - 1)
            return close + 1
        if word == 'for':
            j = self.next_sig(i + 1, b)
            close = self.match(j, '(', ')')
            first = self.find_at_depth(j + 1, close - 1, {';'})
            second = self.find_at_depth(first + 1, close - 1, {';'}) if first is not None else None
            if first is None or second is None:
                self.report(i, 'parse', 'for header without two semicolons')
                return close + 1
            self.statement(j + 1, first - 1)
            steps = {self.tok(k).text for k in range(second + 1, close) if self.tok(k).kind == ID}
            varying = self.varying_bound(first + 1, second - 1, steps)
            head = '[loop] for' if varying or self.long_constant_bound(first + 1, second - 1, steps) else 'for'
            body = self.peek_sig(close + 1, b)
            if self.declares(j + 1, first - 1) and body <= b and self.tok(body).text == '{':
                end = self.match(body, '{', '}')
                self.edit(i, i, '{ ' + head)
                self.edit(end, end, '} }')
            elif head != 'for':
                self.edit(i, i, head)
            self.expression(first + 1, second - 1)
            self.expression(second + 1, close - 1)
            return close + 1
        if word in ('else', 'do'):
            return i + 1
        if word in ('case', 'default'):
            colon = self.find_at_depth(i, b, {':'})
            if word == 'case' and colon is not None:
                self.expression(i + 1, colon - 1)
            return (colon if colon is not None else i) + 1
        end = self.find_at_depth(i, b, {';'})
        if end is None:
            self.report(i, 'parse', f'{word} without a semicolon')
            return b + 1
        if word == 'return':
            self.expression(i + 1, end - 1)
        return end + 1

    def peek_sig(self, i, b):
        while i <= b and is_trivia(self.tok(i)):
            i += 1
        return i

    def declares(self, a, b):
        """Whether a for-loop initialiser declares a variable. FXC scopes it to the enclosing block, not the loop, so a
        later declaration of the same name there is a redefinition; the loop gets its own block."""
        first = self.peek_sig(a, b)
        if first > b or self.tok(first).kind != ID:
            return False
        if self.tok(first).text == 'const':
            return True
        second = self.peek_sig(first + 1, b)
        return second <= b and self.tok(second).kind == ID

    def after_do(self, i):
        """Whether the `while` at i closes a `do` loop."""
        k = self.prev_sig(i)
        if k < 0 or self.tok(k).text != '}':
            return False
        depth = 0
        while k >= 0:
            text = self.tok(k).text
            depth += text == '}'
            depth -= text == '{'
            if depth == 0:
                break
            k -= 1
        k = self.prev_sig(k)
        return k >= 0 and self.tok(k).text == 'do'

    def prev_sig(self, i):
        k = i - 1
        while k >= 0 and (is_trivia(self.tok(k)) or self.tok(k).kind == PP):
            k -= 1
        return k

    def varying_bound(self, a, b, steps):
        """A loop condition that reads a variable, parameter or uniform. FXC unrolls a loop with a gradient
        instruction in it and fails when the trip count is not constant; GLSL runs it with undefined derivatives,
        which `[loop]` asks FXC for too."""
        return any(self.varying_name(self.tok(k).text, steps, 0) for k in range(a, b + 1)
                   if self.tok(k).kind == ID and not self.member_access(k))

    def long_constant_bound(self, a, b, steps):
        """A loop with a constant trip count of UNROLL_TRIP_LIMIT or more. FXC unrolls every loop whose trip count it
        knows, and a long raymarch unrolled around calls that loop again takes minutes to compile (volumetric cloud
        raymarches: about 120 s per program, 6 s with `[loop]`). GLSL leaves unrolling to the driver, as `[loop]` does."""
        texts = [self.tok(k).text for k in range(a, b + 1) if not is_trivia(self.tok(k))]
        compare = next((k for k, t in enumerate(texts) if t in ('<', '<=', '>', '>=', '!=')), None)
        if compare is None or not any(t in steps for t in texts[:compare]):
            return False
        value = self.constant_value(texts[compare + 1:], 0)
        return value is not None and abs(value) >= UNROLL_TRIP_LIMIT

    def init_texts(self, init):
        return [self.tok(j).text for j in init if not is_trivia(self.tok(j))]

    def constant_value(self, texts, depth):
        if depth > MACRO_BOUND_DEPTH or not texts:
            return None
        parts = []
        for t in texts:
            number = TRIP_NUMBER.match(t)
            if number:
                parts.append(number[1])
            elif t in ('int', 'uint', 'float'):
                continue
            elif t in ('(', ')', '+', '-', '*', '/'):
                parts.append(t)
            elif re.fullmatch(r'[A-Za-z_]\w*', t):
                values = [v for v in (self.constant_value(d, depth + 1) for d in self.constant_definitions(t)) if v is not None]
                if not values:
                    return None
                parts.append(repr(max(values)))
            else:
                return None
        try:
            return float(eval(' '.join(parts), {'__builtins__': {}}, {}))
        except (ArithmeticError, SyntaxError, TypeError, ValueError):
            return None

    def constant_definitions(self, name):
        for scope in reversed(self.ctx.scopes):
            if name in scope:
                value = scope.get(('value', name))
                return [value] if value else []
        if name in self.ctx.const_values:
            return [self.ctx.const_values[name]]
        return [[t.text for t in macro.tl.tokens if not is_trivia(t)]
                for macro in self.ctx.macros.get(name, ()) if macro.params is None]

    def member_access(self, k):
        j = self.prev_sig(k)
        return j >= 0 and self.tok(j).text == '.'

    def varying_name(self, name, steps, depth):
        if name in steps:
            return False
        for scope in reversed(self.ctx.scopes):
            if name in scope:
                return ('const', name) not in scope
        if name in self.ctx.globals or (self.ctx.file_globals is not None and name in self.ctx.file_globals):
            return name not in self.ctx.const_globals
        if name in api.UNIFORM_TYPES or name in self.ctx.custom_uniforms:
            return True
        if name in self.ctx.macros and depth < MACRO_BOUND_DEPTH:
            return any(self.varying_name(t.text, steps, depth + 1)
                       for macro in self.ctx.macros[name] if macro.params is None
                       for t in macro.tl.tokens if t.kind == ID and t.text != name)
        return False

    def statement(self, a, b):
        sig, has_pp = significant_range(self.tl, a, b)
        if not sig:
            return
        if has_pp:
            self.pp_statement(a, b, sig, self.statement_sig)
            return
        self.statement_sig(sig)

    def compat_output_write(self, sig):
        """`gl_FrontColor = ...;` and the other fixed-function colour outputs: the compatibility profile feeds them to
        the fragment stage's gl_Color, which NWR fragment stages do not have, so the write is dead and is dropped."""
        name = self.tok(sig[0]).text
        if name not in api.COMPAT_OUTPUTS or any(name in scope for scope in self.ctx.scopes):
            return False
        if name in self.ctx.globals or (self.ctx.file_globals is not None and name in self.ctx.file_globals):
            return False
        if self.find_at_depth(sig[0], sig[-1], ASSIGN) is None:
            return False
        end = sig[-1]
        after = self.peek_trivia(end + 1, len(self.tl.tokens) - 1)
        if not self.variant_mode and after is not None and self.tok(after).text == ';':
            end = after
        self.edit(sig[0], end, self.newlines(sig[0], end))
        self.report(sig[0], 'note', f'{name} is a fixed-function output no NWR stage reads; the write is dropped')
        return True

    def statement_sig(self, sig):
        if self.compat_output_write(sig):
            return
        decl = self.parse_declaration(sig) if self.is_declaration(sig) else None
        if decl is not None:
            for name_k, array_len, init in decl.declarators:
                ty = self.declared_type(decl, sig, array_len, name_k, init)
                if init:
                    try:
                        self.initializer(init)
                    except ParseError as e:
                        self.parse_failure(init, str(e))
                name = self.tok(sig[name_k]).text
                self.ctx.scopes[-1][name] = {ty}
                if 'const' in decl.qualifiers:
                    self.ctx.scopes[-1][('const', name)] = {ty}
                    if init:
                        self.ctx.scopes[-1][('value', name)] = self.init_texts(init)
                if init and self.reads_own_name(init, name):
                    self.tl.overrides[sig[name_k]] = name + LOCAL_SUFFIX
                    self.ctx.scopes[-1][('rename', name)] = name + LOCAL_SUFFIX
                elif name in self.ctx.functions:
                    self.tl.overrides[sig[name_k]] = name + '_'
                    self.ctx.scopes[-1][('rename', name)] = name + '_'
                if not init and array_len is None and not decl.type_array and ty.kind in ('scalar', 'vec', 'mat'):
                    self.zero_local(sig[name_k], ty)
            if decl.type_array:
                self.array_type_syntax(sig, decl)
            return
        self.expression_sig(sig)

    def zero_local(self, name_index, ty):
        """GLSL drivers read an unwritten local as zero; FXC refuses a read it can prove uninitialised (X4575), e.g. a
        local passed to an inout parameter before any write, so a local without an initializer starts at zero."""
        name = self.tl.overrides.get(name_index, self.tok(name_index).text)
        self.tl.overrides[name_index] = f'{name} = ({ty.hlsl()})0'

    def array_type_params(self, sig, open_k, close_k):
        """A GLSL parameter `out float[n] a` becomes `out float a[n]`."""
        for k in range(open_k + 1, close_k):
            if self.tok(sig[k]).text != '[' or self.tok(sig[k - 1]).kind != ID:
                continue
            end = k
            while end < close_k and self.tok(sig[end]).text != ']':
                end += 1
            if end + 1 >= close_k or self.tok(sig[end + 1]).kind != ID:
                continue
            size = ''.join(self.tl.text(sig[j]) for j in range(k + 1, end))
            self.edit(sig[k], sig[end], '')
            name = sig[end + 1]
            self.edit(name, name, f'{self.tl.text(name)}[{size}]')

    def reads_own_name(self, init, name):
        """GLSL starts a local's scope after its initializer, so `float x = f(x);` reads the outer x; HLSL starts it
        before, so the local takes another name."""
        for j in init:
            if self.tl.tokens[j].kind == ID and self.tl.tokens[j].text == name:
                before = self.tl.prev_sig.get(j)
                if (before is None or self.tl.tokens[before].text != '.') and self.tl.next_text(j) != '(':
                    return True
        return False

    def array_type_syntax(self, sig, decl):
        """GLSL `T[n] a, b` becomes `T a[n], b[n]`."""
        open_k = decl.type_index + 1
        close_k = open_k
        while close_k < len(sig) and self.tok(sig[close_k]).text != ']':
            close_k += 1
        if close_k >= len(sig):
            return
        size = ''.join(self.tl.text(sig[k]) for k in range(open_k + 1, close_k))
        self.edit(sig[open_k], sig[close_k], '')
        for name_k, _, _ in decl.declarators:
            index = sig[name_k]
            self.edit(index, index, f'{self.tl.text(index)}[{size}]')

    def pp_statement(self, a, b, sig, handle):
        """A statement with conditional lines in it is converted once per combination of branches. Rewrites that
        stay between two preprocessor lines are applied; one that would span a preprocessor line is reported."""
        for i in range(a, b + 1):
            if self.tok(i).kind == PP:
                self.directive(i, 'convert')
        groups = self.pp_groups(a, b)
        variants = self.pp_expand(groups) if groups is not None else None
        risky = [j for j in sig if self.tok(j).text in ('*', '*=') or self.tok(j).text.startswith('mat')]
        if variants is None:
            if risky:
                self.report(sig[0], 'parse', 'preprocessor line inside a statement with a product: convert by hand')
            return
        edits = {}
        covered = set()
        failed = None
        self.variant_mode = True
        try:
            for excluded in variants:
                variant = [j for j in sig if j not in excluded]
                if variant and self.tok(variant[-1]).text == ';':
                    variant.pop()
                if not variant:
                    continue
                self.variant_edits = {}
                try:
                    handle(variant)
                except ParseError as e:
                    failed = failed or str(e)
                    continue
                for span, text in self.variant_edits.items():
                    if edits.setdefault(span, text) != text:
                        failed = 'preprocessor branches rewrite the same code differently'
                covered.update(variant)
        finally:
            self.variant_mode = False
            self.variant_edits = None
        spans = sorted(edits)
        overlap = any(s[1] >= t[0] for s, t in zip(spans, spans[1:]))
        uncovered = [j for j in risky if j not in covered]
        if overlap or uncovered or (failed and not covered):
            self.report(sig[0], 'parse', 'preprocessor line inside a statement with a product: convert by hand'
                        + (f' ({failed})' if failed else ''))
            return
        for (start, end), text in edits.items():
            self.edit(start, end, text)

    def pp_groups(self, a, b):
        root = []
        stack = [(root, None)]
        for i in range(a, b + 1):
            t = self.tok(i)
            if t.kind != PP:
                stack[-1][0].append(i)
                continue
            m = re.match(r'#\s*(\w+)', t.text)
            word = m.group(1) if m else ''
            if word in ('if', 'ifdef', 'ifndef'):
                group = {'branches': [[]], 'else': False}
                stack[-1][0].append(group)
                stack.append((group['branches'][0], group))
            elif word in ('elif', 'else'):
                group = stack[-1][1]
                if group is None or group['else']:
                    return None
                group['else'] = word == 'else'
                group['branches'].append([])
                stack[-1] = (group['branches'][-1], group)
            elif word == 'endif':
                if stack[-1][1] is None:
                    return None
                stack.pop()
        return root if len(stack) == 1 else None

    @staticmethod
    def pp_tokens(nodes):
        out = set()
        for node in nodes:
            if isinstance(node, int):
                out.add(node)
            else:
                for branch in node['branches']:
                    out |= FileConverter.pp_tokens(branch)
        return out

    def pp_expand(self, nodes, limit=64):
        """Every combination of branches, as the sets of token indices each one leaves out."""
        results = [frozenset()]
        for node in nodes:
            if isinstance(node, int):
                continue
            branch_tokens = [self.pp_tokens(branch) for branch in node['branches']]
            options = []
            for k, branch in enumerate(node['branches']):
                others = set().union(*(t for j, t in enumerate(branch_tokens) if j != k))
                inner = self.pp_expand(branch, limit)
                if inner is None:
                    return None
                options += [frozenset(others | o) for o in inner]
            if not node['else']:
                options.append(frozenset(set().union(*branch_tokens)))
            results = list(dict.fromkeys(r | o for r in results for o in options))
            if len(results) > limit:
                return None
        return results

    def is_declaration(self, sig):
        k = 0
        while k < len(sig) and self.tok(sig[k]).text in GLSL_QUALIFIERS:
            k += 1
        if k + 1 >= len(sig):
            return False
        first = self.tok(sig[k])
        second = self.tok(sig[k + 1])
        if first.kind != ID or self.ctx.type_name(first.text) is None:
            return False
        return second.kind == ID or second.text == '['

    def expression(self, a, b):
        sig, has_pp = significant_range(self.tl, a, b)
        if has_pp:
            if sig:
                self.pp_statement(a, b, sig, self.expression_sig)
            return
        if sig:
            self.expression_sig(sig)

    def expression_sig(self, sig):
        try:
            node = Parser(self.tl, sig, self.ctx).parse_full()
        except ParseError as e:
            self.parse_failure(sig, str(e))
            return
        if node.kind == 'arrayctor' or (node.kind == 'structctor' and node.custom is None):
            self.report(sig[0], 'hand', 'array or struct constructor outside an initializer')
        self.apply(node)

    def parse_failure(self, sig, message):
        if self.variant_mode:
            raise ParseError(message)
        risky = any(self.tok(j).text in ('*', '*=') for j in sig)
        missing = self.undeclared_type(sig)
        if missing is not None:
            self.report(sig[0], 'parse-note', f'{message}: no file of the pack declares the type {missing}, so no '
                        'program that reaches this compiles')
            return
        self.report(sig[0], 'parse' if risky else 'parse-note', message)

    def undeclared_type(self, sig):
        """`T name ...` where T is no GLSL type, struct, macro or function of the pack (a mod's interface type)."""
        if len(sig) < 2 or self.tok(sig[0]).kind != ID or self.tok(sig[1]).kind != ID:
            return None
        name = self.tok(sig[0]).text
        if (self.ctx.type_name(name) is not None or name in self.ctx.macros or name in self.ctx.functions
                or name in GLSL_QUALIFIERS or name in STATEMENT_KEYWORDS):
            return None
        return name

    # ---- macros (after every file was converted)

    def convert_macros(self):
        for i in self.macro_defs:
            for macro in self.ctx.macros.get(self.macro_name(i), []):
                if macro.key != (self.path, i):
                    continue
                self.convert_macro(i, macro)

    def macro_statements(self, tl, sig, key, decisions):
        if not any(tl.tokens[j].text in ATOMICS for j in sig):
            return None
        segments = [[]]
        depth = 0
        for j in sig:
            text = tl.tokens[j].text
            depth += (text in '([{') - (text in ')]}')
            if text == ';' and depth == 0:
                segments.append([])
            else:
                segments[-1].append(j)
        rendered = {}
        try:
            for segment in segments:
                if segment:
                    node = Parser(tl, segment, self.ctx, mode='render', macro=key, decisions=decisions).parse_full()
                    rendered[segment[0]] = (segment[-1], render(tl, node))
        except ParseError:
            return None
        out = []
        k = 0
        while k < len(tl.tokens):
            if k in rendered:
                end, text = rendered[k]
                out.append(text)
                k = end + 1
                continue
            out.append(tl.text(k))
            k += 1
        return ''.join(out)

    def macro_name(self, i):
        m = re.match(r'#\s*define\s+(\w+)', self.tok(i).text)
        return m.group(1) if m else None

    def list_constructors(self, macro, sig):
        """`T[] x = T[] param;` in a function-like macro: GLSL builds the array from the parenthesised list a call passes
        (`m((a, b, c))`). FXC has neither unsized array constructors nor variadic macros, so the list becomes a call to
        an overload per list length that the pack's calls use, returning the items and their count; `x.length()` reads
        the count."""
        tl = macro.tl
        params = set(macro.params or ())
        texts = [tl.tokens[j].text for j in sig]
        helpers = []
        for k in range(len(sig) - 9):
            elem, name = texts[k], texts[k + 3]
            if (texts[k + 1:k + 3] != ['[', ']'] or texts[k + 4] != '=' or texts[k + 5] != elem
                    or texts[k + 6:k + 8] != ['[', ']'] or texts[k + 8] not in params or texts[k + 9] != ';'
                    or tl.tokens[sig[k]].kind != ID or tl.tokens[sig[k + 3]].kind != ID):
                continue
            counts = sorted(self.ctx.list_calls.get(macro.name, ()))
            if not counts:
                self.ctx.report(macro.file, macro.line, 'hand',
                                f'macro {macro.name}: no call passes a parenthesised list to its array constructor')
                return ''
            hlsl = GLSL_TO_HLSL_TYPE.get(elem, elem)
            capacity = counts[-1]
            kind = f'NwrList_{macro.name}_{name}'
            make = f'nwrList_{macro.name}_{name}'
            local = f'nwrList_{name}'
            tl.overrides[sig[k]] = f'{kind} {local} = {make} '
            for j in range(sig[k] + 1, sig[k + 8]):
                tl.overrides[j] = ''
            tl.overrides[sig[k + 9]] = f'; {hlsl} {name}[{capacity}] = {local}.mItems;'
            for m in range(k + 10, len(sig) - 4):
                if texts[m] == name and texts[m + 1:m + 5] == ['.', 'length', '(', ')']:
                    tl.overrides[sig[m]] = f'{local}.mLength'
                    for j in range(sig[m] + 1, sig[m + 4] + 1):
                        tl.overrides[j] = ''
            helpers.append(f'struct {kind} {{ {hlsl} mItems[{capacity}]; int mLength; }};')
            for n in counts:
                args = ', '.join(f'{hlsl} a{a}' for a in range(n))
                fill = ' '.join(f'l.mItems[{a}] = a{a};' for a in range(n))
                helpers.append(f'{kind} {make}({args}) {{ {kind} l = ({kind})0; {fill} l.mLength = {n}; return l; }}')
        if not helpers:
            return ''
        guard = f'NWR_LIST_{macro.name}'
        return f'#ifndef {guard}\n#define {guard}\n' + ' '.join(helpers) + '\n#endif\n'

    def convert_macro(self, i, macro):
        tl = macro.tl
        sig = macro_body(tl)
        helpers = self.list_constructors(macro, sig) if sig and macro.params else ''
        if not sig:
            if macro.name in self.ctx.empty_macros:
                self.edit(i, i, re.sub(r'\(\s*\)$', ' ', macro.header) + tl.raw(0, len(tl.tokens) - 1))
            return
        decisions = {}
        products = [j for j in sig if tl.tokens[j].text in ('*', '*=')]
        for j in products:
            found = self.ctx.decisions.get((macro.key, j), set())
            if found == {'mul'}:
                decisions[j] = 'mul'
            elif 'unknown' in found or ('mul' in found and 'plain' in found):
                self.ctx.report(macro.file, macro.line, 'untyped-product',
                                f'macro {macro.name}: product typed differently at its uses ({sorted(found)})')
        try:
            node = Parser(tl, sig, self.ctx, mode='render', macro=macro.key, decisions=decisions).parse_full()
            text = render(tl, node)
            body = tl.raw(0, sig[0] - 1) + text + tl.raw(sig[-1] + 1, len(tl.tokens) - 1)
        except ParseError:
            body = self.macro_statements(tl, sig, macro.key, decisions)
            if body is not None:
                self.edit(i, i, helpers + macro.header + body)
                return
            body = tl.raw(0, len(tl.tokens) - 1)
            if products and any(self.ctx.decisions.get((macro.key, j)) for j in products):
                self.ctx.report(macro.file, macro.line, 'parse', f'macro {macro.name}: body is not an expression')
        header = macro.header
        if macro.params and any(p in HLSL_RESERVED for p in macro.params):
            params = ', '.join(p + '_' if p in HLSL_RESERVED else p for p in macro.params)
            header = re.sub(r'\([^)]*\)$', f'({params})', header)
        if macro.name in self.ctx.empty_macros:
            header = re.sub(r'\(\s*\)$', ' ', header)
        new = helpers + header + body
        if new != self.tok(i).text:
            self.edit(i, i, new)

    # ---- output

    def output(self):
        out = []
        tokens = self.tl.tokens
        i = 0
        while i < len(tokens):
            if i in self.edits:
                end, text = self.edits[i]
                out.append(text)
                i = end + 1
                continue
            out.append(self.tl.text(i) if tokens[i].kind != PP else host_macros(tokens[i].text))
            i += 1
        return ''.join(out) + self.stage_name_undefs()

    def stage_name_undefs(self):
        """`#define texCoord retroCoord` inside main() (a retro filter option) is harmless in GLSL, where nothing follows the
        pack text, but NWR appends an entry wrapper that copies varyings and outputs by name. A macro that renames one of
        them would rewrite the wrapper, so it ends with the file that defines it."""
        stage_names = set(self.varyings) | set(self.outputs)
        names = []
        for i in self.macro_defs:
            m = re.match(r'#\s*define\s+(\w+)(\()?', self.tok(i).text)
            if m and not m.group(2) and m.group(1) in stage_names and m.group(1) not in names:
                names.append(m.group(1))
        return ''.join(f'\n#undef {name}' for name in names) + ('\n' if names else '')
