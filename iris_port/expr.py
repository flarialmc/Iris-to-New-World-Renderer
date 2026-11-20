# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""Expression parsing, typing and rewriting.

Expressions are parsed over the significant tokens of a TokenList. Each node keeps its first and last token index, so
an unchanged node renders back to exactly its source text; only nodes with a rewrite render differently.
"""
from .api import FOG_MEMBERS, FOG_STRUCT
from .lexer import ID, NUM, OP, PP, STR, is_trivia
from .types import (NONMAT, UNKNOWN, Ty, binary_arith, mat, scalar, swizzle_length, swizzle_to_hlsl, vec)

BINARY = {'||': 4, '^^': 5, '&&': 6, '|': 7, '^': 8, '&': 9, '==': 10, '!=': 10, '<': 11, '>': 11, '<=': 11, '>=': 11,
          '<<': 12, '>>': 12, '+': 13, '-': 13, '*': 14, '/': 14, '%': 14}
ASSIGN = {'=', '+=', '-=', '*=', '/=', '%=', '<<=', '>>=', '&=', '|=', '^='}
UNARY = {'-', '+', '!', '~', '++', '--'}


class ParseError(Exception):
    pass


class TokenList:
    def __init__(self, tokens, renamer=None):
        self.tokens = tokens
        self.overrides = {}
        self.renamer = renamer
        self.sig = [i for i, t in enumerate(tokens) if not is_trivia(t)]
        self.next_sig = {}
        self.prev_sig = {}
        for a, b in zip(self.sig, self.sig[1:]):
            self.next_sig[a] = b
            self.prev_sig[b] = a

    def next_text(self, i):
        j = self.next_sig.get(i)
        return self.tokens[j].text if j is not None else None

    def prev_text(self, i):
        j = self.prev_sig.get(i)
        return self.tokens[j].text if j is not None else None

    def text(self, i):
        if i in self.overrides:
            return self.overrides[i]
        if self.renamer is not None:
            return self.renamer(self, i)
        return self.tokens[i].text

    def raw(self, a, b):
        return ''.join(self.text(i) for i in range(a, b + 1))


class Node:
    __slots__ = ('kind', 'start', 'end', 'children', 'ty', 'custom', 'name', 'op', 'args')

    def __init__(self, kind, start, end, children=(), ty=UNKNOWN, name=None, op=None):
        self.kind = kind
        self.start = start
        self.end = end
        self.children = list(children)
        self.ty = ty if ty is not None else UNKNOWN
        self.custom = None
        self.name = name
        self.op = op
        self.args = None


def render(tl, node):
    if node.custom is not None:
        return node.custom(lambda n: render(tl, n))
    out = []
    i = node.start
    for child in sorted(node.children, key=lambda c: c.start):
        if child.start < i:
            continue
        out.append(tl.raw(i, child.start - 1))
        out.append(render(tl, child))
        i = child.end + 1
    out.append(tl.raw(i, node.end))
    return ''.join(out)


def literal_type(text):
    t = text.lower()
    if t.startswith('0x'):
        return scalar('uint' if t.endswith('u') else 'int')
    if t.endswith('u'):
        return scalar('uint')
    if '.' in t or 'e' in t or t.endswith('f') or t.endswith('lf'):
        return scalar('float')
    return scalar('int')


def decide_product(left, right):
    if left.is_scalar or right.is_scalar:
        return 'plain'
    if left.is_matrix or right.is_matrix:
        other = right if left.is_matrix else left
        if other.kind in ('vec', 'mat'):
            return 'mul'
        return 'unknown'
    if left.kind in ('vec', 'nonmat') and right.kind in ('vec', 'nonmat'):
        return 'plain'
    return 'unknown'


def checked_by_compiler(left, right):
    """A component-wise product of a 2- or 3-component vector and any GLSL matrix (4 to 16 components) does not
    compile in HLSL, so keeping such a product plain can only be right or a compile error, never a silent mistake."""
    vector = left if left.is_vector else right if right.is_vector else None
    return vector is not None and vector.size in (2, 3)


class Parser:
    """mode: 'file' rewrites and reports; 'eval' only types (macro bodies at a use site); 'render' applies the
    product decisions recorded for a macro body."""

    def __init__(self, tl, sig, ctx, mode='file', macro=None, bindings=None, decisions=None):
        self.tl = tl
        self.sig = sig
        self.pos = 0
        self.ctx = ctx
        self.mode = mode
        self.macro = macro
        self.bindings = bindings or {}
        self.decisions = decisions or {}

    def tok(self, k=0):
        j = self.pos + k
        return self.tl.tokens[self.sig[j]] if j < len(self.sig) else None

    def at(self, text, k=0):
        t = self.tok(k)
        return t is not None and t.kind in (OP, ID) and t.text == text

    def idx(self, k=0):
        return self.sig[self.pos + k]

    def take(self):
        if self.pos >= len(self.sig):
            raise ParseError('unexpected end of expression')
        i = self.sig[self.pos]
        self.pos += 1
        return i

    def expect(self, text):
        if not self.at(text):
            t = self.tok()
            raise ParseError(f"expected '{text}', found '{t.text if t else 'end'}'")
        return self.take()

    def report(self, index, category, message):
        if self.mode == 'eval':
            return
        self.ctx.report_at(self.tl, index, category, message)

    def parse_full(self, allow_comma=True):
        node = self.parse_comma() if allow_comma else self.parse_assign()
        if self.pos != len(self.sig):
            raise ParseError(f"unexpected '{self.tok().text}'")
        return node

    def parse_comma(self):
        node = self.parse_assign()
        if not self.at(','):
            return node
        parts = [node]
        while self.at(','):
            self.take()
            parts.append(self.parse_assign())
        return Node('comma', parts[0].start, parts[-1].end, parts, parts[-1].ty)

    def parse_assign(self):
        left = self.parse_ternary()
        t = self.tok()
        if t is not None and t.kind == OP and t.text in ASSIGN:
            op_index = self.take()
            right = self.parse_assign()
            node = Node('assign', left.start, right.end, [left, right], left.ty, op=t.text)
            if t.text == '*=':
                decision = self.product_decision(op_index, left.ty, right.ty)
                if decision == 'mul':
                    node.custom = lambda r, a=left, b=right: f'{r(a)} = mul({r(b)}, {r(a)})'
            return node
        return left

    def parse_ternary(self):
        cond = self.parse_binary(4)
        if not self.at('?'):
            return cond
        self.take()
        a = self.parse_assign()
        self.expect(':')
        b = self.parse_assign()
        ty = a.ty if a.ty.known else b.ty
        return Node('ternary', cond.start, b.end, [cond, a, b], ty)

    def parse_binary(self, min_bp):
        left = self.parse_unary()
        while True:
            t = self.tok()
            if t is None or t.kind != OP or t.text not in BINARY or BINARY[t.text] < min_bp:
                return left
            op_index = self.take()
            right = self.parse_binary(BINARY[t.text] + 1)
            left = self.binary(t.text, op_index, left, right)

    def product_decision(self, op_index, left, right):
        decision = decide_product(left, right)
        if self.mode == 'eval':
            self.ctx.record_decision(self.macro, op_index, decision)
            return None
        if self.mode == 'render':
            return self.decisions.get(op_index)
        if decision == 'unknown' and checked_by_compiler(left, right):
            self.report(op_index, 'untyped-vector-product',
                        f'product of {describe(left)} and {describe(right)} kept component-wise; it fails to compile if '
                        'the unknown side is a matrix')
        elif decision == 'unknown':
            self.report(op_index, 'untyped-product',
                        f'product of {describe(left)} and {describe(right)}: cannot tell whether it is a matrix product')
        return decision

    def binary(self, op, op_index, left, right):
        if op in ('==', '!=', '<', '>', '<=', '>=', '&&', '||', '^^'):
            node = Node('binary', left.start, right.end, [left, right], scalar('bool'), op=op)
            if op in ('==', '!=') and left.ty.is_vector and right.ty.is_vector and self.mode != 'eval':
                fn = 'all' if op == '==' else 'any'
                node.custom = lambda r, a=left, b=right, f=fn, o=op: f'{f}({r(a)} {o} {r(b)})'
            if op == '^^' and self.mode != 'eval':
                node.custom = lambda r, a=left, b=right: f'(bool({r(a)}) != bool({r(b)}))'
            return node
        if op in ('<<', '>>', '&', '|', '^'):
            return Node('binary', left.start, right.end, [left, right], left.ty, op=op)
        ty = binary_arith(op, left.ty, right.ty)
        node = Node('binary', left.start, right.end, [left, right], ty, op=op)
        if op == '*':
            decision = self.product_decision(op_index, left.ty, right.ty)
            if decision == 'mul':
                node.custom = lambda r, a=left, b=right: f'mul({r(b)}, {r(a)})'
        return node

    def parse_unary(self):
        t = self.tok()
        if t is not None and t.kind == OP and t.text in UNARY:
            start = self.take()
            operand = self.parse_unary()
            ty = scalar('bool') if t.text == '!' else operand.ty
            return Node('unary', start, operand.end, [operand], ty, op=t.text)
        return self.parse_postfix(self.parse_primary())

    def parse_args(self):
        self.expect('(')
        args = []
        if self.at(')'):
            return args, self.take()
        if self.at('void') and self.at(')', 1):
            self.take()
            return args, self.take()
        while True:
            args.append(self.parse_assign())
            if self.at(','):
                self.take()
                continue
            return args, self.expect(')')

    def parse_primary(self):
        t = self.tok()
        if t is None:
            raise ParseError('unexpected end of expression')
        if t.kind == NUM:
            i = self.take()
            return Node('literal', i, i, (), literal_type(t.text), name=t.text)
        if t.kind == STR:
            i = self.take()
            return Node('string', i, i, (), UNKNOWN)
        if t.kind == OP and t.text == '(':
            start = self.take()
            inner = self.parse_comma()
            end = self.expect(')')
            return Node('paren', start, end, [inner], inner.ty)
        if t.kind == OP and t.text == '{':
            start = self.take()
            items = []
            while not self.at('}'):
                items.append(self.parse_assign())
                if self.at(','):
                    self.take()
            end = self.take()
            return Node('initlist', start, end, items, UNKNOWN)
        if t.kind == ID:
            name = t.text
            start = self.take()
            if self.at('['):
                element = self.ctx.type_name(name)
                if element is not None:
                    return self.parse_array_constructor(name, start, element)
            if self.at('('):
                args, end = self.parse_args()
                node = Node('call', start, end, args, UNKNOWN, name=name)
                node.args = args
                node.ty = self.ctx.call_type(self, node)
                return node
            node = Node('ident', start, start, (), UNKNOWN, name=name)
            if name in self.bindings:
                node.ty = self.bindings[name]
            else:
                node.ty = self.ctx.lookup_type(self, node)
            return node
        raise ParseError(f"unexpected '{t.text}'")

    def parse_array_constructor(self, name, start, element):
        self.expect('[')
        length = -1
        if not self.at(']'):
            size = self.parse_comma()
            length = int(self.tl.tokens[size.start].text) if size.kind == 'literal' else -1
        self.expect(']')
        if not self.at('('):
            raise ParseError('array type without constructor')
        args, end = self.parse_args()
        if length < 0:
            length = len(args)
        node = Node('arrayctor', start, end, args, Ty('array', element.base, element=element, length=length),
                    name=name)
        node.args = args
        return node

    def parse_postfix(self, node):
        while True:
            if self.at('['):
                self.take()
                index = self.parse_comma()
                end = self.expect(']')
                node = Node('index', node.start, end, [node, index], element_type(node.ty))
                continue
            if self.at('.'):
                self.take()
                t = self.tok()
                if t is None or t.kind != ID:
                    raise ParseError('expected member name')
                member_index = self.take()
                member = t.text
                if member == 'length' and self.at('('):
                    self.expect('(')
                    end = self.expect(')')
                    base = node
                    length_node = Node('length', base.start, end, [base], scalar('int'))
                    if base.ty.kind == 'array' and base.ty.length >= 0:
                        length_node.custom = lambda r, n=base.ty.length: str(n)
                    elif base.ty.is_vector:
                        length_node.custom = lambda r, n=base.ty.size: str(n)
                    else:
                        self.report(base.start, 'hand', 'length() of an array whose size the tool cannot see')
                    node = length_node
                    continue
                if node.ty.kind == 'struct' and node.ty.name == FOG_STRUCT:
                    node = self.fog_member(node, member, member_index)
                    continue
                ty = self.member_type(node.ty, member, member_index)
                node = Node('member', node.start, member_index, [node], ty, name=member)
                continue
            t = self.tok()
            if t is not None and t.kind == OP and t.text in ('++', '--'):
                end = self.take()
                node = Node('postfix', node.start, end, [node], node.ty, op=t.text)
                continue
            return node

    def fog_member(self, base, member, member_index):
        mapped = FOG_MEMBERS.get(member)
        if mapped is None:
            self.report(base.start, 'api-gap', f'gl_Fog.{member} has no format 2 counterpart')
            return Node('member', base.start, member_index, [base], UNKNOWN, name=member)
        node = Node('member', base.start, member_index, [base], mapped[1], name=member)
        if self.mode != 'eval':
            node.custom = lambda r, text=mapped[0]: text
        return node

    def member_type(self, base, member, member_index):
        if base.kind == 'struct':
            return self.ctx.struct_member(base.name, member)
        n = swizzle_length(member)
        if n and base.kind in ('scalar', 'vec', 'unknown', 'nonmat'):
            hlsl = swizzle_to_hlsl(member)
            if hlsl != member and self.mode != 'eval':
                self.tl.overrides[member_index] = hlsl
            return vec(n, base.base if base.kind in ('scalar', 'vec') else 'float')
        if base.kind in ('unknown', 'nonmat'):
            return UNKNOWN
        return UNKNOWN


def element_type(ty):
    if ty.kind == 'array':
        return ty.element if ty.element is not None else UNKNOWN
    if ty.is_matrix:
        return vec(ty.rows)
    if ty.is_vector:
        return scalar(ty.base)
    if ty.kind == 'nonmat':
        return NONMAT
    return UNKNOWN


def describe(ty):
    if ty.kind in ('scalar', 'vec', 'mat'):
        return ty.hlsl()
    if ty.kind == 'nonmat':
        return 'a scalar or vector'
    return 'an unknown type'


def significant_range(tl, a, b):
    """Significant token indices in [a, b], and whether a preprocessor line sits among them."""
    out = []
    has_pp = False
    for i in range(a, b + 1):
        t = tl.tokens[i]
        if t.kind == PP:
            has_pp = True
            continue
        if not is_trivia(t):
            out.append(i)
    return out, has_pp


def constructor_rewrite(parser, node, target):
    """Rewrites a GLSL constructor call whose HLSL form differs."""
    args = node.args
    hlsl = target.hlsl()
    if hlsl is None or not args:
        return
    if target.kind in ('scalar', 'vec') and len(args) == 1:
        a = args[0].ty
        if a.kind == target.kind and a.size == target.size and a.base == target.base:
            return
        if target.is_scalar and a.kind in ('scalar', 'unknown', 'nonmat'):
            return
        if a.is_matrix:
            parser.report(node.start, 'hand', f'{hlsl} built from a matrix')
            return
        node.custom = lambda r, arg=args[0], h=hlsl: f'(({h})({r(arg)}))'
        return
    if target.is_vector:
        sizes = [a.ty.components if a.ty.kind in ('scalar', 'vec') else -1 for a in args]
        if any(a.ty.is_matrix for a in args):
            parser.report(node.start, 'hand', f'{hlsl} built from a matrix')
            return
        if -1 in sizes:
            return
        total = sum(sizes)
        if total <= target.size:
            return
        before = sum(sizes[:-1])
        keep = target.size - before
        if keep <= 0:
            parser.report(node.start, 'hand', f'{hlsl} constructor with unused arguments')
            return
        swz = 'xyzw'[:keep]

        def trimmed(r, n=node, last=args[-1], s=swz):
            return render_call_with(parser.tl, n, r, {id(last): f'({r(last)}).{s}'})
        node.custom = trimmed
        return
    if target.is_matrix and len(args) == 1:
        a = args[0].ty
        if a.is_matrix:
            if a.cols == target.cols and a.rows == target.rows:
                return
            if a.cols >= target.cols and a.rows >= target.rows:
                node.custom = lambda r, arg=args[0], h=hlsl: f'(({h})({r(arg)}))'
                return
            parser.report(node.start, 'hand', f'{hlsl} built from a smaller matrix (GLSL pads with identity)')
            return
        if a.is_scalar:
            arg = args[0]
            if arg.kind == 'literal' or (arg.kind == 'unary' and arg.children[0].kind == 'literal'):
                def diagonal(r, arg=arg, t=target, h=hlsl):
                    v = r(arg).strip()
                    cells = [v if c == row else '0.0' for c in range(t.cols) for row in range(t.rows)]
                    return f'{h}({", ".join(cells)})'
                node.custom = diagonal
                return
            parser.report(node.start, 'hand', f'{hlsl}(scalar) is a diagonal matrix in GLSL; write it out')
            return
        if a.kind == 'unknown' and target.cols == target.rows:
            node.custom = lambda r, arg=args[0], n=target.rows: f'nwrMatrix{n}({r(arg)})'
            return
        if parser.mode == 'file':
            parser.report(node.start, 'untyped-constructor',
                          f'{hlsl}(x) with x of unknown type: a matrix casts, a scalar builds a diagonal')


def render_call_with(tl, node, r, replaced):
    out = []
    i = node.start
    for child in sorted(node.children, key=lambda c: c.start):
        out.append(tl.raw(i, child.start - 1))
        out.append(replaced.get(id(child)) or r(child))
        i = child.end + 1
    out.append(tl.raw(i, node.end))
    return ''.join(out)


def initializer_braces(node):
    """Renders an array or struct constructor as an HLSL initializer list, nested array and struct constructors too."""
    for arg in node.args:
        if arg.kind in ('arrayctor', 'structctor'):
            initializer_braces(arg)

    def braces(r, n=node):
        return '{ ' + ', '.join(r(a).strip() for a in n.args) + ' }'
    node.custom = braces


def strip_trivia_text(tl, a, b):
    return ''.join(tl.text(i) for i in range(a, b + 1) if not is_trivia(tl.tokens[i]))


__all__ = ['ParseError', 'TokenList', 'Node', 'Parser', 'render', 'significant_range', 'constructor_rewrite',
           'initializer_braces', 'element_type', 'describe', 'mat', 'ID', 'OP', 'PP']
