# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""Signed comparisons in #if and #elif conditions.

GLSL evaluates preprocessor conditions in signed arithmetic. fxc (D3DCompile) evaluates them as unsigned 32-bit
values, so `2 > -1` is false there. A relational comparison whose operands may be negative is rewritten as
`((a) + 1073741824 > (b) + 1073741824)`: the 2^30 bias keeps operands from -2^30 to 2^30 - 1 in order both unsigned
(fxc) and signed (DXC, NWR's loader, whose 32-bit signed evaluator would overflow on 2^31), and needs no hex literal.
"""
from .lexer import ID, NUM, OP, is_trivia, tokenize

BIAS = '1073741824'
RELATIONAL = ('<', '>', '<=', '>=')
BINARY_LEVELS = [('||',), ('&&',), ('|',), ('^',), ('&',), ('==', '!='), RELATIONAL, ('<<', '>>'), ('+', '-'),
                 ('*', '/', '%')]


class _Unparsable(Exception):
    pass


class _Parser:
    def __init__(self, tokens, negative):
        self.tokens = tokens
        self.k = 0
        self.negative = negative
        self.changed = False

    def peek(self):
        return self.tokens[self.k].text if self.k < len(self.tokens) else None

    def take(self):
        if self.k >= len(self.tokens):
            raise _Unparsable()
        t = self.tokens[self.k]
        self.k += 1
        return t

    def expression(self):
        cond = self.binary(0)
        if self.peek() == '?':
            self.take()
            a = self.expression()
            if self.peek() != ':':
                raise _Unparsable()
            self.take()
            b = self.expression()
            return (f'{cond[0]} ? {a[0]} : {b[0]}', cond[1] or a[1] or b[1])
        return cond

    def binary(self, level):
        if level == len(BINARY_LEVELS):
            return self.unary()
        left = self.binary(level + 1)
        while self.peek() in BINARY_LEVELS[level]:
            op = self.take().text
            right = self.binary(level + 1)
            signed = left[1] or right[1]
            if op in RELATIONAL and signed:
                self.changed = True
                left = (f'(({left[0]}) + {BIAS} {op} ({right[0]}) + {BIAS})', False)
            else:
                arithmetic = op == '-' or (signed and op in ('+', '*', '/', '%', '<<', '>>'))
                left = (f'{left[0]} {op} {right[0]}', arithmetic)
        return left

    def unary(self):
        t = self.peek()
        if t in ('!', '~', '+'):
            self.take()
            inner = self.unary()
            return (t + inner[0], inner[1] if t == '+' else False)
        if t == '-':
            self.take()
            inner = self.unary()
            return ('-' + inner[0], True)
        return self.primary()

    def primary(self):
        t = self.take()
        if t.kind == NUM:
            return (t.text, False)
        if t.kind == OP and t.text == '(':
            inner = self.expression()
            if self.take().text != ')':
                raise _Unparsable()
            return (f'({inner[0]})', inner[1])
        if t.kind == ID and t.text == 'defined':
            if self.peek() == '(':
                self.take()
                name = self.take()
                if name.kind != ID or self.take().text != ')':
                    raise _Unparsable()
                return (f'defined({name.text})', False)
            name = self.take()
            if name.kind != ID:
                raise _Unparsable()
            return (f'defined {name.text}', False)
        if t.kind == ID:
            if self.peek() == '(':
                self.take()
                depth = 1
                parts = []
                while depth:
                    a = self.take()
                    depth += (a.text == '(') - (a.text == ')')
                    if depth:
                        parts.append(a.text)
                return (f"{t.text}({' '.join(parts)})", True)
            return (t.text, self.negative(t.text))
        raise _Unparsable()


def signed_condition(condition, negative):
    """condition: the text after #if/#elif on one line. negative(name) tells whether a macro may hold a negative value.
    Returns the rewritten condition, or None when nothing needs to change or the condition cannot be parsed."""
    if '\n' in condition or not any(op in condition for op in RELATIONAL):
        return None
    tokens = [t for t in tokenize(condition) if not is_trivia(t)]
    if not tokens:
        return None
    parser = _Parser(tokens, negative)
    try:
        text, _ = parser.expression()
    except _Unparsable:
        return None
    if parser.k != len(tokens) or not parser.changed:
        return None
    return text
