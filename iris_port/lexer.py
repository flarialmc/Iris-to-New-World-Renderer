# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""A lossless GLSL tokenizer: joining every token's text gives the input back."""
from dataclasses import dataclass
import re

WS = 'ws'
NL = 'nl'
COMMENT = 'comment'
PP = 'pp'
ID = 'id'
NUM = 'num'
OP = 'op'
STR = 'str'
OTHER = 'other'

TRIVIA = (WS, NL, COMMENT)

_OPERATORS = sorted([
    '<<=', '>>=', '++', '--', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||', '^^', '+=', '-=', '*=', '/=', '%=',
    '&=', '|=', '^=', '::', '(', ')', '[', ']', '{', '}', ';', ',', '.', '?', ':', '=', '+', '-', '*', '/', '%',
    '<', '>', '!', '~', '&', '|', '^',
], key=len, reverse=True)

_NUMBER = re.compile(r'0[xX][0-9a-fA-F]+[uU]?|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?(?:lf|LF|[fFuU])?')
_IDENT = re.compile(r'[A-Za-z_]\w*')
_SPACE = re.compile(r'[ \t\f\v]+')


@dataclass
class Token:
    kind: str
    text: str
    line: int

    def __repr__(self):
        return f'{self.kind}:{self.text!r}@{self.line}'


def tokenize(source):
    tokens = []
    i = 0
    line = 1
    at_line_start = True
    n = len(source)
    while i < n:
        c = source[i]
        if c == '\r' and i + 1 < n and source[i + 1] == '\n':
            tokens.append(Token(NL, '\r\n', line))
            i += 2
            line += 1
            at_line_start = True
            continue
        if c == '\n':
            tokens.append(Token(NL, '\n', line))
            i += 1
            line += 1
            at_line_start = True
            continue
        if c == '\\' and (source.startswith('\n', i + 1) or source.startswith('\r\n', i + 1)):
            j = i + (2 if source[i + 1] == '\n' else 3)
            tokens.append(Token(WS, source[i:j], line))
            line += 1
            i = j
            continue
        m = _SPACE.match(source, i)
        if m:
            tokens.append(Token(WS, m.group(0), line))
            i = m.end()
            continue
        if c == '#' and at_line_start:
            j = i
            while j < n:
                if source[j] == '\\' and source.startswith('\n', j + 1):
                    j += 2
                    continue
                if source[j] == '\\' and source.startswith('\r\n', j + 1):
                    j += 3
                    continue
                if source[j] == '\n' or (source[j] == '\r' and source.startswith('\n', j + 1)):
                    break
                if source.startswith('/*', j):
                    end = source.find('*/', j + 2)
                    j = n if end < 0 else end + 2
                    continue
                j += 1
            text = source[i:j]
            tokens.append(Token(PP, text, line))
            line += text.count('\n')
            i = j
            continue
        at_line_start = False
        if source.startswith('//', i):
            j = i
            while j < n and source[j] != '\n':
                if source[j] == '\\' and source.startswith('\n', j + 1):
                    j += 2
                    continue
                j += 1
            if j > i and source[j - 1] == '\r':
                j -= 1
            text = source[i:j]
            tokens.append(Token(COMMENT, text, line))
            line += text.count('\n')
            i = j
            continue
        if source.startswith('/*', i):
            end = source.find('*/', i + 2)
            j = n if end < 0 else end + 2
            text = source[i:j]
            tokens.append(Token(COMMENT, text, line))
            line += text.count('\n')
            i = j
            continue
        if c == '"':
            end = source.find('"', i + 1)
            j = n if end < 0 else end + 1
            tokens.append(Token(STR, source[i:j], line))
            i = j
            continue
        m = _NUMBER.match(source, i)
        if m and (c.isdigit() or (c == '.' and i + 1 < n and source[i + 1].isdigit())):
            tokens.append(Token(NUM, m.group(0), line))
            i = m.end()
            continue
        m = _IDENT.match(source, i)
        if m:
            tokens.append(Token(ID, m.group(0), line))
            i = m.end()
            continue
        for op in _OPERATORS:
            if source.startswith(op, i):
                tokens.append(Token(OP, op, line))
                i += len(op)
                break
        else:
            tokens.append(Token(OTHER, c, line))
            i += 1
    return tokens


def is_trivia(token):
    return token.kind in TRIVIA


def directive_word(directive):
    word = re.match(r'#\s*(\w+)', directive)
    return word.group(1) if word else ''


def branch_depth(branches, directive, depth):
    """Bracket depth after a directive. Every #if branch starts at the depth the #if had, and the chain ends at the
    depth its first branch reached, so branches that each open a block closed after #endif count once."""
    word = re.match(r'#\s*(\w+)', directive)
    word = word.group(1) if word else ''
    if word in ('if', 'ifdef', 'ifndef'):
        branches.append([depth, None])
    elif word in ('elif', 'else') and branches:
        if branches[-1][1] is None:
            branches[-1][1] = depth
        return branches[-1][0]
    elif word == 'endif' and branches:
        start, first = branches.pop()
        if first is not None:
            return first
    return depth
