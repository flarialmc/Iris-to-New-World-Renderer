# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""The GLSL type model used to decide which products become mul() and how constructors convert."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Ty:
    kind: str
    base: str = 'float'
    size: int = 1
    cols: int = 0
    rows: int = 0
    name: str = ''
    element: 'Ty' = None
    length: int = -1

    @property
    def is_matrix(self):
        return self.kind == 'mat'

    @property
    def is_scalar(self):
        return self.kind == 'scalar'

    @property
    def is_vector(self):
        return self.kind == 'vec'

    @property
    def known(self):
        return self.kind not in ('unknown', 'nonmat')

    @property
    def components(self):
        if self.kind == 'scalar':
            return 1
        if self.kind == 'vec':
            return self.size
        if self.kind == 'mat':
            return self.cols * self.rows
        return -1

    def hlsl(self):
        prefix = {'float': 'float', 'int': 'int', 'uint': 'uint', 'bool': 'bool', 'double': 'double'}[self.base]
        if self.kind == 'scalar':
            return prefix
        if self.kind == 'vec':
            return f'{prefix}{self.size}'
        if self.kind == 'mat':
            return f'{prefix}{self.cols}x{self.rows}'
        return None


UNKNOWN = Ty('unknown')
NONMAT = Ty('nonmat')
VOID = Ty('void')


def scalar(base='float'):
    return Ty('scalar', base)


def vec(n, base='float'):
    return scalar(base) if n == 1 else Ty('vec', base, n)


def mat(cols, rows):
    return Ty('mat', 'float', 0, cols, rows)


def array(element, length=-1):
    return Ty('array', element.base if element else 'float', element=element, length=length)


def struct(name):
    return Ty('struct', name=name)


def sampler(name):
    return Ty('sampler', name=name)


_VECTOR_PREFIX = {'vec': 'float', 'ivec': 'int', 'uvec': 'uint', 'bvec': 'bool', 'dvec': 'double'}

GLSL_TYPES = {'float': scalar('float'), 'int': scalar('int'), 'uint': scalar('uint'), 'bool': scalar('bool'),
              'double': scalar('double'), 'void': VOID}
for _prefix, _base in _VECTOR_PREFIX.items():
    for _n in (2, 3, 4):
        GLSL_TYPES[f'{_prefix}{_n}'] = vec(_n, _base)
for _c in (2, 3, 4):
    GLSL_TYPES[f'mat{_c}'] = mat(_c, _c)
    for _r in (2, 3, 4):
        GLSL_TYPES[f'mat{_c}x{_r}'] = mat(_c, _r)
        GLSL_TYPES[f'dmat{_c}x{_r}'] = mat(_c, _r)
    GLSL_TYPES[f'dmat{_c}'] = mat(_c, _c)

SAMPLER_TYPES = {
    'sampler1D', 'sampler2D', 'sampler3D', 'samplerCube', 'sampler2DShadow', 'sampler1DShadow', 'samplerCubeShadow',
    'sampler2DArray', 'sampler2DArrayShadow', 'sampler2DRect', 'sampler2DMS', 'samplerBuffer', 'isampler2D',
    'usampler2D', 'isampler3D', 'usampler3D', 'image2D', 'image3D', 'iimage2D', 'uimage2D', 'iimage3D', 'uimage3D',
    'image1D', 'iimage1D', 'uimage1D', 'isampler1D', 'usampler1D', 'image2DArray',
}

GLSL_TO_HLSL_TYPE = {name: ty.hlsl() for name, ty in GLSL_TYPES.items() if ty.kind in ('scalar', 'vec', 'mat')}
GLSL_TO_HLSL_TYPE.update({'double': 'double'})
for _c in (2, 3, 4):
    GLSL_TO_HLSL_TYPE[f'dmat{_c}'] = f'double{_c}x{_c}'
    for _r in (2, 3, 4):
        GLSL_TO_HLSL_TYPE[f'dmat{_c}x{_r}'] = f'double{_c}x{_r}'
for _n in (2, 3, 4):
    GLSL_TO_HLSL_TYPE[f'dvec{_n}'] = f'double{_n}'
GLSL_TO_HLSL_TYPE = {k: v for k, v in GLSL_TO_HLSL_TYPE.items() if k != v}


def parse_type(name, structs=()):
    if name in GLSL_TYPES:
        return GLSL_TYPES[name]
    if name in SAMPLER_TYPES:
        return sampler(name)
    if name in structs:
        return struct(name)
    return None


def widest(types):
    known = [t for t in types if t is not None and t.kind in ('scalar', 'vec')]
    if len(known) != len(types):
        if any(t is not None and t.is_matrix for t in types):
            return UNKNOWN
        return NONMAT if all(t is not None and t.kind in ('scalar', 'vec', 'nonmat') for t in types) else UNKNOWN
    size = max(t.size if t.is_vector else 1 for t in known)
    base = 'float' if any(t.base in ('float', 'double') for t in known) else known[0].base
    return vec(size, base)


def binary_arith(op, left, right):
    if left is None or right is None:
        return UNKNOWN
    if left.is_matrix or right.is_matrix:
        if op != '*':
            return left if left.is_matrix else right
        if left.is_matrix and right.is_matrix:
            return mat(right.cols, left.rows)
        if left.is_matrix and right.is_vector:
            return vec(left.rows)
        if left.is_vector and right.is_matrix:
            return vec(right.cols)
        if left.is_scalar:
            return right
        if right.is_scalar:
            return left
        return UNKNOWN
    if left.kind in ('scalar', 'vec') and right.kind in ('scalar', 'vec'):
        size = max(left.size if left.is_vector else 1, right.size if right.is_vector else 1)
        base = 'float' if 'float' in (left.base, right.base) or 'double' in (left.base, right.base) else left.base
        return vec(size, base)
    if left.kind in ('scalar', 'vec', 'nonmat') and right.kind in ('scalar', 'vec', 'nonmat'):
        return NONMAT
    return UNKNOWN


SWIZZLE_SETS = ('xyzw', 'rgba', 'stpq')


def swizzle_length(name):
    for chars in SWIZZLE_SETS:
        if name and len(name) <= 4 and all(c in chars for c in name):
            return len(name)
    return 0


def swizzle_to_hlsl(name):
    if all(c in 'stpq' for c in name):
        return ''.join('xyzw'['stpq'.index(c)] for c in name)
    return name
