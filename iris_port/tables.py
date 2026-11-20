# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""GLSL built-in functions: HLSL names and result types."""
from .types import NONMAT, UNKNOWN, VOID, mat, scalar, vec, widest

CALL_RENAMES = {
    'fract': 'frac',
    'mix': 'lerp',
    'inversesqrt': 'rsqrt',
    'dFdx': 'ddx',
    'dFdy': 'ddy',
    'dFdxFine': 'ddx_fine',
    'dFdyFine': 'ddy_fine',
    'dFdxCoarse': 'ddx_coarse',
    'dFdyCoarse': 'ddy_coarse',
    'floatBitsToInt': 'asint',
    'floatBitsToUint': 'asuint',
    'intBitsToFloat': 'asfloat',
    'uintBitsToFloat': 'asfloat',
    'fma': 'mad',
    'roundEven': 'round',
    'bitCount': 'countbits',
    'findMSB': 'firstbithigh',
    'findLSB': 'firstbitlow',
    'bitfieldReverse': 'reversebits',
    'isnan': 'nwrIsNan',
    'isinf': 'nwrIsInf',
    'normalize': 'nwrNormalize',
    'pow': 'nwrPow',
    'sqrt': 'nwrSqrt',
    'texture3D': 'texture',
    'texture2DLodARB': 'textureLod',
    'texture2DGradARB': 'textureGrad',
}

SHIM_FUNCTIONS = {'mod'}

UNSUPPORTED_FUNCTIONS = {
    'outerProduct': 'no outerProduct in HLSL',
    'textureOffset': 'textureOffset is not in the format 2 sampling helpers',
    'texelFetchOffset': 'texelFetchOffset is not in the format 2 sampling helpers',
    'textureLodOffset': 'textureLodOffset is not in the format 2 sampling helpers',
    'textureGradOffset': 'textureGradOffset is not in the format 2 sampling helpers',
    'textureGatherOffset': 'textureGatherOffset is not in the format 2 sampling helpers',
    'textureProj': 'textureProj is not in the format 2 sampling helpers',
    'textureProjLod': 'textureProjLod is not in the format 2 sampling helpers',
    'textureQueryLod': 'textureQueryLod is not in the format 2 sampling helpers',
    'textureQueryLevels': 'textureQueryLevels is not in the format 2 sampling helpers',
    'texture1D': '1D textures are not in the format 2 sampling helpers',
    'textureCube': 'cube textures are not in the format 2 sampling helpers',
    'uaddCarry': 'not in SM 5.1',
    'usubBorrow': 'not in SM 5.1',
    'umulExtended': 'not in SM 5.1',
    'imulExtended': 'not in SM 5.1',
    'interpolateAtCentroid': 'use EvaluateAttributeAtCentroid by hand',
    'interpolateAtSample': 'use EvaluateAttributeAtSample by hand',
    'interpolateAtOffset': 'use EvaluateAttributeSnapped by hand',
    'imageSamples': 'multisampled images are not supported',
}

_GENTYPE = {
    'abs', 'sign', 'floor', 'ceil', 'fract', 'trunc', 'round', 'roundEven', 'mod', 'min', 'max', 'clamp', 'mix',
    'sqrt', 'inversesqrt', 'pow', 'exp', 'log', 'exp2', 'log2', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh',
    'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'radians', 'degrees', 'normalize', 'reflect', 'refract',
    'faceforward', 'dFdx', 'dFdy', 'dFdxFine', 'dFdyFine', 'dFdxCoarse', 'dFdyCoarse', 'fwidth', 'fma',
    'saturate', 'lerp', 'frac', 'rsqrt', 'ddx', 'ddy', 'mad', 'rcp',
}
_SCALAR_FLOAT = {'length', 'distance', 'dot', 'determinant'}
_BOOL_SCALAR = {'any', 'all'}
_COMPARE = {'lessThan', 'lessThanEqual', 'greaterThan', 'greaterThanEqual', 'equal', 'notEqual', 'isnan', 'isinf'}
_SAMPLE = {'texture', 'texture2D', 'textureLod', 'texture2DLod', 'textureGrad', 'texture2DGrad', 'texelFetch',
           'texelFetch2D', 'textureOffset', 'textureLodOffset', 'texelFetchOffset', 'textureGather', 'textureProj',
           'texture3D', 'textureCube', 'texture2DLodARB', 'texture2DGradARB'}
PACKING = {'packHalf2x16': (1, 'uint'), 'packUnorm4x8': (1, 'uint'), 'packSnorm4x8': (1, 'uint'),
           'packUnorm2x16': (1, 'uint'), 'packSnorm2x16': (1, 'uint'), 'unpackHalf2x16': (2, 'float'),
           'unpackUnorm4x8': (4, 'float'), 'unpackSnorm4x8': (4, 'float'), 'unpackUnorm2x16': (2, 'float'),
           'unpackSnorm2x16': (2, 'float')}

ATOMICS = {'atomicAdd': 'InterlockedAdd', 'atomicMin': 'InterlockedMin', 'atomicMax': 'InterlockedMax',
           'atomicAnd': 'InterlockedAnd', 'atomicOr': 'InterlockedOr', 'atomicXor': 'InterlockedXor',
           'atomicExchange': 'InterlockedExchange', 'atomicCompSwap': 'InterlockedCompareStore'}

_IMAGE_ATOMICS = {'imageAtomicAdd', 'imageAtomicMin', 'imageAtomicMax', 'imageAtomicAnd', 'imageAtomicOr',
                  'imageAtomicXor', 'imageAtomicExchange', 'imageAtomicCompSwap'}
_BARRIERS = {'barrier', 'memoryBarrier', 'memoryBarrierShared', 'memoryBarrierImage', 'memoryBarrierBuffer',
             'groupMemoryBarrier'}


def _sampled_base(ty):
    if ty is None or ty.kind != 'sampler':
        return 'float'
    return {'u': 'uint', 'i': 'int'}.get(ty.name[0], 'float')


def _volume(ty):
    return ty is not None and ty.kind == 'sampler' and '3D' in ty.name


def builtin_result(name, args, arg_types):
    """Result type of a GLSL built-in call, or None when the name is not a built-in."""
    first = arg_types[0] if arg_types else None
    if name in _SAMPLE:
        if name in ('texture', 'textureLod', 'textureGrad') and args and first is not None \
                and first.kind == 'sampler' and 'Shadow' in first.name:
            return scalar()
        return vec(4, _sampled_base(first))
    if name in ('shadow2D', 'shadow2DLod'):
        return vec(4)
    if name in ('textureSize', 'imageSize'):
        return vec(3 if _volume(first) else 2, 'int')
    if name == 'imageLoad':
        return vec(4, _sampled_base(first))
    if name == 'imageStore' or name in _BARRIERS:
        return VOID
    if name in _IMAGE_ATOMICS:
        return scalar(_sampled_base(first))
    if name in PACKING:
        size, base = PACKING[name]
        return vec(size, base) if size > 1 else scalar(base)
    if name in ATOMICS:
        return first if first is not None and first.known else scalar('uint')
    if name in _SCALAR_FLOAT:
        return scalar()
    if name in _BOOL_SCALAR:
        return scalar('bool')
    if name == 'cross':
        return vec(3)
    if name == 'ftransform':
        return vec(4)
    if name == 'transpose':
        t = arg_types[0] if arg_types else None
        return mat(t.rows, t.cols) if t is not None and t.is_matrix else UNKNOWN
    if name in ('bitfieldExtract', 'bitfieldInsert'):
        return arg_types[0] if arg_types else UNKNOWN
    if name == 'inverse':
        return arg_types[0] if arg_types else UNKNOWN
    if name == 'matrixCompMult':
        return arg_types[0] if arg_types else UNKNOWN
    if name in _COMPARE:
        w = widest(arg_types[:1])
        return vec(w.size, 'bool') if w.known and w.is_vector else (scalar('bool') if w.known else NONMAT)
    if name == 'not':
        return arg_types[0] if arg_types else UNKNOWN
    if name in ('floatBitsToInt', 'floatBitsToUint', 'intBitsToFloat', 'uintBitsToFloat', 'bitCount', 'findMSB',
                'findLSB', 'bitfieldReverse'):
        base = {'floatBitsToInt': 'int', 'floatBitsToUint': 'uint', 'intBitsToFloat': 'float',
                'uintBitsToFloat': 'float'}.get(name, 'int')
        w = widest(arg_types[:1])
        return vec(w.size if w.is_vector else 1, base) if w.known else NONMAT
    if name == 'step':
        return widest(arg_types)
    if name == 'smoothstep':
        return widest(arg_types)
    if name in _GENTYPE:
        if name in ('length',):
            return scalar()
        return widest(arg_types)
    if name in ('nwrIsNan', 'nwrIsInf', 'nwrIsFinite'):
        w = widest(arg_types[:1])
        return vec(w.size, 'bool') if w.known and w.is_vector else scalar('bool')
    return None


def is_builtin_function(name):
    return builtin_result(name, [], []) is not None or name in UNSUPPORTED_FUNCTIONS or name in CALL_RENAMES


HLSL_RESERVED = {
    'line', 'point', 'triangle', 'lineadj', 'triangleadj', 'linear', 'nointerpolation', 'groupshared', 'static',
    'extern', 'register', 'packoffset', 'snorm', 'unorm', 'vector', 'matrix', 'string', 'half', 'dword', 'pass',
    'technique', 'technique10', 'technique11', 'compile', 'cbuffer', 'tbuffer', 'row_major', 'column_major',
    'export', 'inline', 'interface', 'namespace', 'class', 'typedef', 'unsigned', 'sampler', 'SamplerState',
    'Texture2D', 'Texture3D', 'Buffer', 'min16float', 'min10float', 'min16int', 'min12int', 'min16uint', 'float1',
    'int1', 'uint1', 'bool1', 'double1', 'triangleadj', 'globallycoherent', 'stateblock', 'pixelshader',
    'vertexshader', 'shared', 'sample', 'precise', 'centroid', 'texture1D', 'texture3D', 'textureCUBE', 'samplerCUBE',
    'sampler_state', 'auto', 'catch', 'char', 'const_cast', 'delete', 'dynamic_cast', 'enum', 'explicit', 'friend',
    'goto', 'long', 'mutable', 'new', 'operator', 'private', 'protected', 'public', 'reinterpret_cast', 'short',
    'signed', 'sizeof', 'static_cast', 'template', 'this', 'throw', 'try', 'typename', 'union', 'using', 'virtual',
    'NULL',
}
# HLSL intrinsics the tool writes for GLSL calls: a pack name equal to one would hide the intrinsic in its scope.
GLSL_NAMED_TARGETS = {'round', 'texture', 'textureLod', 'textureGrad'}
HLSL_RESERVED |= (set(CALL_RENAMES.values()) | {'mul', 'atan2'}) - GLSL_NAMED_TARGETS

GLSL_QUALIFIERS = {
    'const', 'in', 'out', 'inout', 'uniform', 'varying', 'attribute', 'flat', 'smooth', 'noperspective', 'centroid',
    'invariant', 'highp', 'mediump', 'lowp', 'precise', 'patch', 'sample', 'buffer', 'readonly', 'writeonly',
    'coherent', 'volatile', 'restrict', 'shared',
}

PRECISION = {'highp', 'mediump', 'lowp'}

STATEMENT_KEYWORDS = {'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'return', 'break', 'continue',
                      'discard'}

