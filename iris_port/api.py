# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""What the target API (NWR shader-pack format 2) provides to a ported program, as data.

The converter deletes declarations of anything listed here, because the format 2 library declares it, and reports
every name a pack uses that the API does not provide.
"""
from .types import GLSL_TYPES, array, mat, sampler

UNIFORMS = {
    'mat4': ['gbufferModelView', 'gbufferModelViewInverse', 'gbufferProjection', 'gbufferProjectionInverse',
             'gbufferPreviousModelView', 'gbufferPreviousProjection', 'shadowModelView', 'shadowModelViewInverse',
             'shadowProjection', 'shadowProjectionInverse'],
    'vec4': ['lightningBoltPosition', 'entityColor'],
    'vec3': ['cameraPosition', 'previousCameraPosition', 'cameraPositionFract', 'previousCameraPositionFract',
             'eyePosition', 'relativeEyePosition', 'playerLookVector', 'playerBodyVector', 'upPosition', 'sunPosition',
             'moonPosition', 'shadowLightPosition', 'fogColor', 'skyColor'],
    'ivec4': ['blendFunc'],
    'ivec3': ['cameraPositionInt', 'previousCameraPositionInt', 'currentDate', 'currentTime'],
    'ivec2': ['currentYearTime', 'eyeBrightness', 'eyeBrightnessSmooth', 'atlasSize'],
    'float': ['near', 'far', 'viewWidth', 'viewHeight', 'aspectRatio', 'fov', 'centerDepthSmooth', 'eyeAltitude',
              'frameTime', 'frameTimeCounter', 'sunAngle', 'shadowAngle', 'celestialAngle', 'rainStrength', 'wetness',
              'thunderStrength', 'fogStart', 'fogEnd', 'fogDensity', 'cloudHeight', 'ambientLight', 'nightVision',
              'blindness', 'darknessFactor', 'darknessLightFactor', 'playerMood', 'screenBrightness',
              'currentPlayerHealth', 'maxPlayerHealth', 'currentPlayerAir', 'maxPlayerAir', 'currentPlayerHunger',
              'maxPlayerHunger', 'temperature', 'rainfall', 'alphaTestRef'],
    'int': ['worldTime', 'worldDay', 'moonPhase', 'frameCounter', 'fogMode', 'fogShape', 'isEyeInWater', 'heldItemId',
            'heldItemId2', 'heldBlockLightValue', 'heldBlockLightValue2', 'hideGUI', 'biome', 'biome_category',
            'biome_precipitation', 'dimension', 'logicalHeightLimit', 'bedrockLevel', 'heightLimit', 'entityId',
            'blockEntityId', 'currentRenderedItemId', 'renderStage'],
    'bool': ['firstPersonCamera', 'is_sneaking', 'is_sprinting', 'is_on_ground', 'is_hurt', 'is_invisible',
             'is_burning', 'hasSkylight', 'hasCeiling'],
}

UNIFORM_TYPES = {name: GLSL_TYPES[type_name] for type_name, names in UNIFORMS.items() for name in names}

COLOR_TARGETS = 32

TEXTURES = (
    [f'colortex{i}' for i in range(COLOR_TARGETS)]
    + ['depthtex0', 'depthtex1', 'depthtex2', 'shadowtex0', 'shadowtex1', 'shadowcolor0', 'shadowcolor1',
       'gcolor', 'gdepth', 'gnormal', 'composite', 'gaux1', 'gaux2', 'gaux3', 'gaux4', 'gdepthtex', 'shadow',
       'watershadow', 'shadowcolor', 'noisetex', 'gtexture', 'tex', 'lightmap', 'normals', 'specular']
)

# OptiFine's GLSL 1.20 name for the atlas; a non-call use of `texture` can only mean the sampler.
LEGACY_TEXTURES = {'texture': 'gtexture'}
SHADOWED_UNIFORM_SUFFIX = '_nwrPack'

SAMPLER_PARAMETERS = {'sampler2D', 'isampler2D', 'usampler2D', 'sampler2DShadow'}

SAMPLER_PARAMETER_TYPE = 'Texture2D<float4>'

SAMPLER_COMPANION = '_nwrSampler'

SAMPLER_MACRO = 'nwrSampler_'

SAMPLER_FALLBACK = 'nwrLinearClamp'

SAMPLE_FUNCTIONS = {'texture', 'texture2D', 'textureLod', 'texture2DLod', 'textureGrad', 'texture2DGrad', 'texelFetch',
                    'texelFetch2D', 'textureSize'}

VERTEX_STATICS = {
    'gl_Position': 'vec4', 'gl_Vertex': 'vec4', 'gl_Color': 'vec4', 'gl_Normal': 'vec3', 'gl_MultiTexCoord0': 'vec4',
    'gl_MultiTexCoord1': 'vec4', 'gl_MultiTexCoord2': 'vec4', 'mc_Entity': 'vec4', 'mc_midTexCoord': 'vec4',
    'at_tangent': 'vec4', 'at_midBlock': 'vec4', 'gl_VertexID': 'int',
}

FRAGMENT_STATICS = {'gl_FragCoord': 'vec4', 'gl_FragDepth': 'float', 'gl_FrontFacing': 'bool', 'gl_FragColor': 'vec4'}

COMPAT_OUTPUTS = {'gl_FrontColor', 'gl_BackColor', 'gl_FrontSecondaryColor', 'gl_BackSecondaryColor'}

COMPUTE_STATICS = {'gl_GlobalInvocationID': 'uvec3', 'gl_LocalInvocationID': 'uvec3', 'gl_WorkGroupID': 'uvec3',
                   'gl_LocalInvocationIndex': 'uint', 'gl_WorkGroupSize': 'uvec3', 'gl_NumWorkGroups': 'uvec3'}

ATTRIBUTES = {'gl_Vertex', 'gl_Color', 'gl_Normal', 'gl_MultiTexCoord0', 'gl_MultiTexCoord1', 'gl_MultiTexCoord2',
              'mc_Entity', 'mc_midTexCoord', 'at_tangent', 'at_midBlock'}

MATRIX_BUILTINS = {
    'gl_ModelViewMatrix': mat(4, 4), 'gl_ProjectionMatrix': mat(4, 4), 'gl_ModelViewProjectionMatrix': mat(4, 4),
    'gl_NormalMatrix': mat(3, 3), 'gl_TextureMatrix': array(mat(4, 4), 3),
    'gl_ModelViewMatrixInverse': mat(4, 4), 'gl_ProjectionMatrixInverse': mat(4, 4),
}

PROVIDED_MATRIX_BUILTINS = {'gl_ModelViewMatrix', 'gl_ProjectionMatrix', 'gl_ModelViewProjectionMatrix',
                            'gl_NormalMatrix', 'gl_TextureMatrix'}

UNSUPPORTED_BUILTINS = {
    'gl_InstanceID': 'gl_InstanceID is not available',
    'gl_PrimitiveID': 'gl_PrimitiveID is not available',
    'gl_ModelViewMatrixInverse': 'gl_ModelViewMatrixInverse is not provided; use gbufferModelViewInverse',
    'gl_ProjectionMatrixInverse': 'gl_ProjectionMatrixInverse is not provided; use gbufferProjectionInverse',
    'gl_Fog': 'gl_Fog is not provided; use fogColor/fogStart/fogEnd',
}

CORE_PROFILE = {
    'modelViewMatrix': mat(4, 4), 'projectionMatrix': mat(4, 4), 'modelViewMatrixInverse': mat(4, 4),
    'projectionMatrixInverse': mat(4, 4), 'textureMatrix': mat(4, 4), 'normalMatrix': mat(3, 3),
    'chunkOffset': GLSL_TYPES['vec3'], 'vaPosition': GLSL_TYPES['vec3'], 'vaColor': GLSL_TYPES['vec4'],
    'vaUV0': GLSL_TYPES['vec2'], 'vaUV1': GLSL_TYPES['ivec2'], 'vaUV2': GLSL_TYPES['ivec2'],
    'vaNormal': GLSL_TYPES['vec3'],
}

UNSUPPORTED_BUILTINS.update({name: f'{name} (Iris core-profile input) is not provided' for name in CORE_PROFILE})

UNSUPPORTED_BUILTIN_TYPES = dict(CORE_PROFILE)

FOG_STRUCT = 'gl_Fog'

FOG_MEMBERS = {
    'start': ('fogStart', GLSL_TYPES['float']),
    'end': ('fogEnd', GLSL_TYPES['float']),
    'density': ('fogDensity', GLSL_TYPES['float']),
    'scale': ('(1.0 / (fogEnd - fogStart))', GLSL_TYPES['float']),
    'color': ('float4(fogColor, 1.0)', GLSL_TYPES['vec4']),
}

FRAG_DATA_SLOTS = 8


def builtin_type(name):
    if name in MATRIX_BUILTINS:
        return MATRIX_BUILTINS[name]
    if name in VERTEX_STATICS:
        return GLSL_TYPES[VERTEX_STATICS[name]]
    if name in FRAGMENT_STATICS:
        return GLSL_TYPES[FRAGMENT_STATICS[name]]
    if name in COMPUTE_STATICS:
        return GLSL_TYPES[COMPUTE_STATICS[name]]
    if name == 'gl_FragData':
        return array(GLSL_TYPES['vec4'], FRAG_DATA_SLOTS)
    if name in UNIFORM_TYPES:
        return UNIFORM_TYPES[name]
    if name in TEXTURES:
        return sampler('sampler2DShadow' if name.startswith('shadowtex') else 'sampler2D')
    return None
