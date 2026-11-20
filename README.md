<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Copyright (C) 2026 Flarial contributors
-->
# iris-port

Mechanical GLSL → HLSL conversion of an Iris (OptiFine-style) shader pack to the NWR shader-pack
format 2: Flarial's HLSL API for its New World Renderer. The output uses format 2's
*main shape*: `void main()` programs with `NWR_VARYING`/`NWR_OUTPUT` declarations and GLSL-style
built-ins. The loader generates the entry points.

The tool is Python 3.10+ with the standard library only. It is licensed AGPL-3.0-or-later (see
`LICENSE`). It does not read, link or bundle anything from the client.

## Usage

```
python iris-port.py pack <pack folder> <out folder> [--report FILE] [--drop PATTERN]... [--no-standins] [--force]
python iris-port.py file <program.glsl> [--stage vertex|fragment]
```

- `<pack folder>` is the folder that holds `shaders/`, or `shaders/` itself. `<out folder>` is replaced.
- The report goes to `<out folder>.port-report.txt` by default. It lists every construct the tool did
  not convert, by file and line, grouped by category, plus a summary of API gaps.
- The exit code is 2 while *blocking* entries remain (see below); `--force` returns 0 anyway.
- `--drop` leaves programs out by name pattern. The default is `physics_*` and `voxy_*` (Physics Mod
  and Voxy programs). Distant Horizons programs (`dh_*`) are always left out and reported as
  `dropped`: NWR runs no shader pack while Distant Horizons is on. Include files that match a pattern
  and are only included by dropped programs are dropped too.
- `--no-standins` deletes declarations of uniforms and attributes the API lacks. By default they
  become zero-valued stand-ins, so the rest of the program can still be compiled and checked.

The output is deterministic: the same input gives byte-identical files and report.

**Licences of packs.** Converting a pack makes a modified copy of it. Check the pack's licence before
you share the output.

## Report categories

| Category | Blocking | Meaning |
|---|---|---|
| `untyped-product` | yes | A `*` where an operand's type is unknown and the other could be a matrix. A missed `mat * mat` would compile as a component-wise product, so the tool does not guess |
| `untyped-vector-product` | no | A `*` of a 2- or 3-component vector and an operand of unknown type, kept component-wise. HLSL refuses a component-wise product of such a vector and any matrix, so a wrong guess fails to compile instead of rendering wrongly |
| `untyped-constructor` | yes | `matN(x)` with `x` of unknown type: a matrix casts, a scalar builds a diagonal |
| `parse` | yes | A statement the tool could not parse that contains a product (for example a preprocessor line in the middle of an expression) |
| `api-gap` | no | Something the format 2 API does not provide: a uniform, texture, attribute, built-in variable or function |
| `hand` | no | Needs a person: array varyings, constructors of structs with array members outside initialisers, buffer blocks without a literal binding or with an unsized array that is not the last member, sampler arguments that are not texture names, atomics whose result is used, `shaders.properties` keys |
| `block` | no | A `block.properties` entry that was renamed, or that may not match a Bedrock block |
| `dropped` | no | A program left out by `--drop`, or a Distant Horizons program |
| `parse-note` | no | A statement the tool could not parse, without a product in it, or one declaring a variable of a type no file of the pack declares (a mod's interface type: a program that reaches it cannot compile); only type renames were applied |
| `note` | no | Informational: a uniform declared with a type other than the API's |

## Mapping rules

| GLSL | HLSL (format 2) |
|---|---|
| `#version`, `#extension` | removed; the line stays empty so line numbers keep matching |
| `#include "x.glsl"` | `#include "x.hlsli"`; `.glsl` files are written as `.hlsli`, programs as `<name>.vsh.hlsl` / `<name>.fsh.hlsl` |
| `vecN`, `ivecN`, `uvecN`, `bvecN`, `matN`, `matCxR` | `floatN`, `intN`, `uintN`, `boolN`, `floatNxN`, `floatCxR` |
| `M * v`, `v * M`, `A * B` | `mul(v, M)`, `mul(M, v)`, `mul(B, A)` (format 2 matrices hold GLSL's transpose) |
| `v *= M` | `v = mul(M, v)` |
| `vecN(s)`, `vecN(wider vector)` | `((floatN)(s))` |
| `vecN(a, b)` with spare components | the last argument is cut with a swizzle |
| `matN(M)` from a larger matrix | `((floatNxN)(M))` |
| `matN(1.0)` | the diagonal written out; `matN(x)` with a variable scalar is reported |
| `T[n](a, b)`, struct constructors in an initialiser | `{ a, b }` |
| `uniform …` provided by the API or by `uniform.*` / `texture.*` / `customTexture.*` in `shaders.properties` | removed |
| `uniform image1D/2D/3D x;` named by `image.x`, `uniform sampler1D/3D/usampler3D s;` naming an image's sampler, `uniform sampler3D s;` naming a raw `TEXTURE_3D` texture | removed |
| `layout(std430, binding = n) buffer B { members } b;` | `struct B { members }; NWR_BUFFER(n, B, nwrBufferN);` and `#define b nwrBufferN[0]`; an anonymous block defines each member as `nwrBufferN[0].member` |
| `buffer B { T data[]; };` (one unsized array, anonymous) | `NWR_BUFFER(n, T, data);` |
| `buffer B { T data[]; } b;` (one unsized array behind an instance) | `NWR_BUFFER(n, T, nwrBufferN);`, and `b.data[i]` becomes `nwrBufferN[i]` in every file |
| `buffer B { members; T data[]; } b;` (an unsized array last, after other members) | `struct B { members }; NWR_BUFFER(n, B, nwrBufferN); NWR_BUFFER_TAIL(n, T, nwrBufferN_data, B);`, `#define b nwrBufferN[0]`, and `b.data[i]` becomes `nwrBufferN_data[i]` (anonymous: `data` names the tail) |
| a function with `sampler*` parameters (`float f(sampler2D s, vec2 p)`) | one copy per texture tuple it is called with, `f_nwr_<texture>`, without the sampler parameters; calls through other such functions are followed; copies are written where the original was, each compiled only into programs whose includes call it and only where the textures it names are declared (the copy keeps the `#if` conditions around their `uniform` declarations); a function never called with a known texture is removed (noted) |
| `#if`/`#elif` comparisons whose side may be negative (`#if SHADOW_QUALITY > -1`) | both sides offset by `1073741824` (2^30): fxc's preprocessor is unsigned 32-bit |
| `T a, b;` and `T a[n], b;` members of a buffer block | one struct field per declarator |
| `shared T x[n];`, `shared T x[a][b];` | `groupshared T …;` |
| `atomicAdd(m, v);` … `atomicExchange(m, v);`, `atomicCompSwap(m, c, v);` as statements (also in macro bodies) | `InterlockedAdd(m, v);` … `InterlockedCompareStore(m, c, v);` |
| `pack*`/`unpack*` `Unorm4x8`, `Snorm4x8`, `Unorm2x16`, `Snorm2x16`, `Half2x16` | unchanged; the format 2 library defines them |
| `S(a, b)` outside an initialiser | `nwrMake_S(a, b)`, a maker function written after `struct S` (structs without array members, only when the pack calls `S(`) |
| `T[] x = T[](…);` | `x.length()` is the initializer's element count |
| `texture3D(s, p)` | `texture(s, p)` |
| other `uniform …` | reported; zero stand-in unless `--no-standins` |
| `attribute …`, vertex `in …` of a known attribute (`gl_*`, `mc_Entity`, `mc_midTexCoord`, `at_tangent`, `at_midBlock`) | removed |
| `out`/`in`/`varying` between stages | `NWR_VARYING(T, name)`; `flat` → `NWR_VARYING_FLAT`, `noperspective` → `NWR_VARYING_NOPERSPECTIVE` |
| fragment `layout(location = n) out T name` | `NWR_OUTPUT(n, T, name)` |
| fragment `out T name` without a location | `NWR_OUTPUT(n, T, name)`, n the lowest location no explicit output of the file takes, in declaration order (GLSL's implicit assignment) |
| global non-`const` variables | `static` |
| `precision …;`, `highp`, `mediump`, `lowp` | removed |
| `fract`, `mix`, `inversesqrt`, `dFdx`, `dFdy`, `floatBitsToInt` … | `frac`, `lerp`, `rsqrt`, `ddx`, `ddy`, `asint` … |
| `mix(a, b, bvec)` | `(c ? b : a)` |
| `atan(y, x)` | `atan2(y, x)` |
| `lessThan(a, b)` … `notEqual(a, b)` | `((a) < (b))` … `((a) != (b))` |
| `not(b)`, `matrixCompMult(a, b)` | `(!(b))`, `((a) * (b))` |
| `isnan`, `isinf` | `nwrIsNan`, `nwrIsInf` (bit tests in the format 2 library) |
| vector `a == b`, `a != b` | `all(a == b)`, `any(a != b)` |
| `.stpq` swizzles | `.xyzw` |
| `a.length()` on an array or vector of known size | the size |
| identifiers that are HLSL keywords (`line`, `point`, `sample`, `matrix` …) | suffixed with `_` |
| `gl_Fog.start/end/density/scale/color` | `fogStart`, `fogEnd`, `fogDensity`, `(1.0 / (fogEnd - fogStart))`, `float4(fogColor, 1.0)` |
| `T[n] name` (locals and globals) | `T name[n]` |
| `struct S { … } a;` | `struct S { … }; static S a;` |
| function parameters typed `sampler2D`, `isampler2D`, `usampler2D`, `sampler2DShadow` | `Texture2D<float4> t, SamplerState t_nwrSampler`; texture calls on `t` pass the companion, and callers pass `nwrSampler_<texture>` (the loader's sampler for that texture), so a depth texture stays point-sampled inside the function as GL keeps its unit's filter; a parameter named like an API texture is renamed `name_` |
| `<name>.gsh` beside `<name>.vsh`: `layout(triangles) in`, `layout(triangle_strip, max_vertices = 3K) out`, K `EndPrimitive()` calls | folded into the vertex program: its `main` becomes `nwr_vertex_main`, its outputs plain globals; the geometry body runs after it as `nwr_geometry_main`, with every input array (and `gl_in[i].gl_Position`) reading the vertex's own value. `/* NWR_EMISSIONS: K */` makes the loader draw the program K times with `NWR_EMISSION` = 0..K-1; draw k keeps the outputs of the k-th emitted primitive and gives a vertex that does not emit it a NaN position, so its triangle is discarded. Exact for tests that reject a triangle when any vertex fails; tests comparing different vertices see equal values (noted). Other geometry programs stay `api-gap` |
| `for (int i = …; …) { … }` | `{ for (int i = …; …) { … } }`: FXC scopes a for-loop declaration to the enclosing block, so a later declaration of the same name there would be a redefinition (bodies that are not a block keep their form) |
| `#define F()` (empty parameter list, which FXC rejects) | `#define F`, and `F()` uses become `F` |
| macro parameters that are HLSL keywords | suffixed with `_` |
| a `#include` inside a function body | the included file is converted in the scope of that function |
| global `T g = e;` whose initialiser reads a stage input (`gl_FragCoord`, an attribute, a varying, or another such global) | `static T g; T nwrInit_g() { return e; }` plus a `NWR_INIT_g` macro that `main()` runs first, in dependency order: HLSL runs static initialisers before the entry wrapper fills the inputs; GLSL evaluates them at the start of `main()`; the loader's own fallback (`hlsl.md`) only moves one-line statics, so multi-line initialisers rely on this |
| a `for` or `while` loop whose condition reads a variable, parameter, uniform or a macro of one | `[loop]` in front: FXC would try to unroll it around a texture read and fail; GLSL runs it with undefined derivatives, as `[loop]` does |
| a `for` loop with a constant trip count of 16 or more (a literal, a `const` or a macro's default value) | `[loop]` in front: FXC unrolls every loop with a known count, and a long raymarch unrolled around calls that loop again takes minutes to compile; GLSL leaves unrolling to the driver |
| `#if` branches inside a statement or global initialiser | each branch combination (up to 64) is converted separately; a statement whose branches each end it with their own `;` runs to the `#endif` |
| `MACRO(args)` without a `;` after it, where every definition of `MACRO` ends in `;` or `}` | taken as a whole statement |
| `gl_FrontColor`, `gl_BackColor`, `gl_FrontSecondaryColor`, `gl_BackSecondaryColor` writes | removed (noted): the compatibility profile passes them to the fragment stage's `gl_Color`, which format 2 fragment programs do not have |
| `texture.<stage>.<name> = … TEXTURE_3D …` where `<name>` is a built-in sampler (`colortex1`, `depthtex1`, `noisetex` …) | the override and every read of it in programs that declare `uniform sampler3D <name>` (and in includes only those reach) become `<name>_nwrVolume`: Iris binds the volume only to samplers declared 3D, so programs of the stage that declare `<name>` as 2D keep reading the built-in |
| `IS_IRIS` in preprocessor lines and `shaders.properties` | `IS_NWR`: NWR is the Iris-like host, so the pack takes its Iris paths |
| `shaders.properties` | `nwr.format = 2` and `nwr.weather = columns` (Java's rain and snow column mesh) added when missing |
| `dimension.properties` | `minecraft:the_nether` → `minecraft:nether` |
| `block.properties` | Java names and states mapped by a table (`half=upper` → `upper_block_bit=true`, `lily_pad` → `waterlily` …); everything uncertain is reported |

Left unchanged, because format 2 provides them under the GLSL names: `texture`, `texture2D`,
`textureLod`, `textureGrad`, `texelFetch`, `textureSize`, `textureGather`, `shadow2D` (the loader binds them per
texture), `mod` (GLSL semantics), `gl_Position`, `gl_FragCoord`, `gl_FragData`, `gl_FragColor`,
`gl_FragDepth`, `gl_Vertex`, `gl_Color`, `gl_Normal`, `gl_MultiTexCoord0..2`, `gl_ModelViewMatrix`,
`gl_ProjectionMatrix`, `gl_NormalMatrix`, `gl_TextureMatrix`, `ftransform()`, `tex` (the atlas), and the `RENDERTARGETS`,
`DRAWBUFFERS` and `const` directives. Compute programs (`.csh`) keep `layout(local_size_…) in;`,
`workGroups`, `workGroupsRender`, `imageLoad`, `imageStore`, `imageSize`, `imageAtomic*`, the barriers and the
`gl_*InvocationID`, `gl_WorkGroupID`, `gl_LocalInvocationIndex`, `gl_WorkGroupSize` and `gl_NumWorkGroups` built-ins: the loader reads
or rewrites them (`docs/nwr/iris-api/compute.md`).

## How products are typed

The tool parses every expression and types it from declarations: the API's uniforms and built-ins,
the pack's globals, functions (with overloads), structs and locals in scope. Macros are typed where
they are used: a product inside a macro body becomes `mul()` only when every use agrees, and is
reported otherwise. Preprocessor branches are not evaluated; a name declared with different types in
different branches is typed as the set of them.

## What it does not convert

- Geometry and tessellation programs.
- `uniform` interface blocks, buffer blocks with instance arrays or with an unsized array that is not the last member,
  and atomics whose returned value is used.
- Array images (reported as API gaps).
- Sampler parameters passed anything but a texture name (a macro, an expression, an array element) are reported for a
  person.
- `outerProduct`, `*Offset` and `textureProj` sampling, cube
  textures, pack/unpack built-ins (reported as API gaps).
- Array varyings, constructors of structs with array members outside initialisers.
- Statements with a preprocessor line in the middle that a rewrite would span (reported).
- Java → Bedrock block data beyond the table: tags (`%…`) and most states are left for a person.
- The pack's own render-target formats, programs format 2 does not run yet, and
  `nwr.*` keys in `shaders.properties`.

## Tests

```
python -m unittest discover -s tests
```

`tests/test_rules.py` checks each mapping rule on small snippets. `tests/test_pack.py` converts
`tests/samples/pack` and compiles every program with fxc (shader model 5.1) against a stand-in of the
API built from the tool's own tables (`tests/fxc_stub.py`); set `IRIS_PORT_FXC` if fxc is not on the
Windows SDK path. All samples were written for these tests.
