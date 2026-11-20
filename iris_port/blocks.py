# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""Java Edition block names and states in block.properties, mapped to Bedrock Edition names and states.

The table covers renames that are certain. Everything else passes through unchanged and is listed as a warning, so a
person can check it against the target loader's own unknown-name warnings.
"""
import re

NAME_RENAMES = {
    'grass': 'short_grass',
    'grass_path': 'grass_path',
    'dirt_path': 'grass_path',
    'snow': 'snow_layer',
    'snow_block': 'snow',
    'lily_pad': 'waterlily',
    'sugar_cane': 'reeds',
    'cobweb': 'web',
    'nether_portal': 'portal',
    'spawner': 'mob_spawner',
    'cave_air': 'air',
    'void_air': 'air',
    'tripwire': 'trip_wire',
    'nether_quartz_ore': 'quartz_ore',
    'jack_o_lantern': 'lit_pumpkin',
    'melon': 'melon_block',
    'magma_block': 'magma',
    'note_block': 'noteblock',
    'slime_block': 'slime',
    'terracotta': 'hardened_clay',
    'red_nether_bricks': 'red_nether_brick',
    'bricks': 'brick_block',
    'end_stone_bricks': 'end_bricks',
    'sweet_berry_bush': 'sweet_berry_bush',
    'redstone_wall_torch': 'redstone_torch',
    'wall_torch': 'torch',
    'soul_wall_torch': 'soul_torch',
    'redstone_lamp': 'redstone_lamp',
    'shulker_box': 'undyed_shulker_box',
    'tall_seagrass': 'seagrass',
    'water_cauldron': 'cauldron',
    'lava_cauldron': 'cauldron',
    'powder_snow_cauldron': 'cauldron',
    'wheat': 'wheat',
    'beetroots': 'beetroot',
    'carrots': 'carrots',
    'potatoes': 'potatoes',
    'attached_melon_stem': 'melon_stem',
    'attached_pumpkin_stem': 'pumpkin_stem',
    'cocoa': 'cocoa',
    'kelp_plant': 'kelp',
    'twisting_vines_plant': 'twisting_vines',
    'weeping_vines_plant': 'weeping_vines',
    'cave_vines_plant': 'cave_vines_body_with_berries',
    'big_dripleaf_stem': 'big_dripleaf',
    'moving_piston': 'moving_block',
    'piston_head': 'piston_arm_collision',
    'sticky_piston_head': 'sticky_piston_arm_collision',
    'flower_pot': 'flower_pot',
}

STATE_RENAMES = {
    ('half', 'upper'): ('upper_block_bit', 'true'),
    ('half', 'lower'): ('upper_block_bit', 'false'),
    ('lit', 'true'): None,
    ('lit', 'false'): None,
}

DROPPED_STATES = {'waterlogged', 'snowy', 'powered', 'facing', 'axis', 'shape', 'north', 'south', 'east', 'west',
                  'up', 'down', 'distance', 'persistent', 'level'}

_ENTRY = re.compile(r'^(?:(minecraft):)?([a-z0-9_]+)((?::[a-z0-9_]+=[^:\s]+)*)$')


def map_entry(entry):
    """Returns (bedrock entry, warning or None)."""
    if entry.startswith('%') or entry.startswith('#'):
        return entry, f'tag {entry}: replace with Bedrock names'
    m = _ENTRY.match(entry)
    if not m:
        return entry, f'{entry}: not a plain block name; check by hand'
    namespace, name, states = m.group(1), m.group(2), m.group(3)
    warnings = []
    new_name = NAME_RENAMES.get(name, name)
    if new_name != name:
        warnings.append(f'{name} -> {new_name}')
    new_states = []
    for part in [s for s in states.split(':') if s]:
        key, value = part.split('=', 1)
        mapped = STATE_RENAMES.get((key, value), STATE_RENAMES.get((key, None), (key, value)))
        if mapped is None:
            warnings.append(f'state {key}={value} dropped (Bedrock has a separate block)')
            continue
        new_key = mapped[0]
        new_value = mapped[1] if mapped[1] is not None else value
        if key in DROPPED_STATES:
            warnings.append(f'state {key}={value} has no Bedrock equivalent of that name; dropped')
            continue
        new_states.append(f'{new_key}={new_value}')
    out = (f'{namespace}:' if namespace else '') + new_name + ''.join(f':{s}' for s in new_states)
    return out, ('; '.join(warnings) if warnings else None)


def map_block_properties(text):
    """Maps every block.<id> line. Returns (new text, [(line, warning)])."""
    out_lines = []
    warnings = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        logical = [line]
        while logical[-1].rstrip('\r').endswith('\\') and i + 1 < len(lines):
            i += 1
            logical.append(lines[i])
        first = i - len(logical) + 2
        joined = logical[0]
        m = re.match(r'^(\s*block\.\d+\s*=)', joined)
        if not m:
            out_lines.extend(logical)
            i += 1
            continue
        new_logical = []
        for k, part in enumerate(logical):
            body = part[len(m.group(1)):] if k == 0 else part
            cr = '\r' if body.endswith('\r') else ''
            body = body.rstrip('\r')
            cont = body.endswith('\\')
            if cont:
                body = body[:-1]
            words = re.split(r'(\s+)', body)
            mapped = []
            for w in words:
                if not w or w.isspace():
                    mapped.append(w)
                    continue
                new, warning = map_entry(w)
                if warning:
                    warnings.append((first + k, warning))
                mapped.append(new)
            text_out = ''.join(mapped) + ('\\' if cont else '') + cr
            new_logical.append((m.group(1) if k == 0 else '') + text_out)
        out_lines.extend(new_logical)
        i += 1
    return '\n'.join(out_lines), warnings
