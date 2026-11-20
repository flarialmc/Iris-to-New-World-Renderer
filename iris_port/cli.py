# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
"""Command line: iris-port <pack> <out> [--report FILE] [--drop PATTERN ...] [--no-standins] [--force]."""
import argparse
from pathlib import Path
import shutil
import sys

from .convert import FileConverter, PackContext
from .pack import convert_pack, format_report
from .samplers import specialize


def main(argv=None):
    parser = argparse.ArgumentParser(prog='iris-port', description='Mechanical GLSL to HLSL port of an Iris shader '
                                     'pack to the NWR shader-pack format 2 (main-shape programs).')
    sub = parser.add_subparsers(dest='command', required=True)
    pack = sub.add_parser('pack', help='convert a whole pack')
    pack.add_argument('source', help='pack folder (holding shaders/) or the shaders/ folder itself')
    pack.add_argument('out', help='output pack folder (replaced)')
    pack.add_argument('--report', help='report file (default: <out>.port-report.txt)')
    pack.add_argument('--drop', action='append', help='program name pattern to leave out (default: physics_*, voxy_*); dh_* is always left out')
    pack.add_argument('--no-standins', action='store_true',
                      help='delete declarations of uniforms and attributes the API lacks instead of zero stand-ins')
    pack.add_argument('--force', action='store_true', help='exit 0 even when blocking entries remain')
    one = sub.add_parser('file', help='convert one GLSL file to stdout')
    one.add_argument('path')
    one.add_argument('--stage', choices=['vertex', 'fragment'])
    args = parser.parse_args(argv)

    if args.command == 'file':
        path = Path(args.path)
        ctx = PackContext()
        text = specialize({path.name: path.read_text(encoding='utf-8')}, report=ctx.report)[path.name]
        converter = FileConverter(ctx, path.name, text, args.stage)
        converter.walk_global('collect')
        converter.walk_global('convert')
        converter.convert_macros()
        sys.stdout.write(converter.output())
        for e in sorted(ctx.entries, key=lambda e: (e.line, e.category, e.message)):
            sys.stderr.write(f'{e.file}:{e.line}: {e.category}: {e.message}\n')
        return 0

    out = Path(args.out)
    source = Path(args.source).resolve()
    if out.resolve() == source or source in out.resolve().parents:
        parser.error('the output folder must be outside the source pack')
    if out.exists():
        shutil.rmtree(out)
    summary, entries = convert_pack(args.source, out, drop=args.drop, standins=not args.no_standins)
    report = format_report(summary, entries)
    report_path = Path(args.report) if args.report else out.with_name(out.name + '.port-report.txt')
    report_path.write_text(report, encoding='utf-8')
    head = report.split('\n\n', 1)[0]
    print(head)
    print(f'report: {report_path}')
    if summary['blocking'] and not args.force:
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
