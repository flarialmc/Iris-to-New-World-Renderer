# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Flarial contributors
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iris_port.cli import main  # noqa: E402

sys.exit(main())
