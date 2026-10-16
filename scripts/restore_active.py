#!/usr/bin/env python3
"""Make project N the active project again (metadata set on /projects, with readback); no archive is written.

    uv run python scripts/restore_active.py 1
"""
from __future__ import annotations

import json
import sys

from ko_bridge.ep133 import EP133


def main() -> int:
    project = int(sys.argv[1])
    ep = EP133(lock_owner="ko-bridge restore_active")
    print("before:", ep.active_project())
    print("reload:", json.dumps(ep.client.reload_project(project, cycle=False)))
    print("after:", ep.active_project())
    return 0


if __name__ == "__main__":
    sys.exit(main())
