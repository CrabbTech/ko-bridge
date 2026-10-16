"""Command line for humans: run the server, show status, back up, register with Claude Desktop."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import sys

from . import config as C

CLAUDE_CONFIG = os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json")


def _registration() -> dict:
    uv = shutil.which("uv") or "/opt/homebrew/bin/uv"
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {"command": uv, "args": ["--directory", here, "run", "ko-bridge", "serve"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ko-bridge", description="MCP bridge to a Teenage Engineering EP-133 K.O. II")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the MCP server")
    s.add_argument("--http", action="store_true", help="Streamable HTTP instead of stdio")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8766)
    s.add_argument("--path-secret", default=None, help="random path segment; 'auto' generates one")
    s.add_argument("--public", action="store_true", help="accept any Host header (behind a tunnel)")
    sub.add_parser("status", help="config and device presence")
    b = sub.add_parser("backup", help="back up projects (and sounds) now, in the foreground")
    b.add_argument("--no-sounds", action="store_true")
    r = sub.add_parser("register", help="print (or --write) the Claude Desktop MCP server entry")
    r.add_argument("--write", action="store_true", help="merge the entry into claude_desktop_config.json (a .bak copy is kept)")
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        from .server import run
        secret = secrets.token_urlsafe(18) if a.path_secret == "auto" else a.path_secret
        if a.http and a.public and not secret:
            ap.error("--public needs --path-secret (use 'auto')")
        if a.http:
            print(f"ko-bridge MCP server on http://{a.host}:{a.port}/{secret + '/' if secret else ''}mcp", file=sys.stderr)
        run("streamable-http" if a.http else "stdio", a.host, a.port, secret, a.public)
        return 0
    if a.cmd == "status":
        from .server import get_status
        print(json.dumps(get_status(), indent=1, default=str))
        return 0
    if a.cmd == "backup":
        from .ep133 import EP133
        result = EP133().backup(sounds=not a.no_sounds, stage=lambda t: print(t, file=sys.stderr))
        print(json.dumps(result, indent=1, default=str))
        return 0
    if a.cmd == "register":
        entry = _registration()
        print(json.dumps({"mcpServers": {"ko-bridge": entry}}, indent=2))
        if a.write:
            data = {}
            if os.path.exists(CLAUDE_CONFIG):
                shutil.copy(CLAUDE_CONFIG, CLAUDE_CONFIG + ".bak")
                with open(CLAUDE_CONFIG) as f:
                    data = json.load(f)
            data.setdefault("mcpServers", {})["ko-bridge"] = entry
            with open(CLAUDE_CONFIG, "w") as f:
                json.dump(data, f, indent=2)
            print(f"written to {CLAUDE_CONFIG} (backup: .bak); restart Claude Desktop", file=sys.stderr)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
