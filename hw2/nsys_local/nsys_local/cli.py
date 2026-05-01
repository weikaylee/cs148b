from __future__ import annotations

import subprocess
import sys


def _extract_command(argv: list[str]) -> list[str]:
    if not argv:
        return []

    if argv[0] != "profile":
        return argv

    remaining = argv[1:]
    for index, token in enumerate(remaining):
        if token == "python":
            return [sys.executable, *remaining[index + 1 :]]
        if token == "--":
            command = remaining[index + 1 :]
            if command and command[0] == "python":
                return [sys.executable, *command[1:]]
            return command
    return []


def main() -> int:
    command = _extract_command(sys.argv[1:])
    if not command:
        print("Usage: uv run nsys profile -o <output> python benchmark.py [args...]", file=sys.stderr)
        return 1

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())