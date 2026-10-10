"""Create initial local credentials and send passwords only through encrypted stdin."""
import argparse
import os
import secrets
import shlex
import subprocess
from pathlib import Path


def credentials(path: Path) -> dict:
    return {key: value for line in path.read_text().splitlines()
            if line and not line.startswith("#") and "=" in line
            for key, value in [line.split("=", 1)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", required=True)
    parser.add_argument("--ssh-host", default="ycloud")
    parser.add_argument("--ssh-config")
    parser.add_argument("--env", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if not args.env.exists():
        values = {"PUBLIC_ORIGIN": args.origin, "DAY30_USER": "heimdall",
                  "DAY30_PASSWORD": secrets.token_urlsafe(24), "DAY30_GUEST_USER": "guest",
                  "DAY30_GUEST_PASSWORD": secrets.token_urlsafe(24)}
        fd = os.open(args.env, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as file:
            file.write("".join(f"{key}={value}\n" for key, value in values.items()))
    values = credentials(args.env)
    if values["PUBLIC_ORIGIN"] != args.origin:
        parser.error("Existing credentials belong to a different origin")
    for prefix in ["DAY30", "DAY30_GUEST"]:
        remote = ["sudo", "-u", "day30", "env", "UV_CACHE_DIR=/var/lib/day30/uv-cache",
                  "/usr/local/bin/uv", "run", "--directory", "/opt/ai-advent-day30",
                  "--locked", "--no-sync", "python", "manage.py", values[prefix + "_USER"],
                  "--database", "/var/lib/day30/chat.sqlite3", "--password-stdin"]
        command = ["ssh", "-o", "BatchMode=yes"]
        if args.ssh_config:
            command.extend(["-F", args.ssh_config])
        command.extend([args.ssh_host, shlex.join(remote)])
        subprocess.run(command, input=values[prefix + "_PASSWORD"] + "\n", text=True, check=True)
    print("Two accounts provisioned. Credentials are in the local .env file (not printed).")


if __name__ == "__main__":
    main()
