"""Account administration; passwords are read from stdin/getpass, never arguments."""
import argparse
import getpass
import re
import sys
from pathlib import Path

from chat import Store, password_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("username")
    parser.add_argument("--database", type=Path, default=Path("chat.sqlite3"))
    parser.add_argument("--password-stdin", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{3,32}", args.username):
        parser.error("Username: 3–32 ASCII letters/digits or _.-")
    password = sys.stdin.readline().rstrip("\r\n") if args.password_stdin else getpass.getpass("Password: ")
    if not 10 <= len(password) <= 256:
        parser.error("Password must have 10–256 characters")
    store = Store(args.database)
    try:
        with store.db:
            store.db.execute("INSERT INTO users (name,password) VALUES (?,?) "
                             "ON CONFLICT(name) DO UPDATE SET password=excluded.password",
                             (args.username, password_hash(password)))
            store.db.execute("DELETE FROM sessions WHERE user_id=(SELECT id FROM users WHERE name=?)",
                             (args.username,))
        print("Account saved; previous sessions revoked.")
    finally:
        store.db.close()


if __name__ == "__main__":
    main()
