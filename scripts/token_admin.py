#!/usr/bin/env python3
"""Token administration CLI for MemoryBridge.

Usage:
  bash scripts/token_admin.sh create --label LABEL
  uv run python scripts/token_admin.py create [--label LABEL]
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from memory_bridge.core.tokens import TokenRecord, TokenStore, TokenStoreError  # noqa: E402


def main() -> None:
    args: list[str] = sys.argv[1:]
    if not args:
        print("Usage: token_admin.py {create|list|delete} [--label LABEL] [TOKEN]")
        sys.exit(1)

    db_path: str = os.environ.get("TOKEN_DB_PATH", "data/tokens.db")
    try:
        store: TokenStore = TokenStore(db_path)
    except TokenStoreError as e:
        print(f"Error: {e}")
        sys.exit(1)

    cmd: str = args[0]

    if cmd == "create":
        label: str = ""
        if len(args) >= 3 and args[1] == "--label":
            label = args[2]
        token: str = store.create(label)
        print(f"Token: {token}")
        if label:
            print(f"Label: {label}")
        print()
        print("Use this token in requests:")
        print(f"  Authorization: Bearer {token}")

    elif cmd == "list":
        records: list[TokenRecord] = store.list_all()
        if not records:
            print("No tokens.")
            return
        print(f"{'ID':<5} {'Token':<38} {'Label':<20} {'Created'}")
        print("-" * 90)
        for r in records:
            print(f"{r.id:<5} {r.token:<38} {r.label:<20} {r.created_at}")

    elif cmd == "delete":
        if len(args) < 2:
            print("Usage: token_admin.py delete TOKEN")
            sys.exit(1)
        store.delete(args[1])
        print(f"Token deleted: {args[1]}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
