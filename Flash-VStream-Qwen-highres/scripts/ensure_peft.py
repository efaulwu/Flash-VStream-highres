#!/usr/bin/env python3
import importlib.util
import subprocess
import sys


def main() -> None:
    if importlib.util.find_spec("peft") is not None:
        print("peft_ready=True")
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "peft"])
    ok = importlib.util.find_spec("peft") is not None
    print(f"peft_ready={ok}")
    if not ok:
        raise RuntimeError("Failed to make peft available")


if __name__ == "__main__":
    main()
