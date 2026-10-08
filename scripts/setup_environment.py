"""Install the shared MuJoCo requirements inside the project environment."""
from pathlib import Path
import subprocess
import sys


def main():
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Activate the Python 3.11 robot-sim environment first.")

    root = Path(__file__).resolve().parents[1]
    print(f"Installing project packages into {sys.prefix}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(root / "requirements.txt")],
        cwd=root,
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
    main()
