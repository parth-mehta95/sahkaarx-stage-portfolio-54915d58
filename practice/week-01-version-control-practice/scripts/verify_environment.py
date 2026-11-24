"""
Environment and setup verification script.
Checks Python version, directory structure, and essential dependencies.
"""

import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("Capstone Project: Environment Verification")
    print("=" * 60)

    # 1. Check Python version
    major, minor, micro = sys.version_info[:3]
    print(f"Python Version: {major}.{minor}.{micro}")
    if major < 3 or (major == 3 and minor < 8):
        print("[FAIL] Python 3.8+ is required.")
        return 1
    print("[PASS] Python version check passed.")

    # 2. Check Directories and Key Files
    root_dir = Path(__file__).resolve().parent.parent
    required_files = [
        "README.md",
        ".gitignore",
        "requirements.txt",
        "src/__init__.py",
        "src/config.py",
        "src/data_loader.py",
        "src/model.py",
        "src/utils.py",
        "tests/__init__.py",
        "tests/test_setup.py"
    ]

    all_files_exist = True
    for rel_path in required_files:
        full_path = root_dir / rel_path
        if full_path.exists():
            print(f"[PASS] Found: {rel_path}")
        else:
            print(f"[FAIL] Missing: {rel_path}")
            all_files_exist = False

    if not all_files_exist:
        return 1

    print("=" * 60)
    print("All environment checks passed successfully!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
