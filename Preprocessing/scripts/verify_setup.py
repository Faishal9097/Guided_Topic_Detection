"""
Step 1 verification script — confirms environment is ready.
Run this to check all required libraries import correctly.
"""

import sys
import importlib

REQUIRED_LIBS = [
    "pandas",
    "numpy",
    "networkx",
    "sklearn",
    "requests",
    "gdown",
    "tqdm",
]

def check_environment():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("-" * 50)

    all_ok = True
    for lib in REQUIRED_LIBS:
        try:
            module = importlib.import_module(lib)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {lib:15s} — version {version}")
        except ImportError:
            print(f"❌ {lib:15s} — NOT INSTALLED")
            all_ok = False

    print("-" * 50)
    if all_ok:
        print("All required libraries are installed correctly.")
    else:
        print("Some libraries are missing. Run: pip install -r requirements.txt")

if __name__ == "__main__":
    check_environment()