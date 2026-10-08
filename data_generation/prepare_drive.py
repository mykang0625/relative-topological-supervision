"""Prepare locally obtained DRIVE images and manual annotations."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_generation._lib.drive import prepare_main as main

if __name__ == "__main__":
    main()
