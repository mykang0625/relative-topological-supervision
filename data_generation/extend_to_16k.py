"""Verify paper 4K, then add its missing 12K training scenes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_generation._lib.pathfinder import main

if __name__ == "__main__":
    main(16000)
