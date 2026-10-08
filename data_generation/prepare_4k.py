"""Generate the fixed paper 4K split and the fixed validation/test scenes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_generation._lib.pathfinder import main

if __name__ == "__main__":
    main(4000)
