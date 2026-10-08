"""Build the paper's annotation-derived DRIVE query targets."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_generation._lib.drive import targets_main as main

if __name__ == "__main__":
    main()
