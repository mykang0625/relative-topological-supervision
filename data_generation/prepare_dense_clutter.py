"""Generate the paper's separate dense-clutter corpus."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_generation._lib.dense_clutter import main as main

if __name__ == "__main__":
    main()
