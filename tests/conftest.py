import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend before any other matplotlib import

# Make project root importable so helpers.* works from the tests/ directory
sys.path.insert(0, str(Path(__file__).parents[1]))
