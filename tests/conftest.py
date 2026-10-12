# Lets the tests import simulation.py and analysis.py from the project folder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
