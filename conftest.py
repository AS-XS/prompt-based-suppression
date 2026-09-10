import sys
from pathlib import Path

import numpy  # Load before DSPy installs its lazy module proxy.

sys.path.insert(0, str(Path(__file__).resolve().parent))
