from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial import cKDTree
B=Path('Report/GeometricFidelity_2026-11-02');sys.path.insert(0,str(B.resolve()));import measure_geometry as G
