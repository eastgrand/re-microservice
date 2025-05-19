import numpy as np
import types

# Monkey patch np.int to handle the deprecation
if not hasattr(np, 'int'):
    np.int = int
