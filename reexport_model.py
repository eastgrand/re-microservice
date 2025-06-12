import xgboost as xgb
import pickle
import os

"""Utility script to re-export a legacy pickled Booster into the version-agnostic JSON format.

Run this once in the SAME Python environment / XGBoost version that originally created
`models/xgboost_model.pkl` (or any other .pkl).  It will produce `xgboost_model.json`, which
is safe to load from any newer XGBoost without compatibility warnings.

Usage
-----
$ python reexport_model.py path/to/xgboost_model.pkl  # optional argument
The script writes the json next to the pickle.
"""

import sys

if len(sys.argv) > 1:
    pkl_path = sys.argv[1]
else:
    pkl_path = os.path.join('models', 'xgboost_model.pkl')

if not os.path.exists(pkl_path):
    print(f"Pickle model not found: {pkl_path}")
    sys.exit(1)

json_path = os.path.splitext(pkl_path)[0] + '.json'

print(f"Loading pickled Booster from {pkl_path} …")
with open(pkl_path, 'rb') as f:
    booster = pickle.load(f)

print(f"Saving Booster in JSON format to {json_path} …")
booster.save_model(json_path)
print("Re-export complete. You can now delete the pickle or keep it as backup.") 