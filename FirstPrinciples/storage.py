"""
storage.py -- permanently stores each user's character trajectory maps
on disk (FU 2.6), and re-loads them later for reproduction (FU 2.7).
Plain JSON files, one per user per character -- no database needed for
10 users' worth of single-character trajectories.
"""
import json
from pathlib import Path

STORE_DIR = Path(__file__).resolve().parent / "trajectory_store"
STORE_DIR.mkdir(exist_ok=True)


def save_trajectory(user, character, path_points):
    user_dir = STORE_DIR / user
    user_dir.mkdir(exist_ok=True)
    with open(user_dir / f"{character}.json", "w") as f:
        json.dump(path_points, f)


def load_trajectory(user, character):
    file_path = STORE_DIR / user / f"{character}.json"
    if not file_path.exists():
        return None
    with open(file_path) as f:
        return json.load(f)


def list_users():
    return [d.name for d in STORE_DIR.iterdir() if d.is_dir()]
