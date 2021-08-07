from src.data.batch_loader import load_raw_data, accumulate_box_stats
from src.data.game import Game
import h5py

DB_NAME = "cache/ml_db_0.0.2.h5"

open_db = lambda: h5py.File(DB_NAME, "a")

with h5py.File(DB_NAME, "a") as db:
    if "raw_data_loaded" not in db:
        load_raw_data(db, years=range(2001, 2020))

with h5py.File(DB_NAME, "a") as db:
    if "box_stats_accumulated" not in db:
        accumulate_box_stats(db)
