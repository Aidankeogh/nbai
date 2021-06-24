from typing import OrderedDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import h5py
import atexit
import torch


def get_seasons(years=range(2001, 2020), season_types=["regular", "playoffs"]):
    seasons = []
    for season_type in season_types:
        seasons.extend([(y, season_type) for y in years])
    seasons = [f"{year}_{type}" for year, type in seasons]
    return seasons


all_seasons = get_seasons()


class PlayDataset(Dataset):
    def __init__(self, seasons=all_seasons, db_name="cache/ml_db_0.0.1.h5", dp=False) -> None:
        self.db_name = db_name
        self.seasons = seasons
        with h5py.File(db_name, "r", libver='latest', swmr=True) as db:
            self.length = 0
            self.season_lengths = OrderedDict() 
            for season in seasons:
                self.season_lengths[season] = len(db[f"raw_data/{season}/plays"])
                self.length += self.season_lengths[season]
        self.db = None
        if not dp:
            self.db = h5py.File(self.db_name, "r", libver="latest", swmr=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        season_start = 0
        for season, length in self.season_lengths.items():
            if idx - season_start >= length:
                season_start += length
            else:
                sample = self.db[f"raw_data/{season}/plays"][idx - season_start]
                return sample
        raise Exception("Something is funky with the dataset indexing")

    def h5py_worker_init(self):
        self.db = h5py.File(self.db_name, "r", libver="latest", swmr=True)
        atexit.register(self.cleanup)

    def cleanup(self):
        self.db.close()


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.h5py_worker_init()


class PlayModule(LightningDataModule):
    def __init__(
        self, db_name: str = "cache/ml_db_0.0.1.h5", batch_size: int = 32, val_seasons: int = 4, num_workers: int = 0
    ):
        super().__init__()
        self.db_name = db_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_seasons = get_seasons([2016], ["playoffs"]) #all_seasons[:-val_seasons]
        self.val_seasons = get_seasons([2017], ["playoffs"])

    def setup(self, stage=None):
        self.train_set = PlayDataset(self.train_seasons, self.db_name, dp=self.num_workers > 0)
        self.val_set = PlayDataset(self.val_seasons, self.db_name, dp=self.num_workers > 0)
        self.test_set = self.val_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=worker_init_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=worker_init_fn
        )
