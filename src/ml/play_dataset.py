from typing import OrderedDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from src.loader_pipeline import open_db


def get_seasons(years=range(2001, 2020), season_types=["regular", "playoffs"]):
    seasons = []
    for season_type in season_types:
        seasons.extend([(y, season_type) for y in years])
    seasons = [f"{year}_{type}" for year, type in seasons]
    return seasons


all_seasons = get_seasons()


class PlayDataset(Dataset):
    def __init__(self, seasons=all_seasons) -> None:
        with open_db() as db:
            self.length = 0
            self.season_lengths = OrderedDict()
            for season in seasons:
                self.season_lengths[season] = len(db[f"raw_data/{season}/plays"])
                self.length += self.season_lengths[season]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        season_start = 0
        for season, length in self.season_lengths.items():
            if idx - season_start >= length:
                season_start += length
            else:
                with open_db() as db:
                    sample = db[f"raw_data/{season}/plays"][idx - season_start]
                return sample
        raise Exception("Something is funky with the dataset indexing")


class PlayModule(LightningDataModule):
    def __init__(
        self, batch_size: int = 32, val_seasons: int = 4, num_workers: int = 0
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_seasons = all_seasons[:-val_seasons]
        self.val_seasons = all_seasons[-val_seasons:]

    def setup(self, stage=None):
        self.train_set = PlayDataset(self.train_seasons)
        self.val_set = PlayDataset(self.val_seasons)
        self.test_set = self.val_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size, num_workers=self.num_workers
        )
