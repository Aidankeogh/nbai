from collections import OrderedDict
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from src.utilities.global_timers import timeit, timers
from src.data.play import play_config
import h5py
import atexit
import torch
import random


def get_seasons(years=range(2001, 2020), season_types=["regular", "playoffs"]):
    seasons = []
    for season_type in season_types:
        seasons.extend([(y, season_type) for y in years])
    seasons = [f"{year}_{type}" for year, type in seasons]
    return seasons


all_seasons = get_seasons()


def custom_collate(batch):
    return batch[0]


play_indices = play_config.data_indices
play_indices.update(play_config.slice_keys)
to_extract = [
    "offense_team",
    "defense_team",
    "offense_roster",
    "defense_roster",
    "initial_event",
    "shooter",
    "shot_made",
    "shot_type",
]


def format_data(batch):
    data = {}
    validity = {}
    for key in to_extract:
        temp = torch.tensor(batch[:, play_indices[key]]).long()
        if len(temp.shape) == 1:
            temp = temp.unsqueeze(dim=1)
        if key in play_config.embedding_choices:
            # If it's an embedding choice, get the one hot rather than just the embedding value
            # The one hot should correspond to which player it was that did the action, e.g. shooting
            temp = (data[play_config.embedding_choices[key]] == temp.view(-1, 1)).long()
            temp = torch.argmax(temp, dim=1)
        if key in play_config.is_choice and play_config.is_choice[key]:
            temp = torch.argmax(temp, dim=1)
        data[key] = temp

        if key in play_config.triggers:
            triggers = play_config.triggers[key]
            validity[key] = torch.zeros(data[key].shape[0])
            for trigger in triggers:
                trigger_valid = torch.ones(data[key].shape[0])
                for trigger_key, trigger_value in trigger.items():
                    if play_config.is_choice[trigger_key]:
                        trigger_idx = play_config.choice_indices[trigger_key][
                            trigger_value
                        ]
                        condition_satisfied = data[trigger_key] == trigger_idx
                    else:
                        condition_satisfied = data[trigger_key] == trigger_value
                    trigger_valid = torch.logical_and(
                        trigger_valid, condition_satisfied
                    )
                validity[key] = torch.logical_or(validity[key], trigger_valid)

    return data, validity


class PlayDataset(Dataset):
    def __init__(
        self, seasons=all_seasons, db_name="cache/ml_db_0.0.1.h5", dp=False
    ) -> None:
        self.db_name = db_name
        self.seasons = seasons
        with h5py.File(db_name, "r", libver="latest", swmr=True) as db:
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
                return format_data(sample)
        raise Exception("Something is funky with the dataset indexing")

    def h5py_worker_init(self):
        self.db = h5py.File(self.db_name, "r", libver="latest", swmr=True)
        atexit.register(self.cleanup)

    def cleanup(self):
        self.db.close()


class BatchedPlayDataset(PlayDataset):
    def __init__(self, batch_size=32, shuffle_indices=True, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.seasons = [k for k in self.season_lengths.keys()]
        random.shuffle(self.seasons)

        self.batches_per_season = []
        for season in self.seasons:
            length = self.season_lengths[season]
            self.batches_per_season.append(-(-length // self.batch_size))
        self.n_batches = sum(self.batches_per_season)

        self.current_season = -1
        self.available_batches = 0
        self.shuffle_indices = shuffle_indices

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        while idx >= self.available_batches:
            self.current_season += 1
            self.cache_start_position = self.available_batches
            self.available_batches += self.batches_per_season[self.current_season]
            if idx >= self.available_batches:
                continue
            self.season = self.seasons[self.current_season]
            self.cache = self.db[f"raw_data/{self.season}/plays"][:]
            self.cache_len = len(self.cache)
            self.batch_order = list(range(self.cache_len))
            if self.shuffle_indices:
                random.shuffle(self.batch_order)
        idx_in_cache = idx - self.cache_start_position
        batch_start_idx = idx_in_cache * self.batch_size
        indices = self.batch_order[batch_start_idx : batch_start_idx + self.batch_size]
        raw_data = self.cache[indices]
        return format_data(raw_data)


if __name__ == "__main__":
    b = BatchedPlayDataset(3200)
    bl = DataLoader(b, collate_fn=custom_collate, batch_size=1, shuffle=False)
    timers["loading"].start()
    for batch in bl:
        data, valid = batch
    timers["loading"].stop()
    print(timers.delta())


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.h5py_worker_init()


class PlayModule(LightningDataModule):
    def __init__(
        self,
        db_name: str = "cache/ml_db_0.0.1.h5",
        batch_size: int = 32,
        val_seasons: int = 4,
        num_workers: int = 0,
        **kwargs
    ):
        super().__init__()
        self.db_name = db_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_seasons = get_seasons([2016])
        self.train_seasons = [i for i in get_seasons() if i not in self.val_seasons]
        self.dset_args = kwargs
        print(self.dset_args)

    def setup(self, stage=None):
        self.train_set = BatchedPlayDataset(
            batch_size=self.batch_size,
            seasons=self.train_seasons,
            db_name=self.db_name,
            dp=self.num_workers > 0,
            **self.dset_args
        )
        self.val_set = BatchedPlayDataset(
            batch_size=self.batch_size,
            seasons=self.val_seasons,
            db_name=self.db_name,
            dp=self.num_workers > 0,
            **self.dset_args
        )
        self.test_set = self.val_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=1,
            collate_fn=custom_collate,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            collate_fn=custom_collate,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=1,
            collate_fn=custom_collate,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
        )
