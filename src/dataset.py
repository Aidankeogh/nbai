import os
import torch
import numpy as np
import glob
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch import tensor
from torchvision import transforms, utils
import time

class Nbaset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, games_folder="./game_arrays", years=[2016], name="nba_2016_dataset.h5"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.years = [str(y) for y in years]
        all_games = glob.glob(games_folder + "/*")
        self.valid_games = [g for g in all_games if os.path.basename(g).split('-')[0] in self.years]
        if os.path.exists(name):
            self.database = h5py.File(name, "r")
            self.play_indices = self.database['indices']
        else:
            self.database = h5py.File(name, "a")
            self.play_indices = []

            for game_file in self.valid_games:
                np_game = np.load(open(game_file, 'rb'), allow_pickle=True)
                game_group = self.database.create_group(os.path.basename(game_file))

                game_group['meta'] = np.array(np_game[0])

                for i, play in enumerate(np_game[1]):
                    self.play_indices.append(os.path.basename(game_file) + "/" + str(i))
                    play_group = game_group.create_group(str(i))

                    play_group['meta'] =  np.array(play[0])

                    for j, event_tree in enumerate(play[1]): 
                        play_group[str(j)] = event_tree
            self.play_indices = np.array(self.play_indices, dtype='S')
            self.database.create_dataset('indices', data=self.play_indices)

    def __len__(self):
        return len(self.play_indices)

    def __getitem__(self, idx):
        play_key = self.play_indices[idx]
        play_meta = np.array(self.database[play_key]['meta'])
        game_key = str(play_key).split('/')[0].split("'")[1]
        game_meta = np.array(self.database[game_key]['meta'])
        play = self.database[play_key]
        event_trees = []
        for i in range(len(play) - 1):
            event_trees.append(np.array(play[str(i)]))
        return [game_meta, play_meta, event_trees]

def custom_collate(batch):
    arr = []
    for b in batch:
        arr.append([tensor(b[0]), tensor(b[1]), tensor(b[2])])
    return arr
