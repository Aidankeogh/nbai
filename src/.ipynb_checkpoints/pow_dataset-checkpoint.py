import torch
import pickle
import os
from torch.utils.data import Dataset
from src.data.game import Game
from src.data.possession import Possession


class NbaPosessionDataset(Dataset):
    def __init__(self, db, db_key=None):
        self.items = []
        if db_key in db:
            self.items = db[db_key][:]
        else:
            for season in db:
                print(f"Creating season {season}")
                season_games = db[season + "/games"][:]
                season_possessions = db[season + "/possessions"][:]
                for raw_game in season_games:
                    game = Game(raw_game)
                    game_item_home = torch.Tensor(
                        [game.home_team_score, game.away_team_score]
                    )
                    game_item_away = torch.Tensor(
                        [game.away_team_score, game.home_team_score]
                    )
                    game_possessions = season_possessions[
                        game.possession_start_idx : game.possession_end_idx
                    ]
                    for raw_possession in game_possessions:
                        possession = Possession(raw_possession)
                        start_item = torch.Tensor(
                            [
                                possession.offense_score - possession.score_change,
                                possession.defense_score,
                                possession.start_time,
                            ]
                        )
                        end_item = torch.Tensor(
                            [
                                possession.offense_score,
                                possession.defense_score,
                                possession.end_time,
                            ]
                        )
                        game_item = (
                            game_item_home
                            if (possession.offense_team == game.home_team)
                            else game_item_away
                        )
                        sample = torch.cat([start_item, end_item, game_item])
                        self.items.append(sample)
            if db_key:
                db[db_key] = torch.stack(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]
