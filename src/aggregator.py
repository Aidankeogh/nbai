
from src.data.batch_loader import load_raw_data,accumulate_box_stats
from src.data.possession import Possession
from collections import defaultdict
from src.data.season import Season
from src.data.play import Play
from src.data.game import Game
import os

class Aggregator:
    def __init__(self) -> None:
        
        self.aggregation = defaultdict(dict)
        self.games_indices = defaultdict(set)
        self.possessions_indices = defaultdict(set)
        self.plays_indices = defaultdict(set)
        self.all_players = set()
    
    def aggregate(self, season, db):
        season_info = Season(db[f"raw_data/{season}/season_info"])
        game_idxs = [gms for gms in 
        range(season_info.game_start_idx,season_info.game_end_idx)]

        for game_idx in game_idxs:
            game = Game(db[f"raw_data/{season}/games"][game_idx])
            self.__aggregate_possessions__(season,game,game_idx,db)
        
        for player in self.all_players:
            games = list(self.games_indices[player])
            possessions = list(self.possessions_indices[player])
            plays = list(self.plays_indices[player])
            self.aggregation[player][season] = (games,possessions,plays)
        
        assert len(self.aggregation.keys()) == len(self.all_players)

    def __aggregate_possessions__(self,season,game,game_idx,db):
        possession_idxs = list(range(game.possession_start_idx,
                game.possession_end_idx))

        for possession_idx in possession_idxs:
            possession = Possession(db[f"raw_data/{season}/possessions"][possession_idx])
            self.__aggregate_plays__(season,game_idx,possession,possession_idx,db)

    
    def __aggregate_plays__(self,season,game_idx,possession,possession_idx,db):
        plays_idxs = list(range(possession.play_start_idx,
            possession.play_end_idx))
        
        for play_idx in plays_idxs:
            play = Play(db[f"raw_data/{season}/plays"][play_idx])
            offensive_roster = play.offense_roster
            defensive_roster = play.defense_roster

            for player in offensive_roster:
                self.plays_indices[player].add(play_idx)
                self.possessions_indices[player].add(possession_idx)
                self.games_indices[player].add(game_idx)
                self.all_players.add(player)

            for player in defensive_roster:
                self.plays_indices[player].add(play_idx)
                self.possessions_indices[player].add(possession_idx)
                self.games_indices[player].add(game_idx)
                self.all_players.add(player)
           
    def __getitem__(self, key):
        player, season = key
        if player in self.aggregation.keys():
            if season in self.aggregation[player].keys():
                return self.aggregation[player][season]
            else:
                raise Exception("No {s} for {p}".format(p = player,s = season))
        else:
            raise Exception("No {p} in aggregation".format(p = player))



if __name__ == "__main__":
    import h5py

    db_path = "cache/batch_loader_unit_test.h5"
    if os.path.exists(db_path):
        os.remove(db_path)

    with h5py.File(db_path, "a") as db:
        load_raw_data(db, years=[2018], season_types=["Playoffs"])
        season = "2018_playoffs"
        aggregator = Aggregator()
        aggregator.aggregate(season,db)
        print()
        print()
        print("stephen-curry stats",aggregator["stephen-curry",season])
   
    