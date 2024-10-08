from src.thought_path import DataConfig, ThoughtPath
from src.utilities.global_timers import timeit
from src.data.possession import Possession
from src.data.data_utils import team_name

game_config = DataConfig("src/data/game.yaml")


class Game(ThoughtPath):
    def __init__(self, data=None):
        super().__init__(game_config, data=data)

    def __getattr__(self, key):
        if key == "plays":
            return slice(self.plays_start_idx, self.plays_end_idx)
        if key == "possessions":
            return slice(self.possessions_start_idx, self.possessions_end_idx)
        else:
            return super().__getattr__(key)

    def __repr__(self):
        return f"{self.away_team}: {self.away_team_score} @ {self.home_team}: {self.home_team_score}"


@timeit
def parse_game(in_data, out_data):
    raw_game = in_data["game"]
    game = Game()
    if len(out_data["games"]) > 0:
        prev = Game(out_data["games"][-1])
    else:
        prev = Game()
    game.play_start_idx = prev.play_end_idx
    game.play_end_idx = len(out_data["plays"])

    game.possession_start_idx = prev.possession_end_idx
    game.possession_end_idx = len(out_data["possessions"])

    # game['date'] = raw_game.data['date']
    game.home_team = team_name(raw_game.data["home_team_id"])
    game.away_team = team_name(raw_game.data["away_team_id"])
    possessions = out_data["possessions"][
        int(game.possession_start_idx) : int(game.possession_end_idx)
    ]
    end_possession = Possession(possessions[-1])

    home_on_offense = game.home_team == end_possession.offense_team
    game.home_team_score = (
        end_possession.offense_score
        if home_on_offense
        else end_possession.defense_score
    )
    game.away_team_score = (
        end_possession.defense_score
        if home_on_offense
        else end_possession.offense_score
    )

    out_data["games"].append(game.data)
