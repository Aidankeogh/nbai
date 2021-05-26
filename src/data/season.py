from src.thought_path import DataConfig, ThoughtPath
from src.utilities.global_timers import timeit

season_config = DataConfig("src/data/season.yaml")


class Season(ThoughtPath):
    def __init__(self, data=None):
        super().__init__(season_config, data=data)

    def __repr__(self):
        playoff_str = "playoff" if self.playoffs else "regular season"
        return f"{self.year} {playoff_str}"


@timeit
def parse_season(in_data, out_data):
    season = Season()

    season.year = int(in_data["season"][0])
    season.playoffs = in_data["season"][1] == "Playoffs"

    season.play_start_idx = 0
    season.play_end_idx = len(out_data["plays"])

    season.possession_start_idx = 0
    season.possession_end_idx = len(out_data["possessions"])

    season.game_start_idx = 0
    season.game_end_idx = len(out_data["games"])

    out_data["season_info"] = season.data


if __name__ == "__main__":
    s = Season()
    s.play_start_idx = 5
