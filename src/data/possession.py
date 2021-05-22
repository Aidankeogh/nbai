from src.thought_path import DataConfig, ThoughtPath
from src.utilities.global_timers import timeit
from src.data.play import Play
from src.data.data_utils import time_to_seconds, FOUL

possession_config = DataConfig("src/data/possession.yaml")
class Possession(ThoughtPath):
    def __init__(self, data=None):
        super().__init__(possession_config, data=data)

    def __repr__(self):
        return f"{self.offense_team}: {self.offense_score} -> {self.defense_team}: {self.defense_score}"

@timeit
def parse_possession(in_data, out_data):
    possession = Possession()
    raw_possession = in_data['possession']
    possession.game_idx = len(out_data["games"])
    if len(out_data['possessions']) > 0:
        prev = Possession(out_data['possessions'][-1])
    else:
        prev = Possession()
        prev.game_idx = -1
    
    possession.is_first = possession.game_idx != prev.game_idx
    possession.play_start_idx = prev.play_end_idx
    possession.play_end_idx = len(out_data['plays'])
    plays = out_data['plays'][int(possession.play_start_idx):int(possession.play_end_idx)]
    plays = [Play(p) for p in plays]
    if len(plays) == 0:  # bad possession, skip
        return

    possession.offense_team = plays[0].offense_team
    possession.defense_team = plays[0].offense_team

    switched = (possession.offense_team != prev.offense_team)
    possession.offense_score = prev.defense_score if switched else prev.offense_score 
    possession.defense_score = prev.offense_score if switched else prev.defense_score

    possession.offense_fouls_left = prev.defense_fouls_left if switched else prev.offense_fouls_left 
    possession.defense_fouls_left = prev.offense_fouls_left if switched else prev.defense_fouls_left

    if possession.is_first:
        possession.offense_score = 0
        possession.defense_score = 0

    possession.period = raw_possession.data['period']
    if possession.period != prev.period:
        possession.offense_fouls_left = 6
        possession.defense_fouls_left = 6
        
    possession.score_change = sum([int(play.score_change) for play in plays])
    possession.foul_change = sum(
        [int(play.counts_towards_penalty) for play in raw_possession.events if play.event_type == FOUL]
    )

    possession.offense_score += possession.score_change
    if possession.defense_fouls_left > 0:
        possession.defense_fouls_left -= possession.foul_change

    possession.start_time = time_to_seconds(raw_possession.start_time,  possession.period)
    possession.end_time = time_to_seconds(raw_possession.end_time,  possession.period)
    out_data['possessions'].append(possession.data)