from pbpstats.data_loader import StatsNbaEnhancedPbpLoader, StatsNbaPossessionLoader, StatsNbaShotsLoader, StatsNbaGameFinderLoader
from human_id import generate_id
import msgpack
import pickle
import torch
from collections.abc import Iterable
from collections import defaultdict
from src.data_parsing import split_events, parse_play
from src.data_format import Play
from src.data_utils import team_name
from torch.utils.data import Dataset, DataLoader

FOUL = 6
def time_to_seconds(time, period):
    return ((4 - period) * 12 * 60) + int(time.split(":")[0]) * 60 + int(time.split(":")[1])

def get_prev(prev, key, default=0):
    prev = prev
    if len(prev) == 0: 
        prev_val = default
    else:
        prev_val = prev[-1][key]
    return prev_val

def parse_possession(in_data, out_data):
    possession = {}
    raw_possession = in_data['possession']
    possession['game_idx'] = len(out_data["games"])
    is_first = possession['game_idx'] != get_prev(out_data['possessions'], 'game_idx', -1)
    possession['start_idx'] = get_prev(out_data['possessions'], 'end_idx')
    possession['end_idx'] = len(out_data['plays'])
    plays = out_data['plays'][possession['start_idx']:possession['end_idx']]
    plays = [Play(p) for p in plays]
    offensive_team = None
    offensive_team = plays[0].off_team_name
    defense_team = plays[0].def_team_name

    if len(plays) == 0 or offensive_team is None or defense_team is None:  # bad possession, skip
        return [], possession

    possession['scores'] = get_prev(out_data['possessions'], 'scores', defaultdict(int))
    possession['penalty_fouls'] = get_prev(out_data['possessions'], 'scores', defaultdict(int))
    prev_period = get_prev(out_data['possessions'], 'period', -1)
    if is_first:
        possession['scores'] = defaultdict(int)

    possession['period'] = raw_possession.data['period']
    if possession['period'] != prev_period:  # reset penalty on new period
        possession['penalty_fouls'] = defaultdict(int)
        
    possession['score_change'] = sum([int(play.score_change) for play in plays])
    possession['foul_change'] = sum(
        [int(play.counts_towards_penalty) for play in raw_possession.events if play.event_type == FOUL]
    )

    possession['scores'][offensive_team] += possession['score_change']
    possession['scores'][defense_team] += 0
    possession['penalty_fouls'][defense_team] += possession['foul_change']
    possession['penalty_fouls'][offensive_team] += 0

    possession['offense_team'] = offensive_team
    possession['defense_team'] = defense_team
    possession['period'] = raw_possession.data['period']
    possession['start_time'] = time_to_seconds(raw_possession.start_time,  possession['period'])
    possession['end_time'] = time_to_seconds(raw_possession.end_time,  possession['period'])
    out_data['possessions'].append(possession)

def parse_game(in_data, out_data):
    raw_game = in_data['game']
    game = {}
    game['start_idx'] = get_prev(out_data['games'], 'start_idx')
    game['end_idx'] = len(out_data['plays'])

    game['start_pos_idx'] = get_prev(out_data['games'], 'start_pos_idx')
    game['end_pos_idx'] = len(out_data['plays'])
    
    game['date'] = raw_game.data['date']
    possessions = out_data['possessions'][game['start_pos_idx']: game['end_pos_idx']]
    game['scores'] = defaultdict(int)
    for possession in possessions:
        for k, v in possession['scores'].items():
            game['scores'][k] += v
    game['id'] = raw_game.data['game_id']
    game['home_team'] = team_name(raw_game.data['home_team_id'])
    game['away_team'] = team_name(raw_game.data['visitor_team_id'])
    out_data['games'].append(game)

def parse_season(in_data, out_data):
    season = {}
    season['year'] = in_data['season']
    season['type']
    season['start_idx'] = get_prev(out_data['seasons'], 'start_idx')
    season['end_idx'] = len(out_data['plays'])
    season['start_game_idx'] = get_prev(out_data['seasons'], 'start_game_idx')
    season['end_game_idx'] = len(out_data['games'])
    season['start_pos_idx'] = get_prev(out_data['seasons'], 'start_pos_idx')
    season['end_pos_idx'] = len(out_data['possessions'])
    out_data['seasons'].append(season)

def dump_season(out_data, db):
    pass

def load_stats(config, db, years=[2016]):
    #db.set_namespace(config['loader']['key'])

    out_data = {
        'seasons': [],
        'games': [],
        'possessions': [],
        'plays': []
    }

    in_data = {
        'season': None,
        'game': None,
        'possession': None,
        'play': None
    }

    season_iter = []
    for season_type in ("Regular Season", "Playoffs"):
        season_iter.extend([(str(y-1) + "-" + str(y)[-2:], season_type) for y in years])

    for in_season in season_iter:
        game_iter = StatsNbaGameFinderLoader("nba", in_season[0], in_season[1], "file", "data").items
        in_data['season'] = in_season
        for in_game in game_iter:
            possession_iter = StatsNbaPossessionLoader(in_game.data['game_id'], "file", "data").items
            in_data['game'] = in_game
            for in_possession in possession_iter:
                in_data['possession'] = in_possession
                play_iter = split_events(in_possession.events)
                for in_play in play_iter:
                    in_data['play'] = in_play
                    parse_play(in_data, out_data)
                parse_possession(in_data, out_data)
            parse_game(in_data, out_data)
        parse_season(in_data, out_data)
        dump_season(out_data, db)


load_stats(None, None)