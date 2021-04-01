from pbpstats.data_loader import StatsNbaEnhancedPbpLoader, StatsNbaPossessionLoader, StatsNbaShotsLoader, StatsNbaGameFinderLoader
from human_id import generate_id
import msgpack
import pickle
import redis
import torch
from collections.abc import Iterable
from collections import defaultdict
from src.data_parsing import split_events, parse_event_contents
from src.data_format import Play
from src.data_utils import team_name
from torch.utils.data import Dataset, DataLoader


FOUL = 6
def generate_possession(p, data, db=None):
    word_id = "pos-" + generate_id(word_count=6)
    plays = []
    children = []
    offensive_team = None
    for i, raw_play in enumerate(split_events(p.events)):
        is_second_chance = (i>0)
        play = parse_event_contents(raw_play, is_second_chance)
        plays.append(play)
        if db:
            db[play.id] = pickle.dumps(play.data)
        children.append([word_id , play.id])
        offensive_team = play.off_team_name
        defense_team = play.def_team_name

    if offensive_team is None or defense_team is None:  # bad possession, skip
        return [], data

    if data is None:
        data = {'scores': defaultdict(int), 'penalty_fouls': defaultdict(int), 'period': 1} 
    if p.data['period'] != data['period']:  # reset penalty on new period
        data['penalty_fouls'] = defaultdict(int)
    data['score_change'] = sum([int(play.score_change) for play in plays])
    data['foul_change'] = sum([int(play.counts_towards_penalty) for play in p.events if play.event_type == FOUL])

    data['scores'][offensive_team] += data['score_change']
    data['scores'][defense_team] += 0
    data['penalty_fouls'][defense_team] += data['foul_change']
    data['penalty_fouls'][offensive_team] += 0

    data['offense_team'] = offensive_team
    data['defense_team'] = defense_team
    data['period'] = p.data['period']
    data['start_time'] = ((4 - data['period']) * 12 * 60) + int(p.start_time.split(":")[0]) * 60 + int(p.start_time.split(":")[1])
    data['end_time'] = ((4 - data['period']) * 12 * 60) + int(p.end_time.split(":")[0]) * 60 + int(p.end_time.split(":")[1])
    if db:
        db[word_id] = msgpack.dumps(data)
    return children, data

def generate_game(g, db=None):
    word_id = "game-" + generate_id(word_count=5)
    children = []
    pos_data = None
    for p in StatsNbaPossessionLoader(g.data['game_id'], "file", "data").items:
        grandchildren, pos_data = generate_possession(p, pos_data, db)
        children += [[word_id] + g for g in grandchildren]

    data = {}
    data['date'] = g.data['date']
    data['scores'] = pos_data['scores'] 
    data['id'] = g.data['game_id']
    data['home_team'] = team_name(g.data['home_team_id'])
    data['away_team'] = team_name(g.data['visitor_team_id'])
    if db:
        db[word_id] = msgpack.dumps(data)
    return children

def generate_season(year, db=None):
    word_id = year
    children = []
    for season_type in ["Regular Season", "Playoffs"]:
        for g in StatsNbaGameFinderLoader("nba", year, season_type, "file", "data").items:
            game = generate_game(g, db)
            children += [[word_id] + c for c in game]
    data = {}
    for k, v in g.data.items():
        data[str(k)] = str(v)
    if db:
        db[word_id] = msgpack.dumps(data)
    return children


def load_stats(config, db, years=[2016]):
    db.set_namespace(config['loader']['key'])

    year_ids = [str(y-1) + "-" + str(y)[-2:] for y in years]
    for year_id in year_ids:
        print(f"creating year {year_id}")
        season_keys = generate_season(year_id, db)
        db['item_keys_{}'.format(year_id)] = msgpack.dumps(season_keys)
    
    db['completed'] = 'true'.encode()
