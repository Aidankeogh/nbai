from pbpstats.data_loader import StatsNbaPossessionLoader, StatsNbaGameFinderLoader
from src.utilities.global_timers import timeit, timers
from src.data.play import Play, parse_play, split_events
from src.data.possession import parse_possession
from src.data.game import parse_game
from src.data.season import Season, parse_season
from src.data.box_stats import Box_stats, parse_box_stats
from collections import defaultdict
import torch
import msgpack
import os

def dump_season(out_data, db):
    season = Season(out_data["season_info"])
    base_key = "raw_data/" + str(int(season.year)) + "_" + ("playoffs" if season.playoffs else "regular")

    play_key = base_key + "/" + "plays"
    db[play_key] = torch.stack(out_data["plays"])

    possession_key = base_key + "/" + "possessions"
    db[possession_key] = torch.stack(out_data["possessions"])

    game_key = base_key + "/" + "games"
    db[game_key] = torch.stack(out_data["games"])
    
    season_key = base_key + "/" + "season_info"
    db[season_key] = season.data

@timeit
def load_raw_data(db, years=range(2001,2020), season_types=["Playoffs", "Regular Season"]):
    season_iter = []
    for season_type in season_types:
        season_iter.extend([(y, season_type) for y in years])

    for in_season in season_iter:

        out_data = {
            'season_info': None,
            'games': [],
            'possessions': [],
            'plays': [],
        }

        in_data = {
            'season': None,
            'game': None,
            'possession': None,
            'play': None
        }

        year_string = str(in_season[0]-1) + "-" + str(in_season[0])[-2:]
        game_iter = StatsNbaGameFinderLoader("nba", year_string, in_season[1], "file", "cache/data").items
        in_data['season'] = in_season
        for in_game in game_iter:
            possession_iter = StatsNbaPossessionLoader(in_game.data['game_id'], "file", "cache/data").items
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
    db["raw_data_loaded"] = True
    return out_data

@timeit
def accumulate_box_stats(db):
    all_stats = {}
    for season in db["raw_data"]:
        stats = {}
        rosters = defaultdict(set)
        season_info = Season(db[f"raw_data/{season}/season_info"])
        plays = db[f"raw_data/{season}/plays"][season_info.play_start_idx:season_info.play_end_idx]
        plays = [Play(p) for p in plays]

        for play in plays:
            rosters[play.offense_team].update(play.offense_roster)
            rosters[play.defense_team].update(play.defense_roster)

        for key in rosters.keys():
            rosters[key] = list(rosters[key])
            stats[key] = Box_stats(rosters[key])
        rosters = dict(rosters)
        

        for play in plays:
            try:
                off_stats, def_stats = parse_box_stats(play, rosters)
                stats[play.offense_team] += off_stats
                stats[play.defense_team] += def_stats
            except Exception as e:
                print(e)
        all_stats[season] = stats

        for k, v in stats.items():
            print(season, k)
            db[f"box_stats/{season}/{k}"] = v.data
            os.makedirs(f"metadata/rosters", exist_ok=True)
        with open(f"metadata/rosters/{season}", "wb") as f:
            msgpack.dump(rosters, f)
    db["box_stats_accumulated"] = True
    return all_stats

if __name__ == "__main__":
    import h5py
    
    db_path = "cache/batch_loader_unit_test.h5"
    if os.path.exists(db_path):
        os.remove(db_path)

    with h5py.File(db_path, "a") as db:
        load_raw_data(db, years=[2018], season_types=["Playoffs"])
        accumulate_box_stats(db)
        with open("metadata/rosters/2018_playoffs", "rb") as f:
            rosters = msgpack.load(f)
        box_stats = Box_stats(rosters['GSW'], db['box_stats/2018_playoffs/GSW'][:])

    print(box_stats)
    steph_stats = box_stats['stephen-curry', :]
    incorrect_steph_stats = [379, 142, 73, 64, 162, 41, 43, 9, 82, 81, 26, 11, 43, 34, 1300, -1151, 1228, 1247]
    # This is not quite correct, check out https://www.basketball-reference.com/teams/GSW/2018.html#playoffs_totals
    print(incorrect_steph_stats)
    print(list(steph_stats))
    assert list(steph_stats) == incorrect_steph_stats