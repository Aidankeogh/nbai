from src.utilities.global_timers import timeit
from src.data.play import Play
from collections import defaultdict
from tabulate import tabulate
import torch

categories = [
    "pts",
    "2pa",
    "2pm",
    "3pm",
    "3pa",
    "ftm",
    "fta",
    "orb",
    "drb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pfs",
    "o+-",
    "d+-",
    "o_pos",
    "d_pos",
]
category_idxes = {category: i for i, category in enumerate(categories)}


class Box_stats:
    def __init__(self, data=None) -> None:
        if data is None:
            self.data = {}
        else:
            self.data = data

    def keys(self):
        return categories

    def __getitem__(self, key):
        player, category = key
        if player == "none":
            return None

        if type(category) is slice:
            category_idx = category
        else:
            category_idx = category_idxes[category]
        return self.data[player][category_idx]

    def __setitem__(self, key, value):
        player, category = key
        if player is not None:
            if player not in self.data:
                self.data[player] = torch.zeros(len(categories))
            category_idx = category_idxes[category]
            self.data[player][category_idx] = value

    def __add__(self, other):

        new_data = self.data.copy()
        for k, v in other.data.items():
            if k in new_data:
                new_data[k] += v
            else:
                new_data[k] = v

        return Box_stats(new_data)

    def __iadd__(self, other):
        for k, v in other.data.items():
            if k in self.data:
                self.data[k] += v
            else:
                self.data[k] = v

        return self

    def __repr__(self):
        table = []
        for player in sorted(self.data.keys()):
            if player is not None:
                table.append([player] + list(self[player, :]))
        return tabulate(table, headers=[" "] + categories)


@timeit
def parse_box_stats(play):
    o_stats = Box_stats()
    d_stats = Box_stats()

    for player in play.offense_roster:
        o_stats[player, "o_pos"] = 1 - play.is_second_chance
        o_stats[player, "o+-"] = play.score_change

    for player in play.defense_roster:
        d_stats[player, "d_pos"] = 1 - play.is_second_chance
        d_stats[player, "d+-"] = -play.score_change

    if play.shooter is not None:
        o_stats[play.shooter, "2pa"] = not play.is_3
        o_stats[play.shooter, "2pm"] = play.shot_made and not play.is_3
        o_stats[play.shooter, "3pa"] = play.is_3
        o_stats[play.shooter, "3pm"] = play.shot_made and play.is_3
        o_stats[play.shooter, "pts"] += play.shot_made * (2 + play.is_3)

    if play.free_thrower is not None:
        n_free_throws = play.free_throws_attempted
        free_throws_made = play.free_throws_made
        o_stats[play.free_thrower, "ftm"] = free_throws_made
        o_stats[play.free_thrower, "fta"] = n_free_throws
        o_stats[play.free_thrower, "pts"] += free_throws_made

    if play.defensive_rebounder is not None:
        d_stats[play.defensive_rebounder, "drb"] = 1

    if play.offensive_rebounder is not None:
        o_stats[play.offensive_rebounder, "orb"] = 1

    if play.assister is not None:
        o_stats[play.assister, "ast"] = 1
    if play.stealer is not None:
        d_stats[play.stealer, "stl"] = 1
    if play.blocker is not None:
        d_stats[play.blocker, "blk"] = 1
    if play.turnoverer is not None:
        o_stats[play.turnoverer, "tov"] = 1

    if play.over_limit_fouler is not None:
        d_stats[play.over_limit_fouler, "pfs"] = 1
    if play.shooting_fouler is not None:
        d_stats[play.shooting_fouler, "pfs"] = 1
    if play.common_fouler is not None:
        d_stats[play.common_fouler, "pfs"] = 1

    return o_stats, d_stats


def parse_multiple_plays(plays):
    stats = defaultdict(Box_stats)
    for play in plays:
        if type(play) is not Play:
            play = Play(play)
        off_stats, def_stats = parse_box_stats(play)
        stats[play.offense_team] += off_stats
        stats[play.defense_team] += def_stats

    return stats


if __name__ == "__main__":
    import h5py
    from src.data.game import Game

    db_name = "cache/ml_db_0.0.1.h5"
    with h5py.File(db_name, "r", libver="latest", swmr=True) as db:
        test_game = Game(db["raw_data/2016_playoffs/games"][-4])
        test_plays = db["raw_data/2016_playoffs/plays"][
            test_game.play_start_idx : test_game.play_end_idx
        ]
    stats = parse_multiple_plays(test_plays)
    for k, v in stats.items():
        print(k)
        print(v)
