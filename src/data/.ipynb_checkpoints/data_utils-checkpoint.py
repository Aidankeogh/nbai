import json
import yaml
import os

with open("metadata/team_ids.yaml", "r") as f:
    team_ids = yaml.safe_load(f)

with open("metadata/player_info.yaml", "r") as f:
    player_info = yaml.safe_load(f)

FIELD_GOAL_MADE = 1
FIELD_GOAL_MISSED = 2
FREE_THROW = 3
REBOUND = 4
TURNOVER = 5
FOUL = 6
EJECTION = 11
END_OF_PERIOD = 13

player_name_to_id = {}
for p_id, player in player_info.items():
    player_name_to_id[player["PLAYER_SLUG"]] = p_id


def player_name(player_id):
    if player_id == -1:
        return None
    player_id = int(player_id)
    if player_id in player_info:
        return player_info[player_id]["PLAYER_SLUG"]


def player_id(name):
    if name is None:
        return -1
    else:
        return int(player_name_to_id[name])


def team_name(team_id):
    if team_id < 1610612736:
        team_id = int(team_id) + 1610612736
    team_id = int(team_id)
    if team_id in team_ids:
        return team_ids[team_id]


def team_id(teamname):
    if teamname in team_ids:
        return int(team_ids[teamname]) - 1610612736
    else:
        return 0


def time_to_seconds(time, period):
    return (
        ((4 - period) * 12 * 60)
        + int(time.split(":")[0]) * 60
        + int(time.split(":")[1])
    )


def get_prev(prev, key, default=0):
    prev = prev
    if len(prev) == 0:
        prev_val = default
    else:
        prev_val = prev[-1][key]
    return prev_val
