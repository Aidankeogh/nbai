import json
import yaml
import os

with open("team_ids.json", "r") as f:
    team_ids = json.load(f)

with open("player_info.json", "r") as f:
    player_info = json.load(f)

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
    player_name_to_id[player['PLAYER_SLUG']] = p_id

def player_name(player_id):
    if player_id == -1:
        return None
    player_id = str(int(player_id))
    if player_id in player_info:
        return player_info[player_id]['PLAYER_SLUG']

def player_id(name):
    if name is None:
        return -1
    else:
        return int(player_name_to_id[name])

def team_name(team_id):
    if team_id < 1610612736:
        team_id = int(team_id) + 1610612736
    team_id = str(int(team_id))
    if team_id in team_ids:
        return team_ids[team_id]

def team_id(teamname):
    if teamname in team_ids:
        return int(team_ids[teamname]) - 1610612736
    else:
        return 0

