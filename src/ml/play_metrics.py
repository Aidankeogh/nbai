from typing import DefaultDict
from src.ml.play_dataset import PlayModule, format_data
from src.data.play import Play, play_config
from src.data.box_stats import parse_multiple_plays, Box_stats
from src.thought_path import DataConfig
from src.data.game import Game
from src.utilities.embedding_utilities import get_name, get_idx
import yaml
import torch
import h5py

def get_game(db_name = "cache/ml_db_0.0.1.h5", season="2016_playoffs", idx=-4):
    with h5py.File(db_name, "r", libver="latest", swmr=True) as db:
        test_game = Game(db[f"raw_data/{season}/games"][idx])
        test_plays = db[f"raw_data/{season}/plays"][
            test_game.play_start_idx : test_game.play_end_idx
        ]
    return test_plays

def get_joint_probs(outputs):
    shot_taken_prob = outputs["initial_event"][:, play_config.choice_indices["initial_event"]["shot"]].unsqueeze(1).unsqueeze(2)
    shooter_probs = outputs["shooter"].unsqueeze(2)
    shot_type_probs = outputs["shot_type"]
    shot_made_probs = outputs["shot_made"]
    joint_shot_attempted = shot_taken_prob * shooter_probs * shot_type_probs
    joint_shot_made = joint_shot_attempted * shot_made_probs
    return {
        "shot_attempted": joint_shot_attempted,
        "shot_made": joint_shot_made
    }

def extract_stats(inputs, joint_probs):
    indices_for_2pa = torch.tensor([
        play_config.choice_indices["shot_type"]["ShortMidRange"],
        play_config.choice_indices["shot_type"]["LongMidRange"],
        play_config.choice_indices["shot_type"]["AtRim"],
    ]).long()
    indices_for_3pa = torch.tensor([
        play_config.choice_indices["shot_type"]["Arc3"],
        play_config.choice_indices["shot_type"]["Corner3"],
    ]).long()

    joint_2pa = joint_probs["shot_attempted"][:, :, indices_for_2pa].sum(dim=2)
    joint_3pa = joint_probs["shot_attempted"][:, :, indices_for_3pa].sum(dim=2)

    joint_2pm = joint_probs["shot_made"][:, :, indices_for_2pa].sum(dim=2)
    joint_3pm = joint_probs["shot_made"][:, :, indices_for_3pa].sum(dim=2)

    per_team_box_stats = DefaultDict(Box_stats)
    for idx in range(len(inputs["offense_roster"])):
        roster = inputs["offense_roster"][idx]
        team = inputs["offense_team"][idx]
        for player_idx in range(5):
            team_name = get_name(team)
            name = get_name(roster[player_idx])
            box_stats = per_team_box_stats[team_name]
            box_stats[name, "2pa"] += joint_2pa[idx, player_idx].item()
            box_stats[name, "3pa"] += joint_3pa[idx, player_idx].item()
            box_stats[name, "2pm"] += joint_2pm[idx, player_idx].item()
            box_stats[name, "3pm"] += joint_3pm[idx, player_idx].item()
            box_stats[name, "pts"] += joint_3pm[idx, player_idx].item() * 3 + joint_2pm[idx, player_idx].item() * 2

    return per_team_box_stats

def get_predicted_stats(model, test_plays):
    inputs, validity = format_data(test_plays)
    outputs = model((inputs, validity))
    joint_probs = get_joint_probs(outputs)
    box_stats = extract_stats(inputs, joint_probs)
    return box_stats

if __name__ == "__main__":
    from src.ml.play_model import PlayModel

    test_plays = get_game()
    ground_truth_box_stats = parse_multiple_plays(test_plays)

    model = PlayModel()
    predicted_stats = get_predicted_stats(model, test_plays)

    for item in ground_truth_box_stats.values():
        print(item)
    for item in predicted_stats.values():
        print(item)