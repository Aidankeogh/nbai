from typing import DefaultDict

from torch._C import device
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
    return torch.tensor(test_plays)

indices_for_2pa = torch.tensor([
    play_config.choice_indices["shot_type"]["ShortMidRange"],
    play_config.choice_indices["shot_type"]["LongMidRange"],
    play_config.choice_indices["shot_type"]["AtRim"],
]).long()

indices_for_3pa = torch.tensor([
    play_config.choice_indices["shot_type"]["Arc3"],
    play_config.choice_indices["shot_type"]["Corner3"],
]).long()

def extract_stats(inputs, outputs):
    shot_taken_prob = outputs["initial_event"][:, play_config.choice_indices["initial_event"]["shot"]].unsqueeze(1).unsqueeze(2)
    shooter_probs = outputs["shooter"].unsqueeze(2)
    shot_type_probs = outputs["shot_type"]
    shot_made_probs = outputs["shot_made"]
    joint_shot_attempted = shot_taken_prob * shooter_probs * shot_type_probs
    joint_shot_made = joint_shot_attempted * shot_made_probs

    joint_2pa = joint_shot_attempted[:, :, indices_for_2pa].sum(dim=2)
    joint_3pa = joint_shot_attempted[:, :, indices_for_3pa].sum(dim=2)

    joint_2pm = joint_shot_made[:, :, indices_for_2pa].sum(dim=2)
    joint_3pm = joint_shot_made[:, :, indices_for_3pa].sum(dim=2)
    offense_players_unique = torch.unique(inputs["offense_roster"])

    player_dict = {}
    for unique_player in offense_players_unique:
        stat_dict = {}
        valid_indices = inputs["offense_roster"] == unique_player
        stat_dict['2pa'] = joint_2pa[valid_indices].sum()
        stat_dict['3pa'] = joint_3pa[valid_indices].sum()
        stat_dict['2pm'] = joint_2pm[valid_indices].sum()
        stat_dict['3pm'] = joint_3pm[valid_indices].sum()
        stat_dict['pts'] = stat_dict['2pm'] * 2 + stat_dict['3pm'] * 3
        stat_dict["o_pos"] = valid_indices.sum()
        name = get_name(unique_player)
        player_dict[name] = stat_dict

    return player_dict

def extract_gt_stats(inputs, validity):
    offense_players_unique = torch.unique(inputs["offense_roster"])
    player_dict = {}

    two_pointers = sum([inputs["shot_type"][validity["shot_type"]] == types_2pa for types_2pa in indices_for_2pa])
    three_pointers = sum([inputs["shot_type"][validity["shot_type"]] == types_3pa for types_3pa in indices_for_3pa])
    shots_made = inputs["shot_made"][validity["shot_made"]].squeeze() == 1
    two_pointers_made = two_pointers * shots_made
    three_pointers_made = three_pointers * shots_made

    arange = torch.arange(inputs["shooter"].shape[0])
    shooter_ids = inputs["offense_roster"][arange, inputs["shooter"]][validity["shooter"]]

    for unique_player in offense_players_unique:
        stat_dict = {}
        valid_indices = inputs["offense_roster"] == unique_player

        shooter_indices = shooter_ids == unique_player
        stat_dict["2pa"] = two_pointers[shooter_indices].sum()
        stat_dict["3pa"] = three_pointers[shooter_indices].sum()
        stat_dict["2pm"] = two_pointers_made[shooter_indices].sum()
        stat_dict["3pm"] = three_pointers_made[shooter_indices].sum()
        stat_dict["pts"] = stat_dict["2pm"] * 2 + stat_dict["3pm"] * 3
        stat_dict["o_pos"] = valid_indices.sum()
        name = get_name(unique_player)
        player_dict[name] = stat_dict

    return player_dict

def load_to_box_stats(stats):
    box_stats = Box_stats()
    for name, stats in stats.items():
        box_stats[name, "2pa"] = stats['2pa'].item()
        box_stats[name, "3pa"] = stats['3pa'].item()
        box_stats[name, "2pm"] = stats['2pm'].item()
        box_stats[name, "3pm"] = stats['3pm'].item()
        box_stats[name, "pts"] = stats['pts'].item()
        box_stats[name, "o_pos"] = stats['o_pos'].item()
    return box_stats

def get_predicted_stats(model, test_plays, device="cpu", as_box=True):
    inputs, validity = format_data(test_plays)
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    for k, v in validity.items():
        validity[k] = v.to(device)
    outputs = model((inputs, validity))
    player_stats = extract_stats(inputs, outputs)
    gt_stats = extract_gt_stats(inputs, validity)
    if as_box:
        player_stats = load_to_box_stats(player_stats)
        gt_stats = load_to_box_stats(gt_stats)

    return player_stats, gt_stats

if __name__ == "__main__":
    from src.ml.play_model import PlayModel
    test_plays = get_game()
    model = PlayModel() 
    player_stats, gt_stats = get_predicted_stats(model, test_plays, as_box=True)

    print(player_stats)

    print(gt_stats)
    print(parse_multiple_plays(test_plays))