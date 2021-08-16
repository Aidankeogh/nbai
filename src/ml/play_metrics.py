from collections import defaultdict
from typing import DefaultDict

from torch._C import device
from src.ml.play_dataset import PlayModule, format_data
from src.data.play import Play, play_config
from src.data.box_stats import parse_multiple_plays, Box_stats
from src.thought_path import DataConfig
from src.data.game import Game
from src.utilities.embedding_utilities import get_name, get_idx
from src.loader_pipeline import DB_NAME
import yaml
import torch
import h5py

def get_game(db_name = DB_NAME, season="2016_playoffs", idx=-4):
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
    shooting_fouler_probs = outputs["shooting_fouler"].unsqueeze(2)
    shot_type_probs = outputs["shot_type"]
    shot_made_probs = outputs["shot_made"]
    joint_shot_attempted = shot_taken_prob * shooter_probs * shot_type_probs
    joint_shot_made = joint_shot_attempted * shot_made_probs
    joint_shooting_fouler_prob = shot_taken_prob * shooting_fouler_probs

    joint_2pa = joint_shot_attempted[:, :, indices_for_2pa].sum(dim=2)
    joint_3pa = joint_shot_attempted[:, :, indices_for_3pa].sum(dim=2)

    joint_2pm = joint_shot_made[:, :, indices_for_2pa].sum(dim=2)
    joint_3pm = joint_shot_made[:, :, indices_for_3pa].sum(dim=2)
    offense_players_unique = torch.unique(inputs["offense_roster"])
    defense_players_unique = torch.unique(inputs["defense_roster"])
    player_dict = defaultdict(dict)

    for unique_player in offense_players_unique:
        stat_dict = defaultdict(lambda : torch.zeros(1))
        valid_indices = inputs["offense_roster"] == unique_player
        stat_dict['2pa'] = joint_2pa[valid_indices].sum()
        stat_dict['3pa'] = joint_3pa[valid_indices].sum()
        stat_dict['2pm'] = joint_2pm[valid_indices].sum()
        stat_dict['3pm'] = joint_3pm[valid_indices].sum()
        stat_dict['pts'] = stat_dict['2pm'] * 2 + stat_dict['3pm'] * 3
        stat_dict["o_pos"] = valid_indices.sum()
        name = get_name(unique_player)
        player_dict[name].update(stat_dict)

    for unique_player in defense_players_unique:
        stat_dict = defaultdict(lambda : torch.zeros(1))
        valid_indices = inputs["defense_roster"] == unique_player
        stat_dict["pfs"] = joint_shooting_fouler_prob[valid_indices].sum()
        stat_dict["d_pos"] = valid_indices.sum()
        name = get_name(unique_player)
        player_dict[name].update(stat_dict)
    
    return player_dict

def extract_gt_stats(inputs, validity):
    offense_players_unique = torch.unique(inputs["offense_roster"])
    defense_players_unique = torch.unique(inputs["defense_roster"])
    player_dict = defaultdict(dict)

    validity_mask = validity["shot_made"]
    two_pointers = sum([inputs["shot_type"][validity_mask] == types_2pa for types_2pa in indices_for_2pa])
    three_pointers = sum([inputs["shot_type"][validity_mask] == types_3pa for types_3pa in indices_for_3pa])
    shots_made = inputs["shot_made"][validity_mask].squeeze() == 1
    two_pointers_made = two_pointers * shots_made
    three_pointers_made = three_pointers * shots_made

    arange = torch.arange(inputs["shooter"].shape[0])
    shooter_ids = inputs["offense_roster"][arange, inputs["shooter"]][validity_mask]
    fouler_ids = inputs["defense_roster"][arange, inputs["shooting_fouler"]][validity["shooting_fouler"]]

    for unique_player in offense_players_unique:
        stat_dict = defaultdict(lambda : torch.zeros(1))
        valid_indices = inputs["offense_roster"] == unique_player

        shooter_indices = shooter_ids == unique_player
        stat_dict["2pa"] = two_pointers[shooter_indices].sum()
        stat_dict["3pa"] = three_pointers[shooter_indices].sum()
        stat_dict["2pm"] = two_pointers_made[shooter_indices].sum()
        stat_dict["3pm"] = three_pointers_made[shooter_indices].sum()
        stat_dict["pts"] = stat_dict["2pm"] * 2 + stat_dict["3pm"] * 3
        stat_dict["o_pos"] = valid_indices.sum()
        name = get_name(unique_player)
        player_dict[name].update(stat_dict)

    for unique_player in defense_players_unique:
        stat_dict = defaultdict(lambda : torch.zeros(1))
        valid_indices = inputs["defense_roster"] == unique_player        
        fouler_indices = fouler_ids == unique_player
        stat_dict["pfs"] = fouler_indices.sum()
        stat_dict["d_pos"] = valid_indices.sum()
        name = get_name(unique_player)
        player_dict[name].update(stat_dict)
    
    return player_dict

def load_to_box_stats(stats):
    box_stats = Box_stats()
    for name, player_stats in stats.items():
        for category, value in player_stats.items():
            box_stats[name, category] = value.item()
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
    test_plays = get_game(idx=-1)

    inputs, validity = format_data(test_plays)
    model = PlayModel.load_from_checkpoint("runs/08.12-22:31-wonderful-trout/checkpoints/shot-epoch=17-val_loss=13.15.ckpt")
    player_stats, gt_stats = get_predicted_stats(model, test_plays, as_box=True)
    
    print(player_stats)
    print(gt_stats)
    # print(parse_multiple_plays(test_plays)["GSW"])
    # for p in test_plays:
    #     pp = Play(p)
    #     print(pp)
    #     print(pp.shot_fouled)
    #     print(pp.free_thrower)