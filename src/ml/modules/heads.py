from collections import defaultdict
import torch.nn as nn
from src.ml.modules.shot_heads import ShooterHead, ShotMadeHead, ShotTypeHead, ShotAssistedHead, AssisterHead
from src.ml.modules.foul_heads import ShootingFoulerHead, ShotFouledHead, FreeThrowerHead, FreeThrowsAttemptedHead, FreeThrowPercentageHead
from src.ml.modules.gamestate_heads import InitialEventHead, FreeThrows
import torch

class Heads(nn.ModuleDict):
    used_heads = [
        InitialEventHead,
        ShooterHead,
        ShotMadeHead,
        ShotTypeHead,
        ShotFouledHead,
        ShootingFoulerHead,
        FreeThrows,
        FreeThrowerHead,
        FreeThrowsAttemptedHead,
        FreeThrowPercentageHead,
        ShotAssistedHead,
        AssisterHead,
    ]

    def __init__(
        self, **kwargs
    ):
        super().__init__(
            {head.key: head(**kwargs) for head in self.used_heads}
        )
    
    def forward(self, offense_roster, defense_roster, offense_team, defense_team): 
        inputs = {
            "offense_roster": offense_roster, 
            "defense_roster": defense_roster, 
            "offense_team": offense_team, 
            "defense_team": defense_team
        }
        out_dict = {}
        for key, head in self.items():
            out_dict[key] = head(**inputs)
        return out_dict
    
    def get_loss(self, outputs, targets, validity):

        loss_dict = {}
        for key, head in self.items():
            loss_dict[key] = head.get_loss(outputs, targets, validity)
        return loss_dict

if __name__ == "__main__":
    from src.ml.play_dataset import format_data
    from src.ml.play_metrics import get_game
    
    batch_size = 42
    embed_size = 512
    roster_size = 12
    roster = torch.rand(batch_size, roster_size, embed_size)
    team = torch.rand(batch_size, 1, embed_size)
    heads = Heads(model_dim=embed_size)

    out_dict = heads(roster, roster, team, team)
    key_sizes = {
        "initial_event": [batch_size, 5],
        "shooter": [batch_size, roster_size],
        "shot_type": [batch_size, roster_size, 5],
        "shot_made": [batch_size, roster_size, 5],
        "free_thrower": [batch_size, roster_size],
        "free_throws_attempted": [batch_size, roster_size, 4],
        "free_throw_percentage": [batch_size, roster_size, 1],
        "assisted": [batch_size, roster_size, 5],
        "assister": [batch_size, roster_size]
    }
    for k, size in key_sizes.items():
        print(k, out_dict[k].shape)
        for x, y in zip(size, out_dict[k].shape):
            assert x == y
    from src.data.play import Play
    test_plays = get_game()[:42]
    inputs, validity = format_data(test_plays)
    heads.get_loss(out_dict, inputs, validity)