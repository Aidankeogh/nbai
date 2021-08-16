from collections import defaultdict
import torch.nn as nn
from src.ml.modules.shot_heads import ShooterHead, ShotMadeHead, ShotTypeHead
from src.ml.modules.foul_heads import ShootingFoulerHead, ShotFouledHead
from src.ml.modules.gamestate_heads import InitialEventHead
import torch

class Heads(nn.ModuleDict):
    used_heads = [
        InitialEventHead,
        ShooterHead,
        ShotMadeHead,
        ShotTypeHead,
        #ShotFouledHead,
        ShootingFoulerHead,
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
        out_dict = {key: head(**inputs) for key, head in self.items()}
        return out_dict
    
    def get_loss(self, outputs, targets, validity):
        loss_dict = {key: head.get_loss(outputs, targets, validity) for key, head in self.items()} 
        return loss_dict
    
    def get_pred_stats(self, inputs, outputs):
        for head in self.values():
            stats = head.stats_pred(outputs)

    def get_gt_stats(self, inputs, validity):
        for head in self.values():
            stats, indices = head.stats_gt(inputs, validity)

if __name__ == "__main__":
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
        "shot_made": [batch_size, roster_size, 5]
    }
    for k, size in key_sizes.items():
        print(k, out_dict[k].shape)
        for x, y in zip(size, out_dict[k].shape):
            assert x == y
