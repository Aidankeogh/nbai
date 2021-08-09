from src.ml.modules.gamestate_heads import InitialEventHead
import torch.nn as nn
from src.ml.modules.basic import TransformerPointwise
from src.ml.modules.shot_heads import ShooterHead, ShotMadeHead, ShotTypeHead
import torch


class Heads(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.initial_event = InitialEventHead(**kwargs)
        self.shooter = ShooterHead(**kwargs)
        self.shot_type = ShotTypeHead(**kwargs)
        self.shot_made = ShotMadeHead(**kwargs)
    
    def forward(self, offense_roster, defense_roster, offense_team, defense_team): 
        out_dict = {
            "initial_event": self.initial_event(offense_team, defense_team),
            "shooter": self.shooter(offense_roster),
            "shot_type": self.shot_type(offense_roster),
            "shot_made": self.shot_made(offense_roster),
        }
        return out_dict
    
    def get_loss(self, outputs, targets, validity):
        loss_dict = {
            "initial_event": self.initial_event.get_loss(outputs, targets, validity),
            "shooter": self.shooter.get_loss(outputs, targets, validity),
            "shot_type": self.shot_type.get_loss(outputs, targets, validity),
            "shot_made": self.shot_made.get_loss(outputs, targets, validity),
        }
        return loss_dict

if __name__ == "__main__":
    batch_size = 42
    embed_size = 512
    roster = torch.rand(batch_size, 12, embed_size)
    team = torch.rand(batch_size, 1, embed_size)
    heads = Heads(model_dim=embed_size)

    out_dict = heads(roster, roster, team, team)
    key_sizes = {
        "initial_event": [batch_size, 5],
        "shooter": [batch_size, 12],
        "shot_type": [batch_size, 12, 5],
        "shot_made": [batch_size, 12, 5]
    }
    for k, size in key_sizes.items():
        print(k, out_dict[k].shape)
        for x, y in zip(size, out_dict[k].shape):
            assert x == y
