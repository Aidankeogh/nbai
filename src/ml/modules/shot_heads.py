from src.ml.modules.basic import TransformerPointwise, BaseHead
from src.data.play import play_config
import torch.nn as nn
import torch

indices_for_2pa = torch.tensor([
    play_config.choice_indices["shot_type"]["ShortMidRange"],
    play_config.choice_indices["shot_type"]["LongMidRange"],
    play_config.choice_indices["shot_type"]["AtRim"],
]).long()

indices_for_3pa = torch.tensor([
    play_config.choice_indices["shot_type"]["Arc3"],
    play_config.choice_indices["shot_type"]["Corner3"],
]).long()

class ShooterHead(BaseHead):
    key = "shooter"
    stat_type = "offense"
    def __init__(
        self,
        model_dim=512,
        shooter_hidden=0,
        bn=True,
        **kwargs
    ):
        super().__init__()
        self.conv = TransformerPointwise(model_dim, 1, hidden=shooter_hidden, bn=bn)
        self.sm = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        # 1 output per player, treated as the affinity for shot taking

    def forward(self, offense_roster, **kwargs):
        x = self.conv(offense_roster)
        x = x.squeeze()
        x = self.sm(x)
        return x
    
    def get_loss(self, outputs, targets, validity):
        loss = self.loss(outputs["shooter"], targets["shooter"])
        loss = loss[validity["shooter"]]
        loss = torch.mean(loss)
        return loss

class ShotTypeHead(BaseHead):
    key = "shot_type"
    stat_type = "offense"
    def __init__(
        self,
        model_dim=512,
        shot_type_hidden=0,
        bn=True,
        **kwargs
    ):
        super().__init__()
        self.conv = TransformerPointwise(model_dim, 5, hidden=shot_type_hidden, bn=bn)
        self.sm = nn.Softmax(dim=2)
        # 5 outputs per player, treated as probability of each type of shot
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, offense_roster, **kwargs):
        x = self.conv(offense_roster)
        x = self.sm(x)
        return x
    
    def get_loss(self, outputs, targets, validity):
        arange = torch.arange(targets["shooter"].shape[0])
        shot_type_outputs = outputs["shot_type"][arange, targets["shooter"], :]

        loss = self.loss(shot_type_outputs, targets["shot_type"])
        loss = loss[validity["shot_type"]]
        loss = torch.mean(loss)
        return loss

class ShotMadeHead(BaseHead):
    key = "shot_made"
    stat_type = "offense"
    def __init__(
        self,
        model_dim=512,
        shot_made_hidden=0,
        bn=True,
        **kwargs
    ):
        super().__init__()
        self.conv = TransformerPointwise(model_dim, 5, hidden=shot_made_hidden, bn=bn)
        self.sig = nn.Sigmoid()
        # 5 outputs per player, treated as probability each type of shot is made
        self.loss = nn.BCELoss(reduction="none")

    def forward(self, offense_roster, **kwargs):
        x = self.conv(offense_roster)
        x = self.sig(x)
        return x
    
    def get_loss(self, outputs, targets, validity):
        arange = torch.arange(targets["shooter"].shape[0])
        shot_made_outputs = outputs["shot_made"][arange, targets["shooter"], targets["shot_type"]]

        loss = self.loss(shot_made_outputs, targets["shot_made"].squeeze().float())
        loss = loss[validity["shot_made"]]
        loss = torch.mean(loss)
        return loss
