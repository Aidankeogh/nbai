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
    
    def stats_pred(self, outputs):
        shot_taken_prob = outputs["initial_event"][:, play_config.choice_indices["initial_event"]["shot"]].unsqueeze(1).unsqueeze(2)
        shooter_probs = outputs["shooter"].unsqueeze(2)
        shot_type_probs = outputs["shot_type"]
        joint_shot_attempted = shot_taken_prob * shooter_probs * shot_type_probs

        return {
            "2pa": joint_shot_attempted[:, :, indices_for_2pa].sum(dim=2),
            "3pa": joint_shot_attempted[:, :, indices_for_3pa].sum(dim=2)
        }

    def stats_gt(self, inputs, validity):
        validity_mask = validity["shot_made"]
        two_pointers = sum([inputs["shot_type"][validity_mask] == types_2pa for types_2pa in indices_for_2pa])
        three_pointers = sum([inputs["shot_type"][validity_mask] == types_3pa for types_3pa in indices_for_3pa])

        arange = torch.arange(inputs["shooter"].shape[0])
        shooter_ids = inputs["offense_roster"][arange, inputs["shooter"]][validity_mask]

        return {
            "2pa": two_pointers, 
            "3pa": three_pointers
        }, shooter_ids

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

    def stats_pred(self, outputs):
        shot_taken_prob = outputs["initial_event"][:, play_config.choice_indices["initial_event"]["shot"]].unsqueeze(1).unsqueeze(2)
        shooter_probs = outputs["shooter"].unsqueeze(2)
        shot_type_probs = outputs["shot_type"]
        shot_made_probs = outputs["shot_made"]
        joint_shot_made = shot_taken_prob * shooter_probs * shot_type_probs * shot_made_probs
        
        return {
            "2pm": joint_shot_made[:, :, indices_for_2pa].sum(dim=2),
            "3pm": joint_shot_made[:, :, indices_for_3pa].sum(dim=2)
        }

    def stats_gt(self, inputs, validity):
        validity_mask = validity["shot_made"]
        two_pointers = sum([inputs["shot_type"][validity_mask] == types_2pa for types_2pa in indices_for_2pa])
        three_pointers = sum([inputs["shot_type"][validity_mask] == types_3pa for types_3pa in indices_for_3pa])
        shots_made = inputs["shot_made"][validity_mask].squeeze() == 1
        two_pointers_made = two_pointers * shots_made
        three_pointers_made = three_pointers * shots_made

        arange = torch.arange(inputs["shooter"].shape[0])
        shooter_ids = inputs["offense_roster"][arange, inputs["shooter"]][validity_mask]

        return {
            "2pm": two_pointers_made, 
            "3pm": three_pointers_made
        }, shooter_ids
