from src.ml.modules.basic import TransformerPointwise, BaseHead
import torch.nn as nn
import torch

class ShootingFoulerHead(BaseHead):
    key = "shooting_fouler"
    stat_type = "defense"
    def __init__(
        self,
        model_dim=512,
        bn=True,
        **kwargs
    ):
        super().__init__()
        self.conv = TransformerPointwise(model_dim, 1, hidden=0, bn=bn)
        self.sm = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, defense_roster, **kwargs):
        x = self.conv(defense_roster)
        x = x.squeeze()
        x = self.sm(x)
        return x
    
    def get_loss(self, outputs, targets, validity):
        loss = self.loss(outputs["shooting_fouler"], targets["shooting_fouler"])
        loss = loss[validity["shooting_fouler"]]
        loss = torch.mean(loss)
        return loss

    def stats_pred(self, outputs):
        shot_taken_prob = outputs["initial_event"][:, play_config.choice_indices["initial_event"]["shot"]].unsqueeze(1).unsqueeze(2)
        shooting_fouler_probs = outputs["shooting_fouler"].unsqueeze(2)
        joint_shooting_fouler_prob = shot_taken_prob * shooting_fouler_probs

        return {
            "pfs": joint_shooting_fouler_prob,
        }

    def stats_gt(self, inputs, validity):
        arange = torch.arange(inputs["shooter"].shape[0])
        fouler_ids = inputs["defense_roster"][arange, inputs["shooting_fouler"]][validity["shooting_fouler"]]
        fouler_mask = torch.ones_like(fouler_ids)

        return {
            "pfs": fouler_mask,
        }, fouler_ids

class ShotFouledHead(BaseHead):
    key = "shot_fouled"
    def __init__(
        self,
        model_dim=512,
        bn=True,
        **kwargs
    ):
        super().__init__()
        self.conv = TransformerPointwise(model_dim, 5, hidden=0, bn=bn)
        self.sig = nn.Sigmoid()
        self.loss = nn.BCELoss(reduction="none")

    def forward(self, offense_roster, **kwargs):
        x = self.conv(offense_roster)
        x = self.sig(x)
        return x
    
    def get_loss(self, outputs, targets, validity):
        loss = self.loss(outputs["shot_fouled"], targets["shot_fouled"])
        loss = loss[validity["shot_fouled"]]
        loss = torch.mean(loss)
        return loss

if __name__ == "__main__":
    head = ShotFouledHead
    offense_roster = torch.rand(3, 5, 10)
    defense_team = torch.rand(3, 1, 10)
    x = ShotFouledHead()
    x(offense_roster, defense_team)
    print(x.key)