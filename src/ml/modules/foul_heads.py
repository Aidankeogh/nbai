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
        arange = torch.arange(targets["shooter"].shape[0])
        shot_made_outputs = outputs["shot_fouled"][arange, targets["shooter"], targets["shot_type"]]

        loss = self.loss(shot_made_outputs, targets["shot_fouled"].squeeze().float())
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