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


class FreeThrowerHead(BaseHead):
    key = "free_thrower"
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
        loss = self.loss(outputs["free_thrower"], targets["free_thrower"])
        loss = loss[validity["free_thrower"]]
        loss = torch.mean(loss)
        return loss

class FreeThrowsAttemptedHead(BaseHead):
    key = "free_throws_attempted"
    stat_type = "offense"
    def __init__(
        self,
        model_dim=512,
        shot_type_hidden=0,
        bn=True,
        **kwargs
    ):
        super().__init__()
        self.conv = TransformerPointwise(model_dim, 4, hidden=shot_type_hidden, bn=bn)
        self.sm = nn.Softmax(dim=2)
        # 5 outputs per player, treated as probability of each type of shot
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, offense_roster, **kwargs):
        x = self.conv(offense_roster)
        x = self.sm(x)
        return x
    
    def get_loss(self, outputs, targets, validity):
        arange = torch.arange(targets["free_thrower"].shape[0])
        pred_free_throws_attempted = outputs["free_throws_attempted"][arange, targets["free_thrower"], :]
        gt_free_throws_attempted = targets["free_throws_attempted"].squeeze().clamp(0, 3).long()
        loss = self.loss(pred_free_throws_attempted, gt_free_throws_attempted)
        loss = loss[validity["free_throws_attempted"]]
        loss = torch.mean(loss)
        return loss

class FreeThrowPercentageHead(BaseHead):
    key = "free_throw_percentage"
    stat_type = "offense"
    def __init__(
        self,
        model_dim=512,
        shot_made_hidden=0,
        bn=True,
        **kwargs
    ):
        super().__init__()
        self.conv = TransformerPointwise(model_dim, 1, hidden=shot_made_hidden, bn=bn)
        self.sig = nn.Sigmoid()
        # 5 outputs per player, treated as probability each type of shot is made
        self.loss = nn.BCELoss(reduction="none")

    def forward(self, offense_roster, **kwargs):
        x = self.conv(offense_roster)
        x = self.sig(x)
        return x
    
    def get_loss(self, outputs, targets, validity):
        return 0
        arange = torch.arange(targets["free_thrower"].shape[0])
        gt_ft_percentage = targets["free_throws_made"] / targets["free_throws_attempted"]
        pred_ft_percentage = outputs["free_throw_percentage"][arange, targets["free_thrower"], :]

        loss = self.loss(pred_ft_percentage, gt_ft_percentage)
        loss = loss[validity["free_throws_made"]]
        loss = torch.mean(loss)
        return loss


if __name__ == "__main__":
    head = ShotFouledHead
    offense_roster = torch.rand(3, 5, 10)
    defense_team = torch.rand(3, 1, 10)
    x = ShotFouledHead()
    x(offense_roster, defense_team)
    print(x.key)