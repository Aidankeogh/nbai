from src.ml.modules.basic import TransformerPointwise, BaseHead
import torch.nn as nn
import torch

class InitialEventHead(BaseHead):
    key = "initial_event"
    def __init__(
        self,
        model_dim=512,
        initial_event_hidden=0,
        bn=True,
        **kwargs
    ):
        super().__init__()
        self.conv = TransformerPointwise(model_dim * 2, 5, hidden=initial_event_hidden, bn=bn)
        self.sm = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, offense_team, defense_team, **kwargs):

        teams_combined = torch.cat((offense_team, defense_team), dim=2)
        x = self.conv(teams_combined)
        x = x.squeeze()
        x = self.sm(x)
        return x
    
    def get_loss(self, outputs, targets, validity):
        return self.loss(outputs["initial_event"], targets["initial_event"])


class FreeThrows(BaseHead):
    key = "free_throws"
    stat_type = "offense"
    def __init__(
        self,
        model_dim=512,
        shot_type_hidden=0,
        bn=True,
        **kwargs
    ):
        super().__init__()
        self.conv = TransformerPointwise(model_dim * 2, 1, hidden=shot_type_hidden, bn=bn)
        self.sig = nn.Sigmoid()
        # 5 outputs per player, treated as probability of each type of shot
        self.loss = nn.BCELoss(reduction="none")

    def forward(self, offense_team, defense_team, **kwargs):
        teams_combined = torch.cat((offense_team, defense_team), dim=2)
        x = self.conv(teams_combined)
        x = x.squeeze()
        x = self.sig(x)
        return x
    
    def get_loss(self, outputs, targets, validity):
        pred_free_throws = outputs["free_throws"]
        free_throws = (targets["free_throws_attempted"].squeeze() > 0).float()
        loss = self.loss(pred_free_throws, free_throws)
        loss = loss[validity["free_throws_attempted"]]
        loss = torch.mean(loss)
        return loss