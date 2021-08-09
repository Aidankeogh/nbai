from src.ml.modules.basic import TransformerPointwise
import torch.nn as nn
import torch

class InitialEventHead(nn.Module):
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

    def forward(self, offense_team, defense_team):

        teams_combined = torch.cat((offense_team, defense_team), dim=2)
        x = self.conv(teams_combined)
        x = x.squeeze()
        x = self.sm(x)
        return x
    
    def get_loss(self, outputs, targets, validity):
        return self.loss(outputs["initial_event"], targets["initial_event"])