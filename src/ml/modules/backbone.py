from os import remove
import torch.nn as nn
import torch
from transformers import GPT2Model, BertModel, BertConfig, T5EncoderModel, logging
logging.set_verbosity_error()

class TransformerBackbone(nn.Module):
    def __init__(
            self,
            n_layers=3,
            add_upsampled=False,
            remove_team=False,
            base_model="t5",
            freeze_backbone=False,
            **kwargs
        ):
        super().__init__()
        if base_model == "t5":
            t5 = T5EncoderModel.from_pretrained('t5-small')
            self.transformer = t5.encoder.block[0:n_layers]
        elif base_model == "bert":
            bert = BertModel(BertConfig())
            self.transformer = bert.encoder.layer[0:n_layers]
        elif base_model == "gpt":
            gpt2 = GPT2Model.from_pretrained('gpt2')
            self.transformer = gpt2.encoder.layer[0:n_layers]
        self.add_upsampled = add_upsampled
        self.remove_team = remove_team

        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False
    
    def forward(self, offense_roster, defense_roster, offense_team, defense_team):
        combined_rosters = torch.cat((offense_team, offense_roster, defense_team, defense_roster), dim=1)
        for block in self.transformer:
            hidden = block(combined_rosters)
            combined_rosters = hidden[0]

        if self.add_upsampled:
            offense_team = offense_team + combined_rosters[:, 0:1]
            offense_roster = offense_roster + combined_rosters[:, 1:6]
            defense_team = defense_team + combined_rosters[:, 6:7]
            defense_roster = defense_roster + combined_rosters[:, 7:12]
        else:
            offense_team = combined_rosters[:, 0:1]
            offense_roster = combined_rosters[:, 1:6]
            defense_team = combined_rosters[:, 6:7]
            defense_roster = combined_rosters[:, 7:12]

        return offense_roster, defense_roster, offense_team, defense_team