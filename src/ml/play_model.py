from torch._C import ParameterDict
from src.ml.play_dataset import PlayModule, format_data
from src.data.play import Play
from src.data.box_stats import parse_multiple_plays
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from src.thought_path import DataConfig
from torchmetrics import Accuracy
from pprint import pprint
from src.ml.play_metrics import get_game, get_predicted_stats, extract_stats, extract_gt_stats
from src.loader_pipeline import DB_NAME
from src.utilities.global_ema import ema
import torch.nn as nn
import torch
import transformers
from transformers import GPT2Model, BertModel, BertConfig, T5EncoderModel
transformers.logging.ERROR

def create_encoder(dim, nhead, dim_feedforward, n_layers):
    layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward)
    encoder = nn.TransformerEncoder(layer, n_layers)
    return encoder


def create_decoder(dim, nhead, dim_feedforward, n_layers):
    layer = nn.TransformerDecoderLayer(dim, nhead, dim_feedforward)
    decoder = nn.TransformerDecoder(layer, n_layers)
    return decoder


# class Normshift(nn.Module):T5EncoderModel
#     def __init__(self):
#         super().__init__()
#         self.min = nn.Parameter(torch.rand(1))
#         self.range = nn.Parameter(torch.rand(1))
#         self.bn = nn.BatchNorm1d(1)
    
#     def forward(self, x):
#         prev_shape = x.shape
#         x = self.bn(x.view(-1, 1)).view(prev_shape)

#         return (torch.sigmoid(x) * self.range) + self.min
        

def feedforward(embedding, hidden, out, final=None, activation=nn.Mish(), type="linear", bn=False):
    layers = []
    if bn:
        layers.append(nn.BatchNorm1d(embedding))

    if type == "linear":
        layers += [nn.Linear(embedding, out)]
    elif type == "nonlinear":
        layers += [nn.Linear(embedding, hidden), activation, nn.Linear(hidden, out)]
    else:
        raise f"Usupported head type {type}"

    if final is not None:
        layers.append(final)

    return nn.Sequential(*layers)


def apply_per_player(head, players, softmax_over_players=False):
    # Merge player and batch dim to apply layer to all players
    original_shape = players.shape
    players = players.reshape([original_shape[0] * original_shape[1], -1])
    output = head(players)
    # Get back into desired shape
    output = output.reshape([original_shape[0], original_shape[1], -1])
    output = output.squeeze(dim=2)

    if softmax_over_players:
        # softmax over the player dim
        output = torch.softmax(output, dim=1)

    return output


def apply_mask(outputs, targets, validity):
    mask = validity.nonzero().squeeze()
    return outputs[mask], targets[mask]


class PlayModel(LightningModule):
    def __init__(self, epochs=4, lr=1e-5, lr_backbone=0, weight_decay=0, print_every=1, batch_size=1280, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.epochs = epochs
        self.curr_epoch = 0
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.print_every = print_every
        self.batch_size = batch_size

        self.create_backbone(**kwargs)
        self.create_heads(**kwargs)
        self.create_losses(**kwargs)
        self.create_metrics(**kwargs)

    def create_backbone(
        self,
        embedding_dim=32,
        model_dim=512,
        n_layers=3,
        add_upsampled=False,
        remove_team=False,
        **kwargs
    ):
        # <40 teams
        # <4000 players
        self.team_embeddings = nn.Embedding(40, embedding_dim)
        self.team_offense_upsample = nn.Sequential(
            nn.Conv1d(embedding_dim, model_dim, 1),
            nn.BatchNorm1d(model_dim)
        )
        self.team_defense_upsample = nn.Sequential(
            nn.Conv1d(embedding_dim, model_dim, 1),
            nn.BatchNorm1d(model_dim)
        )
        self.player_embeddings = nn.Embedding(4000, embedding_dim)
        self.player_offense_upsample = nn.Sequential(
            nn.Conv1d(embedding_dim, model_dim, 1),
            nn.BatchNorm1d(model_dim)
        )
        self.player_defense_upsample = nn.Sequential(
            nn.Conv1d(embedding_dim, model_dim, 1),
            nn.BatchNorm1d(model_dim)
        )
        self.add_upsampled = add_upsampled
        self.remove_team = remove_team

        self.embeddings = nn.ModuleDict([
            ["team", self.team_embeddings],
            ["player", self.player_embeddings],
        ])
        
        #gpt2 = GPT2Model.from_pretrained('gpt2')
        #self.backbone = gpt2.h[0:n_layers]
        bert = BertModel(BertConfig())
        t5 = T5EncoderModel.from_pretrained('t5-small')
        #print(bert.encoder)
        self.backbone = t5.encoder.block[0:n_layers]
        #bert.encoder.layer[0:n_layers]
        if self.lr_backbone == 0:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def create_heads(
        self,
        model_dim=512,
        initial_event_hidden=256,
        shooter_hidden=256,
        shot_made_hidden=256,
        shot_type_hidden=256,
        bn=True,
        **kwargs
    ):
        self.initial_event = feedforward(
            model_dim * 2, initial_event_hidden, 5, final=nn.Softmax(dim=1), bn=bn
        )

        self.shooter = feedforward(model_dim, shooter_hidden, 1, bn=bn)
        self.shot_type = feedforward(
            model_dim, shot_type_hidden, 5, final=nn.Softmax(dim=1), bn=bn
        )
        # One for each shot type, P(shot_made | shot_type)
        self.shot_made = feedforward(
            model_dim, shot_made_hidden, 5, final=nn.Sigmoid(), bn=bn
        )
        self.heads = nn.ModuleDict([
            ["initial_event", self.initial_event],
            ["shooter", self.shooter],
            ["shot_type", self.shot_type],
            ["shot_made", self.shot_made]
        ])

    def create_losses(
        self, initial_event_wt=1, shooter_wt=1, shot_made_wt=1, shot_type_wt=1, box_wt=1, **kwargs
    ):
        self.initial_event_loss = nn.CrossEntropyLoss(reduction="mean")
        self.shooter_loss = nn.CrossEntropyLoss(reduction="none")
        self.shot_made_loss = nn.BCELoss(reduction="none")
        self.shot_type_loss = nn.CrossEntropyLoss(reduction="none")
        self.box_loss = nn.MSELoss()
        self.loss_weights = {
            "initial_event": initial_event_wt,
            "shooter": shooter_wt,
            "shot_made": shot_made_wt,
            "shot_type": shot_type_wt,
            "box": box_wt,
        }

    def transformer_backbone(
        self, offense_roster, defense_roster, offense_team, defense_team
    ):
        offense_roster = self.player_embeddings(offense_roster)
        offense_roster = self.player_offense_upsample(offense_roster.swapaxes(1,2)).swapaxes(1,2)

        defense_roster = self.player_embeddings(defense_roster)
        defense_roster = self.player_defense_upsample(defense_roster.swapaxes(1,2)).swapaxes(1,2)

        offense_team = self.team_embeddings(offense_team)
        offense_team = self.team_offense_upsample(offense_team.swapaxes(1,2)).swapaxes(1,2)

        defense_team = self.team_embeddings(defense_team)
        defense_team = self.team_defense_upsample(defense_team.swapaxes(1,2)).swapaxes(1,2)

        combined_rosters = torch.cat((offense_team, offense_roster, defense_team, defense_roster), dim=1)
        for block in self.backbone:
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

    def forward(self, batch):
        play, validity = batch

        # Extract player and team embeddings
        offense_roster = play["offense_roster"]
        defense_roster = play["defense_roster"]

        offense_team = play["offense_team"]
        defense_team = play["defense_team"]
        if self.remove_team:
            offense_team = offense_team * 0
            defense_team = defense_team * 0

        # Run through backbone
        (
            offense_roster, defense_roster, offense_team, defense_team,
        ) = self.transformer_backbone(
            offense_roster, defense_roster, offense_team, defense_team
        )

        # Swap the axes to make the batch dimensions are the traditional batch x player x ...
        # Instead of transformer output player x batch x ...
        # offense_roster, defense_roster = torch.swapaxes(
        #     offense_roster, 0, 1
        # ), torch.swapaxes(defense_roster, 0, 1)

        # For team embeddings get rid of extra dim
        offense_team, defense_team = offense_team.squeeze(dim=1), defense_team.squeeze(
            dim=1
        )
        teams_combined = torch.cat((offense_team, defense_team), dim=1)
        out_dict = {
            "initial_event": self.initial_event(teams_combined),
            "shooter": apply_per_player(
                self.shooter, offense_roster, softmax_over_players=True
            ),
            "shot_type": apply_per_player(self.shot_type, offense_roster),
            "shot_made": apply_per_player(self.shot_made, offense_roster),
        }

        return out_dict

    def get_losses(self, batch):
        targets, validity = batch
        outputs = self(batch)

        # Save outputs and labels for later
        self.outputs = outputs
        self.targets = targets
        self.validity = validity

        loss_dict = {}
        loss_dict["initial_event"] = self.initial_event_loss(
            outputs["initial_event"], targets["initial_event"]
        )

        loss_dict["shooter"] = self.shooter_loss(outputs["shooter"], targets["shooter"])
        loss_dict["shooter"] = torch.mean(loss_dict["shooter"] * validity["shooter"])

        # Just get the shot type and shot made probabilities for the shooter, ignore all other players
        arange = torch.arange(targets["shooter"].shape[0])
        shot_made_outputs = outputs["shot_made"][arange, targets["shooter"], targets["shot_type"]]
        shot_type_outputs = outputs["shot_type"][arange, targets["shooter"], :]

        loss_dict["shot_made"] = self.shot_made_loss(
            shot_made_outputs, targets["shot_made"].squeeze().float()
        )
        loss_dict["shot_made"] = torch.mean(
            loss_dict["shot_made"] * validity["shot_made"]
        )

        loss_dict["shot_type"] = self.shot_type_loss(
            shot_type_outputs, targets["shot_type"]
        )
        loss_dict["shot_type"] = torch.mean(
            loss_dict["shot_type"] * validity["shot_type"]
        )

        pred_box_stats = extract_stats(targets, outputs)
        gt_box_stats = extract_gt_stats(targets, validity)
        loss_dict["box"] = 0
        for player in gt_box_stats.keys():
            for stat in gt_box_stats[player].keys():
                if stat == "o_pos":
                    continue
                loss_dict["box"] += self.box_loss(pred_box_stats[player][stat].float(), gt_box_stats[player][stat].float())
        loss_dict["box"] = loss_dict["box"] / self.batch_size

        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] * self.loss_weights[k]

        return loss_dict

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        self.loss_dict = self.get_losses(batch)
        for k, v in self.loss_dict.items():
            self.log(f"l-{k}", ema(k, v), prog_bar=True, on_step=True)
        return sum(self.loss_dict.values())

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        self.loss_dict = self.get_losses(batch)
        val_loss = sum(self.loss_dict.values())
        self.log('val_loss', val_loss)
        return val_loss

    def create_metrics(self, **kwargs):
        self.initial_event_accuracy = Accuracy(top_k=1)
        self.shooter_accuracy = Accuracy(top_k=1)
        self.shot_made_accuracy = Accuracy(multiclass=False)
        self.shot_type_accuracy = Accuracy(top_k=1)

    def validation_step_end(self, outputs) -> torch.Tensor:
        self.initial_event_accuracy(
            self.outputs["initial_event"], self.targets["initial_event"]
        )

        mask = self.validity["shooter"].nonzero().squeeze()
        valid_shooters = self.targets["shooter"][mask]
        valid_shot_types = self.targets["shot_type"][mask]

        self.shooter_accuracy(self.outputs["shooter"][mask], valid_shooters)

        self.shot_made_accuracy(
            self.outputs["shot_made"][mask, valid_shooters, valid_shot_types],
            self.targets["shot_made"].squeeze()[mask],
        )

        self.shot_type_accuracy(
            self.outputs["shot_type"][mask, valid_shooters, :],
            valid_shot_types,
        )

        self.log("train_loss", outputs)
        return outputs

    def validation_epoch_end(self, outs) -> None:
        iea = self.initial_event_accuracy.compute()
        self.log("initial_event_accuracy", iea)
        sa = self.shooter_accuracy.compute()
        self.log("shooter_accuracy", sa)
        sma = self.shot_made_accuracy.compute()
        self.log("shot_made_accuracy", sma)
        sta = self.shot_type_accuracy.compute()
        self.log("shot_type_accuracy", sta)
        self.log("summed_accuracy", (iea + sa + sma + sta) / 4)
        if self.curr_epoch > 0 and self.curr_epoch % self.print_every == 0:
            self.print(iea, sa, sma, sta)
            predicted_box_stats, gt_stats = get_predicted_stats(self, get_game(), self.device, as_box=True)
            print(predicted_box_stats)

    def training_epoch_end(self, outs) -> None:
        self.curr_epoch += 1
        #self.scheduler.step()

    def configure_optimizers(self):
        #ep_up = self.epochs // 4 + 1
        #ep_down = self.epochs - ep_up
        optim = torch.optim.AdamW([
            {"params": self.embeddings.parameters(), },
            {"params": self.heads.parameters()},
            {"params": self.backbone.parameters(), 'lr': self.lr_backbone, 'weight_decay': 0} 
        ], lr=self.lr, weight_decay=self.weight_decay)
        #self.scheduler = torch.optim.lr_scheduler.CyclicLR(
        #    optim, self.min_lr, self.max_lr, ep_up, ep_down, cycle_momentum=False
        #)
        return optim


if __name__ == "__main__":
    import h5py
    from src.data.game import Game

    db_name = "cache/ml_db_0.0.2.h5"
    datamodule = PlayModule(db_name=db_name, batch_size=320, num_workers=0)

    with h5py.File(db_name, "r", libver="latest", swmr=True) as db:
        test_game = Game(db["raw_data/2016_playoffs/games"][-4])
        test_plays = db["raw_data/2016_playoffs/plays"][
            test_game.play_start_idx : test_game.play_end_idx
        ]

    model = PlayModel(epochs=2)

    # print(parse_multiple_plays(test_plays))
    # print(model(format_data(test_plays)))

    trainer = Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=model.epochs,
        gpus=None,
    )

    trainer.fit(model, datamodule)
