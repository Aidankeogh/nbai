from src.ml.play_dataset import PlayModule, format_data
from src.data.play import Play
from src.data.box_stats import parse_multiple_plays
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from src.thought_path import DataConfig
from torchmetrics import Accuracy
from pprint import pprint
import torch.nn as nn
import torch


def create_encoder(dim, nhead, dim_feedforward, n_layers):
    layer = nn.TransformerEncoderLayer(dim, nhead, dim_feedforward)
    encoder = nn.TransformerEncoder(layer, n_layers)
    return encoder


def create_decoder(dim, nhead, dim_feedforward, n_layers):
    layer = nn.TransformerDecoderLayer(dim, nhead, dim_feedforward)
    decoder = nn.TransformerDecoder(layer, n_layers)
    return decoder


def feedforward(embedding, hidden, out, final=None, activation=nn.ReLU()):
    layers = [nn.Linear(embedding, hidden), activation, nn.Linear(hidden, out)]
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
    def __init__(self, epochs=4, min_lr=0.1, max_lr=1.0, **kwargs):
        super().__init__()

        self.epochs = epochs
        self.curr_epoch = 0
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.create_backbone(**kwargs)
        self.create_heads(**kwargs)
        self.create_losses(**kwargs)
        self.create_metrics(**kwargs)

    def create_backbone(
        self,
        embedding_dim=32,
        n_heads=8,
        # off_el = Offense encoder layers, def_dff = defense dim_feedforward, etc.
        off_el=3,
        off_dl=2,
        off_dff=64,
        def_el=3,
        def_dl=1,
        def_dff=64,
        off_team_dl=3,
        off_team_dff=64,
        def_team_dl=3,
        def_team_dff=64,
        **kwargs
    ):
        # <40 teams
        # <4000 players
        self.team_embeddings = nn.Embedding(40, embedding_dim)
        self.player_embeddings = nn.Embedding(4000, embedding_dim)
        self.offense_encoder = create_encoder(embedding_dim, n_heads, off_dff, off_el)
        self.offense_decoder = create_decoder(embedding_dim, n_heads, off_dff, off_dl)
        self.offense_team_decoder = create_decoder(
            embedding_dim, n_heads, off_team_dff, off_team_dl
        )

        self.defense_encoder = create_encoder(embedding_dim, n_heads, def_dff, def_el)
        self.defense_decoder = create_decoder(embedding_dim, n_heads, def_dff, def_dl)
        self.defense_team_decoder = create_decoder(
            embedding_dim, n_heads, def_team_dff, def_team_dl
        )

    def create_heads(
        self,
        embedding_dim=32,
        initial_event_hidden=256,
        shooter_hidden=256,
        shot_made_hidden=256,
        shot_type_hidden=256,
        **kwargs
    ):
        self.initial_event = feedforward(
            embedding_dim * 2, initial_event_hidden, 5, final=nn.Softmax(dim=1)
        )

        self.shooter = feedforward(embedding_dim, shooter_hidden, 1)
        self.shot_made = feedforward(
            embedding_dim, shot_made_hidden, 1, final=nn.Sigmoid()
        )
        self.shot_type = feedforward(
            embedding_dim, shot_type_hidden, 5, final=nn.Softmax(dim=1)
        )

    def create_losses(
        self, initial_event_wt=1, shooter_wt=1, shot_made_wt=1, shot_type_wt=1, **kwargs
    ):
        self.initial_event_loss = nn.CrossEntropyLoss(reduction="mean")
        self.shooter_loss = nn.CrossEntropyLoss(reduction="none")
        self.shot_made_loss = nn.BCELoss(reduction="none")
        self.shot_type_loss = nn.CrossEntropyLoss(reduction="none")
        self.loss_weights = {
            "initial_event": initial_event_wt,
            "shooter": shooter_wt,
            "shot_made": shot_made_wt,
            "shot_type": shot_type_wt,
        }

    def transformer_backbone(
        self, offense_roster, defense_roster, offense_team, defense_team
    ):

        # Create player encodings (attending to other members on same team)
        offense_roster = self.offense_encoder(offense_roster)
        defense_roster = self.defense_encoder(defense_roster)

        # Update player encodings, letting them attend to the players on the other team.
        # Offense is most important, so compute defense first to feed better defensive embeddings into ofPlayModel
        # Now compute team embeddings
        offense_team = self.offense_team_decoder(offense_team, offense_roster)
        defense_team = self.defense_team_decoder(defense_team, defense_roster)

        return offense_roster, defense_roster, offense_team, defense_team

    def forward(self, batch):
        play, validity = batch

        # Extract player and team embeddings
        offense_roster = play["offense_roster"]
        offense_roster = self.player_embeddings(offense_roster.permute(1, 0))
        defense_roster = play["defense_roster"]
        defense_roster = self.player_embeddings(defense_roster.permute(1, 0))

        offense_team = play["offense_team"]
        offense_team = self.team_embeddings(offense_team)
        defense_team = play["defense_team"]
        defense_team = self.team_embeddings(defense_team)

        # Run through backbone
        (
            offense_roster,
            defense_roster,
            offense_team,
            defense_team,
        ) = self.transformer_backbone(
            offense_roster, defense_roster, offense_team, defense_team
        )

        # Swap the axes to make the batch dimensions are the traditional batch x player x ...
        # Instead of transformer output player x batch x ...
        offense_roster, defense_roster = torch.swapaxes(
            offense_roster, 0, 1
        ), torch.swapaxes(defense_roster, 0, 1)

        # For team embeddings get rid of extra dim
        offense_team, defense_team = offense_team.squeeze(dim=0), defense_team.squeeze(
            dim=0
        )
        teams_combined = torch.cat((offense_team, defense_team), dim=1)

        out_dict = {
            "initial_event": self.initial_event(teams_combined),
            "shooter": apply_per_player(
                self.shooter, offense_roster, softmax_over_players=True
            ),
            "shot_made": apply_per_player(self.shot_made, offense_roster),
            "shot_type": apply_per_player(self.shot_type, offense_roster),
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
        shot_made_outputs = outputs["shot_made"][arange, targets["shooter"]]
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

        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] * self.loss_weights[k]

        return loss_dict

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        self.loss_dict = self.get_losses(batch)
        return sum(self.loss_dict.values())

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        self.loss_dict = self.get_losses(batch)
        return sum(self.loss_dict.values())

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

        self.shooter_accuracy(self.outputs["shooter"][mask], valid_shooters)

        self.shot_made_accuracy(
            self.outputs["shot_made"][mask, valid_shooters],
            self.targets["shot_made"].squeeze()[mask],
        )

        self.shot_type_accuracy(
            self.outputs["shot_type"][mask, valid_shooters, :],
            self.targets["shot_type"][mask],
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
        if self.curr_epoch % 40 == 0:
            self.print(iea, sa, sma, sta)

    def training_epoch_end(self, outs) -> None:
        self.curr_epoch += 1
        self.scheduler.step()

    def configure_optimizers(self):
        ep_up = self.epochs // 4
        ep_down = self.epochs - ep_up
        optim = torch.optim.Adam(self.parameters(), lr=0.02)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            optim, self.min_lr, self.max_lr, ep_up, ep_down, cycle_momentum=False
        )
        return optim


if __name__ == "__main__":
    import h5py
    from src.data.game import Game

    db_name = "cache/ml_db_0.0.1.h5"
    datamodule = PlayModule(db_name=db_name, batch_size=32000, num_workers=4)

    with h5py.File(db_name, "r", libver="latest", swmr=True) as db:
        test_game = Game(db["raw_data/2016_playoffs/games"][-4])
        test_plays = db["raw_data/2016_playoffs/plays"][
            test_game.play_start_idx : test_game.play_end_idx
        ]

    model = PlayModel(epochs=5)

    print(parse_multiple_plays(test_plays))
    print(model(format_data(test_plays)))

    trainer = Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=model.epochs,
        gpus=[0],
    )

    trainer.fit(model, datamodule)
