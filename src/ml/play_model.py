from src.ml.modules.heads import Heads
from src.ml.modules.backbone import TransformerBackbone
from src.ml.modules.embeddings import PlayerEmbeddings
from torch._C import ParameterDict, get_default_dtype
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

class PlayModel(LightningModule):
    def __init__(self, epochs=4, lr=1e-5, lr_backbone=0, weight_decay=0, print_every=1, batch_size=1280, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.epochs = epochs
        self.curr_epoch = 0
        self.step = 0
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.print_every = print_every
        self.batch_size = batch_size

        self.embeddings = PlayerEmbeddings(**kwargs)
        self.backbone = TransformerBackbone(**kwargs)
        self.heads = Heads(**kwargs)
        self.create_losses(**kwargs)
        self.create_metrics(**kwargs)

    def create_losses(
        self,
        box_wt = 1,
        **kwargs
    ):
        self.box_loss = nn.MSELoss()
        self.box_wt = box_wt

    def forward(self, batch):
        play, validity = batch

        # Get embeddings
        offense_roster, defense_roster, offense_team, defense_team = self.embeddings(play)

        # Run through transformer backbone
        offense_roster, defense_roster, offense_team, defense_team = self.backbone(
            offense_roster, defense_roster, offense_team, defense_team
        )

        # Create outputs
        out_dict = self.heads(offense_roster, defense_roster, offense_team, defense_team)
        self.step += 1

        if self.step % 2000 == 0:
            print()
            print("###", self.step, "###")
            for k, v in out_dict.items():
                print(k, v[0])
            predicted_box_stats, gt_stats = get_predicted_stats(self, get_game(), self.device, as_box=True)
            print(predicted_box_stats)

        return out_dict

    def get_losses(self, batch):
        targets, validity = batch
        outputs = self(batch)

        # Save outputs and labels for later
        self.outputs = outputs
        self.targets = targets
        self.validity = validity

        loss_dict = self.heads.get_loss(outputs, targets, validity)

        pred_box_stats = extract_stats(targets, outputs)
        gt_box_stats = extract_gt_stats(targets, validity)

        # Get box loss
        loss_dict["box"] = 0
        for player in gt_box_stats.keys():
            for stat in gt_box_stats[player].keys():
                if stat == "o_pos":
                    continue
                component_loss = self.box_loss(pred_box_stats[player][stat].float(), gt_box_stats[player][stat].float())
                loss_dict["box"] += component_loss
        loss_dict["box"] = loss_dict["box"] * self.box_wt / self.batch_size

        return loss_dict

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        self.loss_dict = self.get_losses(batch)
        for k, v in self.loss_dict.items():
            self.log(f"{k[0:3] + k[-2:]}", ema(k, v), prog_bar=True, on_step=True)
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

    pred_stats, gt_stats = get_predicted_stats(model, test_plays, as_box=True)
    print(pred_stats)
    print(gt_stats)

    # trainer = Trainer(
    #     logger=True,
    #     checkpoint_callback=False,
    #     max_epochs=model.epochs,
    #     gpus=None,
    # )

    # trainer.fit(model, datamodule)
