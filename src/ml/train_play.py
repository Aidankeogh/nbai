from optuna import samplers
from src.ml.play_model import PlayModel
from src.ml.play_dataset import PlayModule
from pytorch_lightning import Trainer
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pprint import pprint


def objective(trial: optuna.trial.Trial) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    # n_layers = trial.suggest_int("n_layers", 1, 3)
    max_lr = trial.suggest_loguniform("max_lr", 1e-2, 1)
    min_lr = trial.suggest_loguniform("min_lr", 1e-2, max_lr)
    args = {
        "epochs": 200,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "n_heads": trial.suggest_int("n_heads", 1, 20),
        "embedding_dim": trial.suggest_uniform(
            "embedding_dim", 20, 200
        ),  # Will be made a multiple of n_heads
        # off_el = Offense encoder layers, def_dff = defense dim_feedforward, etc.
        "off_el": trial.suggest_int("off_el", 1, 4),
        "off_dl": trial.suggest_int("off_dl", 1, 4),
        "off_dff": trial.suggest_int("off_dff", 20, 1000),
        "def_el": trial.suggest_int("def_el", 1, 4),
        "def_dl": trial.suggest_int("def_dl", 1, 4),
        "def_dff": trial.suggest_int("def_dff", 20, 1000),
        "off_team_dl": trial.suggest_int("off_team_dl", 1, 4),
        "off_team_dff": trial.suggest_int("off_team_dff", 20, 1000),
        "def_team_dl": trial.suggest_int("def_team_dl", 1, 4),
        "def_team_dff": trial.suggest_int("def_team_dff", 20, 1000),
        "initial_event_hidden": trial.suggest_int("initial_event_hidden", 20, 1000),
        "shooter_hidden": trial.suggest_int("shooter_hidden", 20, 1000),
        "shot_made_hidden": trial.suggest_int("shot_made_hidden", 20, 1000),
        "shot_type_hidden": trial.suggest_int("shot_type_hidden", 20, 1000),
        "initial_event_wt": trial.suggest_uniform("initial_event_wt", 0, 1),
        "shooter_wt": trial.suggest_uniform("shooter_wt", 0, 1),
        "shot_made_wt": trial.suggest_uniform("shot_made_wt", 0, 1),
        "shot_type_wt": trial.suggest_uniform("shot_type_wt", 0, 1),
    }
    # Force to int and multiple of n_heads
    args["embedding_dim"] = int(
        (args["embedding_dim"] // args["n_heads"]) * args["n_heads"]
    )
    model = PlayModel(**args)
    datamodule = PlayModule(
        db_name="cache/ml_db_0.0.1.h5", batch_size=6400, num_workers=4
    )
    callback = PyTorchLightningPruningCallback(trial, monitor="summed_accuracy")
    trainer = Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=args["epochs"],
        gpus=[0],
        callbacks=[callback],
    )
    hyperparameters = args
    pprint(hyperparameters)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule)
    return trainer.callback_metrics["summed_accuracy"].item()


if __name__ == "__main__":
    pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner()
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=5
    )  # NSGAIISampler(population_size=6)

    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=500, timeout=600000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
