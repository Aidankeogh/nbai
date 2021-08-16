from optuna import samplers
from src.ml.play_model import PlayModel
from src.ml.play_dataset import PlayModule
from pytorch_lightning import Trainer
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from src.loader_pipeline import DB_NAME
from pprint import pprint
from coolname import generate_slug
from datetime import datetime
from src.utilities.embedding_utilities import get_name, get_idx, n_players
import os
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

def objective(trial: optuna.trial.Trial) -> float:
    args = {
        "epochs": 30,
        "lr": 1e-3, #trial.suggest_loguniform("lr", 1e-5, 1e-1),
        "lr_backbone": 1e-5, #trial.suggest_loguniform("lr_backbone", 1e-10, 1e-5),
        "weight_decay": 1e-3, # trial.suggest_loguniform("min_lr", 1e-8, 1e-5),
        "embedding_dim": 64,  # Will be made a multiple of n_heads
        "n_layers": 2, # trial.suggest_int("n_layers", 1, 4), # 12,
        "add_upsampled": True,
        "remove_team": True,
        "batch_size": 640,
        "initial_event_wt": 0.5,  #trial.suggest_uniform("initial_event_wt", 0, 1),
        "shooter_wt": 1,  #trial.suggest_uniform("shooter_wt", 0, 1),
        "shot_made_wt": 4,  #trial.suggest_uniform("shot_made_wt", 0, 1),
        "shot_type_wt": 0.5,  #trial.suggest_uniform("shot_type_wt", 0, 1),
        "box_wt": 1,
    }
    model = PlayModel(**args)
    datamodule = PlayModule(
        db_name=DB_NAME, batch_size=args["batch_size"], num_workers=4, shuffle_indices=False, train_seasons=range(2001, 2020)
    )
    optuna_callback = PyTorchLightningPruningCallback(trial, monitor="summed_accuracy")

    # Set up logging and checkpoints
    run_id = datetime.now().strftime("%m.%d-%H:%M-") + generate_slug(2)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'runs/{run_id}/checkpoints',
        filename='shot-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    tb_logger = TensorBoardLogger(f"runs/{run_id}/logs", name="shot_model")
    print(f"LOGGING TO runs/{run_id}/logs")
    trainer = Trainer(
        default_root_dir='cache',
        logger=tb_logger,
        max_epochs=args["epochs"],
        gpus=[0],
        callbacks=[optuna_callback, checkpoint_callback],
    )

    hyperparameters = args
    pprint(hyperparameters)
    trainer.logger.log_hyperparams(hyperparameters)
    try:
        trainer.fit(model, datamodule)
    except KeyboardInterrupt:
        print(f'Training cancelled early!')

    # Add embeddings to tensorboard for visualization
    player_idxes = torch.arange(32, 2000)
    embeddings = model.player_embeddings(player_idxes)
    names = [get_name(idx) for idx in player_idxes]
    tb_logger.experiment.add_embedding(embeddings, names)
    out_dir = os.path.expanduser(f"~/nbai/runs/{run_id}/logs")
    print()
    print(f"tensorboard --logdir={out_dir}")
    print()

    return trainer.callback_metrics["summed_accuracy"].item()


if __name__ == "__main__":
    pruner: optuna.pruners.BasePruner = optuna.pruners.SuccessiveHalvingPruner()
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=5
    )  # NSGAIISampler(population_size=6)

    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=1, timeout=600000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
