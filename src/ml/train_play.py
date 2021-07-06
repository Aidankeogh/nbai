from optuna import samplers
from src.ml.play_model import PlayModel
from src.ml.play_dataset import PlayModule
from pytorch_lightning import Trainer
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial: optuna.trial.Trial) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
   # n_layers = trial.suggest_int("n_layers", 1, 3)
    max_lr = trial.suggest_loguniform("max_lr", 1e-2, 1)
    min_lr = trial.suggest_loguniform("min_lr", 1e-6, max_lr)
    epochs = 500
    model = PlayProcess("src/ml/play.yaml", "src/data/play.yaml")
    model.set_params(epochs, min_lr, max_lr)
    datamodule = PlayModule(db_name="cache/ml_db_0.0.1.h5", batch_size=32000, num_workers=4)
    callback = PyTorchLightningPruningCallback(trial, monitor="shooter_loss")
    callback.on_train_end = callback.on_validation_end
    trainer = Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=epochs,
        gpus=[0],
        callbacks=[callback],
    )
    hyperparameters = dict(min_lr=min_lr, max_lr=max_lr)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, datamodule)
    return trainer.callback_metrics["shooter_loss"].item()
if __name__ == "__main__":
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.SuccessiveHalvingPruner()
    )
    sampler = optuna.samplers.TPESampler(n_startup_trials= 5)#NSGAIISampler(population_size=6)
    
    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=50, timeout=600000)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))