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
from src.utilities.embedding_utilities import get_name, get_idx, n_players
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

run="mahogany-puma"
epoch="17"
loss="8.23"
print(n_players)
model = PlayModel.load_from_checkpoint(f"runs/{run}/checkpoints/shot-epoch={epoch}-val_loss={loss}.ckpt")

tb_logger = TensorBoardLogger("cache/tb_logs", name="shot_model")

print(get_idx("lebron-james"))
player_idxes = torch.arange(32, 2000)
embeddings = model.player_embeddings(player_idxes)
names = [get_name(idx) for idx in player_idxes]
tb_logger.experiment.add_embedding(embeddings, names)

print(embeddings.shape)
