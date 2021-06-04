from src.thought_process import ThoughtProcess
from src.ml.play_dataset import PlayModule
from pytorch_lightning import Trainer
play_process= ThoughtProcess("src/ml/play.yaml", "src/data/play.yaml")
play_module = PlayModule(batch_size=640, num_workers=0)
trainer = Trainer(gpus=0)
trainer.fit(play_process, play_module)