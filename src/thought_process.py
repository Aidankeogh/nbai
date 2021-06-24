import oyaml
import torch
from typing import OrderedDict
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from src.thought_path import DataConfig
import torch.nn as nn

softmax = nn.Softmax(dim=1)
cross_entropy = nn.CrossEntropyLoss(reduction="none")


class ThoughtProcess(LightningModule):
    def __init__(self, cfg_path: str, data_cfg_path: str) -> None:
        super().__init__()
        cfg = oyaml.load(open(cfg_path, "r"), Loader=oyaml.Loader)
        data_cfg = DataConfig(data_cfg_path)
        self.config = cfg
        self.inputs = []
        self.embeddings = {}
        self.dimensions = {}
        self.data_indices = data_cfg.data_indices
        self.data_indices.update(data_cfg.slice_keys)
        for key, input in cfg["in"].items():
            self.inputs.append(key)
            if input["type"] in ["embedding", "embedding_list"]:
                self.embeddings[key] = nn.Embedding(input["num"], input["dim"])
                self.dimensions[key] = input["dim"]
        for key, input in cfg["in"].items():
            if input["type"] == "embedding_choice":
                self.dimensions[key] = self.dimensions[input["choices"]]

        self.losses = cfg["losses"]
        for loss in self.losses:
           loss["metric"] = Accuracy(top_k=1)

        self.layers = OrderedDict()
        for layer in cfg["compute"]:

            in_dim = self.dimensions[layer["in"][0]]

            if layer["type"] == "transformer":
                if len(layer["in"]) != len(layer["out"]):
                    raise (
                        "Error, number of input embeddings doesn't match output embeddings"
                    )
                d_model = in_dim
                heads = layer["heads"]
                n_layers = layer["layers"]
                d_ff = layer["dim_feedforward"]
                tel = nn.TransformerEncoderLayer(d_model, heads, d_ff)
                te = nn.TransformerEncoder(tel, n_layers)
                layer["module"] = te
                layer["dim"] = d_model
                self.layers[layer["name"]] = layer
                for output in layer["out"]:
                    self.dimensions[output] = d_model

            if layer["type"] == "feedforward":
                hidden = layer["hidden"]
                out_dim = layer["out_dim"]
                layer["module"] = nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
                )
                self.layers[layer["name"]] = layer
                self.dimensions[layer["out"]] = out_dim

        for key, layer in self.layers.items():
            self.__setattr__(key, layer["module"])
        for key, embeds in self.embeddings.items():
            self.__setattr__(f"{key}_embeddings", embeds)

    def forward(self, play):
        data = {}
        for key in self.inputs:
            idx = self.data_indices[key]
            temp = play[:, idx].long()
            if len(temp.shape) == 1:
                temp = temp.unsqueeze(dim=0)
            elif len(temp.shape) == 2:
                temp = temp.permute(1, 0)
            if key in self.embeddings:
                temp = self.embeddings[key](temp)
            data[key] = temp

        for name, layer in self.layers.items():
            # Load/Concatenate data
            x = []
            n_embeddings = []
            for key in layer["in"]:
                temp = data[key]
                n_embeddings.append(temp.shape[0])
                x.append(temp)
            x = torch.cat(x)

            if layer["type"] == "transformer":
                x = layer["module"](x)
                embeddings_used = 0
                for key, curr_size in zip(layer["out"], n_embeddings):
                    data[key] = x[embeddings_used : embeddings_used + curr_size]
                    embeddings_used += curr_size

            if layer["type"] == "feedforward":
                outputs = []
                for component in x[:]:
                    outputs.append(layer["module"](component))
                if len(outputs) == 1:
                    x = outputs[0]
                else:
                    x = torch.cat(outputs, dim=1)
                if layer["end"] == "softmax":
                    x = softmax(x)

                data[layer["out"]] = x
        return data

    def get_losses(self, batch):        
        all_outputs = self(batch)
        loss_dict = {}
        for loss in self.losses:
            if loss["type"] == "classification":
                mask = None
                outputs = all_outputs[loss["out"]]
                labels = batch[:, self.data_indices[loss["label"]]]
                if "options" in loss:
                    options = batch[:, self.data_indices[loss["options"]]]
                    labels = torch.eq(options, labels.view(-1, 1)).long()
                _, labels = labels.max(dim=1)
                loss["outputs"] = outputs
                loss["labels"] = labels
                raw_losses = cross_entropy(outputs, labels)
                if "conditions" in loss:
                    for condition in loss["conditions"]:
                        mask = batch[:, self.data_indices[condition]]
                        raw_losses *= mask
                loss_dict[loss["name"]] = torch.mean(raw_losses)
        return loss_dict

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        self.loss_dict = self.get_losses(batch)
        return sum(self.loss_dict.values())

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        self.loss_dict = self.get_losses(batch)
        return sum(self.loss_dict.values())

    def validation_step_end(self, outputs) -> torch.Tensor:
        #update and log
        for loss in self.losses:
            loss["metric"](loss["outputs"], loss["labels"])
        return outputs

    def validation_epoch_end(self, outs) -> None:
        for loss in self.losses:
            self.log(loss["name"], loss["metric"].compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    from src.loader_pipeline import open_db

    with open_db() as db:
        batch = torch.tensor(db["raw_data/2001_playoffs/plays"][:4])

    tp = ThoughtProcess("src/ml/play.yaml", "src/data/play.yaml")
    outputs = tp(batch)
    for k, v in outputs.items():
        print(k, v.shape)
    tot_params = sum(p.numel() for p in tp.parameters())
    learnable_params = sum(p.numel() for p in tp.parameters() if p.requires_grad)
    loss = tp.training_step(batch, None)
    loss.backward()
