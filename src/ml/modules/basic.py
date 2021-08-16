import torch.nn as nn

class TransformerPointwise(nn.Module):
    def __init__(self, embedding, out, hidden=0, activation=nn.ReLU(), bn=False):
        super().__init__()
        if bn:
            self.bn = nn.BatchNorm1d(embedding)
        else:
            self.bn = nn.Identity()

        # Can handle a hidden dimensions (2 Layer Perceptron) or a simple linear
        if hidden == 0:
            self.lin1 = nn.Conv1d(embedding, out, 1)
            self.activation = nn.Identity()
            self.lin2 = nn.Identity()
        else:
            self.lin1 = nn.Conv1d(embedding, out, 1)
            self.activation = activation
            self.lin2 = nn.Conv1d(embedding, out, 1)
    
    def forward(self, x):
        # Swap axes to use embedding outputs with conv1d
        x = self.bn(x.swapaxes(1, 2))
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x).swapaxes(1, 2)
        return x

class BaseHead(nn.Module):
    key = None
    stat_type = None
    def stats_pred(self, outputs):
        return {}

    def stats_gt(self, inputs, validity):
        return {}, None