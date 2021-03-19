import torch
from datetime import datetime
from torch.nn import functional as F
from torch import nn
from src.pow_dataset import NbaPosessionDataset
torch.manual_seed(42)

class pownet(nn.Module):
        def __init__(self):
            super(pownet, self).__init__()
            self.bn = nn.BatchNorm1d(2)
            self.order_1 = nn.Linear(2, 1, bias=True)
            self.order_2 = nn.Linear(2, 1, bias=True)

        def forward(self, x):
            for _ in range(10):
                x[:, 4][x[:, 4] <= 0] += 300 # overtime
            time_left = (x[:, 4:5] / 2880.0)
            x[:, 2][x[:, 2] > 6] = 6
            x[:, 3][x[:, 3] > 6] = 6
            score_diff = x[:, 0] - x[:, 1]
            penalty_diff = x[:, 2] - x[:, 3]
            time_feature = x[:, 4:5]
            
            state_vec = torch.stack((score_diff, penalty_diff), dim=1)
            state_vec = self.bn(state_vec)
            order_1 = self.order_1(state_vec)
            order_2 = self.order_2(state_vec) / torch.sqrt(time_left)
            out = torch.sigmoid(order_1 + order_2)
            return out

def train_model(
        config,
        db, 
        epochs=20, 
        batch_size=3200,
        base_lr=1e-4,
        max_lr=1e-2,
        momentum=0.9,
    ):
    db.set_namespace(config['dataset']['key'])
    dataset = NbaPosessionDataset(prebuilt_path=db['save_path'].decode())
    print(f"Dataset loaded, len {len(dataset)}")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=3200, shuffle=True, pin_memory=use_cuda, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=3200, shuffle=True, pin_memory=use_cuda, num_workers=0)

    criterion = nn.BCELoss()
    net = pownet()
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=int(epochs/2))
    print(f"Starting training {net}")

    running_loss = 0
    for epoch in range(epochs):
        epoch_start = datetime.now()
        for start_state, _, label in train_loader:
            start_prob = net(start_state.to(device))
            #end_prob = net(end_state.to(device))
            loss = criterion(start_prob, label.to(device))# + criterion(end_prob, label.to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        seconds = int((datetime.now() - epoch_start).total_seconds())
        print(f"e{epoch}:\tloss {running_loss:.3f}\ttime {seconds}\tlabel {label[0].item()}\tpred {start_prob[0].item():.3f}\tfeat{start_state[0].long().tolist()}")
        running_loss = 0