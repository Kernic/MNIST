import torch
from torch import nn
from torch import Tensor
from torch.nn.init import uniform_
import plotly.graph_objects as go 
import copy

class ktanh(nn.Module):
    A = 1.7159
    B = 0.6666
    def forward(self, input: Tensor) -> Tensor:
        return self.A * torch.tanh(self.B * input)

class FFNN(nn.Module):
    def __init__(self, init_weights=True):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 2500),
            ktanh(),
            nn.Linear(2500, 2000),
            ktanh(),
            nn.Linear(2000, 1500),
            ktanh(),
            nn.Linear(1500, 1000),
            ktanh(),
            nn.Linear(1000, 500),
            ktanh(),
            nn.Linear(500, 10),
        )
        if init_weights:
            self.weight_init()

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

    def weight_init(self):
        for i in self.linear_relu_stack:
            try:
                uniform_(i.weight, -0.05, 0.05)
            except:
                pass



if __name__ == "__main__":
    import load_dataset
    from trainer import trainer
    ffnn_1 = FFNN()
    ffnn_2 = FFNN()
    ffnn_3 = FFNN()
    train_data_tranformed, test_data_b = load_dataset.get_datasets(
        transform=True
    )
    train_data, test_data = load_dataset.get_datasets()
    epoch = 50 
    t_1 = trainer(nn.CrossEntropyLoss(), ffnn_1, (1e-3-1e-6)/epoch)
    t_2 = trainer(nn.CrossEntropyLoss(), ffnn_2, (1e-3-1e-6)/epoch)
    t_3 = trainer(nn.CrossEntropyLoss(), ffnn_3, 0)
    t_1.start(test_data, train_data, epoch)
    t_2.start(test_data_b, train_data_tranformed, epoch)
    t_3.start(test_data, train_data, epoch)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=t_1.corrects,
            x=list(range(epoch)), 
            mode="lines", 
            name="Error Rate no transformed data"
        )
    )
    fig.add_trace(
        go.Scatter(
            y=t_1.test_losses,
            x=list(range(epoch)), 
            mode="lines", 
            name="Average Loss no transformed data"
        )
    )
    fig.add_trace(
        go.Scatter(
            y=t_2.corrects,
            x=list(range(epoch)), 
            mode="lines", 
            name="Error Rate transformed data"
        )
    )
    fig.add_trace(
        go.Scatter(
            y=t_2.test_losses,
            x=list(range(epoch)), 
            mode="lines", 
            name="Average Loss transformed data"
        )
    )
    fig.add_trace(
        go.Scatter(
            y=t_3.corrects,
            x=list(range(epoch)), 
            mode="lines", 
            name="Error Rate no regressing Learning Rate"
        )
    )
    fig.add_trace(
        go.Scatter(
            y=t_3.test_losses,
            x=list(range(epoch)), 
            mode="lines", 
            name="Average Loss no regressing Learning Rate"
        )
    )
    fig.show()
