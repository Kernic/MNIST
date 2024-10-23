import torch
from torch import nn
from torch.optim import SGD
from tqdm import tqdm


class layer2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 300),
            nn.ReLU(),
            nn.Linear(300, 10), 
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

class layer6(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 784),
            nn.ReLU(),
            nn.Linear(784, 2500),
            nn.ReLU(),
            nn.Linear(2500, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

class trainer():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    def __init__(self, loss_fn, model):
        self.loss_fn = loss_fn
        self.model = model.to(self.device)        
        self.optimizer = SGD(self.model.parameters(), lr=1e-3)
        print(f"Using {self.device} service\n")
        print(self.model)

    def train(self, dataloader):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation 
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    def test(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n\tAccuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f}\n")
    
    def start(self, test_dataloader, train_dataloader, epoch = 5):
        for t in range(epoch):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train(train_dataloader)
            self.test(test_dataloader)
        print("Done!")


if __name__ == "__main__":
    import load_dataset
    L6 = layer6()
    t = trainer(nn.CrossEntropyLoss(), L6)
    train_data, test_data = load_dataset.get_datasets(100)
    t.start(test_data, train_data, 50)


