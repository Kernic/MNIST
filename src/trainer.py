from torch.optim import SGD
import torch
import copy
from torchvision.transforms import v2, ElasticTransform, RandomRotation, ToTensor
import plotly.express as px

class trainer():
    def __init__(self, loss_fn, model, lrf=1.0):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.loss_fn = loss_fn
        self.model = model.to(self.device)        
        self.optimizer = SGD(self.model.parameters(), lr=1e-3)
        print(f"Using {self.device} service\n")
        print(self.model)
        self.test_losses = []
        self.corrects = []
        self.lrf = lrf

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
        
        curLr = 0
        for g in self.optimizer.param_groups:
            g['lr'] -= self.lrf
        curLr = self.optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {curLr}")

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
        self.test_losses.append(test_loss)
        self.corrects.append(100 * (1 - correct))
        print(
            f"""Test Error: 
    Error Rate: {(100*(1 - correct)):>0.2f}%, Avg loss: {test_loss:>8f}\n"""
        )
    
    def start(self, test_dataloader, train_dataloader, epoch = 5):
        for t in range(epoch):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train(train_dataloader)
            self.test(test_dataloader)
        print("Done!")


