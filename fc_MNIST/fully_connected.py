import torch
import torch.nn.functional as F
from base_model import BaseModel, run_model


class FullyConnected(BaseModel):
    def __init__(self, n_hidden=120):
        super(FullyConnected, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, 10)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def _flatten_mini_batch(self, x):
        num_samples = x.shape[0]
        return x.contiguous().view([num_samples, -1])

    def forward(self, x):
        x = self._flatten_mini_batch(x)
        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        return y


if __name__ == '__main__':
    model = FullyConnected()
    run_model(model, epochs=2)
