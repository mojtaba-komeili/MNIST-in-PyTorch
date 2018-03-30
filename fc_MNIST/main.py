from tqdm import tqdm

import torch
import torchvision
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST


MINI_BATCH_SIZE = 10
LOADING_WORKERS = 0


class Model(torch.nn.Module):
    def __init__(self, n_hidden=120):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, 10)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def flatten_mini_batch(self, x):
        num_samples = x.shape[0]
        return x.contiguous().view([num_samples, -1])

    def forward(self, x):
        x = self.flatten_mini_batch(x)
        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        return y

    def train(self, mode=True, epochs=5):
        for e in range(epochs):
            epoch_loss = 0
            for i, data in enumerate(tqdm(self.get_data_minibatch(train=True))):
                x_train, y_target = data
                x_train = torch.autograd.Variable(x_train)
                y_target = torch.autograd.Variable(y_target)

                self.optimizer.zero_grad()
                out_puts = self.forward(x_train)
                loss = self.criterion(out_puts, y_target)
                epoch_loss += loss.data[0]
                loss.backward()
                self.optimizer.step()
            print('Epoch %d loss %g' %(e+1, epoch_loss))

    def get_data_minibatch(self, train=False):
        mnist_dataset = MNIST('../data', train=train, download=True,
                              transform=torchvision.transforms.ToTensor())
        loader = torch.utils.data.DataLoader(mnist_dataset,
                                                   batch_size=MINI_BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=LOADING_WORKERS)
        return loader


if __name__ == '__main__':
    print('Started')
    model = Model()
    model.train()
