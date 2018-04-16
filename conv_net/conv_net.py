import torch
import torch.nn.functional as F
from base_model import (BaseModel, run_model,
                        DEFAULT_MINI_BATCH_SIZE,
                        DEFAULT_LOADING_WORKERS)


class ConvNet(BaseModel):
    def __init__(self, batch_size=DEFAULT_MINI_BATCH_SIZE,
                 num_loading_worker=DEFAULT_LOADING_WORKERS):
        super(ConvNet, self).__init__(batch_size=batch_size,
                                      num_loading_workers=num_loading_worker)

        self.conv1 = torch.nn.Conv2d(1, 20, 5)
        self.conv2 = torch.nn.Conv2d(20, 10, 5)
        self.fc1 = torch.nn.Linear(10*4*4, 50)
        self.fc2_do = torch.nn.Linear(50, 25)
        self.fc3 = torch.nn.Linear(25, 10)

        optimization_params = list(self.parameters())
        self.optimizer = torch.optim.Adam(optimization_params)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.max_pool2d(y, 2)
        y = F.relu(self.conv2(y))
        y = F.max_pool2d(y, 2)
        y = y.view(-1, 10*4*4)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2_do(y))
        y = F.dropout(y, training=self.training)
        y = self.fc3(y)
        return y


if __name__ == '__main__':
    model = ConvNet()
    run_model(model, epochs=2)
