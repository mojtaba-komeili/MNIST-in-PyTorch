import torch
import torchvision
from torchvision.datasets.mnist import MNIST
from base_model import (BaseModel, run_model,
                        DEFAULT_MINI_BATCH_SIZE,
                        DEFAULT_LOADING_WORKERS)

# Transfer learning scenario
# 1- Fixing the parameters for all layer except the attached layer
# 2-(default) Updating the entire model
TRAINING_MODE = 2


class TransferLearning(BaseModel):
    def __init__(self, batch_size=DEFAULT_MINI_BATCH_SIZE,
                 num_loading_worker=DEFAULT_LOADING_WORKERS):
        super(TransferLearning, self).__init__(batch_size=batch_size,
                                               num_loading_workers=num_loading_worker)
        transferred_model = torchvision.models.resnet18()
        num_last_layer_feats = transferred_model.fc.in_features
        fully_connected = torch.nn.Linear(num_last_layer_feats, 10)

        if TRAINING_MODE == 1:
            print('Only training the parameters in the last layer.')
            for param in transferred_model.parameters():
                param.requires_grad = False
            optimization_params = fully_connected.parameters()
        else:
            print('Training the full model.')
            optimization_params = transferred_model.parameters()

        transferred_model.fc = fully_connected
        self.model = transferred_model
        self.optimizer = torch.optim.Adam(optimization_params)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x_shape = list(x.shape)
        x_shape[1] = 3
        x_higher = torch.zeros(x_shape)
        if self.use_gpu and torch.cuda.is_available():
            x_higher = x_higher.cuda()
        x_higher = torch.autograd.Variable(x_higher)
        x_higher[:, 0, :, :] = x
        return self.model.forward(x_higher)

    def get_data_minibatch(self, train=False):
        """
        overridden for adding the scaling and crop
        """
        tr = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()
        ])
        mnist_dataset = MNIST('../data', train=train,
                              download=True, transform=tr)
        loader = torch.utils.data.DataLoader(mnist_dataset,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             num_workers=self.num_loading_workers)
        return loader


if __name__ == '__main__':
    model = TransferLearning()
    run_model(model, epochs=2)
