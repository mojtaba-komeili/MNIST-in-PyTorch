from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST


MINI_BATCH_SIZE = 16
LOADING_WORKERS = 2
TRAINING_EPOCHS = 2
# Transfer learning scenario
# 1- Fixing the parameters for all layer except the attached layer
# 2-(default) Updating the entire model
TRAINING_MODE = 1


class Model():
    def __init__(self, n_hidden=120):
        transferred_model = torchvision.models.resnet18()
        num_last_layer_feats = transferred_model.fc.in_features
        fully_connected = torch.nn.Linear(num_last_layer_feats, 10)

        if TRAINING_MODE == 1:
            print('Training the parameters in last layer.')
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

    def _convert_data_to_tensors(self, data):
        x_test, y_test = data
        x_test = torch.autograd.Variable(x_test)
        y_test = torch.autograd.Variable(y_test)
        return x_test, y_test

    def forward(self, x):
        x_shape = list(x.shape)
        x_shape[1] = 3
        x_higher = torch.autograd.Variable(torch.zeros(x_shape))
        x_higher[:, 0, :, :]
        return self.model.forward(x_higher)

    def train(self, mode=True, epochs=TRAINING_EPOCHS):
        for e in range(epochs):
            epoch_loss = 0
            for i, data in enumerate(tqdm(self.get_data_minibatch(train=True))):
                x_train, y_target = self._convert_data_to_tensors(data)

                self.optimizer.zero_grad()
                out_puts = self.forward(x_train)
                loss = self.criterion(out_puts, y_target)
                epoch_loss += loss.data[0]
                loss.backward()
                self.optimizer.step()
            sleep(0.01)  # Make sure tqdm prints is finished
            print('Epoch %d out of %d, loss %g'
                  % (e+1, TRAINING_EPOCHS, epoch_loss))

    def predict(self, x):
        y = self.forward(x)
        y = F.softmax(y, dim=1)
        pred_val, pred = torch.max(y, 1)
        return pred

    def estimate_model_accuracy(self):
        total_samples = 0
        correct_pred = 0
        for i, data in enumerate(tqdm(self.get_data_minibatch(train=False))):
            x_test, y_test = self._convert_data_to_tensors(data)
            y_pred = self.predict(x_test)
            y_test = y_test.data.numpy()
            y_pred = y_pred.data.numpy()
            correct_pred += sum(y_pred == y_test)
            total_samples += len(y_test)
        sleep(0.01)  # Make sure tqdm prints is finished
        print('Total samples', total_samples)
        print('Correct predictions:', correct_pred)
        print('Total accuracy %.2f %%'
              % (100. * float(correct_pred) / float(total_samples)))

    def show_sample_prediction(self):
        it = enumerate(self.get_data_minibatch(train=False, n_mini_batch=1))
        _, data = next(it)
        x_test, y_test = self._convert_data_to_tensors(data)
        y_pred = self.predict(x_test)
        if int(y_pred) == int(y_test):
            print("CORRECT!")
        else:
            print("WRONG!")
        print('Model predicted "%d", the actual value is "%d"' % (y_pred, y_test))
        x_numpy_arr = x_test.data.numpy()[0, 0, :, :]
        plt.imshow(x_numpy_arr)
        plt.gray()
        plt.show()

    def get_data_minibatch(self, train=False, n_mini_batch = MINI_BATCH_SIZE):
        tr = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()
        ])
        mnist_dataset = MNIST('../data', train=train,
                              download=True,transform=tr)
        loader = torch.utils.data.DataLoader(mnist_dataset,
                                                   batch_size=n_mini_batch,
                                                   shuffle=True,
                                                   num_workers=LOADING_WORKERS)
        return loader


if __name__ == '__main__':
    model = Model()
    model.train()
    model.estimate_model_accuracy()
    model.show_sample_prediction()