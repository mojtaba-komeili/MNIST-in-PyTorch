from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST


MINI_BATCH_SIZE = 20
LOADING_WORKERS = 2
TRAINING_EPOCHS = 20


class Model(torch.nn.Module):
    def __init__(self, n_hidden=120):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, 10)
        self.optimizer = torch.optim.Adam(self.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()

    def _flatten_mini_batch(self, x):
        num_samples = x.shape[0]
        return x.contiguous().view([num_samples, -1])

    def _convert_data_to_tensors(self, data):
        x_test, y_test = data
        x_test = torch.autograd.Variable(x_test)
        y_test = torch.autograd.Variable(y_test)
        return x_test, y_test

    def forward(self, x):
        x = self._flatten_mini_batch(x)
        y = F.relu(self.fc1(x))
        y = self.fc2(y)
        return y

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
            correct_pred += sum(y_pred == y_test)
            total_samples += len(y_test)
        sleep(0.01)  # Make sure tqdm prints is finished
        print('Total samples', total_samples)
        print('Total accuracy %.2f %%'
              % (100. * float(correct_pred) / float(total_samples)))

    def show_sample_prediction(self):
        it = enumerate(self.get_data_minibatch(train=False, n_mini_batch=1))
        _, data = next(it)
        x_test, y_test = self._convert_data_to_tensors(data)
        y_pred = self.predict(x_test)
        if int(y_pred) == int(y_test) :
            print("CORRECT!")
        else:
            print("WRONG!")
        print('Model predicted "%d", the actual value is "%d"' % (y_pred, y_test))
        x_numpy_arr = x_test.data.numpy()[0, 0, :, :]
        plt.imshow(x_numpy_arr)
        plt.gray()
        plt.show()

    def get_data_minibatch(self, train=False, n_mini_batch = MINI_BATCH_SIZE):
        mnist_dataset = MNIST('../data', train=train, download=True,
                              transform=torchvision.transforms.ToTensor())
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