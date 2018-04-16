from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets.mnist import MNIST


DEFAULT_TRAINING_EPOCHS = 10
DEFAULT_LOADING_WORKERS = 2
DEFAULT_MINI_BATCH_SIZE = 32


class BaseModel(torch.nn.Module):
    def __init__(self, use_gpu=True,
                 batch_size=DEFAULT_MINI_BATCH_SIZE,
                 num_loading_workers=DEFAULT_LOADING_WORKERS):
        super(BaseModel, self).__init__()
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.num_loading_workers = num_loading_workers

    def _convert_data_to_tensors(self, data):
        x_test, y_test = data
        if self.use_gpu and torch.cuda.is_available():
            x_test = x_test.cuda()
            y_test = y_test.cuda()
        x_test = torch.autograd.Variable(x_test)
        y_test = torch.autograd.Variable(y_test)
        return x_test, y_test

    def forward(self, x):
        raise NotImplementedError("Forward pass is not implemented.")

    def train(self, epochs):
        self.training = True
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
                  % (e+1, epochs, epoch_loss))

    def predict(self, x):
        self.training = False
        y = self.forward(x)
        y = F.softmax(y, dim=1)
        pred_val, pred = torch.max(y, 1)
        return pred

    def estimate_model_accuracy(self):
        self.training = False
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
        self.training = False
        it = enumerate(self.get_data_minibatch(train=False, n_mini_batch=1))
        _, data = next(it)
        x_test, y_test = self._convert_data_to_tensors(data)
        y_pred = self.predict(x_test)
        if int(y_pred) == int(y_test):
            print("CORRECT!")
        else:
            print("WRONG!")
        print('Model predicted "%d", the actual value is "%d"'
              % (y_pred, y_test))
        x_numpy_arr = x_test.data.numpy()[0, 0, :, :]
        plt.imshow(x_numpy_arr)
        plt.gray()
        plt.show()

    def get_data_minibatch(self, train=False):
        tr = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        mnist_dataset = MNIST('../data', train=train,
                              download=True, transform=tr)
        loader = torch.utils.data.DataLoader(
            mnist_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_loading_workers)
        return loader


def run_model(model, epochs=DEFAULT_TRAINING_EPOCHS,
              eval=True, show_sample=True, use_gpu=True):
    if use_gpu and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
    model.train(epochs=epochs)
    if eval:
        model.estimate_model_accuracy()
    if show_sample:
        model.show_sample_prediction()
