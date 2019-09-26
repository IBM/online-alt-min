import torch
import torch.nn as nn


class LinMod(nn.Linear):
    '''Linear modules with or without batchnorm, all in one module
    '''
    def __init__(self, n_inputs, n_outputs, bias=False, batchnorm=False):
        super(LinMod, self).__init__(n_inputs, n_outputs, bias=bias)
        if batchnorm:
            self.bn = nn.BatchNorm1d(n_outputs, affine=True)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.batchnorm = batchnorm
        self.bias_flag = bias

    def forward(self, inputs):
        outputs = super(LinMod, self).forward(inputs)
        if hasattr(self, 'bn'):
            outputs = self.bn(outputs)
        return outputs

    def extra_repr(self):
        return '{n_inputs}, {n_outputs}, bias={bias_flag}, batchnorm={batchnorm}'.format(**self.__dict__)


class FFNet(nn.Module):
    '''Feed-forward all-to-all connected network
    '''
    def __init__(self, n_inputs, n_hiddens, n_hidden_layers=2, n_outputs=10, nlin=nn.ReLU, bias=False, batchnorm=False):
        super(FFNet, self).__init__()

        self.features = ()  # Skip convolutional features

        self.classifier = nn.Sequential(LinMod(n_inputs, n_hiddens, bias=bias, batchnorm=batchnorm), nlin())
        for i in range(n_hidden_layers - 1):
            self.classifier.add_module(str(2 * i + 2), LinMod(n_hiddens, n_hiddens, bias=bias, batchnorm=batchnorm))
            self.classifier.add_module(str(2 * i + 3), nlin())
        self.classifier.add_module(str(len(self.classifier)), nn.Linear(n_hiddens, n_outputs))

        self.batchnorm = batchnorm
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def train(model, optimizer, train_loader, criterion=nn.CrossEntropyLoss(), log_times=10):
    model.train()
    device = next(model.parameters()).device

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Outputs to terminal
        if batch_idx % (len(train_loader) // log_times) == 0:
            print('  training progress: {}/{} ({:.0f}%)\tloss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))


def test(model, data_loader, criterion=nn.CrossEntropyLoss(), label=''):
    '''Compute model accuracy
    '''
    model.eval()
    device = next(model.parameters()).device

    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            test_loss += criterion(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

    accuracy = float(correct) / len(data_loader.dataset)
    test_loss /= len(data_loader)  # loss function already averages over batch size
    if label:
        print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            label, test_loss, correct, len(data_loader.dataset), 100. * accuracy))
    return accuracy


class LeNet(nn.Module):
    '''Based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
    '''
    def __init__(self, num_input_channels=3, num_classes=10, window_size=32, bias=True):
        super(LeNet, self).__init__()
        self.bias = bias
        self.window_size = window_size
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channels, 6, 5, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(6, 16, 5, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * int((int((window_size - 4) / 2) - 4) / 2)**2, 120, bias=bias),
            nn.ReLU(),
            nn.Linear(120, 84, bias=bias),
            nn.ReLU(),
            nn.Linear(84, num_classes, bias=bias),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------------------
# Binary net
# --------------------------------------------------------
class StepF(torch.autograd.Function):
    ''' A step function that returns values in {-1, 1} and uses the Straigh-Through Estimator
        to update upstream weights in the network
    '''
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = torch.sign(input_).clamp(min=0) * 2 - 1  # output \in {-1, +1}
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        return grad_input


class Step(nn.Module):
    '''Module wrapper for a step function (StepF).
    '''
    def __init__(self):
        super(Step, self).__init__()

    def __repr__(self):
        s = '{name}(low=-1, high=1)'
        return s.format(name=self.__class__.__name__)

    def forward(self, x):
        return StepF.apply(x)
