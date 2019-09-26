import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from models import test
from utils import get_devices, ddict, load_dataset


# Training settings
parser = argparse.ArgumentParser(description='SGD baseline')
parser.add_argument('--model', default='feedforward', metavar='M',
                    help='name of model: `feedforward` or `LeNet` (default: `feedforward`)')
parser.add_argument('--n-hidden-layers', type=int, default=2, metavar='L',
                    help='number of hidden layers (default: 2; ignored for LeNet)')
parser.add_argument('--n-hiddens', type=int, default=100, metavar='N',
                    help='number of hidden units (default: 100)')
parser.add_argument('--dataset', default='mnist', metavar='D',
                    help='name of dataset: `mnist`, `cifar10`, `higgs` (default: `mnist`)')
parser.add_argument('--data-augmentation', action='store_true', default=False,
                    help='enables data augmentation')
parser.add_argument('--batch-size', type=int, default=200, metavar='B',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                    help='learning rate (default: 0.003)')
parser.add_argument('--lr-decay', type=float, default=1.0, metavar='LD',
                    help='learning rate decay factor per epoch (default: 1.0)')
parser.add_argument('--no-batchnorm', action='store_true', default=False,
                    help='disables batchnormalization (default: False)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before saving test performance (if set to zero, it does not save)')
parser.add_argument('--log-first-epoch', action='store_true', default=False,
                    help='whether or not it should test and log after every mini-batch in first epoch')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA training')
args = parser.parse_args()


# Check cuda
device, num_gpus = get_devices("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu", seed=args.seed)
num_workers = num_gpus if num_gpus>0 else 1


# Load data and model
model_name = args.model.lower()
if model_name == 'feedforward' or model_name == 'binary':
    model_name += '_' + str(args.n_hidden_layers) + 'x' + str(args.n_hiddens)
file_name = 'output/save_' + os.path.basename(__file__).split('.')[0] + '_' + model_name +\
    '_' + args.dataset + '_' + str(args.seed) + '.pt'

print('\nSGD baseline')
print('* Loading dataset {}'.format(args.dataset))
print('* Loading model {}'.format(model_name))
print('     BatchNorm: {}'.format(not args.no_batchnorm))

if args.model.lower() == 'feedforward':
    from models import FFNet

    train_loader, test_loader, n_inputs = load_dataset(args.dataset, batch_size=args.batch_size,
                                                       conv_net=False, num_workers=num_workers)

    model = FFNet(n_inputs, n_hiddens=args.n_hiddens, n_hidden_layers=args.n_hidden_layers,
                  batchnorm=not args.no_batchnorm, bias=True).to(device)

elif args.model.lower() == 'lenet':
    from models import LeNet

    train_loader, test_loader, n_inputs = load_dataset(args.dataset, batch_size=args.batch_size, conv_net=True,
                                                       data_augmentation=args.data_augmentation, num_workers=num_workers)
    if args.data_augmentation:
        print('    data augmentation')

    window_size = train_loader.dataset.data[0].shape[0]
    if len(train_loader.dataset.data[0].shape) == 3:
        num_input_channels = train_loader.dataset.data[0].shape[2]
    else:
        num_input_channels = 1

    model = LeNet(num_input_channels=num_input_channels, window_size=window_size, bias=True).to(device)


# Multi-GPU
if num_workers>1:
    model = nn.DataParallel(model)
criterion = nn.CrossEntropyLoss()


if __name__ == "__main__":

    # Save everything in a `ddict`
    SAV = ddict(args=args.__dict__)

    # Store training and test performance after each training epoch
    SAV.perf = ddict(tr=[], te=[])

    # Store test performance after each iteration in first epoch
    SAV.perf.first_epoch = []

    # Store test performance after each iteration
    SAV.perf.te_vs_iterations = []

    # Model trained with sgd
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

    for epoch in range(1, args.epochs+1):
        print('\nEpoch {} of {}'.format(epoch, args.epochs))
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Store all iterations of first epoch
            if epoch == 1 and args.log_first_epoch:
                SAV.perf.first_epoch += [test(model, data_loader=test_loader, label=" - Test")]

            # Outputs to terminal
            if batch_idx % args.log_interval == 0:
                print(' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

            # After every args.save_interval iterations, evaluate and save full test error
            if args.save_interval > 0 and batch_idx % args.save_interval == 0  and batch_idx > 0:
                SAV.perf.te_vs_iterations += [test(model, data_loader=test_loader, label=" - Test")]

        scheduler.step()

        # Print and save performances for all epochs
        SAV.perf.tr += [test(model, data_loader=train_loader, label="Training")]
        SAV.perf.te += [test(model, data_loader=test_loader, label="Test")]

        # Save intermediate results
        if args.save_interval > 0:
            torch.save(SAV, file_name)

    print('\n- Training performance after each epoch: {}'.format(SAV.perf.tr))
    print('- Test performance after each epoch: {}'.format(SAV.perf.te))
