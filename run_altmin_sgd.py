import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from altmin import get_mods, get_codes, update_codes, update_last_layer_, update_hidden_weights_adam_
from altmin import scheduler_step, post_processing_step
from models import test
from utils import get_devices, ddict, load_dataset


# Training settings
parser = argparse.ArgumentParser(description='Online Alternating-Minimization with SGD')
parser.add_argument('--model', default='feedforward', metavar='M',
                    help='name of model: `feedforward`, `binary` or `LeNet` (default: `feedforward`)')
parser.add_argument('--n-hidden-layers', type=int, default=2, metavar='L',
                    help='number of hidden layers (default: 2; ignored for LeNet)')
parser.add_argument('--n-hiddens', type=int, default=100, metavar='N',
                    help='number of hidden units (default: 100; ignored for LeNet)')
parser.add_argument('--dataset', default='mnist', metavar='D',
                    help='name of dataset')
parser.add_argument('--data-augmentation', action='store_true', default=False,
                    help='enables data augmentation')
parser.add_argument('--batch-size', type=int, default=200, metavar='B',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=50, metavar='E',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--n-iter-codes', type=int, default=5, metavar='N',
                    help='number of internal iterations for codes optimization')
parser.add_argument('--n-iter-weights', type=int, default=1, metavar='N',
                    help='number of internal iterations in learning weights')
parser.add_argument('--lr-codes', type=float, default=0.3, metavar='LR',
                    help='learning rate for codes updates')
parser.add_argument('--lr-out', type=float, default=0.008, metavar='LR',
                    help='learning rate for last layer weights updates')
parser.add_argument('--lr-weights', type=float, default=0.008, metavar='LR',
                    help='learning rate for hidden weights updates')
parser.add_argument('--lr-half-epochs', type=int, default=8, metavar='LH',
                    help='number of epochs after which learning rate if halfed')
parser.add_argument('--no-batchnorm', action='store_true', default=False,
                    help='disables batchnormalization')
parser.add_argument('--lambda_c', type=float, default=0.0, metavar='L',
                    help='codes sparsity')
parser.add_argument('--lambda_w', type=float, default=0.0, metavar='L',
                    help='weight sparsity')
parser.add_argument('--mu', type=float, default=0.003, metavar='M',
                    help='initial mu parameter')
parser.add_argument('--d-mu', type=float, default=0.0/300, metavar='M',
                    help='increase in mu after every mini-batch')
parser.add_argument('--postprocessing-steps', type=int, default=0, metavar='N',
                    help='number of Carreira-Perpinan post-processing steps after training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before saving test performance (if set to zero, it does not save)')
parser.add_argument('--log-first-epoch', action='store_true', default=False,
                    help='whether or not it should test and log after every mini-batch in first epoch')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()


# Check cuda
device, num_gpus = get_devices("cuda:0" if not args.no_cuda and torch.cuda.is_available() else "cpu", seed=args.seed)


# Load data and model
model_name = args.model.lower()
if model_name == 'feedforward' or model_name == 'binary':
    model_name += '_' + str(args.n_hidden_layers) + 'x' + str(args.n_hiddens)
file_name = 'output/save_' + os.path.basename(__file__).split('.')[0] + '_' + model_name +\
    '_' + args.dataset + '_' + str(args.seed) + '.pt'

print('\nOnline alternating-minimization with sgd')
print('* Loading dataset {}'.format(args.dataset))
print('* Loading model {}'.format(model_name))
print('     BatchNorm: {}'.format(not args.no_batchnorm))

if args.model.lower() == 'feedforward' or args.model.lower() == 'binary':
    from models import FFNet

    train_loader, test_loader, n_inputs = load_dataset(args.dataset, batch_size=args.batch_size, conv_net=False)

    model = FFNet(n_inputs, n_hiddens=args.n_hiddens, n_hidden_layers=args.n_hidden_layers,
                  batchnorm=not args.no_batchnorm, bias=False).to(device)

elif args.model.lower() == 'lenet':
    from models import LeNet

    train_loader, test_loader, n_inputs = load_dataset(args.dataset, batch_size=args.batch_size, conv_net=True,
                                                       data_augmentation=args.data_augmentation)
    if args.data_augmentation:
        print('    data augmentation')

    window_size = train_loader.dataset.data[0].shape[0]
    if len(train_loader.dataset.data[0].shape) == 3:
        num_input_channels = train_loader.dataset.data[0].shape[2]
    else:
        num_input_channels = 1

    model = LeNet(num_input_channels=num_input_channels, window_size=window_size, bias=True).to(device)

criterion = nn.CrossEntropyLoss()


if __name__ == "__main__":

    # Save everything in a `ddict`
    SAV = ddict(args=args.__dict__)

    # Store training and test performance after each training epoch
    SAV.perf = ddict(tr=[], te=[])

    # Store test performance after each iteration in first epoch
    SAV.perf.first_epoch = []

    # Store test performance after each args.save_interval iterations
    SAV.perf.te_vs_iterations = []

    # Expose model modules that has_codes
    model = get_mods(model, optimizer='Adam', optimizer_params={'lr': args.lr_weights},
                     scheduler=lambda epoch: 1/2**(epoch//args.lr_half_epochs))
    model[-1].optimizer.param_groups[0]['lr'] = args.lr_out

    if args.model.lower() == 'binary':
        from models import Step
        # Add Dropout and discretize first hidden layer (as in Diff Target propagation paper)
        model[2] = nn.Sequential(nn.Dropout(p=0.2), nn.Tanh(), Step())
        # Add Dropout before last linear layer
        model[4][0] = nn.Sequential(nn.Dropout(p=0.2), nn.Tanh())

    # Initial mu and increment after every mini-batch
    mu = args.mu
    mu_max = 10 * args.mu

    for epoch in range(1, args.epochs+1):
        print('\nEpoch {} of {}. mu = {:.4f}, lr_out = {}'.format(epoch, args.epochs, mu, model[-1].scheduler.get_lr()))

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            # (1) Forward
            model.train()
            with torch.no_grad():
                outputs, codes = get_codes(model, data)

            # (2) Update codes
            codes = update_codes(codes, model, targets, criterion, mu, lambda_c=args.lambda_c, n_iter=args.n_iter_codes, lr=args.lr_codes)

            # (3) Update weights
            update_last_layer_(model[-1], codes[-1], targets, criterion, n_iter=args.n_iter_weights)

            update_hidden_weights_adam_(model, data, codes, lambda_w=args.lambda_w, n_iter=args.n_iter_weights)

            # Store all iterations of first epoch
            if epoch == 1 and args.log_first_epoch:
                SAV.perf.first_epoch += [test(model, data_loader=test_loader, label=" - Test")]

            # Outputs to terminal
            if batch_idx % args.log_interval == 0:
                loss = criterion(outputs, targets)
                print(' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

            # After every args.save_interval iterations compute and save test error
            if args.save_interval > 0 and batch_idx % args.save_interval == 0 and batch_idx > 0:
                SAV.perf.te_vs_iterations += [test(model, data_loader=test_loader, label=" - Test")]

            # Increment mu
            if mu < mu_max:
                mu = mu + args.d_mu

        scheduler_step(model)

        # Print performances
        SAV.perf.tr += [test(model, data_loader=train_loader, label="Training")]
        SAV.perf.te += [test(model, data_loader=test_loader, label="Test")]

        # Save intermediate results
        if args.save_interval > 0:
            torch.save(SAV, file_name)

    # ----------------------------------------------------------------
    # Post-processing step from Carreira-Perpinan (fit last layer):
    # ----------------------------------------------------------------
    if args.postprocessing_steps > 0:

        print('\nPost-processing step:\n')

        for epoch in range(1, args.postprocessing_steps+1):
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                post_processing_step(model, data, targets, criterion, args.lambda_w)

                # Outputs to terminal
                if batch_idx % args.log_interval == 0:
                    print(' Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))

    # Print performances
    SAV.perf.tr_final = test(model, data_loader=train_loader, label="  Training set after post-processing")
    SAV.perf.te_final = test(model, data_loader=test_loader, label="  Test set after post-processing    ")

    # Save final results
    if args.save_interval > 0:
        torch.save(SAV, file_name)
