import torch
from torchvision import datasets, transforms


def load_dataset(namedataset='mnist', batch_size=200, data_augmentation=False, conv_net=False, num_workers=1):
    '''data_augmentation: use data augmentation, if it is available for dataset
       conv_net: set to `True` if the dataset is being used with a conv net (i.e. the inputs have to be 3d tensors and not flattened)
    '''

    # Load mnist dataset
    if namedataset == 'mnist':

        DIR_DATASET = '~/data'

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]

        if not conv_net:
            transform_list.append(transforms.Lambda(lambda x: x.view(x.size(1) * x.size(2))))

        transform = transforms.Compose(transform_list)

        trainset = datasets.MNIST(DIR_DATASET, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = datasets.MNIST(DIR_DATASET, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=True, num_workers=num_workers)

        classes = tuple(range(10))
        n_inputs = 784

    # Load mnist_tf dataset (mnist with tensorflow validation split, i.e. remove
    # first 5000 samples from training set for validation split)
    elif namedataset == 'mnist_tf':

        from .mnist_tf import MNIST_TF
        DIR_DATASET = '~/data/mnist'

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(x.size(1) * x.size(2))),
        ])

        trainset = MNIST_TF(DIR_DATASET, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = MNIST_TF(DIR_DATASET, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=True, num_workers=num_workers)

        classes = tuple(range(10))
        n_inputs = 784

    # Load cifar10 (same preprocessing as https://github.com/kuangliu/pytorch-cifar)
    elif namedataset == 'cifar10':

        DIR_DATASET = '~/data/cifar10'

        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

        if not conv_net:
            transform_list.append(
                transforms.Lambda(lambda x: x.view(x.size(0) * x.size(1) * x.size(2))))

        transform_test = transforms.Compose(transform_list)

        if data_augmentation:
            transform_train_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

            if not conv_net:
                transform_train_list.append(
                    transforms.Lambda(lambda x: x.view(x.size(0) * x.size(1) * x.size(2))))

            transform_train = transforms.Compose(transform_train_list)

        else:
            transform_train = transform_test

        trainset = datasets.CIFAR10(DIR_DATASET, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = datasets.CIFAR10(DIR_DATASET, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=num_workers)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        n_inputs = 3 * 32 * 32

    # Load Higgs (first 10'000 samples of dataset are test, the rest are training)
    elif namedataset == 'higgs':

        from .higgs import HIGGS_LOADER
        DIR_DATASET = '~/data/higgs'

        train_loader = HIGGS_LOADER(DIR_DATASET, train=True, download=True, batch_size=batch_size)
        test_loader = HIGGS_LOADER(DIR_DATASET, train=False, download=True, batch_size=200)
        n_inputs = train_loader.n_inputs

    else:
        raise ValueError('Dataset {} not recognized'.format(namedataset))

    return train_loader, test_loader, n_inputs
