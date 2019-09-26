from __future__ import print_function
import torch.utils.data as data
import sys
import os
import os.path
import errno
import torch
import codecs
from torchvision.datasets.utils import download_url
import itertools
import pandas as pd


class HIGGS_LOADER(object):
    """HIGGS dataset
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'

    n_test = 500000     # (as in paper use last 500k samples for test)
    test_file = 'test.csv'
    n_train = 11000000-n_test
    chunk_size = 100000 # chuck_size must be a divisor of n_test and n_train
    n_inputs = 28
    classes = [0.0, 1.0]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, batch_size=100):
        self.root = os.path.expanduser(root)
        self.filename = os.path.basename(self.url)
        self.filepath = os.path.join(self.root, self.filename)
        self.testfilepath = os.path.join(self.root, self.test_file)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.batch_size = batch_size

        self.columns = ['y'] + list(range(self.n_inputs))

        if download:
            self.download()

        if not (self._check_exists_raw() and self._check_exists_test()):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        # Quick and dirty, so that len(dataloader.dataset) works
        class dl:
            def __init__(self, n):
                self.n = n
            def __len__(self):
                return self.n
        if self.train:
            self.dataset = dl(self.n_train)
        else:
            self.dataset = dl(self.n_test)

    def __len__(self):
        if self.train:
            return int(self.n_train/self.batch_size)
        else:
            return int(self.n_test/self.batch_size)

    def _check_exists_raw(self):
        return os.path.exists(self.filepath)

    def _check_exists_test(self):
        # Check test set and raw data for training
        return os.path.exists(os.path.join(self.root, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        if not self._check_exists_raw():

            # download files
            try:
                os.makedirs(self.root)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    pass
                else:
                    raise
            download_url(self.url, root=self.root, md5=None)
            print('Done!')

        # Make the test set if file does not exist
        if not self._check_exists_test():
            print('\nCreating HIGGS test dataset (this will take some time):')
            iter_start = 0
            iter_end = int((self.n_train+self.n_test)/self.chunk_size)
            test_start = int(self.n_train/self.chunk_size)

            test_data = pd.DataFrame()
            for i, data_chunk in enumerate(itertools.islice(
                pd.read_csv(self.filepath, compression='gzip', sep=',', chunksize=self.chunk_size, names=self.columns),
                iter_start, iter_end)):

                sys.stdout.write("  Processes %d of %d data chunks (%0.2f%%)\r" %
                    (i+1, iter_end, (i+1)*100/iter_end))
                if i >= iter_end-1:
                    sys.stdout.write('\n')

                if i >= test_start:
                    test_data = pd.concat([test_data, data_chunk])

            print('Writing test file'+self.testfilepath)
            test_data.to_csv(self.testfilepath)
            print('Done!')

    def __iter__(self):
        if self.train:
            iter_start = 0
            iter_end = int(self.n_train/self.chunk_size)
            csv_file = self.filepath
        else:
            iter_start = 0
            iter_end = int(self.n_test/self.chunk_size)
            csv_file = self.testfilepath

        self.data_iter = itertools.islice(
            pd.read_csv(self.filepath, compression='gzip', sep=',', chunksize=self.chunk_size, names=self.columns),
            iter_start, iter_end)

        return _DataLoaderIter(self)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class _DataLoaderIter(object):
    r"""Iterates once over the DataLoader's dataset"""

    def __init__(self, loader):
        self.data_iter = loader.data_iter
        self.batch_size = loader.batch_size
        self.n_inputs = loader.n_inputs
        self.n_batches = len(loader)

        self.data_chunk = next(self.data_iter)

    def __next__(self):
        if len(self.data_chunk) < self.batch_size:
            try:
                new_data_chunk = next(self.data_iter)
                self.data_chunk = pd.concat([self.data_chunk, new_data_chunk])
            except StopIteration:
                raise StopIteration

        while True:
            inputs = torch.from_numpy(
                self.data_chunk.head(self.batch_size)[list(range(self.n_inputs))].values).float()
            targets  = torch.from_numpy(
                self.data_chunk.head(self.batch_size)['y'].values).long()

            self.data_chunk = self.data_chunk.iloc[self.batch_size:]

            return inputs, targets

    next = __next__ # Python 2 compatibility

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        return self
