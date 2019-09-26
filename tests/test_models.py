import unittest
import torch

class test_models(unittest.TestCase):

    def test_lenet(self):
        from models import LeNet

        n_outputs = 10
        model = LeNet(num_classes=n_outputs)
        model.eval()
        x = torch.randn(20,3,32,32)
        outputs = model(x)

        self.assertTrue(outputs.shape[0] == x.shape[0])
        self.assertTrue(outputs.shape[1] == n_outputs)


if __name__ == '__main__':
    unittest.main()
