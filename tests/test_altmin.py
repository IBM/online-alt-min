import unittest
import numpy as np
import torch
import torch.nn.functional as F


class test_altmin(unittest.TestCase):

    def test_bcd(self):
        # Pseudo-inverse solution: w_ps = B.mm(A.inverse())
        from altmin import BCD
        c = torch.randn(10,5)
        a = F.relu(c)
        A = a.t().mm(a)
        B = c.t().mm(a)
        A0, B0 = A.clone(), B.clone()
        w, errors = BCD(torch.zeros(A.shape[1], A.shape[1]), A, B, 0.0, return_errors=True)
        error = (w.mm(A) - B)/A.diag()

        # Checked that data was passed by copy
        self.assertTrue((A-A0).abs().max().item() == 0)
        self.assertTrue((B-B0).abs().max().item() == 0)
        # Check that error is small
        self.assertTrue(error.abs().mean().item() < 1e-2)


class test_conv(unittest.TestCase):

    def test_get_mods(self):
        from models import LeNet
        from altmin import get_mods

        model = LeNet()
        model.eval()
        x = torch.randn(20,3,32,32)
        outputs = model(x)

        model_mods = get_mods(model)

        self.assertTrue(len(model.features) + len(model.classifier) >= len(model_mods))

    def test_get_codes(self):
        from models import LeNet
        from altmin import get_mods, get_codes

        model = LeNet()
        model.eval()
        x = torch.randn(20,3,32,32)
        outputs = model(x)

        model_mods = get_mods(model)
        out1, codes = get_codes(model_mods, x)
        out2 = model_mods(x)

        self.assertAlmostEqual((outputs - out1).abs().mean().item(), 0)
        self.assertAlmostEqual((out1 - out2).abs().mean().item(), 0)


if __name__ == '__main__':
    unittest.main()
