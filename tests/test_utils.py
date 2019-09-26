import unittest
import numpy as np

class test_utils(unittest.TestCase):

    def test_import(self):
        import utils
        from utils import ddict

    def test_ddict(self):
        from utils import ddict

        # Test basics
        dd = ddict(a=1.0, b=[1,2])
        self.assertTrue(dd.a == 1.0)
        self.assertTrue(dd.b == [1,2])
        self.assertTrue(dd.a == dd['a'])
        self.assertTrue(dd.b == dd['b'])

        dd.c = 'c'
        self.assertTrue(dd.c == 'c')
        dd['c'] = 'efg'
        self.assertTrue(dd.c == 'efg')

        dd_sum = dd + ddict(**{'a':10, 'd': 4.0})
        self.assertTrue(dd.a == 1.0)
        self.assertTrue(dd_sum.a == [1.0, 10])
        self.assertTrue(dd_sum.d == 4.0)

        dd_sum_with_dict = dd + {'a':10, 'd': 4.0}
        self.assertTrue(dd.a == 1.0)
        self.assertTrue(dd_sum.a == [1.0, 10])
        self.assertTrue(dd_sum.d == 4.0)

        # Test saving to disk
        dd._save('/tmp/tmp', date=True)
        dd_load = ddict()._load(dd._filename)

        for k in dd._keys():
            self.assertTrue(dd_load[k] == dd[k])

        for k,v in dd._items():
            self.assertTrue(dd_load[k] == v)

        self.assertTrue(ddict(a=1, b=2)._to_dict() == {'a':1, 'b':2})

        # Flatten
        dd = ddict(a=1, c=ddict(a=2, b=ddict(x=5, y=10)), d=[1, 2, 3])
        self.assertTrue(dd._flatten() == {'a': 1, 'c_a': 2, 'c_b_x': 5, 'c_b_y': 10, 'd': [1, 2, 3]})


if __name__ == '__main__':
    unittest.main()
