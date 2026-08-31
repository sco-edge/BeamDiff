import ast
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1]/'scripts/train_beamsense_har1.py'
tree = ast.parse(path.read_text())
node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='nested_augmentation_subset')
namespace={'np':np}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
select=namespace['nested_augmentation_subset']


class NestedSelectionTests(unittest.TestCase):
    def test_prefixes_are_nested_and_repeatable(self):
        values=np.arange(1000,dtype=np.int64)
        a=select(values,.1,111)
        b=select(values,.25,111)
        c=select(values,.5,111)
        self.assertEqual((len(a),len(b),len(c)),(100,250,500))
        np.testing.assert_array_equal(a,b[:100])
        np.testing.assert_array_equal(b,c[:250])
        np.testing.assert_array_equal(a,select(values,.1,111))
        self.assertFalse(np.array_equal(a,select(values,.1,42)))

    def test_zero_and_full(self):
        values=np.arange(7)
        self.assertEqual(len(select(values,0,1)),0)
        full=select(values,1,1)
        np.testing.assert_array_equal(np.sort(full),values)
        with self.assertRaises(ValueError):
            select(values,1.1,1)


if __name__=='__main__':
    unittest.main()
