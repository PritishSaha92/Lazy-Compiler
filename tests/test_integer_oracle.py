"""Zero tolerance must mean exact integer output, including all 64 bits."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_compiler import compare_outputs


class IntegerOracleTests(unittest.TestCase):
    def test_adjacent_large_integers_and_exact_failure_json(self):
        for dtype, values in ((np.int64, [2**53, 2**63-2, -2**63]),
                              (np.uint64, [2**53, 2**64-2])):
            for value in values:
                with self.subTest(dtype=dtype, value=value):
                    row = compare_outputs({'x':np.array([value+1], dtype)},
                                          {'x':np.array([value], dtype)}, 0, 0)[0]
                    self.assertFalse(row['pass'])
                    self.assertEqual(row['mismatches'], 1)
                    self.assertEqual(row['max_abs_error'], 1)
                    decoded = json.loads(json.dumps(row))['first_failure']
                    self.assertIs(type(decoded['actual']), int)
                    self.assertEqual(decoded['actual'], value+1)
                    self.assertEqual(decoded['expected'], value)

    def test_mixed_signedness_and_scalar_buffer_shape(self):
        for left, right, equal in ((2**63-1,2**63,False), (-1,2**64-1,False),
                                   (2**53+1,2**53,False), (2**53+1,2**53+1,True)):
            a, b = np.array([left], np.int64), np.array(right, np.uint64)
            for actual, expected in ((a,b),(b,a)):
                row = compare_outputs({'x':actual},{'x':expected},0,0)[0]
                self.assertEqual(row['pass'],equal)
        row = compare_outputs({'x':np.array([-2**63],np.int64)},
                              {'x':np.array([2**64-1],np.uint64)},0,0)[0]
        self.assertEqual(row['max_abs_error'],float(3*2**63-1))

    def test_integer_reference_does_not_accept_rounded_float(self):
        with self.assertRaisesRegex(ValueError, 'requires exact integer'):
            compare_outputs({'x':np.array([float(2**53)])},
                            {'x':np.array([2**53+1],np.int64)},0,0)


if __name__ == '__main__':
    unittest.main()
