import digitclassifier.DigitClassifier as DigitClassifier
import unittest
import numpy as np


class DigitClassifierTest(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def setUp(self):
        DigitClassifier.NUM_INPUTS = 4
        DigitClassifier.HIDDEN_LAYERS = 2
        DigitClassifier.HIDDEN_LAYER_SIZE = 2
        DigitClassifier.NUM_OUTPUTS = 2

    def test_all_zeroes(self):
        dc = DigitClassifier.DigitClassifier()
        result = dc.classify([0, 0, 0, 0])
        self.assertNotEqual(dc.biases, [])

    def test_sigmoid_and_inverse(self):
        x = 4.521
        self.assertAlmostEqual(x, DigitClassifier.inverse_sigmoid(DigitClassifier.sigmoid(x)))

    def test_sum_changes(self):
        self.assertEqual(23, DigitClassifier.sum_changes([np.array([1, 2, 3]), np.array([3])]))
        self.assertEqual(23, DigitClassifier.sum_changes([np.array([1, 2, -3]), np.array([3])]))

    def test_merge_changes_into(self):
        self.assertEqual(DigitClassifier.merge_changes_into([np.array([1, 2])], [np.array([3, 4])])[0][0], 4)
        self.assertEqual(DigitClassifier.merge_changes_into([np.array([1, 2])], [np.array([3, 4])])[0][1], 6)
