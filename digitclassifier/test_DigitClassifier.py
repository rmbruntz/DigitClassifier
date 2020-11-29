import digitclassifier.DigitClassifier as DigitClassifier
import unittest


class DigitClassifierTest(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def setUp(self):
        DigitClassifier.NUM_INPUTS = 4
        DigitClassifier.HIDDEN_LAYERS = 2
        DigitClassifier.HIDDEN_LAYER_SIZE = 2
        DigitClassifier.NUM_OUTPUTS = 2

    def test_all_zeroes(self):
        dc = DigitClassifier()
        self.assertEqual(dc.biases, [])
        print(dc)


unittest.main()
