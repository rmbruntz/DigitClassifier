import csv
from digitclassifier.DigitClassifier import DigitClassifier

NUM_TESTS = 50


def test(neural_net: DigitClassifier):
    shown = 0
    correct = 0

    with open('data\\mnist_test.csv', newline='') as test_file:
        reader = csv.reader(test_file)
        for test_number in range(0, NUM_TESTS):
            # print("Test {}".format(test_number + 1))
            line = next(reader)
            label = int(line[0])
            data = [int(x) for x in line[1:]]
            # print("Label: {}".format(label))
            # print(data)

            #result = neural_net.classify(data)

            # print("Answered {}, answer was {}".format(result, label))
            if result == label:
                correct += 1
            shown += 1

            # print("{} shown so far, {} correct".format(shown, correct))
            # print()


#test(DigitClassifier())

DigitClassifier().classify([1, 1, 0, 0])
