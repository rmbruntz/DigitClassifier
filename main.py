import csv
from digitclassifier.DigitClassifier import DigitClassifier

NUM_TESTS = 50

TRAINING_BATCH_SIZE = 100


def test(neural_net: DigitClassifier):
    shown = 0
    correct = 0

    with open('data\\mnist_test.csv', newline='') as test_file:
        reader = csv.reader(test_file)
        for test_number in range(0, NUM_TESTS):
            # print("Test {}".format(test_number + 1))
            line = next(reader)
            label = int(line[0])
            data = [int(x) / 256 for x in line[1:]]

            # print("Label: {}".format(label))
            # print(data)

            result = neural_net.classify(data)

            # print("Answered {}, answer was {}".format(result, label))
            if result == label:
                correct += 1
            shown += 1

            # print("{} shown so far, {} correct".format(shown, correct))
            # print()


def train(neural_net: DigitClassifier):
    while True:
        try:
            with open('data\\mnist_train.csv', newline='') as test_file:
                reader = csv.reader(test_file)
                while True:
                    data_sets = []
                    for i in range(0, TRAINING_BATCH_SIZE):
                        line = next(reader)
                        label = int(line[0])
                        data = [int(x) / 256 for x in line[1:]]

                        # print("\n\n\n{}\n\n\n".format(len(data)))

                        # print(data, flush=True)

                        data_sets.append((label, data))

                    num_correct = neural_net.train(data_sets)
                    print("{} correct out of {}".format(num_correct, TRAINING_BATCH_SIZE))
        except StopIteration:
            pass


# test(DigitClassifier())
dc = DigitClassifier()
train(dc)
