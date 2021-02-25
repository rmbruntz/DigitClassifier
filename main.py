import csv
import pickle
import atexit
import random
from digitclassifier.DigitClassifier import DigitClassifier
import graphics as Graphics

NUM_TESTS = 10000

TRAINING_BATCH_SIZE = 100


def save_network():
    with open('data/neural_net.json', mode='wb') as net_state_file:
        pickle.dump(dc, net_state_file)
    print("\n\n\nNetwork config saved\n\n\n")


def test(neural_net: DigitClassifier):
    shown = 0
    correct = 0

    graphics = Graphics.GraphWin("Digit Classifier", 500, 508)

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

            if test_number % 1000 == 0:
                print("{} tests done".format(test_number))

            # print("{} shown so far, {} correct".format(shown, correct))
            # print()

    print("{} / {}, {}%".format(correct, shown, correct/shown * 100))


def train(neural_net: DigitClassifier):
    first_pass = True
    while True:
        try:
            with open('data\\mnist_train.csv', newline='') as test_file:
                reader = csv.reader(test_file)

                # Start at a random point on the first pass
                if first_pass:
                    for n in range(0, 100 * random.randrange(0, 500) + 50):
                        next(reader)
                    first_pass = False

                a = 0
                while True:
                    data_sets = []
                    for i in range(0, TRAINING_BATCH_SIZE):
                        # for n in range(0, random.randint(0, 3)):
                        #     next(reader)
                        line = next(reader)
                        label = int(line[0])
                        data = [int(x) / 256 for x in line[1:]]

                        # print("\n\n\n{}\n\n\n".format(len(data)))

                        # print(data, flush=True)

                        data_sets.append((label, data))

                    num_correct = neural_net.train(data_sets)
                    print("{} correct out of {}".format(num_correct, TRAINING_BATCH_SIZE))
                    a += 1
        except StopIteration:
            pass


# test(DigitClassifier())
dc = None
try:
    with open('data/neural_net.json', mode='rb') as net_state_file:
        dc = pickle.load(net_state_file)
except IOError:
    dc = DigitClassifier()

test(dc)

# atexit.register(save_network)
# train(dc)
