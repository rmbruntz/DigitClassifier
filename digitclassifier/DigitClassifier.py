import numpy as np
import math
from unittest import TestCase


def get_cost(data, label):
    cost = 0
    for pos, value in enumerate(data):
        if pos == label:
            cost += (1 - value)**2
        else:
            cost += (0 - value)**2

    return cost

def sigmoid(x):
    return 1 / (1 + math.e**(-x))

NUM_INPUTS = 4
HIDDEN_LAYERS = 2
HIDDEN_LAYER_SIZE = 2
NUM_OUTPUTS = 10

class DigitClassifier:

    def __init__(self):
        self.num_inputs = NUM_INPUTS
        self.hidden_layers = HIDDEN_LAYERS
        self.weights = []
        self.biases = []

        # Initialize weights to random float (0, 1)
        input_to_hidden = np.random.rand(NUM_INPUTS, HIDDEN_LAYER_SIZE)
        self.weights.append(input_to_hidden)
        for i in range(0, HIDDEN_LAYERS - 1):
            hidden_to_hidden = np.random.rand(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE)
            self.weights.append(hidden_to_hidden)
        hidden_to_output = np.random.rand(HIDDEN_LAYER_SIZE, NUM_OUTPUTS)
        self.weights.append(hidden_to_output)

        print("Weights:\n", self.weights)

        # Initialize biases to 0
        for i in range(0, HIDDEN_LAYERS):
            hidden_biases = np.zeros(HIDDEN_LAYER_SIZE)
            self.biases.append(hidden_biases)
        output_biases = np.zeros(NUM_OUTPUTS)
        self.biases.append(output_biases)

        print("Biases:\n", self.biases)

    def _classify_to_vector(self, data):
        activations = []
        current_layer_values = np.array(data)
        next_layer_values = np.matmul(current_layer_values, self.weights[0])
        print("First layer:")
        print(current_layer_values)
        print()
        activations.append(next_layer_values)

        for layer_pair in range(0, HIDDEN_LAYER_SIZE - 1):
            current_layer_values = next_layer_values
            next_layer_values = np.matmul(current_layer_values, self.weights[layer_pair + 1])
            activations.append(next_layer_values)

        current_layer_values = next_layer_values
        next_layer_values = np.matmul(current_layer_values, self.weights[-1])
        activations.append(next_layer_values)


        print("Final Answer:")
        print()

        result = next_layer_values
        print(result)
        return result

    def classify(self, data):
        choice_vector = self._classify_to_vector(data)
        print("Cost:", get_cost(choice_vector, 7))
        result = np.argmax(choice_vector)

        return result;

    def train(self, labeled_data_sets):
        weights = []



        biases = []



        change_vector = (weights, biases)
        for labeled_data_set in labeled_data_sets:
            label = labeled_data_set[0]
            data = labeled_data_set[1:]
            result = self.classify(data)
            self.backpropagate(result, label, data)

    def backpropagate(self, result, label, data):
        pass




