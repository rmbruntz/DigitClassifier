import numpy as np
import random

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

        print(self.weights)

        # Initialize biases to 0
        for i in range(0, HIDDEN_LAYERS):
            hidden_biases = np.zeros(HIDDEN_LAYER_SIZE)
            self.biases.append(hidden_to_hidden)
        output_biases = np.zeros(NUM_OUTPUTS)
        self.biases.append(output_biases)

        print(self.biases)

    def classify(self, data):
        input_vector = np.array(data)
        current_layer_values = np.matmul(input_vector, self.weights[0])
        print("First layer:")
        print()
        print(current_layer_values)

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