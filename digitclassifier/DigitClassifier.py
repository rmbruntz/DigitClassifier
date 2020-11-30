import numpy as np
import math


def get_cost(data, label):
    cost = 0
    for pos, value in enumerate(data):
        if pos == label:
            cost += (1 - value) ** 2
        else:
            cost += (0 - value) ** 2

    return cost


def sigmoid(x):
    result = 1 / (1 + math.e ** (-x))
    return result


def inverse_sigmoid(x):
    result = math.log(x / (1 - x), math.e)
    return result


def expand_range_to_negatives(x):
    return (x * 2) - 1


def merge_changes_into(from_array_list, to_array_list):
    new_array_list = []
    for i in range(0, len(to_array_list)):
        new_array_list.append(np.add(from_array_list[i], to_array_list[i]))
    return new_array_list


def sum_changes(array_list):
    total = 0
    for array in array_list:
        total += np.sum(np.vectorize(lambda x: x ** 2)(array))
    return total


NUM_INPUTS = 28 * 28
HIDDEN_LAYERS = 2
HIDDEN_LAYER_SIZE = 16
NUM_OUTPUTS = 10

STEP_VECTOR_SCALE = 0.3


class DigitClassifier:

    def __init__(self):
        self.num_inputs = NUM_INPUTS
        self.hidden_layers = HIDDEN_LAYERS
        self.weights = []
        self.biases = []
        self.last_total_cost = 1000

        # Initialize weights to random float (0, 1)
        input_to_hidden = np.vectorize(expand_range_to_negatives)(np.random.rand(NUM_INPUTS, HIDDEN_LAYER_SIZE))
        self.weights.append(input_to_hidden)
        for i in range(0, HIDDEN_LAYERS - 1):
            hidden_to_hidden = np.vectorize(expand_range_to_negatives)(
                np.random.rand(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE))
            self.weights.append(hidden_to_hidden)
        hidden_to_output = np.vectorize(expand_range_to_negatives)(np.random.rand(HIDDEN_LAYER_SIZE, NUM_OUTPUTS))
        self.weights.append(hidden_to_output)

        # print("Weights:\n", self.weights)

        # Initialize biases to 0
        for i in range(0, HIDDEN_LAYERS):
            hidden_biases = np.zeros(HIDDEN_LAYER_SIZE)
            self.biases.append(hidden_biases)
        output_biases = np.zeros(NUM_OUTPUTS)
        self.biases.append(output_biases)

        # print("Biases:\n", self.biases)

    def _classify_to_vector(self, data):
        activations = []

        vectorized_sigmoid = np.vectorize(sigmoid)

        current_layer_values = np.array(data)
        next_layer_values = np.matmul(current_layer_values, self.weights[0])
        # print(next_layer_values)
        next_layer_values = vectorized_sigmoid(next_layer_values)

        # print("First layer:")
        # print(next_layer_values)
        # print()

        activations.append(next_layer_values)

        for layer_pair in range(0, HIDDEN_LAYERS - 1):
            current_layer_values = next_layer_values
            next_layer_values = np.matmul(current_layer_values, self.weights[layer_pair + 1])
            next_layer_values = vectorized_sigmoid(next_layer_values)

            activations.append(next_layer_values)

        current_layer_values = next_layer_values
        next_layer_values = np.matmul(current_layer_values, self.weights[-1])
        next_layer_values = vectorized_sigmoid(next_layer_values)
        activations.append(next_layer_values)

        # print("Activations:\n", activations)

        # print("Final Answer:")
        # print()
        # print(result)

        return activations

    def classify(self, data):
        choice_vector = self._classify_to_vector(data)[-1]
        # print("Cost:", get_cost(choice_vector, 7))
        result = np.argmax(choice_vector)

        return result

    def train(self, labeled_data_sets):

        weight_changes = [np.zeros(mat.shape) for mat in self.weights]
        bias_changes = [np.zeros(arr.shape) for arr in self.biases]

        correct = 0
        total_cost = 0

        for label, data in labeled_data_sets:

            activations = self._classify_to_vector(data)
            choice_vector = activations[-1]
            total_cost += get_cost(choice_vector, label)
            result = np.argmax(choice_vector)
            # print("Choosing {} with correct answer {}, choice vector:".format(result, label), choice_vector)
            # print("cost:", get_cost(choice_vector, label))

            if result == label:
                # print("Correct!")
                correct += 1

            if self.last_total_cost > 25 or result != label:
                requested_bias_changes, requested_weight_changes = self.get_adjustments(data, activations, label)
                weight_changes = merge_changes_into(requested_weight_changes, weight_changes)

                bias_changes = merge_changes_into(requested_bias_changes, bias_changes)

        total_change_vector_length = math.sqrt(sum_changes(weight_changes) + sum_changes(bias_changes))

        print("TOTAL CHANGE VECTOR LENGTH:", total_change_vector_length)

        # print("BIAS CHANGES:", bias_changes)

        # Normalize the vector length, then multiply by step size
        # And of course, don't forget to negate
        weight_changes = [np.divide(layer_weight_changes, - total_change_vector_length / (STEP_VECTOR_SCALE)) for
                          layer_weight_changes in weight_changes]

        bias_changes = [np.divide(layer_bias_changes, - total_change_vector_length / (STEP_VECTOR_SCALE)) for
                        layer_bias_changes in bias_changes]

        self.weights = merge_changes_into(weight_changes, self.weights)
        self.biases = merge_changes_into(bias_changes, self.biases)

        # print("WEIGHT CHANGES:", weight_changes)

        print("total cost: ", total_cost)
        print("biases: ", self.biases)

        self.last_total_cost = total_cost

        return correct

    def get_adjustments(self, dataset, activations, correct):

        # print(dataset)

        weight_changes = [np.zeros(mat.shape) for mat in self.weights]
        bias_changes = [np.zeros(arr.shape) for arr in self.biases]

        derivatives = []
        for layer in activations:
            derivatives.append(np.zeros(layer.shape[0]))

        activations = [np.array(dataset)] + activations

        # set the last set of derivatives to cost function * derivative of sigmoid function, since that will be the
        # derivative multiplied by each weight, bias, and further layer

        for digit in range(0, len(derivatives[-1])):
            digit_activation = activations[-1][digit]
            # derivative of cost function for a digit: 2 * (result - correct)
            derivatives[-1][digit] = 2 * (digit_activation - 1) if digit == correct else 2 * digit_activation

            # print(digit == correct, derivatives[-1][digit])
            # derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            # sigmoid(x) is just the activation
            derivatives[-1][digit] *= digit_activation * (1 - digit_activation)

        # Iterate backward over the activations
        for target_layer in range(len(activations) - 1, 0, -1):

            # go through each node on the current target layer and repeat this process
            for node in range(0, len(activations[target_layer])):
                # bias derivative is equal to the derivative of the next layer (ie cost) times sigmoid derivative
                bias_changes[target_layer - 1][node] = derivatives[target_layer - 1][node]

                # now iterate over previous nodes, request a weight change, and set the new derivative for the next
                # iteration (if it's changeable)
                for prev_node in range(0, len(activations[target_layer - 1])):

                    weight_changes[target_layer - 1][prev_node][node] = \
                        derivatives[target_layer - 1][node] * activations[target_layer - 1][prev_node]

                    # No reason to compute derivatives for inputs (target_layer = 1) - can't change the inputs
                    if target_layer >= 2:
                        # print(derivatives[target_layer - 1][node])
                        # print(activations[target_layer - 1][prev_node])
                        # We can apply this at the same time, since it's always coupled with the derivative farther
                        # up ( cost, last layer's combined derivatives, etc.) And since they're added,
                        # we can multiply it each time

                        last_layer_sigmoid_derivative = \
                            activations[target_layer - 1][prev_node] * (1 - activations[target_layer - 1][prev_node])
                        # Update next layer of derivatives
                        derivatives[target_layer - 2][prev_node] += \
                            self.weights[target_layer - 1][prev_node][node] * derivatives[target_layer - 1][node] * \
                            last_layer_sigmoid_derivative
                    else:
                        pass
                        # print("row 1:", repr(weight_changes[0]))

            # At this point the weight and bias request should be set, and the total derivative (all paths) should
            # be added up to use in the next iteration
        # print("Derivatives:\n", derivatives)
        # print("Activations:\n", activations, "\n\n\n")
        # print("Bias Changes:\n", bias_changes[-1])
        # print("Weight Changes:\n", weight_changes[0])

        # print("REQ BIAS CHANGES:", bias_changes)

        return bias_changes, weight_changes
