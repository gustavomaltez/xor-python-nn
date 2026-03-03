import random
import math

# Training data (XOR Table)

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected_outputs = [0, 1, 1, 0]

# Neurons (5 total)
# -----------------
# Neurons are the values that get computed as the data flows through the network
#
# For this network we'll have:
# 2 input neurons (because XOR takes 2 inputs)
# 1 output neuron (because XOR produces 1 output)
# 2 hidden neurons (because XOR needs at least 2 hidden neurons to solve it
# otherwise it can't learn the pattern)
# inputs -> hidden neurons -> output

# Weights and biases
# ------------------
# The data that will change during the training to achieve the desired output
#
# -> Weights are numbers that controls how much influence one neuron has on the next
# Like a volume knob between two neurons
# -> Biases are extra numbers added to a neuron's input before it activates, it
# allows the neuron shift its output up or down regardless of the input. Without it
# the network won't be flexible enough to learn.
#
# Mathematically, we could say that y = mx + b, where:
# y -> output
# m -> weight
# b -> bias

weights_input_hidden = [
    [random.uniform(-1, 1), random.uniform(-1, 1)],
    [random.uniform(-1, 1), random.uniform(-1, 1)],
]

weights_hidden_output = [random.uniform(-1, 1), random.uniform(-1, 1)]

bias_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
bias_output = random.uniform(-1, 1)


# Sigmoid function
# ----------------
# It is basically the activation function, after each neuron computes its weighted
# sum, this function will squash the result between 0 and 1. Without it, the
# stacking layers would just stack linear math (which can't solve XOR)
#
# Examples
# -> Considering two layers with weights 2 and 3
#
# (without sigmoid)
# layer1_output = input * 2
# layer2_output = layer1_output * 3
#                = input * 2 * 3
#                = input * 6
#
# (with sigmoid)
# layer1_output = input * 2
# layer1_activated = sigmoid(input * 2)
#                   = 1 / (1 + e^(-(input * 2)))
# layer2_output = layer1_activated * 3
#
# In summary, after each layer, we call the sigmoid function to "activate" it,
# so the input of the next layer, instead of linear math, will always be a value
# between 0 and 1
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Forward pass
# ------------
# This function takes one input and runs it through the network to get a prediction
# 1. For each hidden neuron we:
# - Multiply each input by its weight
# - Sum them upp
# - Add the bias
# - Apply sigmoid
# 2. For the output neuron we (using the hidden layer's output):
# - Multiply each input by its weight
# - Sum them upp
# - Add the bias
# - Apply sigmoid
# 3. Return the final value
def forward(inputs):
    hidden = [0, 0]

    # Step 1
    for i in range(len(hidden)):
        input_a_times_weight = inputs[0] * weights_input_hidden[0][i]
        input_b_times_weight = inputs[1] * weights_input_hidden[1][i]
        hidden[i] = sigmoid(
            input_a_times_weight + input_b_times_weight + bias_hidden[i]
        )

    # Step 2
    hidden_a_times_weight = hidden[0] * weights_hidden_output[0]
    hidden_b_times_weight = hidden[1] * weights_hidden_output[1]
    output = sigmoid(hidden_a_times_weight + hidden_b_times_weight + bias_output)

    # Step 3
    return output
