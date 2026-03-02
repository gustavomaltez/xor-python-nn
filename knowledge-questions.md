# Knowledge Questions

Questions and answers from building this project.

---

**What is a weight?**

A number that controls how much influence one neuron has on the next. Like a volume knob between two neurons. Each input is multiplied by a weight before being passed forward.

---

**Why do weights start random?**

The network doesn't know the right values yet. Training adjusts them. Starting at zero would make every neuron learn the exact same thing. Random breaks that symmetry.

---

**What are biases?**

Extra numbers added to a neuron's input before it activates. They let the neuron shift its output up or down regardless of the inputs. Without them the network has less flexibility to learn. Mathematically: `y = mx + b`, where `m` is the weight and `b` is the bias.

---

**What is a neuron?**

A value that gets computed as data flows through the network. Input neurons hold the raw inputs. Hidden and output neurons hold the result of combining the previous layer's values using weights and biases.

---

**Why 5 neurons?**

2 input neurons because XOR takes 2 inputs. 1 output neuron because XOR produces 1 output. 2 hidden neurons because XOR needs at least 2 to solve it. Less than that and it can't learn the pattern.

---

**What is the flow of data through the network?**

Both inputs feed into every hidden neuron. Then every hidden neuron feeds into the output. Direction is always left to right: inputs → hidden neurons → output.

---

**What changes during training?**

Only the weights and biases. Neurons are temporary values computed each pass. Weights and biases are the network's memory. They get nudged slightly after each prediction until the network gets the right answers.

---

**How does the network know that [0, 0] → 0 and [0, 1] → 1?**

`inputs` and `expected_outputs` are always paired by index. `inputs[0]` is `[0, 0]` and `expected_outputs[0]` is `0`. During training the network takes each input, makes a prediction, compares it to the expected output at the same index, and adjusts weights to reduce the error.

---

**If I wanted to train AND instead of XOR, what would change?**

Only `expected_outputs`. Change it to `[0, 0, 0, 1]`. The inputs, architecture, and all code stays the same. The network doesn't know or care what problem it's solving. It just adjusts weights until predictions match whatever expected outputs you give it.

---

**Why is it called `weights_input_hidden` and not `weights_output_hidden`?**

The name describes the direction of the connection: source → destination. `weights_input_hidden` connects input to hidden. `weights_output_hidden` would imply data flows from output back to hidden, which is the wrong direction.

---

**Does the network retrain every time the program runs?**

Yes. Weights start random each run, the network trains from scratch, then you test it. There's no memory between runs. For XOR this is fine because training is fast. In real ML you'd save the trained weights to a file to avoid retraining every time.

---

**What is the sigmoid function and why do we need it?**

Sigmoid is the activation function. After each neuron computes its weighted sum, sigmoid squashes the result to a value between 0 and 1. Without it, stacking layers is just stacking linear math, which collapses into a single operation and can't solve non-linear problems like XOR.

---

**What happens without an activation function?**

Say you have two layers with weights `2` and `3`:

```
layer1_output = input * 2
layer2_output = layer1_output * 3
              = input * 2 * 3
              = input * 6
```

Two layers collapsed into one multiplication by `6`. No matter how many layers you add, it's always equivalent to a single layer. The hidden layer buys you nothing.

---

**What happens with sigmoid applied?**

```
layer1_output    = input * 2
layer1_activated = sigmoid(input * 2)
                 = 1 / (1 + e^(-(input * 2)))
layer2_output    = layer1_activated * 3
```

You can't simplify that into a single multiplication anymore. The `e^(-x)` inside sigmoid makes it impossible to collapse. The network can now represent curves, not just straight lines.

---

**What are the layers in this network?**

The input layer just holds the raw values you feed in. It doesn't compute anything.

Layer 1 is the hidden layer. It takes the raw inputs, multiplies by `weights_input_hidden`, adds `bias_hidden`, then applies sigmoid.

Layer 2 is the output layer. It takes the hidden layer's results, multiplies by `weights_hidden_output`, adds `bias_output`, then applies sigmoid again.
