# xor-python-nn

A neural network built from scratch. No frameworks, no abstractions, just Python and math.

The goal isn't to build something useful. It's to understand what's actually happening inside a neural network when it learns something.

---

## The problem: XOR

XOR (exclusive or) is a logical operation. Given two inputs, it returns `1` if exactly one of them is `1`, and `0` otherwise:

| A | B | A XOR B |
|---|---|---------|
| 0 | 0 |    0    |
| 0 | 1 |    1    |
| 1 | 0 |    1    |
| 1 | 1 |    0    |

Simple enough. But if you try to draw a straight line that separates the `1`s from the `0`s on a 2D grid, you can't. That's called a *linearly non-separable* problem, and it's exactly the kind of thing a single-layer network can't solve.

XOR is the canonical test for whether a neural network is actually learning non-linear patterns through hidden layers. If it can solve XOR, the core mechanics are working.

---

## What this project builds

- A feedforward neural network with one hidden layer
- Forward pass: computing predictions from inputs
- Backpropagation: computing how wrong we were, layer by layer
- Gradient descent: adjusting weights to be less wrong next time

No TensorFlow. No PyTorch. No scikit-learn. Not even `math` for the activation functions. Those get implemented manually too.

---

## Why

Every ML framework hides the same 50 lines of math behind abstractions. This project is about writing those 50 lines and actually knowing what they do.

---

## Notes on the code

The code is heavily commented. This is a study project, so comments explain the reasoning behind each decision, not just what the code does.

Questions that came up during the study sessions, and their answers, are in [knowledge-questions.md](knowledge-questions.md).
