# Day 12: A Neural-Network Layer Is a Matrix Multiply

**Phase 1 · Week 2 · ~120 minutes**

## Objectives

- [ ] Go from one neuron, \( z = \mathbf{w} \cdot \mathbf{x} + b \), to a whole layer, \( \mathbf{z} = W\mathbf{x} + \mathbf{b} \), and explain why row \( i \) of \( W \) is neuron \( i \).
- [ ] Run a layer on a whole batch with \( Z = XW^\top + \mathbf{b} \) and predict every shape involved.
- [ ] Apply ReLU and write a complete 2-layer forward pass in NumPy.
- [ ] Count a layer's parameters, its memory in `float32` and `int8`, and its FLOPs.
- [ ] Recognize the same layer as PyTorch's `nn.Linear` and LiteRT's `FULLY_CONNECTED` op.
- [ ] Read the three most common shape errors and fix them.

**Prerequisites.** [Day 5](day-05-vectors.md) (a neuron is \( \mathbf{w} \cdot \mathbf{x} + b \)), [Day 9](day-09-broadcasting.md) (broadcasting) and [Day 10](day-10-matrices-matmul.md) (matrix multiplication).

---

## The big idea

On Day 5 you met one neuron: a dot product plus a number. A **layer** is just many neurons that all read the *same* input. Stack their weight vectors on top of each other and you have a matrix; run all of them at once and you have a matrix multiply. Stack several layers, with a simple bend between them, and you have a **neural network**.

That's the whole secret. The "network" in a phone's image classifier is, to a first approximation, a chain of `@` operations, each followed by an addition and a bend.

**Analogy.** Picture a panel of judges at a talent show. Every judge watches the same performance (the input). Each judge has their own scoring sheet: "how much do I care about singing, dancing, stage presence" (their weights) and their own mood offset (their bias). Each produces one score. Collect the panel's scoring sheets as rows of a table and you have \( W \). Collect their scores and you have \( \mathbf{z} \). A second panel then scores the first panel's scores: that's layer 2.

## From one neuron to many

Recap from Day 5. One neuron has a weight vector \( \mathbf{w} \) and a bias \( b \):

\[
z = \mathbf{w} \cdot \mathbf{x} + b = w_1 x_1 + w_2 x_2 + w_3 x_3 + b
\]

With \( \mathbf{x} = [2, 1, 3] \), \( \mathbf{w} = [0.5, -0.2, 0.1] \) and \( b = 0.4 \): \( z = 1 - 0.2 + 0.3 + 0.4 = 1.5 \).

Now take **4 neurons**, all reading the same 3 inputs. Each has its own weight vector of length 3 and its own bias. Write the weight vectors as the **rows** of a matrix, and the biases as a vector:

\[
W = \begin{bmatrix} 0.5 & -0.2 & 0.1 \\ 0 & 1 & -1 \\ 0.3 & 0.3 & 0.3 \\ -0.4 & 0.2 & 0.6 \end{bmatrix}
\qquad
\mathbf{b} = \begin{bmatrix} 0.4 \\ 0 \\ -1 \\ 0.1 \end{bmatrix}
\]

Row 1 of \( W \) is exactly the \( \mathbf{w} \) above; rows 2–4 belong to the other three neurons. New symbols:

- \( W \) is the **weight matrix**. Its shape is **(out, in)**: one row per neuron (4), one column per input feature (3).
- \( \mathbf{b} \) is the **bias vector**: one number per neuron (4).
- \( \mathbf{z} \) is the layer's output: one number per neuron.

From Day 10, a matrix times a vector is one dot product per row. So all four neurons fire at once:

\[
\mathbf{z} = W\mathbf{x} + \mathbf{b}
\qquad
z_i = (\text{row } i \text{ of } W) \cdot \mathbf{x} + b_i
\]

<figure class="diagram" markdown>
![Animation: a 4 by 3 weight matrix W times the vector x = [2, 1, 3] plus bias b equals z = [1.5, -2, 0.8, 1.3]; each row of W is highlighted in turn together with its bias, its output and its neuron, while the full arithmetic for that neuron is shown below](../../assets/img/phase-1/day12-rows-are-neurons.svg)
<figcaption>Animated: row i of W, dotted with x, plus b_i, is neuron i's output. One W @ x + b does all four.</figcaption>
</figure>

```python
import numpy as np

x = np.array([2.0, 1.0, 3.0])               # one input with 3 features

W = np.array([[ 0.5, -0.2,  0.1],           # neuron 1's weights (Day 5's w)
              [ 0.0,  1.0, -1.0],           # neuron 2
              [ 0.3,  0.3,  0.3],           # neuron 3
              [-0.4,  0.2,  0.6]])          # neuron 4
b = np.array([0.4, 0.0, -1.0, 0.1])         # one bias per neuron

# The slow way: one dot product per neuron
z_loop = np.array([W[i] @ x + b[i] for i in range(4)])
print(z_loop)                 # [ 1.5 -2.   0.8  1.3]

# The layer way: one matrix-vector product
z = W @ x + b
print(z)                      # [ 1.5 -2.   0.8  1.3]
print(W.shape, x.shape, z.shape)   # (4, 3) (3,) (4,)
```

Shape rule: \( (\text{out}, \text{in}) \) @ \( (\text{in},) \) gives \( (\text{out},) \). Three numbers in, four numbers out. A layer *changes the size* of the vector, which is how a network goes from 150,528 pixel values down to 1,000 class scores.

## A batch: many inputs at once

In practice you never push one example through at a time. You push a **batch**: a matrix \( X \) with one example per row, shape **(batch, in)**. That's the same "rows are samples, columns are features" layout as a spreadsheet or a pandas DataFrame.

We want every row of \( X \) to go through the layer. Each output entry is "example \( n \) dotted with neuron \( j \)'s weights". In Day 10 terms, that's row \( n \) of the left matrix times column \( j \) of the right matrix, so the right matrix needs neurons as *columns*. That's \( W^\top \) (W transposed, Day 11), shape (in, out):

\[
Z = X W^\top + \mathbf{b}
\qquad
\underbrace{(\text{batch}, \text{in})}_{X} \; @ \; \underbrace{(\text{in}, \text{out})}_{W^\top} \;\to\; \underbrace{(\text{batch}, \text{out})}_{Z}
\]

The bias \( \mathbf{b} \) has shape (out,). Adding it to a (batch, out) matrix uses Day 9's broadcasting: shapes align from the right, so \( \mathbf{b} \) is added to **every row**. Every example gets the same biases.

<figure class="diagram" markdown>
![Animation: a 5 by 3 batch X times the 3 by 4 matrix W transpose, plus a bias row of 4, equals a 5 by 4 output Z; the rows of X light up one at a time together with the matching row of Z](../../assets/img/phase-1/day12-batch-matmul.svg)
<figcaption>Animated: example n (row n of X) produces row n of Z. All examples share the same W and b.</figcaption>
</figure>

```python
X = np.array([[2.0,  1.0, 3.0],             # example 1 (same x as before)
              [1.0, -1.0, 2.0]])            # example 2
Z = X @ W.T + b
print(X.shape, W.T.shape, Z.shape)   # (2, 3) (3, 4) (2, 4)
print(Z)
# [[ 1.5 -2.   0.8  1.3]
#  [ 1.3 -3.  -0.4  0.7]]
```

Row 1 of `Z` is exactly the `z` from before. Same weights, more data, one call.

!!! note "This is exactly PyTorch's convention"
    PyTorch's `nn.Linear` stores its weight with shape **(out_features, in_features)** and computes \( y = xW^\top + b \). Learn the shapes here and PyTorch code reads naturally. (Some books and Keras store the transpose, shape (in, out), and write \( XW + b \). Same math, different storage. Always check which one you're looking at.)

## ReLU: the bend between layers

If you stack two layers with nothing between them, you gain nothing: \( W_2(W_1\mathbf{x}) = (W_2 W_1)\mathbf{x} \), which is just *one* layer with matrix \( W_2W_1 \). (You'll prove this numerically in Exercise 6.) Stacking matrix multiplies only ever gives another matrix multiply: straight lines in, straight lines out.

To make depth useful, we apply a simple non-linear function to every number between layers. The most common is **ReLU** (Rectified Linear Unit):

\[
\text{ReLU}(z) = \max(0, z)
\]

Negative numbers become 0; positive numbers pass through unchanged. It's applied **element-wise**: each number on its own, shape unchanged. The output of a layer after ReLU is called its **activations**, written \( \mathbf{h} \) (for "hidden").

```python
def relu(z):
    return np.maximum(0, z)   # element-wise, keeps the shape

H = relu(Z)
print(H)
# [[1.5 0.  0.8 1.3]
#  [1.3 0.  0.  0.7]]
```

Intuition: each neuron is a pattern detector (Day 5). ReLU says "if your pattern isn't present, stay silent". That on/off behavior, combined across many neurons, is what lets a network carve up input space into complicated regions.

## A full 2-layer forward pass

Let's build a tiny classifier: 4 input features → 8 hidden neurons → 3 output scores (one per class). Running the input through all the layers is called the **forward pass**.

<figure class="diagram" markdown>
![Animation of a small network with 3 inputs, 4 hidden neurons and 2 outputs: the inputs light up, pulses travel along the 12 edges into the hidden layer, the hidden neurons light up, pulses travel along the 8 edges to the outputs, and the outputs light up; labels under the layers read W1: 4×3 and W2: 2×4](../../assets/img/phase-1/day12-network-forward.svg)
<figcaption>Animated: activation flows left to right. Every bundle of edges between two layers is one weight matrix of shape (out, in).</figcaption>
</figure>

The diagram is smaller than our code, but count the edges: 3 inputs × 4 hidden = 12 edges, and \( W_1 \) has 4 × 3 = 12 entries. **Each edge is one weight.** The math:

\[
\mathbf{h} = \text{ReLU}(W_1\mathbf{x} + \mathbf{b}_1) \qquad \mathbf{z} = W_2\mathbf{h} + \mathbf{b}_2
\]

Batched, for the code:

```python
import numpy as np

rng = np.random.default_rng(12)

def relu(z):
    return np.maximum(0, z)

# Layer sizes: 4 inputs -> 8 hidden -> 3 outputs
n_in, n_hidden, n_out = 4, 8, 3

# Parameters, PyTorch convention: W has shape (out, in)
W1 = rng.normal(0, 0.5, size=(n_hidden, n_in)).astype(np.float32)
b1 = np.zeros(n_hidden, dtype=np.float32)
W2 = rng.normal(0, 0.5, size=(n_out, n_hidden)).astype(np.float32)
b2 = np.zeros(n_out, dtype=np.float32)

# A batch of 5 examples, 4 features each
X = rng.normal(0, 1, size=(5, n_in)).astype(np.float32)
print("X  ", X.shape)       # X   (5, 4)

Z1 = X @ W1.T + b1
print("Z1 ", Z1.shape)      # Z1  (5, 8)
H = relu(Z1)
print("H  ", H.shape)       # H   (5, 8)
Z2 = H @ W2.T + b2
print("Z2 ", Z2.shape)      # Z2  (5, 3)

print(np.round(Z2, 3))
# [[-0.484  1.35   0.429]
#  [ 0.178 -0.778  0.759]
#  [-0.079 -0.317  0.771]
#  [ 0.581 -1.153  1.703]
#  [-0.239  0.463 -0.065]]
print("predicted class per example:", Z2.argmax(axis=1))
# predicted class per example: [1 2 2 2 1]
print("fraction of hidden values zeroed by ReLU:", (H == 0).mean())
# fraction of hidden values zeroed by ReLU: 0.575
```

Read the shape trail: (5, 4) → (5, 8) → (5, 8) → (5, 3). The batch size never changes; only the feature dimension does. The final `Z2` holds 3 raw scores per example (called **logits**), and `argmax` picks the biggest as the prediction.

!!! warning "These predictions mean nothing (yet)"
    The weights came from a random number generator, so the "predictions" are noise. **Training** (Month 2 for the idea with linear models, Month 4 for neural networks) is the process of nudging every number in `W1`, `b1`, `W2`, `b2` so that the scores come out right on real data. The forward pass you just wrote never changes. Only the numbers inside the matrices do.

## Counting parameters and memory

A layer with `in` inputs and `out` neurons has `in × out` weights plus `out` biases:

\[
\text{params} = \text{in} \cdot \text{out} + \text{out}
\]

```python
def count_params(sizes):
    total = 0
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        p = n_in * n_out + n_out
        print(f"{n_in:>4} -> {n_out:<4} weights {n_in*n_out:>7}  biases {n_out:>4}  total {p:>7}")
        total += p
    return total

count_params([4, 8, 3])
#    4 -> 8    weights      32  biases    8  total      40
#    8 -> 3    weights      24  biases    3  total      27      -> 67 parameters

total = count_params([784, 128, 10])    # a classic digit classifier (28×28 images)
#  784 -> 128  weights  100352  biases  128  total  100480
#  128 -> 10   weights    1280  biases   10  total    1290
print("float32 bytes:", total * 4, " int8 bytes:", total * 1)
# float32 bytes: 407080  int8 bytes: 101770
```

Tie back to Day 4's dtype table: each `float32` weight is 4 bytes, each `int8` weight is 1 byte. You can check with `.nbytes`:

```python
W = np.zeros((128, 784), dtype=np.float32)
print(W.nbytes, W.astype(np.int8).nbytes)   # 401408 100352
```

That 4× shrink is **quantization**, and it's why on-device models are usually shipped as `int8`. Notice where the parameters live: the first layer, which touches the big input, holds 99% of them.

## FLOPs: how much work a layer is

Each output number is a dot product of length `in`: `in` multiplications and about `in` additions. There are `batch × out` output numbers. So:

\[
\text{FLOPs} \approx 2 \cdot \text{batch} \cdot \text{in} \cdot \text{out}
\]

(A **FLOP** is one floating-point operation. One multiply plus one add is often counted as one **multiply-add**, or MAC, which is 2 FLOPs.) This is Day 10's cost \( m \cdot n \cdot p \) multiply-adds with \( m = \text{batch} \), \( n = \text{in} \), \( p = \text{out} \).

```python
def flops(batch, sizes):
    return sum(2 * batch * i * o for i, o in zip(sizes[:-1], sizes[1:]))

print(flops(1, [784, 128, 10]))    # 203264      ~0.2 MFLOPs per image
print(flops(32, [784, 128, 10]))   # 6504448     ~6.5 MFLOPs for a batch of 32
```

**A real phone-sized model for scale (approximate figures, from the MobileNetV2 paper):** MobileNetV2 at 224×224 has about **3.4 million parameters** and needs about **300 million multiply-adds per image** (roughly 600 million FLOPs). Most of that work is convolutions (Month 5), which are matrix multiplies in disguise; its final classifier is equivalent to a 1280 → 1000 fully connected layer. Run it at 30 frames per second and you need about 18 billion FLOPs every second, on a battery. That's why phones have GPUs and NPUs, and why model size and FLOPs come up in every on-device AI discussion.

## Preview: the same layer in PyTorch and LiteRT

PyTorch isn't installed yet (you'll set it up in Month 4), so here's the code without output. Read it for the shapes:

```python
import torch

layer = torch.nn.Linear(4, 8)       # in_features=4, out_features=8
print(layer.weight.shape)           # torch.Size([8, 4])   -> (out, in), like our W1
print(layer.bias.shape)             # torch.Size([8])

x = torch.randn(5, 4)               # a batch of 5 examples
y = layer(x)                        # computes x @ layer.weight.T + layer.bias
print(y.shape)                      # torch.Size([5, 8])
```

The shape comments follow from PyTorch's documented convention, not from a run on this machine. A whole network is these layers in sequence: `torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))` is our NumPy forward pass, line for line.

When that model is converted to a `.tflite` file for LiteRT, each `Linear` shows up as a **`FULLY_CONNECTED`** operator (open a `.tflite` file in the free Netron viewer and you'll see the boxes), and ReLU usually gets fused into it as its activation function.

!!! android "Android bridge"
    When your app calls `interpreter.run(input, output)` on an image classifier, this is what happens under the hood. The input `ByteBuffer` (shape `[1, 224, 224, 3]`, NHWC) is fed to the first operator; then LiteRT executes a chain of operators like `CONV_2D`, `DEPTHWISE_CONV_2D` and `FULLY_CONNECTED`, each one essentially a matrix multiply plus bias plus activation, on the CPU, or on the GPU/NPU if you attached a delegate. The `output` you passed in, say an `Array(1) { FloatArray(1000) }`, is filled with the **last layer's \( Z \)**: batch of 1, 1,000 class scores. Your `maxBy` over that `FloatArray` is our `argmax`. (Many exported classifiers include a final softmax, so the scores may already be probabilities; a quantized model gives you `uint8`/`int8` values you de-quantize. Either way, it's the last layer's output.)

## Shape debugging

Almost every bug you'll hit while writing layers is a shape bug. Here are the three most common, with the exact messages NumPy 2.x prints. Assume `X` is `(5, 4)`, `W1` is `(8, 4)`, `b1` is `(8,)`.

**1. Forgetting the transpose.**

```python
X @ W1
# ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 8 is different from 4)
```

Read it as: "the inner sizes `k` must match, but the left has 4 columns and the right has 8 rows". Fix: `X @ W1.T`. You get exactly the same message when the *input* has the wrong number of features (e.g. 3 instead of 4), which in a real app usually means the preprocessing differs from training.

**2. A bias of the wrong shape.**

```python
X @ W1.T + np.zeros(4)        # bias sized to the INPUT, not the output
# ValueError: operands could not be broadcast together with shapes (5,8) (4,)
```

The bias has one entry per **output** neuron, so it must be `(out,)` = `(8,)`.

**3. A single example passed as a 1-D vector.**

```python
x = np.ones(4)                # one example, shape (4,), not (1, 4)
(x @ W1.T).argmax(axis=1)
# numpy.exceptions.AxisError: axis 1 is out of bounds for array of dimension 1
```

`x @ W1.T` quietly works and returns shape `(8,)`, so the error only shows up later, in code that expects a batch axis. Fix: make a batch of one with `x[np.newaxis, :]` or `x.reshape(1, -1)` (Day 8), giving `(1, 4)` → `(1, 8)`. This is the NumPy version of LiteRT insisting on `[1, 224, 224, 3]` rather than `[224, 224, 3]`.

!!! warning "Common mistake: the bug that doesn't crash"
    A bias stored as a column, shape `(8, 1)`, crashes with a batch of 5: `operands could not be broadcast together with shapes (5,8) (8,1)`. But with a batch of exactly 8, `(8, 8) + (8, 1)` broadcasts happily and adds the *wrong* bias to every row. The shape is right, the numbers are wrong, and nothing tells you. Keep biases 1-D, `(out,)`, and print shapes while you build.

!!! warning "Common mistake: (in, out) vs (out, in)"
    If you copy weights between frameworks (Keras stores `(in, out)`, PyTorch and LiteRT's `FULLY_CONNECTED` use `(out, in)`) and a square layer like 128 → 128 is involved, a missing transpose won't raise any error. The model just gives garbage. When numbers look wrong but shapes look fine, suspect a transpose.

---

## Exercises

1. A digit classifier has layers 784 → 256 → 128 → 10. Count the parameters per layer and in total. How many KB is it in `float32` and in `int8`?
2. By hand, then in NumPy: \( W = \begin{bmatrix} 1 & 2 \\ 0 & -1 \\ 3 & 1 \end{bmatrix} \), \( \mathbf{x} = [1, 2] \), \( \mathbf{b} = [0, 1, -2] \). Compute \( \mathbf{z} = W\mathbf{x} + \mathbf{b} \) and \( \text{ReLU}(\mathbf{z}) \). How many neurons does this layer have?
3. Write `dense(X, W, b)` returning `X @ W.T + b`. Check it against a double `for` loop over examples and neurons using `np.allclose`, for `X` of shape (6, 5) and `W` of shape (2, 5).
4. How many FLOPs does a forward pass of the 784 → 128 → 10 network take for a batch of 32? And how many FLOPs does a 1280 → 1000 classifier layer (like MobileNetV2's last layer) take for one image?
5. You have `X` of shape (10, 6) and `W` of shape (4, 6). What error does `X @ W` give, how do you fix it, and what is the output shape? What shape must `b` be?
6. Prove the "no ReLU, no point" claim numerically: with random `W1` (8, 4), `W2` (3, 8) and `x` (4,), show that `W2 @ (W1 @ x)` equals `(W2 @ W1) @ x`. What's the shape of `W2 @ W1`? Then show it's no longer true once you put `relu` in the middle.

??? success "Solutions"
    ```python
    import numpy as np

    # 1
    sizes = [784, 256, 128, 10]
    per_layer = [i * o + o for i, o in zip(sizes[:-1], sizes[1:])]
    print(per_layer, sum(per_layer))   # [200960, 32896, 1290] 235146
    total = sum(per_layer)
    print(total * 4 / 1024, total * 1 / 1024)   # 918.5390625 229.634765625
    # about 919 KB in float32, about 230 KB in int8

    # 2   z1 = 1*1 + 2*2 + 0 = 5;  z2 = 0*1 - 1*2 + 1 = -1;  z3 = 3*1 + 1*2 - 2 = 3
    W = np.array([[1, 2], [0, -1], [3, 1]]); x = np.array([1, 2]); b = np.array([0, 1, -2])
    z = W @ x + b
    print(z, np.maximum(0, z))   # [ 5 -1  3] [5 0 3]
    # 3 neurons (3 rows of W), each reading 2 inputs

    # 3
    def dense(X, W, b):
        return X @ W.T + b

    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 5)); W = rng.normal(size=(2, 5)); b = rng.normal(size=2)
    Z = dense(X, W, b)
    Zloop = np.zeros((6, 2))
    for n in range(6):
        for j in range(2):
            Zloop[n, j] = W[j] @ X[n] + b[j]
    print(Z.shape, np.allclose(Z, Zloop))   # (6, 2) True

    # 4
    print(2 * 32 * 784 * 128 + 2 * 32 * 128 * 10)   # 6504448   ~6.5 MFLOPs
    print(2 * 1 * 1280 * 1000)                       # 2560000   ~2.6 MFLOPs
    # Small next to MobileNetV2's ~600 MFLOPs total: the convolutions do most of the work.

    # 5   X @ W -> ValueError: matmul: ... (size 4 is different from 6)
    X = np.zeros((10, 6)); W = np.zeros((4, 6)); b = np.zeros(4)
    print((X @ W.T + b).shape)   # (10, 4)    b must be (4,): one bias per neuron

    # 6
    rng = np.random.default_rng(6)
    W1 = rng.normal(size=(8, 4)); W2 = rng.normal(size=(3, 8)); x = rng.normal(size=4)
    W_combined = W2 @ W1
    print(W_combined.shape, np.allclose(W2 @ (W1 @ x), W_combined @ x))   # (3, 4) True
    relu = lambda z: np.maximum(0, z)
    print(np.allclose(W2 @ relu(W1 @ x), W_combined @ x))                 # False
    ```
    Exercise 6 is the reason every network has activation functions: without them, a 100-layer network collapses into one (out, in) matrix.

## Resources

- [3Blue1Brown: "But what is a neural network?"](https://www.youtube.com/watch?v=aircAruvnKk): the best visual explanation of layers, weights and biases, including the "weights as a matrix" moment. 19 minutes.
- [Michael Nielsen: Neural Networks and Deep Learning, chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html): a free book that builds exactly this forward pass, then trains it.
- [CS231n: Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/): Stanford's notes on neurons, layers, ReLU and parameter counting.
- [Dive into Deep Learning: Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/mlp.html): the same material with runnable code, and a preview of training.
- [PyTorch docs: torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html): the official shapes and formula \( y = xA^\top + b \).
- [MobileNetV2 paper (arXiv)](https://arxiv.org/abs/1801.04381): source of the parameter and multiply-add figures; skim the tables, not the whole paper.
- [LiteRT overview (Google AI Edge)](https://ai.google.dev/edge/litert): where `FULLY_CONNECTED` and the other operators run on Android.

## Checklist

- [ ] I can explain why row \( i \) of \( W \) is neuron \( i \)'s weight vector.
- [ ] I can write \( \mathbf{z} = W\mathbf{x} + \mathbf{b} \) and \( Z = XW^\top + \mathbf{b} \) and give every shape.
- [ ] I know why ReLU (or some non-linearity) is needed between layers.
- [ ] I wrote and ran a 2-layer forward pass in NumPy and followed its shape trail.
- [ ] I can count a layer's parameters, bytes and FLOPs.
- [ ] I can map `nn.Linear(in, out)` and LiteRT's `FULLY_CONNECTED` to the same math.
- [ ] I can read the three shape errors above and know the fix for each.

**Next:** [Day 13: Week 2 review and mini-project](day-13-week2-review.md)
