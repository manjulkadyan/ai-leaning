# Day 33: Gradients of Vectors and Matrices, Checked Numerically

**Phase 1 · Week 5 · ~120 minutes**

## Objectives

- [ ] Write the gradient of a function of a whole vector of parameters, and state the shape rule: the gradient has the parameter's shape.
- [ ] Derive \( \nabla_{\mathbf{w}}(\mathbf{w} \cdot \mathbf{x}) = \mathbf{x} \) and \( \nabla_{\mathbf{w}}\|\mathbf{w}\|^2 = 2\mathbf{w} \) by writing out the entries.
- [ ] Show that the gradient of a loss with respect to a weight matrix \( W \) in \( \mathbf{z} = W\mathbf{x} \) is the outer product \( \boldsymbol{\delta}\,\mathbf{x}^\top \).
- [ ] Derive the linear-regression MSE gradient \( \frac{2}{n} X^\top (X\mathbf{w} - \mathbf{y}) \) step by step and check every shape.
- [ ] Write a reusable `grad_check(f, w)` that compares an analytic gradient with central differences using the relative error, and read its result.
- [ ] Say in one paragraph what a Jacobian is.
- [ ] Compute and check a real gradient on the Play Store data.

**Prerequisites.** [Day 5](day-05-vectors.md) (dot product, length), [Day 10](day-10-matrices-matmul.md) (matrix products, matrix-vector as a weighted sum of columns), [Day 12](day-12-layer-is-matmul.md) (a layer is \( W\mathbf{x} + \mathbf{b} \), batches as \( XW^\top + \mathbf{b} \)), [Day 25](day-25-projections-pca.md) (least squares, the normal equation) and this week so far: [Day 29](day-29-derivatives.md) (derivatives, central differences), [Day 30](day-30-derivative-rules.md) (derivative rules), [Day 31](day-31-chain-rule.md) (the chain rule, computational graphs) and [Day 32](day-32-partial-derivatives-gradient.md) (partial derivatives, the gradient).

---

## The big idea

On Day 32 the gradient had two entries, one per parameter of a line. A real model has thousands or millions of parameters, stored in vectors and matrices. Today you learn to take the gradient of a loss with respect to a **whole vector or matrix at once**, and to check the answer with a few lines of code.

**Analogy.** Think of a sound engineer's mixing desk with one slider per parameter. Your loss is "how bad the mix sounds". The gradient is a sticky note on **every** slider saying: "push this one up a hair, and the badness changes by this much". One note per slider, so if the sliders are laid out as a 2×3 block, the notes form a 2×3 block too. That is the whole **shape rule**: *the gradient has the same shape as the thing you differentiate with respect to*.

Two habits make matrix calculus safe for someone who has never done it:

| Habit | What it buys you |
|---|---|
| **Write it out with indices.** Turn \( \mathbf{w} \cdot \mathbf{x} \) into \( w_1x_1 + w_2x_2 + \dots \), take the derivative of one entry, then pack the pattern back into vector form. | You never need to memorize a formula you can't rederive in two lines. |
| **Check it numerically.** Nudge every entry and measure, the Day 29 way, then compare. | You find the forgotten factor of 2 before it costs you a day of "why won't this train?". |

## The gradient with respect to a vector

Collect the parameters into one vector \( \mathbf{w} = [w_1, w_2, \dots, w_d] \), where \( d \) is how many there are. For a function \( f(\mathbf{w}) \) that returns one number, the **gradient** is the vector of its partial derivatives (Day 32), one per entry:

\[
\nabla_{\mathbf{w}} f = \left[ \frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \dots, \frac{\partial f}{\partial w_d} \right]
\]

- \( \nabla \) ("nabla" or "del") means "gradient of". The subscript \( \mathbf{w} \) says what we differentiate with respect to; it is dropped when obvious.
- \( \frac{\partial f}{\partial w_j} \) is the partial derivative: nudge \( w_j \) alone, keep every other entry frozen, and measure how fast \( f \) changes.
- \( \nabla_{\mathbf{w}} f \) has \( d \) entries, exactly like \( \mathbf{w} \). That is the shape rule.

**Worked example.** \( f(\mathbf{w}) = 3w_1 + 5w_2 - 2w_3 + w_1 w_2 \) at \( \mathbf{w} = [1, 2, 3] \). Freeze all but one entry at a time:

- \( \partial f / \partial w_1 = 3 + w_2 = 5 \) (the \( w_1 w_2 \) term acts like "\( w_1 \) times the constant \( w_2 \)").
- \( \partial f / \partial w_2 = 5 + w_1 = 6 \).
- \( \partial f / \partial w_3 = -2 \).

So \( \nabla f = [5, 6, -2] \): three entries for three parameters.

<figure class="diagram" markdown>
![Animation: a column of three blue boxes holding w = 1, 2, 3; one box at a time is highlighted in yellow, and a middle panel shows f nudged up and down by h = 0.1 for that entry (9.5 and 8.5, then 9.6 and 8.4, then 8.8 and 9.2) and the resulting slope 5, 6, -2, which fills the matching box of a teal gradient column; at the end the panel says 3 entries, 3 nudges, 3 slopes, gradient = [5, 6, -2], and that a 2 by 3 matrix would take 6 nudges](../../assets/img/phase-1/day33-nudge-each-entry.svg)
<figcaption>Animated: nudge one entry, measure the slope, write it in the same slot. The gradient comes out the same shape as w.</figcaption>
</figure>

The animation is also an algorithm: Day 29's central difference, applied to one entry at a time. Here it is as a function that works for a parameter of **any** shape, vector or matrix. You'll use it for the rest of the course to check gradients.

```python
import numpy as np

def f(w):
    return 3 * w[0] + 5 * w[1] - 2 * w[2] + w[0] * w[1]

w = np.array([1.0, 2.0, 3.0])
f(w)                     # 9.0
grad_by_hand = np.array([3 + w[1], 5 + w[0], -2.0])
grad_by_hand             # array([ 5.,  6., -2.])
```

```python
def numerical_grad(f, w, h=1e-5):
    """Central-difference gradient of a scalar function f at w. Works for any shape of w."""
    w = np.array(w, dtype=float)          # a float64 copy we are free to poke
    grad = np.zeros_like(w)               # the shape rule: same shape as w
    for idx in np.ndindex(w.shape):       # (0,), (1,), ... or (i, j) for a matrix
        old = w[idx]
        w[idx] = old + h
        f_plus = f(w)
        w[idx] = old - h
        f_minus = f(w)
        w[idx] = old                      # put it back before the next entry
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad

g_num = numerical_grad(f, w)
g_num                                   # array([ 5.,  6., -2.])
np.abs(g_num - grad_by_hand).max()      # 7.57156559672012e-11   (rounding noise, Day 29)
```

`np.ndindex(shape)` yields every index of an array of that shape: `(0,), (1,), (2,)` for a vector, `(0, 0), (0, 1), ...` for a matrix. That one line is what makes the function shape-agnostic. Each entry costs two calls to `f`, so a model with a million parameters would need two million calls: fine for testing a small case, hopeless for training. Training uses the formulas below (and on Day 34, backpropagation computes them for you).

## Three rules you'll use constantly

### Rule 1: the gradient of a dot product is the other vector

\[
\nabla_{\mathbf{w}} (\mathbf{w} \cdot \mathbf{x}) = \mathbf{x}
\]

**Derivation, with indices.** Write the dot product out: \( \mathbf{w} \cdot \mathbf{x} = w_1x_1 + w_2x_2 + \dots + w_dx_d \). Differentiate with respect to one entry \( w_j \): every term without \( w_j \) is a frozen constant and vanishes, and the term \( w_jx_j \) has slope \( x_j \) (Day 30's constant-multiple rule). So \( \partial/\partial w_j = x_j \) for every \( j \). Pack those back up and you get the vector \( \mathbf{x} \).

It is the vector version of "the derivative of \( a\,w \) is \( a \)". With \( \mathbf{x} = [2, -1, 0.5] \):

```python
x = np.array([2.0, -1.0, 0.5])

def dot_x(w):
    return w @ x

dot_x(w)                   # 1.5   (2 - 2 + 1.5)
numerical_grad(dot_x, w)   # array([ 2. , -1. ,  0.5])   = x, wherever w is
```

**Where this shows up in ML.** A neuron's pre-activation is \( z = \mathbf{w} \cdot \mathbf{x} + b \) (Day 12). By rule 1, \( \partial z / \partial \mathbf{w} = \mathbf{x} \): **each weight's gradient is proportional to the input it sees.** A feature that is always 0 gives its weight a zero gradient, so that weight never learns. A feature measured in big units (megabytes, milliseconds) gets a huge gradient, and its weight gets yanked around. That is the calculus reason behind Day 22's "standardize your features"; you'll see it on real data at the end of the lesson.

### Rule 2: the gradient of the squared length is twice the vector

\[
\nabla_{\mathbf{w}} \|\mathbf{w}\|^2 = 2\mathbf{w}
\]

**Derivation.** \( \|\mathbf{w}\|^2 = \mathbf{w} \cdot \mathbf{w} = w_1^2 + w_2^2 + \dots + w_d^2 \) (Day 22: squared L2 norm). The derivative of \( w_j^2 \) with respect to \( w_j \) is \( 2w_j \) (Day 30's power rule), and all other terms are frozen. So the gradient is \( [2w_1, \dots, 2w_d] = 2\mathbf{w} \).

At \( \mathbf{w} = [1, 2, 3] \): \( \|\mathbf{w}\|^2 = 14 \) and the gradient is \( [2, 4, 6] \).

```python
def sq_norm(w):
    return w @ w               # ||w||^2

sq_norm(w)                     # 14.0
numerical_grad(sq_norm, w)     # array([2., 4., 6.])
2 * w                          # array([2., 4., 6.])
```

Geometrically, \( \|\mathbf{w}\|^2 \) is a round bowl with its bottom at the origin, and \( 2\mathbf{w} \) points straight away from the origin: uphill, as Day 32 promised a gradient always does. Walking downhill means moving towards the origin.

**Where this shows up in ML.** **L2 regularization** (previewed on Day 22) adds \( \lambda\|\mathbf{w}\|^2 \) to the loss to discourage big weights, where \( \lambda \) ("lambda") is a small number you choose. By rule 2 it adds \( 2\lambda\mathbf{w} \) to the gradient. A gradient-descent step with learning rate \( \eta \) ("eta") then becomes \( \mathbf{w} \leftarrow \mathbf{w} - \eta(\nabla L + 2\lambda\mathbf{w}) = (1 - 2\eta\lambda)\,\mathbf{w} - \eta\nabla L \): every step first shrinks the weights by a fixed fraction. That is why it is also called **weight decay**.

### The scalar rules and their vector twins

| One variable (Days 29–30) | Many variables (today) | Shape of the gradient |
|---|---|---|
| \( \frac{d}{dw}(a\,w) = a \) | \( \nabla_{\mathbf{w}}(\mathbf{a} \cdot \mathbf{w}) = \mathbf{a} \) | like \( \mathbf{w} \) |
| \( \frac{d}{dw}(w^2) = 2w \) | \( \nabla_{\mathbf{w}}\|\mathbf{w}\|^2 = 2\mathbf{w} \) | like \( \mathbf{w} \) |
| \( \frac{d}{dw}(a\,w\,x) = a\,x \) | \( \nabla_W\big(\boldsymbol{\delta} \cdot (W\mathbf{x})\big) = \boldsymbol{\delta}\,\mathbf{x}^\top \) | like \( W \) |
| \( \frac{d}{dw}\,\tfrac{1}{n}\sum(x_i w - y_i)^2 = \tfrac{2}{n}\sum x_i(x_i w - y_i) \) | \( \nabla_{\mathbf{w}}\,\tfrac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2 = \tfrac{2}{n}X^\top(X\mathbf{w} - \mathbf{y}) \) | like \( \mathbf{w} \) |

The last two rows are the next two sections.

!!! warning "Common mistake: memorizing formulas instead of the method"
    Matrix-calculus tables online disagree about whether a gradient is a row or a column (they use different "layout conventions"), so a formula copied from one source can come out transposed in your code. Don't depend on them. Write the scalar expression for one entry, differentiate it, pack it back up, and check the shape rule. Then check the numbers with `numerical_grad`. In NumPy, a 1-D array has no row/column distinction anyway; the shape rule is the convention.

## Gradients with respect to a matrix

A layer computes \( \mathbf{z} = W\mathbf{x} \) with \( W \) of shape (out, in) (Day 12). Training needs \( \partial L / \partial W \): one number per weight, so by the shape rule a matrix the shape of \( W \).

Strictly speaking, the derivative of the *vector* \( W\mathbf{x} \) with respect to the *matrix* \( W \) is a 3-D block of numbers (every output against every weight), and it is mostly zeros. You never need that block, because training always ends in **one number**, the loss \( L \). Then everything collapses to something simple.

**Derivation, with indices.** Output \( i \) is row \( i \) of \( W \) dotted with \( \mathbf{x} \):

\[
z_i = W_{i1}x_1 + W_{i2}x_2 + \dots + W_{in}x_n
\]

The weight \( W_{ij} \) appears in exactly one output, \( z_i \), and with slope \( x_j \) (rule 1 again). Let \( \boldsymbol{\delta} = \partial L / \partial \mathbf{z} \) ("delta", a vector with one entry per output) be the gradient that comes back from the loss. By the chain rule (Day 31), a nudge to \( W_{ij} \) moves \( z_i \) by \( x_j \) per unit, and each unit of \( z_i \) moves \( L \) by \( \delta_i \), so:

\[
\frac{\partial L}{\partial W_{ij}} = \delta_i\, x_j
\qquad\Longrightarrow\qquad
\frac{\partial L}{\partial W} = \boldsymbol{\delta}\,\mathbf{x}^\top
\]

\( \boldsymbol{\delta}\,\mathbf{x}^\top \) is the **outer product** from Day 23: a column times a row. Shapes: (out, 1) times (1, in) gives (out, in), the shape of \( W \). The shape rule holds.

**Worked example.** \( W = \begin{bmatrix} 0.5 & -1 & 2 \\ 1 & 0 & -0.5 \end{bmatrix} \), \( \mathbf{x} = [2, -1, 0.5] \), target \( \mathbf{t} = [2, 2] \) and loss \( L = \tfrac{1}{2}\|\mathbf{z} - \mathbf{t}\|^2 \) (the \( \tfrac{1}{2} \) cancels the 2 from rule 2, a common convenience).

- Forward: \( \mathbf{z} = W\mathbf{x} = [1 + 1 + 1,\; 2 + 0 - 0.25] = [3, 1.75] \), and \( L = \tfrac{1}{2}(1^2 + 0.25^2) = 0.53125 \).
- From the loss: \( \boldsymbol{\delta} = \mathbf{z} - \mathbf{t} = [1, -0.25] \) (rule 2 with the \( \tfrac{1}{2} \)).
- The weight gradient: every cell is "its row's \( \delta \)" times "its column's \( x \)".

<figure class="diagram" markdown>
![Animation: the input x = [2, -1, 0.5] runs along the top in blue and delta = [1, -0.25] runs down the side in orange; one cell at a time of a 2 by 3 teal grid is outlined in red while its row's delta and column's x are highlighted, and the cell fills with their product: 2, -1, 0.5 in the first row and -0.5, 0.25, -0.125 in the second; a side panel explains row i is how much the loss wants output i to change and column j is how big input j was, then shows (2, 1) times (1, 3) gives (2, 3) and np.outer(delta, x)](../../assets/img/phase-1/day33-outer-product.svg)
<figcaption>Animated: each weight's gradient is "error at its output" times "signal on its input", which is exactly an outer product.</figcaption>
</figure>

```python
W = np.array([[0.5, -1.0,  2.0],
              [1.0,  0.0, -0.5]])     # shape (2, 3): 2 outputs, 3 inputs
t = np.array([2.0, 2.0])              # the target output

def layer_loss(W):
    z = W @ x
    return 0.5 * np.sum((z - t) ** 2)

z = W @ x
z                                     # array([3.  , 1.75])
layer_loss(W)                         # 0.53125
delta = z - t                         # dL/dz
delta                                 # array([ 1.  , -0.25])
dL_dW = np.outer(delta, x)            # delta x^T, shape (2, 3) like W
dL_dW
# [[ 2.     -1.      0.5  ]
#  [-0.5     0.25   -0.125]]
numerical_grad(layer_loss, W)         # the same matrix, from 6 nudges
# [[ 2.     -1.      0.5  ]
#  [-0.5     0.25   -0.125]]
W.T @ delta                           # array([ 0.25 , -1.   ,  2.125])   dL/dx
```

That last line is the gradient with respect to the **input**, \( \partial L / \partial \mathbf{x} = W^\top \boldsymbol{\delta} \) (shape (in,), like \( \mathbf{x} \)): output \( i \) depends on \( x_j \) with slope \( W_{ij} \), and the chain rule adds those paths over \( i \) (Day 31: add across paths). In a multi-layer network, this is what one layer hands to the layer below it during backpropagation. A layer with a bias, \( \mathbf{z} = W\mathbf{x} + \mathbf{b} \), simply has \( \partial L / \partial \mathbf{b} = \boldsymbol{\delta} \), because each \( b_i \) moves \( z_i \) one-for-one (Exercise 3).

### A whole batch at once

With a batch, Day 12 writes the layer as \( Z = XW^\top + \mathbf{b} \): \( X \) is (batch, in), one example per row, and \( Z \) is (batch, out). Let \( \Delta = \partial L / \partial Z \), shape (batch, out). Each example contributes its own outer product, and the chain rule adds them up. Stacked, that sum of outer products is one matrix product:

\[
\frac{\partial L}{\partial W} = \sum_{k} \boldsymbol{\delta}_k\, \mathbf{x}_k^\top = \Delta^\top X
\qquad
\frac{\partial L}{\partial \mathbf{b}} = \sum_k \boldsymbol{\delta}_k
\]

Shapes: (out, batch) @ (batch, in) gives (out, in), like \( W \). Check it all on random numbers:

```python
rng = np.random.default_rng(33)
N, n_in, n_out = 5, 3, 2
Xb = rng.normal(size=(N, n_in))       # a batch of 5 examples, one per row
Wb = rng.normal(size=(n_out, n_in))   # (out, in), like Day 12
bb = rng.normal(size=n_out)
Tb = rng.normal(size=(N, n_out))      # targets

def batch_loss(Wb, bb):
    Z = Xb @ Wb.T + bb                # Day 12's batched layer, shape (5, 2)
    return 0.5 * np.sum((Z - Tb) ** 2) / N

Delta = (Xb @ Wb.T + bb - Tb) / N     # dL/dZ, shape (5, 2)
dW = Delta.T @ Xb                     # (2, 5) @ (5, 3) -> (2, 3), like Wb
db = Delta.sum(axis=0)                # (2,), like bb
dW.shape, db.shape                    # ((2, 3), (2,))
sum_of_outers = sum(np.outer(Delta[i], Xb[i]) for i in range(N))
np.allclose(dW, sum_of_outers)                                     # True
np.allclose(dW, numerical_grad(lambda V: batch_loss(V, bb), Wb))   # True
np.allclose(db, numerical_grad(lambda v: batch_loss(Wb, v), bb))   # True
```

!!! tip "Let the shapes write the formula"
    You need \( \partial L / \partial W \) with shape (2, 3), and you hold \( \Delta \) (5, 2) and \( X \) (5, 3). The only way to multiply those two into a (2, 3) is \( \Delta^\top X \). For products like these, "make the shapes fit" usually *is* the derivation. It is a hint, not a proof: when the sizes are equal (a square \( W \), or batch = in), a wrong arrangement can fit too. Confirm with `numerical_grad`.

!!! android "Android bridge"
    The usual split is "train in Python, run on the phone", but LiteRT (TensorFlow Lite) also supports **on-device training** for personalization: the model is exported with a `train` signature whose gradients TensorFlow worked out at export time, and the app calls it through `Interpreter.runSignature(inputs, outputs, "train")`. For a tiny model you can also write the gradient yourself. Here is the MSE gradient you'll derive in the next section, for a linear model that re-ranks items from a user's own taps, in plain Kotlin:

    ```kotlin
    /** Gradient of (1/n) * ||X w - y||^2. x is the n-by-d matrix X, row-major: X[i][j] = x[i * d + j]. */
    fun mseGradient(x: FloatArray, y: FloatArray, w: FloatArray, n: Int, d: Int): FloatArray {
        require(x.size == n * d && y.size == n && w.size == d)
        val grad = FloatArray(d)                        // the shape rule: same size as w
        for (i in 0 until n) {
            var pred = 0f
            for (j in 0 until d) pred += x[i * d + j] * w[j]
            val r = pred - y[i]                         // residual of example i
            for (j in 0 until d) grad[j] += r * x[i * d + j]   // add r_i times row i of X
        }
        val scale = 2f / n
        for (j in 0 until d) grad[j] *= scale
        return grad
    }
    ```

    One step of learning is then `w[j] -= learningRate * grad[j]` for each `j`. Note the pattern: the gradient is a `FloatArray` exactly as long as the weights, just as a `diff` of a list has one entry per element.

## The MSE gradient for linear regression

Day 25 fitted a line by projection. Gradient descent (Day 51) will fit it by walking downhill instead, and for that it needs the gradient of the **mean squared error**:

\[
L(\mathbf{w}) = \frac{1}{n}\,\|X\mathbf{w} - \mathbf{y}\|^2 = \frac{1}{n}\sum_{i=1}^{n} \big(\mathbf{x}_i \cdot \mathbf{w} - y_i\big)^2
\]

- \( X \) is the data matrix, shape \( (n, d) \): \( n \) examples (rows), \( d \) features (columns, including a column of ones for the intercept).
- \( \mathbf{x}_i \) is row \( i \) of \( X \), one example, and \( y_i \) its target.
- \( \mathbf{w} \) holds the \( d \) parameters. \( X\mathbf{w} \) is every prediction at once.

Day 32 took this gradient for two parameters with two separate partial derivatives. Here is the version for any number of parameters, in four small steps.

**Step 1: name the residuals.** \( \mathbf{r} = X\mathbf{w} - \mathbf{y} \), shape \( (n,) \): how far off each prediction is. Then \( L = \frac{1}{n}\,\mathbf{r} \cdot \mathbf{r} \).

**Step 2: from the loss back to the residuals.** By rule 2, \( \partial L / \partial \mathbf{r} = \frac{2}{n}\,\mathbf{r} \). Each residual pushes on the loss in proportion to its own size.

**Step 3: from one residual back to the weights.** \( r_i = \mathbf{x}_i \cdot \mathbf{w} - y_i \), so by rule 1, \( \nabla_{\mathbf{w}}\, r_i = \mathbf{x}_i \) (\( y_i \) is a constant).

**Step 4: chain rule, adding across paths.** \( \mathbf{w} \) reaches \( L \) through all \( n \) residuals, so (Day 31) the contributions add:

\[
\nabla_{\mathbf{w}} L = \sum_{i=1}^{n} \frac{2}{n}\, r_i\, \mathbf{x}_i = \frac{2}{n}\, X^\top \mathbf{r} = \frac{2}{n}\, X^\top (X\mathbf{w} - \mathbf{y})
\]

The middle step is Day 10's "a matrix times a vector is a weighted sum of columns": the columns of \( X^\top \) are the rows \( \mathbf{x}_i \), weighted by the residuals \( r_i \). In words: **every example pulls on the gradient along its own feature vector, as hard as it misses.**

**Worked example.** Day 25's three points \( (0, 1), (1, 2), (2, 2) \), at the starting guess \( \mathbf{w} = [0, 0] \) (slope 0, intercept 0):

<figure class="diagram" markdown>
![Animation in stages: X, a 3 by 2 blue grid with rows (0, 1), (1, 1), (2, 1), times w = [0, 0] gives the orange predictions Xw = [0, 0, 0]; minus y = [1, 2, 2] gives the red residuals r = [-1, -2, -2]; on the second row X transpose, a 2 by 3 grid, times r gives the teal vector [-6, -5], and times 2/3 gives the gradient [-4, -3.33]; a side panel lists each shape step, (3, 2) @ (2,) to (3,), (3,) - (3,) to (3,), (2, 3) @ (3,) to (2,), number times (2,) to (2,), then ticks that (2,) is the shape of w and notes that both entries are negative, so raise the slope and the intercept](../../assets/img/phase-1/day33-mse-shapes.svg)
<figcaption>Animated: residuals of shape (n,) go in, and Xᵀ turns them back into a vector of shape (d,), the shape of w.</figcaption>
</figure>

- Predictions \( X\mathbf{w} = [0, 0, 0] \), residuals \( \mathbf{r} = [-1, -2, -2] \), loss \( L = (1 + 4 + 4)/3 = 3 \).
- \( X^\top \mathbf{r} = [0\cdot(-1) + 1\cdot(-2) + 2\cdot(-2),\; -1 - 2 - 2] = [-6, -5] \).
- \( \nabla L = \frac{2}{3}[-6, -5] = [-4, -3.333] \).

Both entries are negative, so increasing either the slope or the intercept lowers the loss. Sensible: the line \( y = 0 \) sits below all three points.

```python
X = np.array([[0., 1.],
              [1., 1.],
              [2., 1.]])              # Day 25's three points: column of x values, column of ones
y = np.array([1., 2., 2.])
n = len(y)

def mse(w):
    r = X @ w - y
    return r @ r / n

def mse_grad(w):
    return 2 / n * X.T @ (X @ w - y)

w0 = np.zeros(2)
mse(w0)                    # 3.0
mse_grad(w0)               # array([-4.        , -3.33333333])
numerical_grad(mse, w0)    # array([-4.        , -3.33333333])
```

Now the payoff. At the bottom of the bowl the gradient is zero. Set it to zero:

\[
\frac{2}{n} X^\top (X\hat{\mathbf{w}} - \mathbf{y}) = \mathbf{0}
\quad\Longrightarrow\quad
X^\top X\, \hat{\mathbf{w}} = X^\top \mathbf{y}
\]

That is the **normal equation** of Days 11 and 25, reached a third way. Projection ("the residual is perpendicular to the columns") and calculus ("the slope is zero at the bottom") give the same answer, because they are the same condition: \( X^\top \mathbf{r} = \mathbf{0} \).

```python
w_best = np.linalg.lstsq(X, y)[0]
w_best                     # array([0.5       , 1.16666667])   Day 25's line
mse_grad(w_best)           # array([ 2.83723662e-16, -2.88349591e-16])   zero, up to rounding
```

## Gradient checking, properly

You now have formulas. Formulas have typos. **Gradient checking** compares an analytic gradient (the formula) with a numerical one (the nudges) and reports how far apart they are, as a single number.

Comparing "is the difference small?" doesn't work on its own, because "small" depends on how big the gradient is. So we use the **relative error**:

\[
\text{relative error} = \frac{\|\mathbf{g}_a - \mathbf{g}_n\|}{\|\mathbf{g}_a\| + \|\mathbf{g}_n\|}
\]

- \( \mathbf{g}_a \) is the analytic gradient, \( \mathbf{g}_n \) the numerical one, and \( \|\cdot\| \) the L2 norm from Day 22 (for a matrix, NumPy's default norm treats it as one long vector).
- The result is between 0 (identical) and 1, whatever units the loss is in.

The function below expects `f(w)` to return **two things**, the loss and its analytic gradient. That's a common contract for optimizers (SciPy's `minimize` accepts exactly this with `jac=True`), and it keeps the loss and its gradient side by side in your code, where a mismatch is easiest to spot.

```python
def grad_check(f, w, h=1e-5):
    """f(w) must return (loss, analytic_gradient).
    Returns the relative error between that gradient and a central-difference one."""
    _, g_analytic = f(w)
    g_numeric = numerical_grad(lambda v: f(v)[0], w, h)
    diff = np.linalg.norm(g_analytic - g_numeric)
    size = np.linalg.norm(g_analytic) + np.linalg.norm(g_numeric)
    return diff / size if size > 0 else 0.0

def mse_and_grad(w):
    r = X @ w - y
    return r @ r / n, 2 / n * X.T @ r

w_test = np.array([0.3, -0.7])        # any ordinary point, not a special one
grad_check(mse_and_grad, w_test)      # 3.123288396289289e-12  (last digits vary)
```

How to read the number, in float64 with \( h = 10^{-5} \):

| Relative error | Verdict |
|---|---|
| below \( 10^{-7} \) | Correct. A smooth function usually lands around \( 10^{-9} \) to \( 10^{-12} \). |
| \( 10^{-7} \) to \( 10^{-4} \) | Look closer. Normal near a kink (ReLU at 0), with float32, or when the gradient is tiny; otherwise suspicious. |
| above \( 10^{-4} \) | Treat it as a bug. |

Now break the gradient on purpose, three classic ways:

```python
def bug_missing_2(w):
    r = X @ w - y
    return r @ r / n, 1 / n * X.T @ r          # forgot the 2

def bug_sign(w):
    r = X @ w - y
    return r @ r / n, 2 / n * X.T @ (y - X @ w)   # residual the wrong way round

def bug_no_n(w):
    r = X @ w - y
    return r @ r / n, 2 * X.T @ r              # forgot to divide by n

grad_check(bug_missing_2, w_test), grad_check(bug_sign, w_test), grad_check(bug_no_n, w_test)
# (0.3333333333353785, 0.9999999999999999, 0.4999999999982744)
```

Those aren't random. If your gradient is the right one times a constant \( c \), the relative error is \( |c - 1| / (|c| + 1) \): \( c = \tfrac{1}{2} \) gives exactly \( 1/3 \), a flipped sign (\( c = -1 \)) gives exactly 1, and forgetting to divide by \( n = 3 \) (\( c = 3 \)) gives \( 2/4 = 0.5 \). **A clean 0.333 means a factor of 2; a clean 1.0 means a sign.** Learn to recognize them.

Why relative and not absolute? Measure the same loss in different units (times a million) and the absolute difference grows a million-fold while the relative error doesn't move:

```python
def big_loss(w):
    L, g = mse_and_grad(w)
    return 1e6 * L, 1e6 * g                    # same function, different units

_, g_a = big_loss(w_test)
np.abs(g_a - numerical_grad(lambda v: big_loss(v)[0], w_test)).max()   # 1.5521887689828873e-05
grad_check(big_loss, w_test)                   # 1.2855793111755902e-12   still "correct"
```

An absolute test with a threshold like `1e-7` would have called this correct gradient wrong.

!!! warning "Common mistake: gradient checking in float32"
    Models usually run in float32, but float32 keeps only about 7 significant digits. With \( h = 10^{-5} \), \( f(w + h) - f(w - h) \) is mostly rounding noise: Day 29's error curve, at its worst. Here the *correct* gradient fails:

    ```python
    X32, y32 = X.astype(np.float32), y.astype(np.float32)

    def mse_and_grad_32(w):
        w = w.astype(np.float32)                   # the model computes in float32
        r = X32 @ w - y32
        return r @ r / n, 2 / n * X32.T @ r

    grad_check(mse_and_grad_32, w_test)            # 0.0015359212651238896   looks like a bug, isn't
    grad_check(mse_and_grad_32, w_test, h=1e-2)    # 1.6764930097463573e-06  bigger h helps, but only so far
    ```

    Always gradient-check in **float64**, on a small problem. That's also what PyTorch's `torch.autograd.gradcheck` expects: double-precision inputs.

!!! warning "Common mistake: checking right next to a kink"
    ReLU's slope jumps from 0 to 1 at 0 (Day 30). If an input sits within \( h \) of the kink, the central difference straddles it and averages the two slopes:

    ```python
    def relu_sum(w):
        return np.maximum(w, 0).sum(), (w > 0).astype(float)

    grad_check(relu_sum, np.array([0.5, 2.0]))     # 3.275602011843332e-12
    grad_check(relu_sum, np.array([1e-6, 2.0]))    # 0.17609183310539098
    numerical_grad(lambda v: relu_sum(v)[0], np.array([1e-6, 2.0]))   # array([0.55, 1.  ])
    ```

    The analytic gradient `[1, 1]` is right; the nudge measured a blend, 0.55. If a check fails on a model with ReLUs, test again at a different random point before you hunt for a bug.

!!! android "Android bridge"
    Gradient checking is a **unit test** with a reference implementation. You know the pattern: a fast, clever production function (here the formula) checked against a slow, obviously-correct one (here the nudges), on small inputs, in CI. Write it that way: a JUnit or pytest test that builds a tiny model, calls `grad_check` at a few random points, and asserts the result is below `1e-7`. Do the math in `Double` (or float64), exactly as the warning above says, even if production code uses `Float`.

## The Jacobian, in one paragraph

So far the function returned one number. When a function takes a vector of size \( n \) and returns a vector of size \( m \), like a layer \( \mathbf{z} = W\mathbf{x} \), its derivative is a whole table: the **Jacobian** \( J \), an \( m \times n \) matrix whose entry \( J_{ij} = \partial z_i / \partial x_j \) says how output \( i \) responds to input \( j \). Row \( i \) is just the gradient of output \( i \). For \( \mathbf{z} = W\mathbf{x} \) the Jacobian with respect to \( \mathbf{x} \) is \( W \) itself, and the "backward" step \( \partial L / \partial \mathbf{x} = W^\top \boldsymbol{\delta} \) you saw earlier is \( J^\top \boldsymbol{\delta} \), a *vector-Jacobian product*. Backpropagation never builds Jacobians (they would be enormous); it computes these products layer by layer.

```python
def layer(v):
    return W @ v                           # vector in (3,), vector out (2,)

J = np.array([numerical_grad(lambda v: layer(v)[i], x) for i in range(2)])   # one gradient per output
J
# [[ 0.5 -1.   2. ]
#  [ 1.   0.  -0.5]]
np.allclose(J, W)                          # True
J.T @ delta                                # array([ 0.25 , -1.   ,  2.125])   = W.T @ delta
```

## A real gradient: Play Store ratings

Now a real loss on real (synthetic) data: predict an app's star rating from two features, `log_installs` (log10 of the installs, Day 25) and `size_mb`. Load it with the short cleaning from Day 17:

```python
import pandas as pd

BASE = "https://manjulkadyan.github.io/ai-leaning/assets/data/"
apps = pd.read_csv(BASE + "play_store_apps.csv")
apps = apps.drop_duplicates().assign(
    installs=lambda d: d["installs"].str.replace(",", "").str.rstrip("+").astype(int),
    size_mb=lambda d: d["size_mb"].replace("Varies with device", np.nan).astype(float),
)
data = pd.DataFrame({
    "log_installs": np.log10(apps["installs"]),
    "size_mb": apps["size_mb"],
    "rating": apps["rating"],
}).dropna()
data.shape                 # (442, 3)
print(data.describe().loc[["mean", "std"]].round(2).to_string())
#       log_installs  size_mb  rating
# mean          4.45    43.70    4.17
# std           0.93    37.68    0.46
```

No internet? Download the file from the [Datasets page](../../datasets.md) and pass the local path to `read_csv`. (Tested on pandas 3.0.)

Standardize both features (Day 22), add a column of ones for the intercept, and compute the loss and gradient at \( \mathbf{w} = \mathbf{0} \):

```python
F = data[["log_installs", "size_mb"]].to_numpy()
F_std = (F - F.mean(axis=0)) / F.std(axis=0, ddof=1)      # Day 22: mean 0, std 1 per column
Xp = np.column_stack([F_std, np.ones(len(F_std))])        # (442, 3): two features + intercept
yp = data["rating"].to_numpy()                            # (442,)
Xp.shape, yp.shape                                        # ((442, 3), (442,))

def rating_loss_and_grad(w):
    r = Xp @ w - yp
    return r @ r / len(yp), 2 / len(yp) * Xp.T @ r

wp = np.zeros(3)
Lp, gp = rating_loss_and_grad(wp)
Lp.round(4), gp.round(4)          # (17.5794, array([-0.0206, -0.1096, -8.3353]))
grad_check(rating_loss_and_grad, wp)   # 2.0103375920982322e-11   correct
```

Read the gradient entry by entry. It has three entries because \( \mathbf{w} \) has three.

- **The intercept entry, −8.34, dominates.** \( \mathbf{w} = \mathbf{0} \) predicts 0 stars for every app, so the loudest message is "raise everything". For the column of ones, \( \partial L/\partial w_3 = \frac{2}{n}\sum(0 - y_i) = -2\bar{y} \), minus twice the mean rating.
- **The feature entries are small.** Each standardized column has mean 0, so its entry works out to \( -\frac{2}{n}\sum_i x_{ij}(y_i - \bar{y}) \): proportional to how that feature co-varies with rating. Size co-varies a little (the −0.11); installs barely (−0.02).

```python
-2 * yp.mean()                                       # -8.33529411764706
-2 * np.mean(F_std[:, 1] * (yp - yp.mean()))         # -0.1096461853874422   the size entry
for lr in [0.1, 0.4]:                                # one downhill step, two step sizes
    w_next = wp - lr * gp
    print(lr, w_next.round(4), rating_loss_and_grad(w_next)[0].round(4))
# 0.1 [0.0021 0.011  0.8335] 11.3254
# 0.4 [0.0082 0.0439 3.3341] 0.902
```

One step against the gradient cuts the loss from 17.58 to 11.33, or to 0.90 with a bigger step. How big a step to take is the subject of Day 51. Where does it end? At the least-squares solution, where the gradient is zero:

```python
w_star = np.linalg.lstsq(Xp, yp)[0]
w_star.round(4)                      # array([3.4000e-03, 5.4500e-02, 4.1676e+00])
L_star, g_star = rating_loss_and_grad(w_star)
L_star.round(4), np.abs(g_star).max()    # (0.2071, 2.972448676086259e-15)
```

The best intercept is 4.17, the mean rating, and the two slopes are tiny: 0.003 and 0.05 stars per standard deviation. The final loss, 0.207, is barely below the variance of the ratings (\( 0.46^2 \approx 0.21 \)), so these two features hardly predict rating at all in this dataset. That is a real finding, and the gradient still did its job: it pointed to the best these features can do.

Finally, rule 1 on real data. Skip the standardization and compute the gradient again:

```python
Xraw = np.column_stack([F, np.ones(len(F))])
g_raw = 2 / len(yp) * Xraw.T @ (Xraw @ np.zeros(3) - yp)
g_raw.round(2)             # array([ -37.1 , -368.43,   -8.34])
gp.round(2)                # array([-0.02, -0.11, -8.34])
```

The `size_mb` entry jumps from −0.11 to −368, not because size matters more, but because its values are around 44 instead of around 0 (rule 1: the gradient is proportional to the input). One learning rate can't suit entries that differ by a factor of 3,000: small enough for `size_mb` is far too slow for everything else. Standardized features keep the gradient's entries on comparable scales.

!!! warning "Common mistake: a gradient with the wrong shape that still runs"
    NumPy broadcasting (Day 9) can hide a shape bug. If `y` is stored as a column, shape `(n, 1)`, then `X @ w - y` with `X @ w` of shape `(n,)` broadcasts to an `(n, n)` matrix, and `X.T @ r` silently returns a `(d, n)` "gradient". The shape rule is your alarm: assert `grad.shape == w.shape` in every gradient function you write.

---

## Exercises

1. By hand, find the gradient of \( f(\mathbf{w}) = 4w_1 - w_2 + 2w_3 + w_1^2 \) at \( \mathbf{w} = [1, 2, -1] \). Check it with `numerical_grad`.
2. For a 2×2 matrix \( A \), the function \( q(\mathbf{v}) = \mathbf{v}^\top A \mathbf{v} \) is a number. Write it out with indices for \( d = 2 \), and show that \( \nabla q = (A + A^\top)\mathbf{v} \). Check with \( A = \begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} \) and \( \mathbf{v} = [1, -2] \). What does it simplify to when \( A \) is symmetric, and which earlier rule is the case \( A = I \)?
3. For the layer loss \( L = \tfrac{1}{2}\|W\mathbf{x} + \mathbf{b} - \mathbf{t}\|^2 \), with this lesson's `W`, `x`, `t` and \( \mathbf{b} = [0.5, -1] \), write three functions returning `(loss, grad)` with respect to \( \mathbf{b} \), \( \mathbf{x} \) and \( W \), and `grad_check` all three.
4. Before running anything, predict what `grad_check` returns if the MSE gradient is written as `4 / n * X.T @ r`. Then predict what happens if it is written as `X @ r` instead of `X.T @ r`. Run both. Derive the formula \( |c - 1| / (|c| + 1) \) for a gradient that is off by a factor \( c \).
5. **Ridge regression.** Add L2 regularization to the Play Store loss: \( L(\mathbf{w}) = \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2 \) with \( \lambda = 0.1 \). Write its gradient, `grad_check` it, set it to zero to get a linear system, solve it with `np.linalg.solve`, and confirm the gradient is zero there. What happened to the intercept, and why?
6. The **mean absolute error** \( L = \frac{1}{n}\sum_i |r_i| \) has gradient \( \frac{1}{n}X^\top \operatorname{sign}(\mathbf{r}) \) wherever no residual is exactly 0. Check it on the tiny `X`, `y` at `w_test`. Then check at \( \mathbf{w} = [1, 1] \), where one residual is exactly 0, and at \( [1, 1 + 3\times10^{-6}] \). Explain the three results.

??? success "Solutions"
    ```python
    # 1: df/dw1 = 4 + 2 w1 = 6, df/dw2 = -1, df/dw3 = 2
    def f1(w):
        return 4 * w[0] - w[1] + 2 * w[2] + w[0] ** 2

    w1 = np.array([1.0, 2.0, -1.0])
    np.array([4 + 2 * w1[0], -1.0, 2.0])   # array([ 6., -1.,  2.])
    numerical_grad(f1, w1)                 # array([ 6., -1.,  2.])
    ```

    **2.** With indices, \( q = A_{11}v_1^2 + (A_{12} + A_{21})v_1v_2 + A_{22}v_2^2 \). So \( \partial q/\partial v_1 = 2A_{11}v_1 + (A_{12} + A_{21})v_2 \) and \( \partial q/\partial v_2 = (A_{12} + A_{21})v_1 + 2A_{22}v_2 \), which is exactly \( (A + A^\top)\mathbf{v} \). For symmetric \( A \) it is \( 2A\mathbf{v} \), and \( A = I \) gives \( \mathbf{v}^\top\mathbf{v} = \|\mathbf{v}\|^2 \) with gradient \( 2\mathbf{v} \): rule 2.

    ```python
    A = np.array([[2.0, 1.0],
                  [0.0, 3.0]])
    v2 = np.array([1.0, -2.0])
    quad = lambda v: v @ A @ v
    quad(v2)                  # 12.0
    (A + A.T) @ v2            # array([  2., -11.])
    numerical_grad(quad, v2)  # array([  2., -11.])
    S = np.array([[2.0, 1.0], [1.0, 3.0]])
    2 * S @ v2, numerical_grad(lambda v: v @ S @ v, v2)   # (array([  0., -10.]), array([  0., -10.]))
    ```

    ```python
    # 3: delta = W x + b - t; dL/db = delta, dL/dx = W^T delta, dL/dW = outer(delta, x)
    b3 = np.array([0.5, -1.0])

    def loss_b(b):
        d = W @ x + b - t
        return 0.5 * d @ d, d

    def loss_x(v):
        d = W @ v + b3 - t
        return 0.5 * d @ d, W.T @ d

    def loss_W(M):
        d = M @ x + b3 - t
        return 0.5 * d @ d, np.outer(d, x)

    grad_check(loss_b, b3), grad_check(loss_x, x), grad_check(loss_W, W)
    # (3.2755583311397037e-12, 5.312892013853388e-12, 4.160719977850615e-12)
    loss_b(b3)[1], loss_x(x)[1]     # (array([ 1.5 , -1.25]), array([-0.5  , -1.5  ,  3.625]))
    ```

    **4.** `4 / n * X.T @ r` is twice the right answer, \( c = 2 \), so the prediction is \( |2 - 1|/(2 + 1) = 1/3 \). `X @ r` fails before any check: `X` is (3, 2) and `r` is (3,), so the inner sizes don't match. (On a square `X` it would run and give garbage, which `grad_check` would catch.)

    ```python
    def bug_double(w):
        r = X @ w - y
        return r @ r / n, 4 / n * X.T @ r

    grad_check(bug_double, w_test)   # 0.3333333333312882

    r_test = X @ w_test - y
    try:
        X @ r_test                   # (3, 2) @ (3,): the inner sizes 2 and 3 don't match
    except ValueError as err:
        print(err)
    # matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)
    ```

    The formula: with \( \mathbf{g}_a = c\,\mathbf{g}_n \), the numerator is \( \|c\,\mathbf{g}_n - \mathbf{g}_n\| = |c - 1|\,\|\mathbf{g}_n\| \) and the denominator is \( |c|\,\|\mathbf{g}_n\| + \|\mathbf{g}_n\| \). The \( \|\mathbf{g}_n\| \) cancels.

    ```python
    # 5: gradient = (2/n) X^T (X w - y) + 2 lam w. Zero when (X^T X + n lam I) w = X^T y
    lam = 0.1

    def ridge(w):
        r = Xp @ w - yp
        return r @ r / len(yp) + lam * w @ w, 2 / len(yp) * Xp.T @ r + 2 * lam * w

    grad_check(ridge, np.array([0.2, -0.3, 1.0]))   # 2.220191794112763e-11
    n_p = len(yp)
    w_ridge = np.linalg.solve(Xp.T @ Xp + n_p * lam * np.eye(3), Xp.T @ yp)
    w_ridge.round(4)                     # array([3.6000e-03, 4.9500e-02, 3.7888e+00])
    np.abs(ridge(w_ridge)[1]).max()      # 1.4432899320127035e-15
    ```

    Setting \( \frac{2}{n}X^\top(X\mathbf{w} - \mathbf{y}) + 2\lambda\mathbf{w} = \mathbf{0} \) and multiplying by \( n/2 \) gives \( (X^\top X + n\lambda I)\,\mathbf{w} = X^\top\mathbf{y} \). The intercept dropped from 4.17 to 3.79, because the penalty pulls *every* entry of \( \mathbf{w} \) towards 0, the intercept included, and predicting below the mean rating is the price. In practice the intercept is usually left out of the penalty, which you'll see in Phase 2.

    ```python
    # 6
    def mae(w):
        r = X @ w - y
        return np.abs(r).mean(), X.T @ np.sign(r) / n

    grad_check(mae, w_test)                           # 7.826683745609991e-12
    mae(np.array([1.0, 1.0]))                         # (0.3333333333333333, array([0.66666667, 0.33333333]))
    grad_check(mae, np.array([1.0, 1.0]))             # 3.2756297674189476e-12
    grad_check(mae, np.array([1.0, 1.0 + 3e-6]))      # 0.22219661637598478
    ```

    At `w_test` no residual is near 0, so the formula is exact and the check passes. At \( [1, 1] \) the predictions are \( [1, 2, 3] \), so two residuals are exactly 0. It still passes, by luck: `np.sign(0)` is 0, and the central difference of \( |r| \) at exactly 0 is \( (h - h)/2h = 0 \) as well. At \( [1, 1 + 3\times10^{-6}] \) those residuals are \( 3\times10^{-6} \), closer to the kink than \( h = 10^{-5} \), so the nudges straddle the kink and the check fails although the formula is right. Same lesson as ReLU: a failure next to a kink is not proof of a bug.

## Resources

- [3Blue1Brown: Backpropagation calculus (Deep learning, chapter 4)](https://www.youtube.com/watch?v=tIeHLnjs5U8): the chain rule through a network, with the "error times input" pattern for every weight that today's outer product packages up.
- [The Matrix Calculus You Need For Deep Learning (Terence Parr and Jeremy Howard)](https://explained.ai/matrix-calculus/): a free, patient article that builds vector and matrix derivatives, and Jacobians, from scalar ones.
- [Mathematics for Machine Learning (free book)](https://mml-book.github.io/): chapter 5, "Vector Calculus", covers gradients of vectors and matrices, Jacobians and the gradient of least squares.
- [CS231n: Neural Networks Part 3](https://cs231n.github.io/neural-networks-3/): its "Gradient Checks" section is the classic practical guide (relative error, kinks, double precision).
- [Dive into Deep Learning: Calculus](https://d2l.ai/chapter_preliminaries/calculus.html): derivatives, partial derivatives, gradients and the chain rule, with code.
- [Khan Academy: Multivariable calculus](https://www.khanacademy.org/math/multivariable-calculus): gradients and Jacobians with exercises, if you want more practice by hand.
- [PyTorch docs: torch.autograd](https://pytorch.org/docs/stable/autograd.html): the automatic version of everything today, including `gradcheck` for numerical gradient checking.

## Checklist

- [ ] I can state the shape rule and use it to sanity-check any gradient I write.
- [ ] I can derive \( \nabla(\mathbf{w}\cdot\mathbf{x}) = \mathbf{x} \) and \( \nabla\|\mathbf{w}\|^2 = 2\mathbf{w} \) by writing out the entries.
- [ ] I can explain why \( \partial L/\partial W = \boldsymbol{\delta}\,\mathbf{x}^\top \) for a layer, and why a batch gives \( \Delta^\top X \).
- [ ] I can derive \( \frac{2}{n}X^\top(X\mathbf{w} - \mathbf{y}) \) and explain why setting it to zero gives the normal equation.
- [ ] I wrote `numerical_grad` and `grad_check`, and I know what 0.333 and 1.0 mean.
- [ ] I know to gradient-check in float64, at random points, away from kinks.
- [ ] I can say what a Jacobian is and what a vector-Jacobian product has to do with backpropagation.
- [ ] I computed and checked a real gradient on the Play Store data, and saw why standardizing features matters for it.

**Next:** [Day 34: Week 5 review and mini-project](day-34-week5-review.md)
