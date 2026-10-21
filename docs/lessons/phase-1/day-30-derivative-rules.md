# Day 30: Derivative Rules You'll Actually Use

**Phase 1 · Week 5 · ~90 minutes**

## Objectives

- [ ] Differentiate any polynomial in your head with four rules: constant, power, constant multiple and sum.
- [ ] Use the product rule, and explain it with a growing rectangle.
- [ ] Know why \( e^x \) is its own derivative and why the derivative of \( \ln x \) is \( 1/x \).
- [ ] Derive \( \sigma'(x) = \sigma(x)(1 - \sigma(x)) \) for the sigmoid, and know the derivatives of tanh, ReLU (including what happens at 0) and softplus.
- [ ] Check every rule numerically with Day 29's central difference, and plot functions next to their derivatives.
- [ ] Explain why the sigmoid's slope is never above 0.25, and why that makes deep sigmoid networks hard to train.

**Prerequisites.** [Day 29](day-29-derivatives.md) (the derivative as a slope and as sensitivity, the notation \( f'(x) \) and \( \frac{df}{dx} \), and the central difference). [Day 12](day-12-layer-is-matmul.md) helps for the activation functions: that is where you met ReLU.

---

## The big idea

Yesterday you computed derivatives the slow way: nudge \( x \) by a tiny \( h \), see how far \( f \) moves, divide. That always works, and it is how you'll **check** your work forever. But nobody differentiates \( 3x^4 - 5x^2 + 7x - 2 \) by taking limits. Instead you learn the derivatives of a handful of building blocks, plus a few rules for how derivatives behave when you combine blocks, and you snap them together.

**Analogy.** You don't write your own `sort()` for every screen. You learn what the standard library gives you (`sortedBy`, `filter`, `map`) and how its pieces compose. Derivatives work the same way: a small "standard library" of known slopes (\( x^n \), \( e^x \), \( \ln x \)) and composition rules (sum, constant multiple, product, and tomorrow the chain rule). With those, you can differentiate anything a neural network is built from.

Why should an ML engineer care? Training a model means asking "if I nudge this weight, how does the error change?" for millions of weights at once. Frameworks like PyTorch answer that with exactly the rules in this lesson, applied automatically. Today you learn the rules by hand so the automatic version is never magic.

| Rule or function | Where it shows up in ML |
|---|---|
| Power, sum, constant multiple | The squared error \( (p - y)^2 \) and its gradient \( 2(p - y) \) |
| \( e^x \) and \( \ln x \) | Softmax, sigmoid, log-loss (cross-entropy) |
| Sigmoid, tanh | Output probabilities, gates in older sequence models (LSTMs) |
| ReLU, softplus | The default hidden-layer activation in modern networks |

First, the checking tool from Day 29. Every rule below gets tested with it.

```python
import numpy as np

def central_diff(f, x, h=1e-5):
    """Day 29's central difference: the slope of f at x, measured numerically."""
    return (f(x + h) - f(x - h)) / (2 * h)

central_diff(lambda x: x**2, 3.0)    # 6.000000000039306   the slope of x² at 3 is 2·3 = 6
```

## The four everyday rules

### Constant rule: flat lines have slope 0

If \( f(x) = c \) for some fixed number \( c \) (say \( f(x) = 7 \)), the graph is a flat horizontal line. Nudging \( x \) changes nothing, so

\[
\frac{d}{dx}\, c = 0
\]

(\( \frac{d}{dx} \) means "the derivative with respect to \( x \) of whatever follows".)

### Power rule: bring the power down, lower it by one

For \( f(x) = x^n \):

\[
\frac{d}{dx}\, x^n = n\, x^{n-1}
\]

Reading it: the old power \( n \) moves to the front as a multiplier, and the new power is one less. Examples:

- \( x^2 \to 2x^1 = 2x \) (Day 29's result)
- \( x^3 \to 3x^2 \)
- \( x^5 \to 5x^4 \)
- \( x^1 = x \to 1 \cdot x^0 = 1 \): the line \( y = x \) has slope 1 everywhere
- \( x^0 = 1 \to 0 \): the constant rule again

**Why it works, for \( n = 3 \).** Nudge \( x \) by \( h \) and multiply out \( (x + h)^3 = (x + h)(x + h)(x + h) \):

\[
(x + h)^3 = x^3 + 3x^2 h + 3x h^2 + h^3
\]

The change in the output is everything except the original \( x^3 \). Divide it by the nudge \( h \):

\[
\frac{(x + h)^3 - x^3}{h} = \frac{3x^2 h + 3x h^2 + h^3}{h} = 3x^2 + 3xh + h^2
\]

Now shrink \( h \) toward 0: \( 3xh \) and \( h^2 \) disappear, and \( 3x^2 \) is left. The pattern is general: in \( (x + h)^n \), exactly \( n \) terms have a single \( h \) (one for each bracket you could take the \( h \) from), each worth \( x^{n-1} h \). Everything else has \( h^2 \) or more and vanishes.

The rule also works for negative and fractional powers (we'll accept that without proof and check it numerically):

- \( \frac{1}{x} = x^{-1} \to -1 \cdot x^{-2} = -\frac{1}{x^2} \)
- \( \sqrt{x} = x^{1/2} \to \frac{1}{2} x^{-1/2} = \frac{1}{2\sqrt{x}} \)

**Worked example at \( x = 2 \):** \( x^3 \) has slope \( 3 \cdot 2^2 = 12 \); \( 1/x \) has slope \( -1/4 = -0.25 \); \( \sqrt{x} \) has slope \( 1/(2\sqrt{2}) \approx 0.3536 \).

```python
x = 2.0
for n in [2, 3, 5, -1, 0.5]:
    rule = n * x**(n - 1)
    numeric = central_diff(lambda t: t**n, x)
    print(f"n = {n:>4}:  rule {rule:.6f}   numeric {numeric:.6f}")
# n =    2:  rule 4.000000   numeric 4.000000
# n =    3:  rule 12.000000   numeric 12.000000
# n =    5:  rule 80.000000   numeric 80.000000
# n =   -1:  rule -0.250000   numeric -0.250000
# n =  0.5:  rule 0.353553   numeric 0.353553
```

### Constant multiple and sum rules

Two more rules, both of which say "derivatives pass straight through":

\[
\frac{d}{dx}\big(c\, f(x)\big) = c\, f'(x)
\qquad\qquad
\frac{d}{dx}\big(f(x) + g(x)\big) = f'(x) + g'(x)
\]

- **Constant multiple.** Stretching a graph 3× taller makes every slope 3× steeper. So \( 5x^2 \to 5 \cdot 2x = 10x \).
- **Sum.** If \( x \) moves a little, \( f \) moves by about \( f' \cdot h \) and \( g \) by about \( g' \cdot h \). Their sum moves by both: \( (f' + g') \cdot h \). Subtraction works the same way (it is adding \( -1 \times g \)).

Together, the four rules differentiate any polynomial term by term. **Worked example:**

\[
f(x) = 3x^4 - 5x^2 + 7x - 2
\quad\Rightarrow\quad
f'(x) = 3 \cdot 4x^3 - 5 \cdot 2x + 7 \cdot 1 - 0 = 12x^3 - 10x + 7
\]

At \( x = 2 \): \( 12 \cdot 8 - 10 \cdot 2 + 7 = 96 - 20 + 7 = 83 \).

```python
f  = lambda x: 3*x**4 - 5*x**2 + 7*x - 2
df = lambda x: 12*x**3 - 10*x + 7          # by the rules
df(2.0), central_diff(f, 2.0)              # (83.0, 83.00000000289742)
central_diff(lambda x: 7.0, 2.0)           # 0.0   the constant rule
```

### Your first ML derivative: the squared error

A model predicts \( p \) and the true answer is \( y \). The **squared error** is \( L = (p - y)^2 \) (\( L \) for "loss"). How does the loss change if the model nudges its prediction? Treat \( y \) as a fixed number and expand:

\[
L(p) = p^2 - 2y\,p + y^2
\quad\Rightarrow\quad
\frac{dL}{dp} = 2p - 2y + 0 = 2(p - y)
\]

Only the power, constant multiple, sum and constant rules were used (\( y^2 \) is a constant, because \( y \) doesn't change when \( p \) does). With \( y = 3 \) and \( p = 5 \): \( dL/dp = 2(5 - 3) = 4 \). It is positive, so making \( p \) **bigger** makes the loss bigger; to reduce the loss, move \( p \) down. And the further off the prediction, the bigger the push. This little formula is the starting point of almost every regression model you'll train.

```python
y = 3.0
loss = lambda p: (p - y)**2
for p in [5.0, 3.5, 3.0, 1.0]:
    print(p, 2 * (p - y), round(central_diff(loss, p), 6))
# 5.0 4.0 4.0
# 3.5 1.0 1.0
# 3.0 0.0 0.0
# 1.0 -4.0 -4.0
```

At \( p = y \) the slope is 0: the bottom of the loss curve, where there is nowhere better to go.

!!! warning "Common mistake: forgetting the constant disappears, or that the power moves"
    \( \frac{d}{dx}(x^3 + 10) \) is \( 3x^2 \), not \( 3x^2 + 10 \): a constant added on shifts the graph up but doesn't tilt it. And \( \frac{d}{dx}\,x^3 \) is \( 3x^2 \), not \( 3x^3 \) or \( x^2 \): both halves of the rule happen (multiply by \( n \), then lower the power).

## The product rule

What is the derivative of a product, like \( (3x + 1)(x^2 + 2) \)? The tempting answer, "multiply the derivatives", is wrong. Picture the product as the **area of a rectangle** with width \( u(x) \) and height \( v(x) \):

<figure class="diagram" markdown>
![Animation: a rectangle of width u and height v, labelled u times v; a teal strip of width delta u appears on its right side, labelled delta u times v, then an orange strip of height delta v on top, labelled u times delta v, then a tiny yellow corner labelled delta u delta v; the text alongside adds up the three new pieces and concludes that the derivative of u v is u prime v plus u v prime](../../assets/img/phase-1/day30-product-rule.svg)
<figcaption>Animated: nudging x grows both sides; the extra area is two strips plus a corner so small it vanishes.</figcaption>
</figure>

Nudge \( x \). The width grows by \( \Delta u \) and the height by \( \Delta v \) (\( \Delta \), "delta", means "a small change in"). The new area minus the old is the two strips plus the corner: \( \Delta u \cdot v + u \cdot \Delta v + \Delta u \cdot \Delta v \). Divide by the nudge and shrink it: the strips give \( u' v \) and \( u v' \), and the corner, a tiny number times a tiny number, disappears. So:

\[
\big(u\,v\big)' = u'\,v + u\,v'
\]

In words: "derivative of the first times the second, plus the first times the derivative of the second".

**Worked example** with \( u = 3x + 1 \) and \( v = x^2 + 2 \), at \( x = 1 \):

- \( u = 4 \), \( u' = 3 \); \( v = 3 \), \( v' = 2x = 2 \)
- \( (uv)' = u'v + uv' = 3 \cdot 3 + 4 \cdot 2 = 17 \)

Check by expanding first: \( (3x + 1)(x^2 + 2) = 3x^3 + x^2 + 6x + 2 \), whose derivative \( 9x^2 + 2x + 6 \) is \( 9 + 2 + 6 = 17 \) at \( x = 1 \). Same answer.

```python
u = lambda x: 3*x + 1
v = lambda x: x**2 + 2
x = 1.0
central_diff(lambda t: u(t) * v(t), x)     # 17.000000000422233   numeric
3 * v(x) + u(x) * (2 * x)                  # 17.0   product rule: u'v + uv'
3 * (2 * x)                                # 6.0    u'·v' : wrong
```

!!! warning "Common mistake: (uv)′ = u′v′"
    Multiplying the derivatives gives 6 above instead of 17. The rectangle shows why: growing a rectangle adds area along **both** sides, and each strip is one side's growth times the **other** side's full length.

!!! note "The quotient rule, for reference"
    Division has its own rule, \( \left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2} \). You won't need to memorize it: it follows from the product rule plus tomorrow's chain rule, and below we'll get the sigmoid's derivative without it.

## The exponential eˣ and the natural log ln x

### The function that is its own slope

Exponentials \( b^x \) (\( 2^x \), \( 3^x \), ...) grow faster and faster. How fast? Nudge \( x \):

\[
\frac{b^{x+h} - b^x}{h} = b^x \cdot \frac{b^h - 1}{h}
\]

The second factor doesn't depend on \( x \) at all: it is the slope of \( b^x \) at \( x = 0 \). So **the slope of an exponential is the exponential itself, times a fixed number** (its slope at 0). Measure that number:

```python
for b in [2.0, np.e, 3.0]:
    print(f"b = {b:.5f}:  slope at 0 = {central_diff(lambda t: b**t, 0.0):.6f}")
# b = 2.00000:  slope at 0 = 0.693147
# b = 2.71828:  slope at 0 = 1.000000
# b = 3.00000:  slope at 0 = 1.098612
```

For \( b = 2 \) the slope at 0 is 0.69, for \( b = 3 \) it is 1.10. Somewhere between them is a base whose slope at 0 is exactly 1. That base is the number \( e \approx 2.71828 \) (`np.e`), and for it the fixed number is 1:

\[
\frac{d}{dx}\, e^x = e^x
\]

The height of the curve *is* its slope. At \( x = 0 \) the curve is at height 1 and climbs at slope 1; at \( x = 2 \) it is at height 7.389 and climbs at slope 7.389. This is why \( e \) is everywhere in ML: it is the exponential with the simplest derivative. (`np.exp(x)` means \( e^x \).)

```python
for x in [-1.0, 0.0, 1.0, 2.0]:
    print(x, np.exp(x), central_diff(np.exp, x))
# -1.0 0.36787944117144233 0.3678794411765551
# 0.0 1.0 1.0000000000121023
# 1.0 2.718281828459045 2.718281828517632
# 2.0 7.38905609893065 7.38905609911633
```

**Flipped: \( e^{-x} \).** The graph of \( e^{-x} \) is the mirror image of \( e^x \) (left and right swapped). A mirror turns every uphill into a downhill of the same steepness, so the slope of \( e^{-x} \) is \( -e^{-x} \). (Tomorrow's chain rule gives the same answer more formally.) We'll need this one for the sigmoid.

### The natural log: (ln x)′ = 1/x

The **natural logarithm** \( \ln x \) (`np.log`) undoes \( e^x \): \( \ln(e^a) = a \). For example, \( \ln(7.389) = 2 \) because \( e^2 = 7.389 \). It only exists for \( x > 0 \).

Undoing a function mirrors its graph across the diagonal line \( y = x \), which swaps "across" and "up". A slope is up ÷ across, so mirroring turns slope \( s \) into \( 1/s \). The point \( (a, e^a) \) on the exp curve, with slope \( e^a \), becomes the point \( (e^a, a) \) on the log curve, with slope \( 1/e^a \). Calling that point's first coordinate \( x = e^a \):

\[
\frac{d}{dx}\, \ln x = \frac{1}{x}
\]

At \( x = 0.5 \) the slope is 2, at \( x = 10 \) only 0.1: the log curve keeps climbing, more and more slowly.

```python
for x in [0.5, 1.0, 2.0, 10.0]:
    print(x, 1 / x, central_diff(np.log, x))
# 0.5 2.0 2.0000000002629026
# 1.0 1.0 1.0000000000343332
# 2.0 0.5 0.5000000000088267
# 10.0 0.1 0.09999999999177332
```

**Where this shows up: log-loss.** When a classifier says "the right answer has probability \( p \)", the standard loss is \( -\ln p \) (Month 2 and Phase 2 explain why). Its derivative is \( -1/p \). At \( p = 0.9 \) (confident and right) the slope is \( -1.1 \): a gentle push. At \( p = 0.01 \) (confident and wrong) it is \( -100 \): a shove. That steepness is exactly what makes log-loss train classifiers well.

## The activation derivatives ML uses every day

On Day 12 a layer was \( W\mathbf{x} + \mathbf{b} \) followed by an **activation function** applied to each number separately. Training needs the slope of that activation at every unit, on every step, so these four derivatives get computed trillions of times a day.

### Sigmoid: σ′(x) = σ(x)(1 − σ(x))

The **sigmoid** squashes any number into the range 0 to 1, which is why it is used to turn a score into a probability:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

(\( \sigma \) is the Greek letter "sigma".) Big positive \( x \): \( e^{-x} \) is tiny, so \( \sigma \approx 1 \). Big negative \( x \): \( e^{-x} \) is huge, so \( \sigma \approx 0 \). At \( x = 0 \): \( 1/(1 + 1) = 0.5 \).

**Deriving its slope with the product rule.** Multiply both sides of the definition by \( 1 + e^{-x} \):

\[
\sigma(x)\,\big(1 + e^{-x}\big) = 1
\]

The right side is a constant, so its derivative is 0. The left side is a product, so use the product rule, with \( (1 + e^{-x})' = -e^{-x} \) from the mirror argument above:

\[
\sigma'(x)\,\big(1 + e^{-x}\big) + \sigma(x)\,\big(-e^{-x}\big) = 0
\quad\Rightarrow\quad
\sigma'(x) = \sigma(x) \cdot \frac{e^{-x}}{1 + e^{-x}}
\]

The last fraction is just \( 1 - \sigma(x) \), because \( 1 - \frac{1}{1 + e^{-x}} = \frac{1 + e^{-x} - 1}{1 + e^{-x}} = \frac{e^{-x}}{1 + e^{-x}} \). So:

\[
\sigma'(x) = \sigma(x)\,\big(1 - \sigma(x)\big)
\]

This is the famous shortcut: once the forward pass has computed \( s = \sigma(x) \), the slope costs one subtraction and one multiplication, \( s(1 - s) \). **Worked example at \( x = 0 \):** \( s = 0.5 \), slope \( 0.5 \times 0.5 = 0.25 \). At \( x = 4 \): \( s = 0.982 \), slope \( 0.982 \times 0.018 = 0.0177 \). Far from 0 the curve is nearly flat: the sigmoid is **saturated**.

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

xs = np.array([-4., -1., 0., 1., 4.])
sigmoid(xs).round(4)                 # array([0.018 , 0.2689, 0.5   , 0.7311, 0.982 ])
sigmoid_prime(xs).round(4)           # array([0.0177, 0.1966, 0.25  , 0.1966, 0.0177])
central_diff(sigmoid, xs).round(4)   # array([0.0177, 0.1966, 0.25  , 0.1966, 0.0177])  agrees
```

`central_diff` works on a whole array at once: every operation inside it is element-wise NumPy (Day 4).

### tanh: tanh′(x) = 1 − tanh²(x)

**tanh** ("hyperbolic tangent", say "tanch") is the sigmoid's centred cousin: an S-curve from −1 to 1, passing through 0 at \( x = 0 \).

\[
\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
\qquad\qquad
\tanh'(x) = 1 - \tanh^2(x)
\]

(\( \tanh^2(x) \) means \( (\tanh x)^2 \).) The same multiply-out-and-use-the-product-rule trick derives it; that's Exercise 3. Like the sigmoid, its slope is computed from its own output: \( t = \tanh(x) \), slope \( 1 - t^2 \). At \( x = 0 \): \( t = 0 \), slope 1, four times steeper than the sigmoid's best.

```python
def tanh_prime(x):
    return 1 - np.tanh(x)**2

np.tanh(xs).round(4)                 # array([-0.9993, -0.7616,  0.    ,  0.7616,  0.9993])
tanh_prime(xs).round(4)              # array([0.0013, 0.42  , 1.    , 0.42  , 0.0013])
central_diff(np.tanh, xs).round(4)   # array([0.0013, 0.42  , 1.    , 0.42  , 0.0013])
```

### ReLU: slope 0 or 1 (and the kink at 0)

**ReLU**, \( \max(0, x) \), is the default activation in modern networks (Day 12). It is two straight lines glued together, so its derivative is two constants:

\[
\text{ReLU}'(x) = \begin{cases} 0 & x < 0 \\ 1 & x > 0 \end{cases}
\]

At exactly \( x = 0 \) there is a **kink**: the slope from the left is 0 and from the right is 1, so there is no single tangent and, strictly, no derivative. Frameworks simply pick a value, and PyTorch and TensorFlow both use **0** at \( x = 0 \). It almost never matters: a float landing on exactly `0.0` is rare, and when it happens either choice still lets training work.

Look at what the central difference does at the kink. It straddles both sides and reports their average:

```python
def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(float)       # 1.0 where x > 0, else 0.0 (including x = 0)

relu_prime(xs)                         # array([0., 0., 0., 1., 1.])
central_diff(relu, xs)                 # array([0. , 0. , 0.5, 1. , 1. ])   0.5 at the kink!
h = 1e-5
(relu(0 + h) - relu(0)) / h, (relu(0) - relu(0 - h)) / h
# (np.float64(1.0), np.float64(0.0))   slope from the right, slope from the left
```

!!! warning "Common mistake: gradient-checking exactly at a kink"
    A numerical check at \( x = 0 \) "fails" for ReLU (0.5 versus 0), even though nothing is wrong with your code. When you gradient-check a network (Day 33), use random inputs that don't sit on a kink, or expect a few mismatches there.

The other famous ReLU quirk: a unit whose input is negative for every example has slope 0 everywhere it is used, so its weights never get a push and never change. That's a **dead ReLU**. Variants such as *leaky ReLU* (Exercise 5) give the negative side a small slope to avoid it.

### Softplus: a smooth ReLU whose slope is the sigmoid

**Softplus** rounds off ReLU's kink:

\[
\text{softplus}(x) = \ln\big(1 + e^{x}\big)
\qquad\qquad
\text{softplus}'(x) = \sigma(x)
\]

For big positive \( x \), \( 1 + e^x \approx e^x \), so softplus \( \approx x \); for big negative \( x \), \( e^x \approx 0 \) and softplus \( \approx \ln 1 = 0 \). Its slope rises smoothly from 0 to 1 instead of jumping, and that slope is exactly the sigmoid. Proving it needs the chain rule (log of something that is itself a function), so it's a Day 31 exercise; for now, check it numerically:

```python
def softplus(x):
    return np.log1p(np.exp(x))          # log1p(z) = ln(1 + z), accurate for small z

softplus(xs).round(4)                   # array([0.0181, 0.3133, 0.6931, 1.3133, 4.0181])
central_diff(softplus, xs).round(4)     # array([0.018 , 0.2689, 0.5   , 0.7311, 0.982 ])
sigmoid(xs).round(4)                    # array([0.018 , 0.2689, 0.5   , 0.7311, 0.982 ])  identical
```

!!! warning "Common mistake: overflow in eˣ"
    `np.exp(800)` is too big for a float64, so the formula above breaks for large inputs. This block prints a `RuntimeWarning: overflow` on purpose:

    ```python
    softplus(np.array([800.]))          # array([inf])   plus an overflow RuntimeWarning
    np.logaddexp(0, np.array([800.]))   # array([800.])  ln(e⁰ + e⁸⁰⁰), computed safely
    ```

    Real libraries rewrite these formulas so that \( e^x \) never sees a huge \( x \). The sigmoid has the same issue for very negative inputs (it still returns the right answer, 0, but warns). When you write activations yourself, use the stable forms (`np.logaddexp`); when you use PyTorch's, it already has.

## The reference table

| Function | Derivative | Check at |
|---|---|---|
| \( c \) (constant) | \( 0 \) | any \( x \) |
| \( x^n \) | \( n x^{n-1} \) | \( x = 2 \) |
| \( c\,f(x) \) | \( c\,f'(x) \) | |
| \( f(x) + g(x) \) | \( f'(x) + g'(x) \) | |
| \( u(x)\,v(x) \) | \( u'v + uv' \) | |
| \( e^x \) | \( e^x \) | any \( x \) |
| \( e^{-x} \) | \( -e^{-x} \) | any \( x \) |
| \( \ln x \) | \( 1/x \) | \( x > 0 \) |
| \( \sigma(x) = 1/(1 + e^{-x}) \) | \( \sigma(x)(1 - \sigma(x)) \), at most 0.25 | any \( x \) |
| \( \tanh x \) | \( 1 - \tanh^2 x \), at most 1 | any \( x \) |
| \( \text{ReLU}(x) = \max(0, x) \) | 0 for \( x < 0 \), 1 for \( x > 0 \) (0 at \( x = 0 \) by convention) | away from 0 |
| \( \text{softplus}(x) = \ln(1 + e^x) \) | \( \sigma(x) \) | any \( x \) |

"Trust, but verify": here is every building block in the table checked on 200 points at once. The grid deliberately avoids \( x = 0 \) (ReLU's kink) and uses only positive \( x \) for \( \ln \).

```python
grid = np.linspace(-3, 3, 200)          # 200 points; 0 is not one of them
pos = np.linspace(0.1, 5, 200)          # ln needs x > 0

checks = [
    ("x^3",      lambda x: x**3,         lambda x: 3 * x**2,       grid),
    ("1/x",      lambda x: 1 / x,        lambda x: -1 / x**2,      pos),
    ("sqrt",     np.sqrt,                lambda x: 0.5 / np.sqrt(x), pos),
    ("exp",      np.exp,                 np.exp,                   grid),
    ("exp(-x)",  lambda x: np.exp(-x),   lambda x: -np.exp(-x),    grid),
    ("ln",       np.log,                 lambda x: 1 / x,          pos),
    ("x^2*e^x",  lambda x: x**2 * np.exp(x), lambda x: 2*x*np.exp(x) + x**2*np.exp(x), grid),
    ("sigmoid",  sigmoid,                sigmoid_prime,            grid),
    ("tanh",     np.tanh,                tanh_prime,               grid),
    ("relu",     relu,                   relu_prime,               grid),
    ("softplus", softplus,               sigmoid,                  grid),
]
for name, f, df, pts in checks:
    err = np.max(np.abs(df(pts) - central_diff(f, pts)))
    print(f"{name:9s} max error {err:.1e}")
```

```text
x^3       max error 3.7e-10
1/x       max error 1.0e-06
sqrt      max error 2.0e-09
exp       max error 5.5e-10
exp(-x)   max error 5.5e-10
ln        max error 3.3e-08
x^2*e^x   max error 1.4e-08
sigmoid   max error 1.1e-11
tanh      max error 3.3e-11
relu      max error 6.6e-12
softplus  max error 2.2e-11
```

(The exact digits can differ slightly between machines.) Every error is tiny compared with the slopes themselves; the largest, for \( 1/x \), comes from \( x = 0.1 \), where the slope is \( -100 \) and the curve bends most sharply: the central difference's leftover error (Day 29's truncation error) grows with how fast the curve bends.

## Reading a function next to its derivative

The derivative is a new function: at every \( x \) it tells you the slope of the original. Plotting the two together builds the instinct you'll use for the rest of the course. Start with \( f(x) = x^3 - 3x \), whose derivative by the rules is \( f'(x) = 3x^2 - 3 \):

```python
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.facecolor": "white", "savefig.facecolor": "white"})

x = np.linspace(-2.3, 2.3, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, x**3 - 3*x, color="#3949ab", lw=2.5, label="f(x) = x³ − 3x")
ax.plot(x, 3*x**2 - 3, color="#00897b", lw=2.5, ls="--", label="f′(x) = 3x² − 3")
ax.axhline(0, color="#90a4ae", lw=1)
for x0 in [-1, 1]:                                  # where f′(x) = 0
    ax.plot([x0, x0], [0, x0**3 - 3*x0], color="#ef6c00", lw=1.5, ls=":")
    ax.plot(x0, x0**3 - 3*x0, "o", color="#ef6c00", ms=8)
    ax.plot(x0, 0, "o", color="#00897b", ms=6)
ax.annotate("peak: slope 0", xy=(-1, 2), xytext=(-2.2, 3.4), color="#ef6c00",
            arrowprops=dict(arrowstyle="->", color="#ef6c00"))
ax.annotate("valley: slope 0", xy=(1, -2), xytext=(1.1, -4.2), color="#ef6c00",
            arrowprops=dict(arrowstyle="->", color="#ef6c00"))
ax.set_ylim(-5, 5)
ax.set_xlabel("x")
ax.legend(loc="upper center")
ax.grid(alpha=0.3)
ax.set_title("Where f′ crosses zero, f is flat")
fig.savefig("day30-cubic.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Line plot of f(x) = x cubed minus 3x as a solid indigo S-shaped curve with a peak at x = -1 and a valley at x = 1, and its derivative 3x squared minus 3 as a dashed teal parabola; the parabola crosses zero exactly at x = -1 and x = 1, marked with orange dots and dotted lines up to the peak and the valley](../../assets/img/phase-1/day30-cubic.svg)
<figcaption>The derivative (dashed) is positive where f climbs, negative where it falls, and zero at the peak and the valley.</figcaption>
</figure>

Three things to read off it:

- Where the dashed curve is **above zero** (\( x < -1 \) and \( x > 1 \)), \( f \) is climbing. Where it is **below zero**, \( f \) is falling.
- Where it **crosses zero** (\( 3x^2 - 3 = 0 \), so \( x = \pm 1 \)), \( f \) is momentarily flat: a peak or a valley. Training a model is a hunt for the valley of the loss, and "slope = 0" is how you recognize one.
- The **size** of the derivative is how steep \( f \) is. At \( x = 2.3 \) it is \( 3 \cdot 5.29 - 3 = 12.87 \): the curve is shooting up.

Now the four activations, each with its derivative:

```python
z = np.linspace(-5, 5, 500)
panels = [("sigmoid", sigmoid(z), sigmoid_prime(z)),
          ("tanh", np.tanh(z), tanh_prime(z)),
          ("ReLU", relu(z), relu_prime(z)),
          ("softplus", softplus(z), sigmoid(z))]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharex=True)
for ax, (name, f_vals, d_vals) in zip(axes, panels):
    ax.plot(z, f_vals, color="#3949ab", lw=2.5, label=name)
    ax.plot(z, d_vals, color="#00897b", lw=2.5, ls="--", label="its derivative")
    ax.axhline(0, color="#90a4ae", lw=1)
    ax.set_title(name)
    ax.set_ylim(-1.15, 2.2)
    ax.grid(alpha=0.3)
    ax.set_xlabel("x")
axes[0].axhline(0.25, color="#e53935", lw=1, ls=":")
axes[0].text(-4.8, 0.5, "max slope 0.25", color="#e53935", fontsize=10)
axes[0].legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig("day30-activations.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Four side-by-side plots from x = -5 to 5, each showing an activation as a solid indigo line and its derivative as a dashed teal line: the sigmoid S-curve from 0 to 1 with a low bump of height 0.25 for its derivative, marked by a red dotted line; the tanh S-curve from -1 to 1 with a taller bump of height 1; ReLU, flat at 0 then rising at 45 degrees, with a step derivative from 0 to 1; and softplus, a smoothed ReLU, whose derivative is a sigmoid S-curve](../../assets/img/phase-1/day30-activations.svg)
<figcaption>Solid: the activation. Dashed: its slope. The sigmoid's slope never rises above 0.25; ReLU's is exactly 1 for every positive input.</figcaption>
</figure>

And the same story in motion: a point sliding along each curve, with its tangent line, and the tangent's slope drawn underneath.

<figure class="diagram" markdown>
![Animation in two columns: on the left a point slides back and forth along the sigmoid S-curve with an orange tangent line, while below a teal dot traces the sigmoid's derivative, a low bump that never rises above a red dashed line at 0.25; on the right a point slides along ReLU, whose tangent is flat left of zero and slopes upward right of zero, while below a teal dot jumps between 0 and 1 on a step-shaped derivative, with a note that frameworks use 0 at the kink](../../assets/img/phase-1/day30-slope-tracer.svg)
<figcaption>Animated: the tangent's slope (orange) becomes the height of the derivative curve (teal). Both bottom panels use the same 0 to 1 scale.</figcaption>
</figure>

## Why the 0.25 matters: a vanishing-gradient preview

Tomorrow's **chain rule** says that when functions are stacked (a deep network is a stack of layers), their slopes **multiply** along the way. When the network learns, it sends an error signal backward through every layer, and each sigmoid it passes through multiplies the signal by its slope, which is at most 0.25:

```python
for layers in [1, 2, 5, 10, 20]:
    print(f"{layers:2d} layers: at most {0.25**layers:.2e}")
#  1 layers: at most 2.50e-01
#  2 layers: at most 6.25e-02
#  5 layers: at most 9.77e-04
# 10 layers: at most 9.54e-07
# 20 layers: at most 9.09e-13
```

That's the **best** case, with every input sitting exactly at 0. With realistic inputs it's worse. Here are ten random layer inputs, and what each activation's slope does to a signal that passes through all ten:

```python
rng = np.random.default_rng(30)
z10 = rng.normal(0, 1, size=10)          # one input per layer
z10.round(2)
# array([ 1.57,  0.74, -0.97, -0.21, -0.29,  2.36, -0.94,  1.38,  0.12,
#         1.02])
for name, d in [("sigmoid", sigmoid_prime), ("tanh", tanh_prime), ("relu", relu_prime)]:
    print(f"{name:8s} {d(z10).round(3)}  product {np.prod(d(z10)):.2e}")
# sigmoid  [0.142 0.218 0.199 0.247 0.245 0.079 0.202 0.161 0.249 0.195]  product 4.64e-08
# tanh     [0.158 0.602 0.44  0.956 0.922 0.035 0.457 0.226 0.985 0.406]  product 5.29e-05
# relu     [1. 1. 0. 0. 0. 1. 0. 1. 1. 1.]  product 0.00e+00
```

<figure class="diagram" markdown>
![Animation: an orange pulse travels from the output on the right to the input on the left through five layer boxes; in the sigmoid row each box multiplies by 0.25 and the bars showing the gradient size shrink from 1 to 0.25, 0.0625, 0.0156, 0.0039 and 0.00098, almost invisible; in the ReLU row each box multiplies by 1 and every bar stays at full height 1](../../assets/img/phase-1/day30-vanishing.svg)
<figcaption>Animated: in the backward pass each layer multiplies the gradient by its slope. Sigmoids shrink it ×0.25 or worse per layer; an active ReLU passes it on untouched.</figcaption>
</figure>

Read the three lines:

- **Sigmoid:** after ten layers the signal is about \( 5 \times 10^{-8} \) of what it was. The early layers get almost no push and barely learn. This is the **vanishing gradient problem**, and it is a big reason deep networks were so hard to train before about 2010.
- **tanh:** a peak slope of 1 instead of 0.25 makes it about 1,000 times better here, but saturated units (like the 0.035 at \( z = 2.36 \)) still choke the signal.
- **ReLU:** every factor is exactly 1 or exactly 0. Along this single path one 0 blocks everything; but a real network has many paths through many units (Day 31 shows that path contributions **add**), and along the active ones the signal is passed on with no shrinking at all. That property is a big part of why ReLU became the default.

This is a simplified picture: real backward passes also multiply by the weights, and they can make things better or worse (on Day 24 you saw repeated multiplication by a matrix explode or fade). Phase 3 covers the full story and the fixes (careful initialization, normalization layers, residual connections).

!!! android "Android bridge: animation interpolators are functions you can differentiate"
    An Android `Interpolator` maps animation progress \( t \in [0, 1] \) to eased progress, and its **derivative is the animation's speed**. The docs for `AccelerateInterpolator` say a factor of `1.0f` "produces a y=x^2 parabola", and `DecelerateInterpolator(1.0f)` an upside-down one, \( 1 - (1 - t)^2 = 2t - t^2 \). With today's rules:

    - Accelerate: \( (t^2)' = 2t \). Speed 0 at the start (it eases in), and 2 at the end: twice the average speed.
    - Decelerate: \( (2t - t^2)' = 2 - 2t \). Speed 2 at the start, 0 at the end (it eases out).

    Concretely, if a view slides 300 px in 300 ms (1 px/ms on average), an accelerate animation is moving at 2 px/ms when it stops, which is why it can look like it slams into place. If you write your own `TimeInterpolator`, `getInterpolation(input)` is \( f(t) \) and you can sanity-check its speed profile with exactly the `central_diff` above:

    ```kotlin
    // A custom ease-in-out: smoothstep, f(t) = 3t² − 2t³, so f'(t) = 6t − 6t²
    class SmoothStep : TimeInterpolator {
        override fun getInterpolation(input: Float): Float =
            input * input * (3f - 2f * input)
    }
    // f'(0) = 0 and f'(1) = 0: it starts and stops gently; top speed f'(0.5) = 1.5
    ```

    The same idea runs the other way in ML on device: when LiteRT executes a `LOGISTIC` (sigmoid), `TANH` or `RELU` op it only needs the function, but any on-device training or fine-tuning also needs exactly the derivatives in today's table.

---

## Exercises

1. By hand, differentiate \( f(x) = 4x^3 - \frac{2}{x} + 5\sqrt{x} \) (rewrite each term as a power first). Evaluate \( f'(4) \), then check with `central_diff`.
2. Use the product rule on \( g(x) = x^2 e^x \). Factor your answer and find the two values of \( x \) where the slope is 0. Check both numerically.
3. Derive \( \tanh'(x) = 1 - \tanh^2(x) \). Hint: write \( t(x) = \tanh(x) \), multiply both sides of the definition by \( e^x + e^{-x} \), and differentiate both sides with the product rule, as we did for the sigmoid.
4. For which \( x \) does the sigmoid's slope drop below 0.01? Solve \( s(1 - s) = 0.01 \) for \( s \) with the quadratic formula, turn \( s \) back into \( x \) using \( x = \ln\frac{s}{1 - s} \) (the inverse of the sigmoid), and confirm with a grid search over `np.linspace(0, 10, 100001)`.
5. **Leaky ReLU** is \( x \) for \( x > 0 \) and \( 0.01x \) for \( x \le 0 \). Write `leaky_relu` and `leaky_relu_prime` (use `np.where`), check them against `central_diff` at \( x = -2 \) and \( x = 3 \), and predict, before running it, what `central_diff` reports at exactly \( x = 0 \).
6. A model that ignores its input and always predicts the same number \( p \) has loss \( L(p) = \frac{1}{n}\sum_{i=1}^{n} (p - y_i)^2 \) on targets \( y_1, \dots, y_n \). Use the sum rule and the squared-error derivative to find \( L'(p) \), then find the \( p \) where \( L'(p) = 0 \). Check with `y = np.array([4.1, 3.8, 4.6, 4.5])`. What well-known statistic did you just derive?

??? success "Solutions"
    ```python
    # 1: f = 4x³ − 2x⁻¹ + 5x^(1/2)  →  f' = 12x² + 2x⁻² + (5/2)x^(-1/2)
    # f'(4) = 12·16 + 2/16 + 5/(2·2) = 192 + 0.125 + 1.25 = 193.375
    f1 = lambda x: 4*x**3 - 2/x + 5*np.sqrt(x)
    df1 = lambda x: 12*x**2 + 2/x**2 + 2.5/np.sqrt(x)
    df1(4.0), central_diff(f1, 4.0)
    # (np.float64(193.375), np.float64(193.37500000062843))

    # 2: g' = 2x·eˣ + x²·eˣ = eˣ·x·(x + 2). eˣ is never 0, so g' = 0 at x = 0 and x = −2
    g = lambda x: x**2 * np.exp(x)
    central_diff(g, 0.0), central_diff(g, -2.0)
    # (np.float64(1.0000000000085784e-10), np.float64(0.0))   both 0 up to rounding
    ```

    3: Multiply \( t = \frac{e^x - e^{-x}}{e^x + e^{-x}} \) by the bottom: \( t\,(e^x + e^{-x}) = e^x - e^{-x} \). Differentiate both sides, using \( (e^x)' = e^x \) and \( (e^{-x})' = -e^{-x} \):

    \[
    t'\,(e^x + e^{-x}) + t\,(e^x - e^{-x}) = e^x + e^{-x}
    \]

    Divide everything by \( e^x + e^{-x} \). The middle fraction \( \frac{e^x - e^{-x}}{e^x + e^{-x}} \) is \( t \) itself, so \( t' + t \cdot t = 1 \), that is \( t' = 1 - t^2 \). (The numerical check is the `tanh` line of the table check above.)

    ```python
    # 4: s − s² = 0.01  →  s² − s + 0.01 = 0  →  s = (1 ± √0.96) / 2. Take the + root (x > 0).
    s = (1 + np.sqrt(0.96)) / 2
    x_cut = np.log(s / (1 - s))
    s, x_cut                                     # (np.float64(0.9898979485566356), np.float64(4.584863339122353))
    xg = np.linspace(0, 10, 100001)
    xg[np.argmax(sigmoid_prime(xg) < 0.01)]      # np.float64(4.5849)   first grid point below 0.01
    # by symmetry the slope is also below 0.01 for x < −4.585: |x| > 4.585 means "saturated"

    # 5
    def leaky_relu(x):
        return np.where(x > 0, x, 0.01 * x)

    def leaky_relu_prime(x):
        return np.where(x > 0, 1.0, 0.01)

    pts5 = np.array([-2.0, 3.0])
    leaky_relu_prime(pts5), central_diff(leaky_relu, pts5)
    # (array([0.01, 1.  ]), array([0.01, 1.  ]))
    central_diff(leaky_relu, 0.0)                # np.float64(0.505)   the average of the two sides: (1 + 0.01) / 2
    ```

    6: each term \( (p - y_i)^2 \) has derivative \( 2(p - y_i) \); the sum and constant multiple rules give

    \[
    L'(p) = \frac{1}{n}\sum_{i=1}^{n} 2(p - y_i) = 2\Big(p - \frac{1}{n}\sum_{i=1}^{n} y_i\Big)
    \]

    which is 0 exactly when \( p \) equals the **mean** of the targets. Under squared error, the best constant prediction is the average. (That is why "predict the mean" is the standard baseline in Month 2's milestone.)

    ```python
    y6 = np.array([4.1, 3.8, 4.6, 4.5])
    L = lambda p: np.mean((p - y6)**2)
    y6.mean()                                    # np.float64(4.25)
    central_diff(L, y6.mean())                   # np.float64(-6.938893903907227e-13)   0 up to rounding: flat at the mean
    central_diff(L, 4.0), 2 * (4.0 - y6.mean())
    # (np.float64(-0.4999999999921733), np.float64(-0.5))   below the mean the slope is negative: move up
    ```

## Resources

- [3Blue1Brown: Derivative formulas through geometry (Essence of Calculus, ch. 3)](https://www.youtube.com/watch?v=S0_qX4VJhMQ): the power rule and the product rule as growing squares and rectangles, exactly today's pictures.
- [3Blue1Brown: What's so special about Euler's number e? (Essence of Calculus, ch. 5)](https://www.youtube.com/watch?v=m2MIpDrF7Es): why \( e^x \) is its own derivative, visually.
- [Khan Academy: Differential Calculus](https://www.khanacademy.org/math/differential-calculus): the "Derivatives: definition and basic rules" unit has short practice sets for every rule in the table.
- [Dive into Deep Learning: Calculus](https://d2l.ai/chapter_preliminaries/calculus.html): the same rules, with plotting code, from an ML textbook.
- [Mathematics for Machine Learning (free book), ch. 5 "Vector Calculus"](https://mml-book.github.io/): section 5.1 covers derivatives, the rules and Taylor series, formally.
- [PyTorch docs: `torch.nn.ReLU`](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) and [`torch.nn.Softplus`](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html): the activation definitions as used in practice (Softplus's page shows its numerically stable threshold).
- [Android reference: `AccelerateInterpolator`](https://developer.android.com/reference/android/view/animation/AccelerateInterpolator): the "y=x^2 parabola" from the Android bridge.

## Checklist

- [ ] I can differentiate a polynomial term by term with the constant, power, constant multiple and sum rules.
- [ ] I can derive \( \frac{d}{dp}(p - y)^2 = 2(p - y) \) and say which way it tells the prediction to move.
- [ ] I can state the product rule and explain it with the growing rectangle, and I don't multiply derivatives.
- [ ] I know \( (e^x)' = e^x \), \( (e^{-x})' = -e^{-x} \) and \( (\ln x)' = 1/x \).
- [ ] I can derive \( \sigma' = \sigma(1 - \sigma) \) and know the derivatives of tanh, ReLU (and the convention at 0) and softplus.
- [ ] I check derivatives numerically with a central difference, away from kinks.
- [ ] I can explain, with the number 0.25, why deep stacks of sigmoids suffer from vanishing gradients.

**Next:** [Day 31: The chain rule and computational graphs](day-31-chain-rule.md)
