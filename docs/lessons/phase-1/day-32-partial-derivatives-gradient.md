# Day 32: Partial Derivatives and the Gradient

**Phase 1 · Week 5 · ~120 minutes**

## Objectives

- [ ] Picture a function of two variables as a surface and read it as a contour map.
- [ ] Compute a partial derivative by hand by freezing every variable except one, and check it numerically.
- [ ] Assemble the partial derivatives into the gradient vector \( \nabla f \), and write a `num_grad` function that works in any number of dimensions.
- [ ] Compute a directional derivative as \( \nabla f \cdot \mathbf{u} \), and use it to explain why the gradient points uphill and is perpendicular to contour lines.
- [ ] Derive the gradient of the mean squared error of a line fit \( \hat{y} = wx + b \), and check it numerically.
- [ ] Take a few steps downhill on that loss by hand, and say why this is how models learn.

**Prerequisites.** [Day 5](day-05-vectors.md) (dot product and the cosine of the angle), [Day 9](day-09-broadcasting.md) (broadcasting), [Day 19](day-19-matplotlib.md) (matplotlib), [Day 22](day-22-norms-distances.md) (length and unit vectors), [Day 25](day-25-projections-pca.md) (the least-squares line through four points), and this week's [Day 29](day-29-derivatives.md) (the derivative and the central difference), [Day 30](day-30-derivative-rules.md) (derivative rules) and [Day 31](day-31-chain-rule.md) (the chain rule).

---

## The big idea

So far this week every function had one input: nudge \( x \), watch \( f(x) \) move, and the derivative told you how much. A model's loss has many inputs. A line fit has two parameters (slope and intercept); a small neural network has thousands; a phone-sized language model has billions. Training means: **find the settings of all those knobs that make the loss small.**

**Analogy.** You are on a hillside in thick fog. You can't see the valley, but you can feel the ground under your boots. Take a small step east: does the ground rise or fall, and how steeply? Take a small step north: same question. Those two answers are **partial derivatives**, one per direction you can move. Put them together as an arrow and you get the **gradient**: it points in the direction where the ground rises fastest, and its length says how steep that is. To get down to the valley, step the opposite way, feel again, repeat.

That fog walk is, almost word for word, how every neural network is trained. The "ground" is the loss, the "position" is the list of parameters, and the "feel the slope" step is computing the gradient. Today you learn the gradient for two variables, where you can still draw everything; Day 33 makes it work for whole vectors and matrices.

| Idea | Hillside | Model training |
|---|---|---|
| Function value | Height of the ground | Loss |
| Inputs | Your east and north position | The parameters (weights) |
| Partial derivative | Slope if you step only east (or only north) | How the loss changes if you nudge one weight |
| Gradient | Arrow pointing straight uphill | The vector of all those nudges |
| Step against the gradient | Walk downhill | One training update |

## Functions of two variables: surfaces and contour maps

A function of two variables takes two numbers and gives back one. Here is the one we'll use most today:

\[
f(x, y) = x^2 + 3y^2
\]

Read it as a recipe: square \( x \), square \( y \) and triple it, add. At the point \( (1, 1) \): \( f(1, 1) = 1 + 3 = 4 \). At \( (2, 0) \): \( 4 + 0 = 4 \). At \( (0, 2) \): \( 0 + 12 = 12 \).

```python
import numpy as np

def f(x, y):
    return x**2 + 3 * y**2

f(1, 1)                        # 4
f(2, 0), f(0, 2), f(-1, -1)    # (4, 12, 4)
```

If you think of \( (x, y) \) as a position on a floor and \( f(x, y) \) as a height above it, the function is a **surface**: a landscape. To draw it you need its height at many points. `np.meshgrid` builds a grid of every \( (x, y) \) combination, and because `f` is made of NumPy operations it evaluates the whole grid at once (Day 4's vectorization):

```python
xs = np.array([-2., -1., 0., 1., 2.])
ys = np.array([-1., 0., 1.])
XX, YY = np.meshgrid(xs, ys)
XX.shape                 # (3, 5): one ROW per y value, one COLUMN per x value
print(XX)
# [[-2. -1.  0.  1.  2.]
#  [-2. -1.  0.  1.  2.]
#  [-2. -1.  0.  1.  2.]]
print(YY)
# [[-1. -1. -1. -1. -1.]
#  [ 0.  0.  0.  0.  0.]
#  [ 1.  1.  1.  1.  1.]]
print(f(XX, YY))         # the height at all 15 grid points
# [[7. 4. 3. 4. 7.]
#  [4. 1. 0. 1. 4.]
#  [7. 4. 3. 4. 7.]]
```

The lowest value, 0, sits at \( (0, 0) \) in the middle of the grid, and the heights rise in every direction: \( f \) is a bowl.

A bowl is a little dull to look at, so for pictures we'll also use a landscape with two hills. Below, the same `hill` function is drawn twice: as a 3-D surface, and as a **contour map**, the view from directly above with a line drawn through all the points at the same height, exactly like the contour lines on a hiking map.

```python
import matplotlib
matplotlib.use("Agg")            # draw into files, no window (leave this out in Jupyter)
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

def hill(x, y):
    """Two smooth hills: a tall one near (1, 0.5) and a lower one near (-1.5, -0.8)."""
    return 3 * np.exp(-((x - 1)**2 + (y - 0.5)**2)) + 2 * np.exp(-((x + 1.5)**2 + (y + 0.8)**2) / 0.6)

gx = np.linspace(-3, 3, 121)
GX, GY = np.meshgrid(gx, gx)
H = hill(GX, GY)
H.shape              # (121, 121)
H.max().round(3)     # np.float64(3.0)   the top of the tall hill

fig = plt.figure(figsize=(11, 4.6))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(GX, GY, H, cmap="viridis", rstride=3, cstride=3, linewidth=0.2, edgecolor="#333", alpha=0.95)
ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("height")
ax1.set_title("The surface z = hill(x, y)")
ax1.view_init(elev=32, azim=-60)

ax2 = fig.add_subplot(1, 2, 2)
cs = ax2.contour(GX, GY, H, levels=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9], cmap="viridis")
ax2.clabel(cs, fontsize=9)                       # write the height on each line
ax2.plot([1, -1.5], [0.5, -0.8], "^", color="#e53935", ms=8)   # the two summits
ax2.set_aspect("equal")
ax2.set_xlabel("x"); ax2.set_ylabel("y")
ax2.set_title("The same hills as a contour map (from above)")
ax2.grid(alpha=0.3)
fig.savefig("day32-surface-contour.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Left: a 3-D surface plot over x and y from -3 to 3 showing a flat purple floor with two smooth bumps, a tall yellow peak of height 3 and a lower green one of height 2; right: the same function as a contour map, with nested closed rings labelled 0.25 to 2.9 around each of two red triangle summits at (1, 0.5) and (-1.5, -0.8), the outer 0.25 and 0.5 rings joining the two hills like a figure of eight](../../assets/img/phase-1/day32-surface-contour.svg)
<figcaption>One function, two views. Each contour line joins points of equal height; where the lines bunch together the ground is steep.</figcaption>
</figure>

How to read a contour map:

- **Each line is one height.** Walk along a line and you neither climb nor descend.
- **Closely spaced lines mean steep ground**; widely spaced lines mean gentle ground. Around the tall hill the rings are packed tighter than around the low one.
- **Small closed rings** surround a peak (or, for a loss, a valley floor). For a loss function, the place you want to reach is the middle of the innermost ring.

Contour maps are how you'll look at loss functions from now on: a 3-D surface hides its own back, a contour map hides nothing, and it's what training-visualization tools draw.

!!! warning "Common mistake: mixing up meshgrid's axes"
    `np.meshgrid(xs, ys)` returns arrays of shape `(len(ys), len(xs))`: **rows follow y, columns follow x**, like an image (row = vertical position). So `H[i, j]` is the height at `x = gx[j]`, `y = gx[i]`, not the other way round. With a square grid both shapes are the same and the bug hides until you change one range. Print `XX[:2, :3]` and `YY[:2, :3]` when in doubt.

## Partial derivatives: freeze everything but one

On Day 29 the derivative answered "if I nudge the input, how fast does the output move?". With two inputs there are two different nudges, so there are two answers.

The trick to get each one: **freeze the other variable at a fixed number**, and what is left is an ordinary one-variable function, which you already know how to differentiate.

Stand at \( P = (1, 1) \) on \( f(x, y) = x^2 + 3y^2 \).

- **Freeze \( y = 1 \)** and only move along \( x \): \( f(x, 1) = x^2 + 3 \). Its derivative is \( 2x \) (Day 30's power rule; the \( + 3 \) is a constant and contributes nothing). At \( x = 1 \) that is **2**.
- **Freeze \( x = 1 \)** and only move along \( y \): \( f(1, y) = 1 + 3y^2 \). Its derivative is \( 6y \). At \( y = 1 \) that is **6**.

<figure class="diagram" markdown>
![Animation: on the left a contour map of f = x squared plus 3 y squared with ellipses for f = 1, 4 and 7 and a red point P = (1, 1); first an orange dashed horizontal line y = 1 is drawn through P while the top-right orange panel shows the curve f(x, 1) = x squared + 3 with an orange tangent of slope 2 at x = 1; then a teal dashed vertical line x = 1 appears while the bottom-right teal panel shows f(1, y) = 1 + 3 y squared with a tangent of slope 6 at y = 1; finally a red arrow labelled gradient of f = [2, 6] grows out of P at a right angle to the dashed tangent of the ellipse through P](../../assets/img/phase-1/day32-partial-slices.svg)
<figcaption>Animated: each partial derivative is the slope of a 1-D slice through the surface; stacking the two slopes gives the gradient, which crosses the contour at a right angle.</figcaption>
</figure>

### The notation, one symbol at a time

\[
\frac{\partial f}{\partial x} \qquad \frac{\partial f}{\partial y}
\]

- \( \partial \) is a curly "d", read "partial". It plays the role of Day 29's \( d \) in \( \frac{df}{dx} \), and the different letter is a warning: *other variables exist and are being held fixed*.
- \( \frac{\partial f}{\partial x} \) is read "partial f partial x" (or "the partial derivative of f with respect to x"). It is the slope of \( f \) when you move only in the \( x \) direction.
- Books also write \( f_x \) for the same thing. To say where it is measured, write \( \frac{\partial f}{\partial x}(1, 1) = 2 \).

As a limit it is exactly Day 29's derivative with \( y \) left alone:

\[
\frac{\partial f}{\partial x}(x, y) = \lim_{h \to 0} \frac{f(x + h, y) - f(x, y)}{h}
\]

For \( f = x^2 + 3y^2 \), treating \( y \) as a constant everywhere:

\[
\frac{\partial f}{\partial x} = 2x
\qquad
\frac{\partial f}{\partial y} = 6y
\]

### A worked example where the variables mix

The bowl was easy because \( x \) and \( y \) live in separate terms. Try one where they multiply:

\[
g(x, y) = x^2 y + 3y
\]

**\( \partial g / \partial x \): treat \( y \) as a number.** If it helps, pretend \( y = 5 \): then \( g = 5x^2 + 15 \), whose derivative is \( 10x = 2x \cdot 5 \). So in general \( \frac{\partial g}{\partial x} = 2xy \). The \( 3y \) term has no \( x \) in it, so it is a constant and vanishes.

**\( \partial g / \partial y \): treat \( x \) as a number.** Now \( x^2 \) is just a coefficient: \( g = (x^2 + 3)\,y \), a straight line in \( y \) with slope \( x^2 + 3 \). So \( \frac{\partial g}{\partial y} = x^2 + 3 \).

At the point \( (2, 1) \): \( \frac{\partial g}{\partial x} = 2 \cdot 2 \cdot 1 = 4 \) and \( \frac{\partial g}{\partial y} = 2^2 + 3 = 7 \).

Check both with Day 29's central difference, nudging one variable at a time:

```python
def g(x, y):
    return x**2 * y + 3 * y

h = 1e-5
x0, y0 = 2.0, 1.0
dg_dx = (g(x0 + h, y0) - g(x0 - h, y0)) / (2 * h)   # nudge x, freeze y
dg_dy = (g(x0, y0 + h) - g(x0, y0 - h)) / (2 * h)   # nudge y, freeze x
dg_dx, dg_dy     # (4.000000000026205, 7.000000000001449)
```

Both match the hand answers to about 10 decimal places, the accuracy Day 29 led you to expect from a central difference with \( h = 10^{-5} \).

!!! warning "Common mistake: treating the frozen variable as zero"
    "Freeze \( y \)" means "treat \( y \) as a constant", **not** "set \( y = 0 \)". For \( g = x^2 y \), \( \frac{\partial g}{\partial x} = 2xy \): the \( y \) stays in the answer as a multiplier. Writing \( 2x \) (dropping \( y \)) or \( 0 \) (setting it to zero) are the two classic slips. A numerical check at a point where \( y \neq 0, 1 \) catches both instantly.

## The gradient: all the partials in one vector

Stack the partial derivatives into a vector and you get the **gradient**:

\[
\nabla f(x, y) = \begin{bmatrix} \dfrac{\partial f}{\partial x} \\[2mm] \dfrac{\partial f}{\partial y} \end{bmatrix}
\]

- \( \nabla \) is read "nabla" or "del", or just "grad": \( \nabla f \) is "grad f".
- It is a **vector with one entry per input**. A function of 2 variables has a 2-entry gradient; a loss with a million weights has a million-entry gradient.
- It is different at every point. It is a vector *field*: an arrow attached to every spot on the map.

For the bowl, \( \nabla f(x, y) = [2x,\ 6y] \), so \( \nabla f(1, 1) = [2, 6] \). For \( g \), \( \nabla g(2, 1) = [4, 7] \).

The numerical version generalizes neatly to any number of inputs: put the point in a vector `p`, and nudge one coordinate at a time.

```python
def num_grad(F, p, h=1e-5):
    """Numerical gradient of F at point p, one central difference per coordinate."""
    p = np.asarray(p, dtype=float)
    grad = np.zeros_like(p)
    for i in range(p.size):
        step = np.zeros_like(p)
        step[i] = h                     # nudge only coordinate i ...
        grad[i] = (F(p + step) - F(p - step)) / (2 * h)   # ... and freeze the others
    return grad

def grad_f(x, y):
    """The gradient of f = x² + 3y², worked out by hand."""
    return np.array([2 * x, 6 * y])

grad_f(1, 1)                                  # array([2, 6])
num_grad(lambda p: f(p[0], p[1]), [1, 1])     # array([2., 6.])
num_grad(lambda p: g(p[0], p[1]), [2, 1])     # array([4., 7.])
num_grad(lambda p: f(p[0], p[1]), [0, 0])     # array([0., 0.])   flat at the bottom of the bowl
```

`num_grad` wants a function of **one vector**, so the `lambda p: f(p[0], p[1])` unpacks it into the two numbers `f` expects. The last line is worth noticing: at the bottom of the bowl the gradient is zero. **A zero gradient means a flat spot**, and at the bottom of a valley the ground is flat. That is how you'll recognize a trained model: the loss gradient is (close to) zero.

Day 33 grows this same idea into the gradient checker you'll use for the rest of the course. Each entry costs two function evaluations, so for a million parameters it costs two million loss evaluations: fine for testing, far too slow for training. That's why backpropagation (Day 31) matters: it gets the whole gradient for about the cost of one extra pass.

## Which way is uphill? The directional derivative

The two partial derivatives are the slopes going east (\( +x \)) and north (\( +y \)). But you can walk in any direction. What is the slope if you head north-east, or at 37°?

Describe a direction by a **unit vector** \( \mathbf{u} = [u_1, u_2] \) (length 1, Day 22). The slope of \( f \) in that direction, the **directional derivative**, is

\[
D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = \frac{\partial f}{\partial x}\, u_1 + \frac{\partial f}{\partial y}\, u_2
\]

**Why that formula?** Zoom in far enough on a smooth surface and it looks like a flat, tilted plane, just as a curve looks like its tangent line when you zoom in (Day 29). On a plane, the effects of moving in \( x \) and moving in \( y \) simply add: a step \( \Delta x \) changes the height by about \( \frac{\partial f}{\partial x} \Delta x \), and a step \( \Delta y \) by about \( \frac{\partial f}{\partial y} \Delta y \). So

\[
f(x + \Delta x,\ y + \Delta y) \approx f(x, y) + \frac{\partial f}{\partial x}\,\Delta x + \frac{\partial f}{\partial y}\,\Delta y = f(x, y) + \nabla f \cdot \Delta
\]

with \( \Delta = [\Delta x, \Delta y] \). A step of size \( t \) along \( \mathbf{u} \) is \( \Delta = t\,\mathbf{u} \), so the height changes by about \( t\,(\nabla f \cdot \mathbf{u}) \): the slope is \( \nabla f \cdot \mathbf{u} \).

**Worked example.** At \( P = (1, 1) \), \( \nabla f = [2, 6] \). Head in the direction \( \mathbf{u} = [0.6, 0.8] \) (a unit vector, since \( 0.36 + 0.64 = 1 \)):

\[
\nabla f \cdot \mathbf{u} = 2 \cdot 0.6 + 6 \cdot 0.8 = 1.2 + 4.8 = 6.0
\]

And the "tilted plane" estimate: from \( (1, 1) \) step to \( (1.01, 1.02) \). The estimate says \( 4 + 2 \cdot 0.01 + 6 \cdot 0.02 = 4.14 \); the true value is \( 1.0201 + 3 \cdot 1.0404 = 4.1413 \). Close, and closer still for smaller steps.

```python
p = np.array([1., 1.])
grad = grad_f(*p)                 # array([2, 6])
u = np.array([0.6, 0.8])
np.linalg.norm(u)                 # np.float64(1.0)   a unit vector
grad @ u                          # np.float64(6.0)   the formula
t = 1e-5
(f(*(p + t * u)) - f(*(p - t * u))) / (2 * t)   # np.float64(5.999999999994897)   nudge along u
f(1.01, 1.02), 4 + grad @ np.array([0.01, 0.02])  # (4.1413, 4.14)   the tilted-plane estimate
```

(`f(*p)` unpacks the vector into `f(p[0], p[1])`.)

### The three facts that follow

Now use Day 5's other way to write a dot product: \( \mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\,\|\mathbf{b}\| \cos\theta \), where \( \theta \) (theta) is the angle between them. Since \( \|\mathbf{u}\| = 1 \),

\[
D_{\mathbf{u}} f = \|\nabla f\| \cos\theta
\]

where \( \theta \) is the angle between your walking direction and the gradient. Cosine is at most 1 (at 0°), exactly 0 at 90°, and at least −1 (at 180°). So:

1. **The gradient points straight uphill.** The slope is largest, \( \|\nabla f\| \), when you walk along \( \nabla f \) (\( \theta = 0 \)). Its length is the steepness of the steepest route.
2. **Minus the gradient points straight downhill.** The slope is most negative, \( -\|\nabla f\| \), at \( \theta = 180° \). This is the direction training steps in.
3. **The gradient is perpendicular to the contour line.** Walking along a contour keeps the height constant, so the slope that way is 0, which needs \( \cos\theta = 0 \): \( \theta = 90° \).

Check fact 1 by brute force, walking out of \( P \) in 12 directions:

```python
for deg in range(0, 360, 30):
    u = np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])
    print(f"{deg:4d} deg   slope = {grad @ u:7.3f}")
#    0 deg   slope =   2.000
#   30 deg   slope =   4.732
#   60 deg   slope =   6.196
#   90 deg   slope =   6.000
#  120 deg   slope =   4.196
#  150 deg   slope =   1.268
#  180 deg   slope =  -2.000
#  210 deg   slope =  -4.732
#  240 deg   slope =  -6.196
#  270 deg   slope =  -6.000
#  300 deg   slope =  -4.196
#  330 deg   slope =  -1.268
np.linalg.norm(grad)                                   # np.float64(6.324555320336759)
np.rad2deg(np.arctan2(grad[1], grad[0])).round(2)      # np.float64(71.57)
```

The steepest of the 12 is 6.196 at 60°, just under \( \|\nabla f\| = \sqrt{40} \approx 6.325 \), which you'd get by walking exactly at 71.57°, the gradient's own angle. At 0° the slope is 2 and at 90° it is 6: the two partial derivatives are just the directional derivatives along the axes. And every direction 180° away gives the same slope with the sign flipped.

Check fact 3. The contour through \( P \) is the ellipse \( x^2 + 3y^2 = 4 \); its tangent direction at \( (1, 1) \) is \( [-3, 1] \) (Exercise 5 asks you to confirm that this is a tangent). Walking that way, the slope is zero and the height barely changes, while the same small step along the gradient climbs:

```python
along = np.array([-3., 1.]) / np.sqrt(10)    # unit vector along the contour at P
grad @ along                                  # np.float64(1.1102230246251565e-16)   zero: perpendicular
f(*(p + 0.01 * along)) - f(*p)                # np.float64(0.00011999999999989797)   barely moves
f(*(p + 0.01 * grad / np.linalg.norm(grad))) - f(*p)   # np.float64(0.06352555320336783)   climbs
```

The 0.00012 isn't exactly zero only because the step is a straight line and the contour curves away from it; halve the step and it drops about four times (a second-order effect, as Day 29 would say).

### Seeing it: the gradient field of the hills

Here is the gradient at a grid of points on the two-hill landscape. `grad_grid` is the same central difference as `num_grad`, written to work on whole meshgrid arrays at once:

```python
def grad_grid(F, X, Y, h=1e-5):
    """Both partial derivatives of F(x, y) at every grid point, by central differences."""
    dF_dx = (F(X + h, Y) - F(X - h, Y)) / (2 * h)
    dF_dy = (F(X, Y + h) - F(X, Y - h)) / (2 * h)
    return dF_dx, dF_dy

qx = np.linspace(-2.75, 2.75, 12)            # a coarser grid for the arrows
QX, QY = np.meshgrid(qx, qx)
U, V = grad_grid(hill, QX, QY)
U.shape          # (12, 12)

fig, ax = plt.subplots(figsize=(6.4, 6.0))
cs = ax.contour(GX, GY, H, levels=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9], cmap="viridis")
ax.clabel(cs, fontsize=9)
ax.quiver(QX, QY, U, V, color="#e53935", angles="xy", scale_units="xy", scale=5, width=0.005)
ax.plot([1, -1.5], [0.5, -0.8], "^", color="#3949ab", ms=8)
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Red arrows: the gradient of hill(x, y)")
fig.savefig("day32-gradient-field.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

`ax.quiver(X, Y, U, V)` draws an arrow at each \( (X, Y) \) with components \( (U, V) \). `angles="xy", scale_units="xy"` make the arrows point and measure in data units, and `scale=5` shrinks each arrow to a fifth of the gradient so they don't overlap.

<figure class="diagram" markdown>
![Contour map of the two hills with a 12 by 12 grid of red arrows: on the slopes the arrows point inward toward the nearest summit, each crossing the contour rings at a right angle; arrows are longest where the rings are packed most tightly and shrink to dots far from the hills and right at the two summits](../../assets/img/phase-1/day32-gradient-field.svg)
<figcaption>Every arrow points toward higher ground and crosses the contours at a right angle. Long arrows sit where the contours are close together (steep); at the summits and on the flat plain they shrink to nothing.</figcaption>
</figure>

All three facts are visible: arrows point uphill, cross the contours at right angles, and are long where the lines bunch up. At each summit the gradient is zero, just as at the bottom of the bowl: a zero gradient means "flat", not necessarily "lowest".

!!! warning "Common mistake: a contour plot with unequal axis scales"
    "Perpendicular to the contours" only *looks* true if one unit of \( x \) is drawn as long as one unit of \( y \). Matplotlib's default stretches each axis to fill the box, which tilts every arrow off the right angle. Use `ax.set_aspect("equal")` whenever angles matter. And remember the directional derivative formula needs a **unit** vector: \( \nabla f \cdot [3, 4] \) is 5 times the slope in that direction, not the slope.

## Where this shows up in ML: the gradient of a loss

On Day 25 you fitted a line \( \hat{y} = wx + b \) to four points with `np.linalg.lstsq` and got slope 1.4 and intercept 0.5. Now the same problem, seen as a landscape.

The four points are \( (1, 2), (2, 3), (3, 5), (4, 6) \). For any slope \( w \) and intercept \( b \), the line predicts \( \hat{y}_i = w x_i + b \) for point \( i \), misses by the **error** \( e_i = w x_i + b - y_i \), and the **mean squared error** (MSE) averages the squared misses:

\[
L(w, b) = \frac{1}{n} \sum_{i=1}^{n} \left( w x_i + b - y_i \right)^2
\]

- \( n = 4 \) is the number of points, and \( \sum_{i=1}^{n} \) means "add up the following for \( i = 1, 2, \dots, n \)".
- The data \( x_i, y_i \) are **fixed numbers**. The variables are \( w \) and \( b \): the loss is a function of two variables, a surface over the \( (w, b) \) plane, exactly like \( f(x, y) \).

**Worked example.** At \( w = 0, b = 0 \) (the flat line \( \hat{y} = 0 \)) the errors are \( -2, -3, -5, -6 \), so \( L = (4 + 9 + 25 + 36)/4 = 18.5 \).

### Deriving the gradient

Each term is (something)², and the something depends on \( w \). That is Day 31's chain rule: the outside is "square it", with derivative \( 2e_i \); the inside \( e_i = w x_i + b - y_i \) has \( \frac{\partial e_i}{\partial w} = x_i \) (freeze \( b \): it is a line in \( w \) with slope \( x_i \)) and \( \frac{\partial e_i}{\partial b} = 1 \). Multiply, and average over the points:

\[
\frac{\partial L}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} e_i\, x_i
\qquad\qquad
\frac{\partial L}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} e_i
\]

**Worked example, at \( (w, b) = (0, 0) \).** The errors are \( [-2, -3, -5, -6] \).

- \( \sum e_i x_i = (-2)(1) + (-3)(2) + (-5)(3) + (-6)(4) = -2 - 6 - 15 - 24 = -47 \), so \( \frac{\partial L}{\partial w} = \frac{2}{4}(-47) = -23.5 \).
- \( \sum e_i = -16 \), so \( \frac{\partial L}{\partial b} = \frac{2}{4}(-16) = -8 \).

Reading the signs: both partials are negative, so **increasing** \( w \) or \( b \) would make the loss **smaller**. That makes sense: the flat line \( \hat{y} = 0 \) sits below every point, and tilting it up or lifting it helps. The \( w \) partial is about 3 times bigger, because changing the slope moves the predictions at \( x = 3 \) and \( x = 4 \) a lot.

```python
x_data = np.array([1., 2., 3., 4.])
y_data = np.array([2., 3., 5., 6.])

def mse(w, b):
    errors = w * x_data + b - y_data
    return np.mean(errors**2)

def mse_grad(w, b):
    """[dL/dw, dL/db], derived by hand above."""
    errors = w * x_data + b - y_data
    return np.array([2 * np.mean(errors * x_data), 2 * np.mean(errors)])

mse(0, 0)                                        # np.float64(18.5)
mse_grad(0, 0)                                   # array([-23.5,  -8. ])
num_grad(lambda p: mse(p[0], p[1]), [0, 0])      # array([-23.5,  -8. ])   the derivation checks out
mse(1.4, 0.5), mse_grad(1.4, 0.5)
# (np.float64(0.050000000000000065), array([-1.99840144e-15, -6.66133815e-16]))
```

At Day 25's best line \( (1.4, 0.5) \) the gradient is zero (to 15 decimal places): the bottom of the loss valley is flat, like the bottom of the bowl. That's no coincidence. Setting both partials to zero gives two equations in \( w \) and \( b \), and they are exactly Day 25's **normal equation**, \( X^\top X \hat{\mathbf{w}} = X^\top \mathbf{y} \). "Perpendicular residual" and "zero gradient" are two descriptions of the same best line. Day 33 writes this gradient in matrix form, \( \frac{2}{n} X^\top (X\mathbf{w} - \mathbf{y}) \).

!!! note "Why not just plot the landscape and look for the bottom?"
    With two parameters you can: evaluate the loss on a 200×200 grid (40,000 evaluations) and pick the smallest. With 10 parameters the same grid needs \( 200^{10} \approx 10^{23} \) evaluations; with a million, forget it. The gradient tells you which way is downhill **from where you are**, using only local information, and that works in any number of dimensions. That is the entire reason calculus is in an ML course.

## A first walk downhill

The gradient points uphill, so to reduce the loss, step the other way. With a small number \( \eta \) (eta, the **learning rate**) controlling the step size:

\[
\begin{bmatrix} w \\ b \end{bmatrix} \leftarrow \begin{bmatrix} w \\ b \end{bmatrix} - \eta\, \nabla L(w, b)
\]

The arrow \( \leftarrow \) means "replace with", like `w = w - lr * gw` in code. Here is the idea on the bowl first, where you can watch the point step across the contour lines:

<figure class="diagram" markdown>
![Animation: a contour map of f = x squared plus 3 y squared with five nested ellipses; an orange dot starts at (-4, 1.5); a red arrow labelled gradient appears pointing outward, perpendicular to the ellipse through the dot; then the dot jumps against it along a teal arrow to the next ellipse in; this repeats four times, the steps getting shorter as the dot nears the minimum; a table on the right fills in step by step: (-4.00, 1.50) height 22.75, (-3.20, 0.60) 11.32, (-2.56, 0.24) 6.73, (-2.05, 0.10) 4.22, (-1.64, 0.04) 2.69](../../assets/img/phase-1/day32-walk-downhill.svg)
<figcaption>Animated: at each point the gradient points uphill across the contour; a step of −0.1 × gradient lands on a lower contour, and the steps shrink as the ground flattens.</figcaption>
</figure>

Now do it on the line-fit loss, from the flat line \( (w, b) = (0, 0) \), with \( \eta = 0.05 \). By hand, the first step: \( w = 0 - 0.05 \cdot (-23.5) = 1.175 \), \( b = 0 - 0.05 \cdot (-8) = 0.4 \). The loss drops from 18.5 to 0.552. The code repeats it:

```python
lr = 0.05                      # the learning rate, eta
w, b = 0.0, 0.0
for step in range(6):
    gw, gb = mse_grad(w, b)
    print(f"step {step}: w = {w:6.3f}  b = {b:6.3f}  loss = {mse(w, b):7.3f}  grad = [{gw:7.3f}, {gb:7.3f}]")
    w, b = w - lr * gw, b - lr * gb     # one step downhill
# step 0: w =  0.000  b =  0.000  loss =  18.500  grad = [-23.500,  -8.000]
# step 1: w =  1.175  b =  0.400  loss =   0.552  grad = [ -3.875,  -1.325]
# step 2: w =  1.369  b =  0.466  loss =   0.064  grad = [ -0.638,  -0.224]
# step 3: w =  1.401  b =  0.477  loss =   0.050  grad = [ -0.103,  -0.042]
# step 4: w =  1.406  b =  0.480  loss =   0.050  grad = [ -0.015,  -0.012]
# step 5: w =  1.407  b =  0.480  loss =   0.050  grad = [ -0.001,  -0.007]
```

Five steps took the loss from 18.5 to 0.050, the best possible value, using nothing but "feel the slope, step downhill". Notice two things. The steps shrink on their own, because the gradient shrinks as the ground flattens. And \( b \) is still at 0.480, short of 0.5, creeping up slowly: the last part of the walk is slow. To see why, walk from a worse start too, and draw both walks on the loss landscape:

```python
path = [(0.0, 0.0)]
for step in range(30):
    gw, gb = mse_grad(*path[-1])
    path.append((path[-1][0] - lr * gw, path[-1][1] - lr * gb))
path = np.array(path)                     # shape (31, 2): every (w, b) visited
path[-1].round(3), round(mse(*path[-1]), 4)   # (array([1.405, 0.486]), np.float64(0.05))

path2 = [(2.5, 3.0)]                      # a much worse starting line
for step in range(40):
    gw, gb = mse_grad(*path2[-1])
    path2.append((path2[-1][0] - lr * gw, path2[-1][1] - lr * gb))
path2 = np.array(path2)
path2[[0, 1, 2, 10, 40]].round(2)
# [[2.5  3.  ]
#  [1.05 2.48]
#  [0.82 2.37]
#  [0.84 2.14]
#  [1.05 1.54]]
mse(*path2[0]), mse(*path2[1]).round(3), mse(*path2[40]).round(3)
# (np.float64(29.125), np.float64(1.413), np.float64(0.231))
```

The second walk crashes from 29.1 to 1.41 in one step, then 39 more steps only get it down to 0.23, still far from the best 0.05. The picture shows why:

```python
ws = np.linspace(-0.5, 3.0, 201)
bs = np.linspace(-2.0, 3.5, 201)
W, B = np.meshgrid(ws, bs)
# the loss at every (w, b) on the grid: broadcasting (Day 9) adds a points axis, then we average it away
L = ((W[..., None] * x_data + B[..., None] - y_data) ** 2).mean(axis=-1)
L.shape          # (201, 201)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1, 0.9]})
ax1.scatter(x_data, y_data, color="#3949ab", s=50, zorder=3, label="data")
xl = np.array([0, 5])
for (wi, bi), col, lab in [((0, 0), "#9e9e9e", "start: w=0, b=0"),
                           (path[1], "#ef6c00", "after 1 step"),
                           ((1.4, 0.5), "#e53935", "best: w=1.4, b=0.5")]:
    ax1.plot(xl, wi * xl + bi, color=col, lw=2, label=lab)
ax1.set_xlim(0, 5); ax1.set_ylim(-0.5, 8)
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.set_title("The data and three candidate lines")
ax1.legend(frameon=False, loc="upper left")
ax1.grid(alpha=0.3)

cs = ax2.contour(W, B, L, levels=[0.3, 1, 2, 4, 8, 16], cmap="viridis")
ax2.clabel(cs, fontsize=9, fmt="%g")
ax2.plot(path[:, 0], path[:, 1], "o-", color="#ef6c00", ms=4, lw=1.5, label="walk from (0, 0)")
ax2.plot(path2[:, 0], path2[:, 1], "o-", color="#3949ab", ms=3, lw=1.2, label="walk from (2.5, 3)")
ax2.plot(1.4, 0.5, "*", color="#e53935", ms=16, label="minimum (1.4, 0.5)")
ax2.set_xlabel("w (slope)"); ax2.set_ylabel("b (intercept)")
ax2.set_title("The loss L(w, b) and two walks downhill")
ax2.set_aspect("equal")            # equal scales, so right angles look like right angles
ax2.legend(loc="lower left", fontsize=9, framealpha=0.9)
fig.savefig("day32-mse-landscape.svg", format="svg", bbox_inches="tight")
plt.close(fig)
```

<figure class="diagram" markdown>
![Left: four blue data points with three lines, a grey flat line at y = 0, an orange line after one step that sits just under the points, and the red best line w = 1.4, b = 0.5 through them; right: contour map of the loss over w from -0.5 to 3 and b from -2 to 3.5, long thin tilted ellipses around a red star at (1.4, 0.5); an orange path from (0, 0) reaches the star in a couple of big steps, while a blue path from (2.5, 3) jumps across the contours into the narrow valley and then creeps slowly down the valley floor toward the star](../../assets/img/phase-1/day32-mse-landscape.svg)
<figcaption>The loss landscape is a long, narrow valley. The first step drops straight into it (steepest descent is across the valley); then the gradient along the gently sloping valley floor is small, so progress is slow.</figcaption>
</figure>

The contours are long thin ellipses: the valley is steep across and nearly flat along its floor. Steepest descent heads straight down the steep walls first, then has only a gentle slope left to follow along the floor. Too large a learning rate makes the steps bounce from wall to wall and blow up (Exercise 6). Choosing \( \eta \), why narrow valleys are slow, and the fixes (standardizing features, momentum, Adam) are **Day 51's** subject: gradient descent in full. Today's point is only this: **the gradient, and nothing else, told us which way to change both parameters.**

!!! warning "Common mistake: adding the gradient instead of subtracting it"
    The gradient points **uphill**. `w = w + lr * gw` climbs the loss and makes it explode within a few steps. To minimize, subtract: `w = w - lr * gw`. If your loss goes *up* every step, check this sign before anything else.

!!! android "Android bridge: the gradient of an image is an edge detector"
    A grayscale photo is a function of two variables: brightness \( I(x, y) \) at pixel column \( x \) and row \( y \). Its gradient \( \nabla I \) is large where brightness changes fast, which is exactly at **edges**, and it points across the edge, from dark to bright. The edge itself runs along a contour of brightness, perpendicular to \( \nabla I \), just like today's arrows. The same central difference you wrote in `num_grad`, with \( h = 1 \) pixel:

    ```kotlin
    import android.graphics.Bitmap
    import android.graphics.Color
    import kotlin.math.sqrt

    /** Brightness gradient at an interior pixel (1 <= x < width-1, 1 <= y < height-1). */
    fun brightnessGradient(bitmap: Bitmap, x: Int, y: Int): FloatArray {
        fun lum(px: Int, py: Int): Float {
            val c = bitmap.getPixel(px, py)
            return 0.299f * Color.red(c) + 0.587f * Color.green(c) + 0.114f * Color.blue(c)
        }
        val dIdx = (lum(x + 1, y) - lum(x - 1, y)) / 2f   // freeze y, nudge x
        val dIdy = (lum(x, y + 1) - lum(x, y - 1)) / 2f   // freeze x, nudge y (image y grows DOWNWARDS)
        return floatArrayOf(dIdx, dIdy)
    }

    fun edgeStrength(g: FloatArray): Float = sqrt(g[0] * g[0] + g[1] * g[1])   // the gradient's length
    ```

    Real code doesn't call `getPixel` per pixel (it's slow); it copies the image into an `IntArray` once with `bitmap.getPixels(...)`, or uses OpenCV's `Imgproc.Sobel`, or a GPU shader. The **Sobel filter** is this central difference with a little smoothing across neighbouring rows. And the first layer of a trained convolutional network, the kind that runs in LiteRT for image classification, typically learns small filters that look a lot like these gradient detectors, because edges are the first useful thing to know about a picture.

    On the training side: in PyTorch (Phase 3), after `loss.backward()` every parameter tensor has a `.grad` holding \( \partial L / \partial \text{parameter} \) for each of its entries. That is today's gradient, one entry per knob, and the optimizer step is today's `w - lr * gw`.

---

## Exercises

1. By hand, find both partial derivatives of \( f(x, y) = 3x^2 y - 2y + 5 \), and evaluate the gradient at \( (1, 2) \). Check with `num_grad`.
2. For \( q(x, y) = x^2 + y^2 \) at the point \( (3, 4) \): give \( \nabla q \) and its length. What is the directional derivative toward the origin, \( \mathbf{u} = [-0.6, -0.8] \)? And along \( \mathbf{u} = [-0.8, 0.6] \)? Explain both numbers using the three facts.
3. Gradients work in any dimension. For \( s(x, y, z) = xy + yz + zx \), find \( \nabla s \) by hand at \( (1, 2, 3) \), and check it with the same `num_grad` (no changes needed).
4. Compute the MSE gradient by hand at \( (w, b) = (1, 1) \) and at \( (w, b) = (2, 0) \) for today's four points. For each, say in words whether the slope and the intercept should go up or down, and why that makes sense for the line. Check with `mse_grad`.
5. Use the tilted-plane estimate \( f(\mathbf{p} + \Delta) \approx f(\mathbf{p}) + \nabla f \cdot \Delta \) to estimate \( f(1.1, 0.9) \) for \( f = x^2 + 3y^2 \), starting from \( (1, 1) \). Compare with the true value. Then show that \( [-3, 1] \) really is along the contour at \( (1, 1) \): use the tilted-plane estimate to show the height doesn't change (to first order) when you step that way.
6. Rerun the six-step walk on the MSE loss with `lr = 0.1`, `0.115`, `0.12` and `0.13` for 100 steps each. Which converge to \( (1.4, 0.5) \) and which blow up? Find the rough threshold between them.

??? success "Solutions"
    **1.** Freeze \( y \): \( \frac{\partial f}{\partial x} = 6xy \). Freeze \( x \): \( 3x^2 y \) is \( (3x^2) \cdot y \), slope \( 3x^2 \); \( -2y \) has slope \( -2 \); \( 5 \) is constant. So \( \frac{\partial f}{\partial y} = 3x^2 - 2 \). At \( (1, 2) \): \( \nabla f = [12, 1] \).

    ```python
    num_grad(lambda p: 3 * p[0]**2 * p[1] - 2 * p[1] + 5, [1, 2])   # array([12.,  1.])
    ```

    **2.** \( \nabla q = [2x, 2y] = [6, 8] \), length \( \sqrt{36 + 64} = 10 \).

    ```python
    gq = num_grad(lambda p: p[0]**2 + p[1]**2, [3, 4])
    gq, np.linalg.norm(gq)         # (array([6., 8.]), np.float64(9.999999999976694))
    gq @ np.array([-0.6, -0.8])    # np.float64(-9.999999999976692)
    gq @ np.array([-0.8, 0.6])     # about -1.8e-10, zero up to rounding
    ```

    Toward the origin the slope is \( -6 \cdot 0.6 - 8 \cdot 0.8 = -10 = -\|\nabla q\| \): the most negative possible, because \( [-0.6, -0.8] \) is exactly opposite the gradient (fact 2: steepest descent). The contours of \( q \) are circles around the origin, and \( [-0.8, 0.6] \) is tangent to the circle through \( (3, 4) \), so the slope is \( -4.8 + 4.8 = 0 \) (fact 3). The tiny non-zero values are floating-point error from the numerical gradient.

    **3.** \( \frac{\partial s}{\partial x} = y + z \) (the term \( yz \) has no \( x \)), \( \frac{\partial s}{\partial y} = x + z \), \( \frac{\partial s}{\partial z} = x + y \). At \( (1, 2, 3) \): \( [5, 4, 3] \).

    ```python
    num_grad(lambda p: p[0] * p[1] + p[1] * p[2] + p[2] * p[0], [1, 2, 3])   # array([5., 4., 3.])
    ```

    **4.** At \( (1, 1) \) the predictions are \( [2, 3, 4, 5] \), errors \( [0, 0, -1, -1] \). \( \sum e_i x_i = -3 - 4 = -7 \), so \( \frac{\partial L}{\partial w} = \frac{2}{4}(-7) = -3.5 \); \( \sum e_i = -2 \), so \( \frac{\partial L}{\partial b} = -1 \). Both negative: increase both. The line is right at \( x = 1, 2 \) but too low at \( x = 3, 4 \), so it should tilt up; the slope matters more because the misses are at large \( x \).

    At \( (2, 0) \) the predictions are \( [2, 4, 6, 8] \), errors \( [0, 1, 1, 2] \). \( \sum e_i x_i = 0 + 2 + 3 + 8 = 13 \), so \( \frac{\partial L}{\partial w} = 6.5 \); \( \sum e_i = 4 \), so \( \frac{\partial L}{\partial b} = 2 \). Both positive: decrease both. The line is too steep and over-predicts everywhere except \( x = 1 \).

    ```python
    mse_grad(1, 1), mse_grad(2, 0)     # (array([-3.5, -1. ]), array([6.5, 2. ]))
    ```

    **5.** \( \Delta = [0.1, -0.1] \), \( \nabla f(1, 1) = [2, 6] \), so the estimate is \( 4 + 0.2 - 0.6 = 3.6 \). The true value is \( 1.21 + 3 \cdot 0.81 = 3.64 \): off by 0.04, the curvature the flat plane ignores.

    ```python
    f(1.1, 0.9), f(1, 1) + grad_f(1, 1) @ np.array([0.1, -0.1])   # (3.6400000000000006, np.float64(3.6))
    ```

    For a step \( t\,[-3, 1] \), the estimated change is \( t\,(\nabla f \cdot [-3, 1]) = t\,(2 \cdot (-3) + 6 \cdot 1) = 0 \). Zero first-order change in height is what "along the contour" means, so \( [-3, 1] \) is a tangent direction there.

    **6.**

    ```python
    for lr_try in [0.1, 0.115, 0.12, 0.13]:
        w, b = 0.0, 0.0
        for _ in range(100):
            gw, gb = mse_grad(w, b)
            w, b = w - lr_try * gw, b - lr_try * gb
        print(f"lr = {lr_try}: w = {w:.3f}, b = {b:.3f}, loss = {mse(w, b):.4g}")
    # lr = 0.1: w = 1.400, b = 0.499, loss = 0.05
    # lr = 0.115: w = 1.400, b = 0.499, loss = 0.05
    # lr = 0.12: w = -0.713, b = -0.219, loss = 41.65
    # lr = 0.13: w = -10162599.887, b = -3456523.234, loss = 9.622e+14
    ```

    0.1 and 0.115 converge; 0.12 already diverges, slowly: the loss rises a little every step, from 18.5 at the start to 41.6 after 100 steps. At 0.13 it has exploded. The threshold sits just under 0.12. Each step jumps across the narrow valley; once the jump is so long that it lands higher up the opposite wall than where it started, every step makes things worse. (The exact limit is about 0.1198 and comes from the valley's steepest curvature, which you'll compute on Day 51.)

## Resources

- [3Blue1Brown: Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w): the "walk downhill on the loss" picture, animated, for a real network.
- [Khan Academy: Multivariable calculus](https://www.khanacademy.org/math/multivariable-calculus): the "Derivatives of multivariable functions" unit covers partial derivatives, the gradient and directional derivatives with short videos and practice.
- [MIT OpenCourseWare 18.02SC: Multivariable Calculus](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/): a full free course; Unit 2 is partial derivatives, gradients and contour plots.
- [Mathematics for Machine Learning (free book)](https://mml-book.github.io/): chapter 5, "Vector Calculus", covers partial derivatives and gradients and leads into the matrix forms of Day 33.
- [Dive into Deep Learning: Calculus](https://d2l.ai/chapter_preliminaries/calculus.html): partial derivatives and gradients in a deep-learning context, with code.
- [Matplotlib docs: `contour`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html) and [`quiver`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html), plus [NumPy docs: `meshgrid`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html).
- [Android reference: `android.graphics.Bitmap`](https://developer.android.com/reference/android/graphics/Bitmap): `getPixel` and `getPixels` for reading the brightness values the bridge differentiates.

## Checklist

- [ ] I can read a contour map: one height per line, close lines mean steep, small rings mean a peak or a valley floor.
- [ ] I can compute a partial derivative by freezing the other variables (as constants, not zeros), and check it with a central difference.
- [ ] I can write the gradient \( \nabla f \) as the vector of partial derivatives, and use `num_grad` in any number of dimensions.
- [ ] I can compute a directional derivative \( \nabla f \cdot \mathbf{u} \) with a unit vector \( \mathbf{u} \).
- [ ] I can explain, from \( \|\nabla f\| \cos\theta \), why the gradient points uphill, why minus the gradient is steepest descent, and why the gradient is perpendicular to contours.
- [ ] I derived \( \partial L / \partial w \) and \( \partial L / \partial b \) for the MSE of a line fit, and checked them numerically.
- [ ] I took steps \( \mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L \) and watched the loss fall, and I know a too-large \( \eta \) makes it blow up.

**Next:** Day 33: Gradients of vectors and matrices, checked numerically (coming soon)
