# Day 5: Vectors, the Atoms of Machine Learning

**Phase 1 · Week 1 · ~150 minutes**

This is the last teaching day of Week 1. It turns the NumPy arrays of [Days 4–4c](day-04-numpy-arrays.md) into geometry: arrows, lengths, angles and similarity. It's long; a good break point is after "Unit vectors and normalising", before the dot product.

## Objectives

- [ ] Describe a vector in three ways (a list of numbers, an arrow, a point), read the notation \( \mathbf{a} \), \( a_i \) and \( \mathbb{R}^n \), and tell the two meanings of "dimension" apart.
- [ ] Explain why a Python list is not a vector (`+` concatenates), and store vectors as 1-D float NumPy arrays.
- [ ] Add, subtract and scale vectors by hand and in NumPy, and draw what each operation does.
- [ ] Write a vector as a linear combination of other vectors.
- [ ] Compute a vector's length (its norm) by Pythagoras in 2-D, 3-D and n-D, and the distance between two points.
- [ ] Turn any non-zero vector into a unit vector, and guard against the zero vector.
- [ ] Compute a dot product by hand, with a loop and with `@`, and explain what its sign means.
- [ ] Derive \( \mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta \) from the law of cosines, and use it to find the "shadow" (projection length) of one vector on another.
- [ ] Recover the angle between two vectors with `np.arccos`, clipping first so float error can't produce `nan`.
- [ ] Compute cosine similarity, interpret its value, and show that for unit vectors \( \|\hat{\mathbf{a}} - \hat{\mathbf{b}}\|^2 = 2 - 2\cos\theta \).
- [ ] Rank documents against a query with toy embeddings, and explain why cosine beats the raw dot product there.
- [ ] Compute a neuron's output \( \mathbf{w} \cdot \mathbf{x} + b \), explain why features are scaled first, and run a batch of inputs with one matrix product.

**Prerequisites.**

- [Day 2](day-02-python-types.md): `float` and why `0.1 + 0.2 != 0.3`; `**` for powers; the `math` module (`math.sqrt`, `math.pi`, `math.isclose`); `nan`.
- [Day 2b](day-02b-collections.md): lists, and the fact that `+` and `*` on lists concatenate and repeat; list comprehensions; `zip`.
- [Day 3](day-03-functions-classes.md): writing functions with `def` and type hints.
- [Day 3b](day-03b-errors-files-json.md): `try` / `except`, and `raise ValueError(...)`.
- [Day 3d](day-03d-iterators-generators.md): generator expressions inside `sum(...)`, and `zip(..., strict=True)`.
- [Day 4](day-04-numpy-arrays.md): `np.array`, `shape`, `ndim`, `dtype`, vectorised arithmetic, `np.sqrt`, reductions such as `.sum()` with `axis=` and `keepdims=True`.
- [Day 4b](day-04b-numpy-toolkit.md): `np.clip`, `np.argsort`, `nan` and the `RuntimeWarning` it comes with, `np.isclose` / `np.allclose`.
- [Day 4c](day-04c-random-io-stats.md): `np.random.default_rng` and seeded random arrays.

!!! info "How to read the code on this page"
    Every Python block has a **▶ Run** button, and blocks share their variables from top to bottom like notebook cells, so run them in order. A bare expression on its own line shows its value, exactly as the `# comment` beside or below it says. Lines that fail on purpose are wrapped in `try` / `except` so the rest of the block still runs, and the comment shows the real error message.

---

## The big idea

Every piece of data a model sees is first turned into a list of numbers of a fixed length. A house becomes `[bedrooms, area_m2, age_years, distance_to_city_km]`. A 28×28 grayscale image becomes 784 brightness values in a row. A sentence goes through an embedding model and comes out as 384, 768 or 1,536 numbers. A user of your app becomes counts of the features they touched.

That fixed-length list of numbers is a **vector**, and the reason ML loves vectors is that once data is a vector, questions about data become questions about **geometry**:

- "Are these two support tickets about the same problem?" becomes "do these two arrows point the same way?"
- "Which saved photo looks most like this one?" becomes "which point is closest to this point?"
- "Does this input match the pattern this neuron has learned?" becomes "how far does this arrow lean along that one?"

Three operations answer almost all of those questions: the **length** of a vector, the **distance** between two vectors and the **dot product**, which measures how much two vectors point the same way. Today you learn each one by hand, in a picture and in NumPy, and see how they become semantic search and a neuron.

**Analogy.** Think of a vector as walking directions in a city grid: "3 blocks east, 2 blocks north". The list `[3, 2]` is the instructions. The arrow is the walk. The place you end up, if you start from the town hall, is the point. With more numbers you're walking in more directions at once. You can't picture 768 of them, but every formula today works the same way in 768 dimensions as in 2, which is why we learn it in 2 and then trust the arithmetic.

---

**Part 1: What a vector is**

## Three views of one vector

**What it is.** A **vector** is an ordered list of numbers. Each number is a **component** (or **element**, or **entry**) of the vector. "Ordered" matters: `[3, 2]` and `[2, 3]` are different vectors. The number of components is the vector's **dimension**.

The same list can be read three ways:

1. **A list of numbers**: a record with a fixed set of fields, like a row in a table.
2. **An arrow**: a movement with a direction and a length. "3 east, 2 north" is the same movement wherever you start, so an arrow can be drawn from any starting point and still be the same vector.
3. **A point**: where the arrow ends if it starts from the **origin**, the point `[0, 0]` where the axes cross. Used this way a vector is a *location*, such as a house's position in "feature space".

**Why it exists.** Each view answers different questions. The list view is how you store and compute. The arrow view explains addition, scaling and angles. The point view explains distances and "nearest neighbours". ML practitioners switch between them constantly, often in the same sentence: "the query embedding (a point) is closest to these documents, because their vectors (arrows) point the same way".

**Intuition first.** Picture graph paper. Draw the axes. Mark the point 3 squares right and 2 squares up. Now draw an arrow from the origin to that point. Now draw the same arrow (3 right, 2 up) starting somewhere else, say from `[1, 1]`. It's a different *drawing* but the same *vector*, because a vector records only the movement, not where it started.

<figure class="diagram" markdown>
![Animation in three steps: the list [3, 2] is shown as two boxes holding 3 and 2; then an arrow is drawn from the origin 3 squares right and 2 squares up, with dashed guides labelled 3 and 2; then the arrow's tip becomes a dot labelled point (3, 2), and a second, identical arrow appears starting from (1, 3), labelled same vector, different start](../../assets/img/phase-1/day05-three-views.svg)
<figcaption>Animated: one vector, three readings. The numbers say how far to move along each axis; the arrow is that movement; the point is where it ends if you start at the origin.</figcaption>
</figure>

**Notation.** Maths writes vectors as bold lowercase letters, and their components with a plain letter and a subscript (a small number below the line) that counts from 1:

\[
\mathbf{a} = [a_1, a_2, \dots, a_n]
\qquad\text{for example}\qquad
\mathbf{a} = [3, 2], \quad a_1 = 3, \quad a_2 = 2
\]

Symbol by symbol:

- \( \mathbf{a} \) (bold) is the whole vector. On paper people draw an arrow over the letter, \( \vec{a} \), instead.
- \( a_1 \) (not bold) is one number, the first component. Maths counts from 1; NumPy counts from 0, so \( a_1 \) is `a[0]` in code.
- \( n \) is the dimension, and \( \dots \) means "and so on up to".
- \( \mathbb{R} \) (a blackboard-bold R) is the set of all real numbers: every number on the number line, fractions and decimals included. \( \mathbb{R}^n \) ("R n") is the set of all vectors with \( n \) real components. \( \mathbb{R}^2 \) is the flat plane, \( \mathbb{R}^3 \) is ordinary space, and "an embedding in \( \mathbb{R}^{768} \)" means "a vector of 768 floats".
- A single number, to tell it apart from a vector, is called a **scalar**. Scalars are written in plain italics: \( c \), \( \lambda \), \( b \).

**Example.** In NumPy a vector is a 1-D array (Day 4). Make it a float array from the start; you'll see why in a minute.

```python
import numpy as np

a = np.array([3.0, 2.0])
a            # array([3., 2.])
a[0], a[1]   # the components a₁ and a₂ (NumPy counts from 0)
a.shape      # (2,)   one axis, holding 2 numbers
a.ndim       # 1
a.dtype      # dtype('float64')
```

`a[0], a[1]` prints `(np.float64(3.0), np.float64(2.0))`: these are the **NumPy scalars** from Day 4, single numbers that keep their dtype and echo as `np.float64(...)` (or `np.int64(...)`, `np.True_`). They behave like Python numbers in arithmetic. Every bare NumPy number on this page shows in that form; wrap it in `print()` or `float()` to see the plain value.

**How it works underneath.** A NumPy vector is one contiguous block of memory holding \( n \) numbers of one dtype, plus a shape `(n,)`. A 768-dimensional float32 embedding is 768 × 4 = 3,072 bytes in a row. Nothing in memory says "arrow" or "point"; those are ways of reading the same bytes.

### The two meanings of "dimension"

This trips everyone up, so here it is head-on. "Dimension" means two different things:

- **The dimension of the vector** (maths): how many components it has. A 768-number embedding is a "768-dimensional vector".
- **The number of axes of the array** (NumPy's `ndim`): how many indices you need to pick one number. A vector, however long, is a 1-D array.

```python
emb = np.zeros(768)    # a 768-dimensional vector
emb.shape              # (768,)
emb.ndim               # 1     ...stored in a 1-D array
```

When someone says "a 768-dimensional array", they almost always mean shape `(768,)`, not 768 axes. Read from context.

!!! warning "Common mistake: expecting a 1-D array to be a row or a column"
    Textbooks often draw vectors as columns. A 1-D NumPy array is neither a row nor a column: it has one axis and no orientation, so `.T` (transpose, Day 11) does nothing to it.

    ```python
    a.T.shape          # (2,)   unchanged
    a[:, None].shape   # (2, 1) an explicit column (Day 8)
    a[None, :].shape   # (1, 2) an explicit row
    ```

    `None` inside an index inserts a new axis of length 1 at that position; Day 8 explains it under its alias `np.newaxis`. For everything on this page, 1-D arrays are what you want. Day 8 shows when you need the explicit `(n, 1)` column, and Day 9 shows the bug you get if you mix them up.

## Why a Python list is not a vector

**What it is.** A Python list holds numbers, but its operators mean *list* things, not *vector* things. `+` glues two lists together and `*` repeats a list.

**Why it matters.** The vector operations you're about to learn are element-by-element. If you write them on lists, Python doesn't complain; it quietly does something else.

**Kotlin comparison.** Kotlin does the same with `List`:

```kotlin
val a = listOf(3, 2)
val b = listOf(1, 4)
println(a + b)          // [3, 2, 1, 4]   List.plus concatenates
// println(2 * a)       // does not compile: Int has no times(List)
```

```python
a_list = [3, 2]
b_list = [1, 4]
a_list + b_list         # [3, 2, 1, 4]   concatenation, not addition
2 * a_list              # [3, 2, 3, 2]   repetition, not scaling

a = np.array([3.0, 2.0])
b = np.array([1.0, 4.0])
a + b                   # array([4., 6.])   element by element
2 * a                   # array([6., 4.])
```

The same: in both languages a list's `+` concatenates. The difference: Kotlin refuses `2 * list` at compile time, while Python happily repeats the list. NumPy arrays redefine `+`, `-`, `*` and `/` to work element by element, which is exactly vector arithmetic. Convert lists with `np.array(...)` before doing maths.

**Underneath.** `list.__add__` builds a new, longer list of references to the same number objects. `ndarray.__add__` runs one C loop over two blocks of floats and writes a new block of the same length. Same symbol, different method on a different type (Day 3a's dunder methods).

!!! warning "Common mistake: `if a == b:` on two vectors"
    `==` on arrays compares element by element and returns an array of booleans, and `if` doesn't know whether you mean "all equal" or "any equal":

    ```python
    a == np.array([3.0, 5.0])   # array([ True, False])
    try:
        if a == np.array([3.0, 5.0]):
            print("same")
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    ```

    **Fix:** say which one you mean: `(a == b).all()`, or `np.array_equal(a, b)`. For float vectors that come out of arithmetic, compare with a tolerance, `np.allclose(a, b)` (Day 4b), since `0.1 + 0.2` is not exactly `0.3`.

---

**Part 2: Moving vectors around**

## Adding and subtracting vectors

**What it is.** To add two vectors of the same dimension, add matching components. To subtract, subtract matching components:

\[
\mathbf{a} + \mathbf{b} = [a_1 + b_1,\; a_2 + b_2,\; \dots,\; a_n + b_n]
\qquad
\mathbf{a} - \mathbf{b} = [a_1 - b_1,\; \dots,\; a_n - b_n]
\]

**Why it exists.** Adding is "do one movement, then the other". It's how a position is updated (old position + movement), how a model's weights are nudged (weights + change), and how a bias vector is added to a layer's outputs. Subtracting gives the **displacement** from one point to another, which is the first step of every distance calculation.

**Intuition first.** Walk 3 east and 1 north, then walk 1 east and 3 north. Where are you? 4 east, 4 north. Geometrically: put the tail of \( \mathbf{b} \) on the tip of \( \mathbf{a} \) ("tip to tail"); the sum is the arrow from the very start to the very end.

<figure class="diagram" markdown>
![Animation of vector addition: a = [3, 1] is drawn from the origin; b = [1, 3] slides so its tail sits at a's tip; the sum [4, 4] appears from the origin to b's new tip](../../assets/img/phase-1/day05-vector-addition.svg)
<figcaption>Animated: adding vectors means following one arrow, then the other.</figcaption>
</figure>

**Worked by hand.** \( [3, 1] + [1, 3] = [3 + 1,\; 1 + 3] = [4, 4] \). Doing it the other way round, \( [1, 3] + [3, 1] \), gives the same \( [4, 4] \): addition is **commutative** (order doesn't matter), because each component is an ordinary number sum. In the picture, walking \( \mathbf{b} \) first and then \( \mathbf{a} \) traces the other two sides of a parallelogram and ends at the same corner. For the same reason addition is **associative**, \( (\mathbf{a} + \mathbf{b}) + \mathbf{c} = \mathbf{a} + (\mathbf{b} + \mathbf{c}) \), so you can add a list of vectors in any grouping.

**Subtraction as "the arrow from b to a".** \( \mathbf{a} - \mathbf{b} \) answers "what do I add to \( \mathbf{b} \) to get to \( \mathbf{a} \)?", because \( \mathbf{b} + (\mathbf{a} - \mathbf{b}) = \mathbf{a} \). So if \( \mathbf{a} \) and \( \mathbf{b} \) are two points, \( \mathbf{a} - \mathbf{b} \) is the arrow that starts at \( \mathbf{b} \) and ends at \( \mathbf{a} \). Example: you're at \( \mathbf{p} = [2, 1] \) and the coffee shop is at \( \mathbf{q} = [5, 5] \). The walk is \( \mathbf{q} - \mathbf{p} = [3, 4] \): 3 east, 4 north. Order matters here: \( \mathbf{p} - \mathbf{q} = [-3, -4] \) is the walk back.

```python
a = np.array([3.0, 1.0])
b = np.array([1.0, 3.0])
a + b            # array([4., 4.])
b + a            # array([4., 4.])   commutative

p = np.array([2.0, 1.0])     # where you are
q = np.array([5.0, 5.0])     # where the coffee shop is
q - p            # array([3., 4.])   the walk from p to q
p + (q - p)      # array([5., 5.])   start at p, do the walk, arrive at q
```

**How it works underneath.** `a + b` checks that the shapes match (Day 9's broadcasting rules, which for two `(n,)` arrays means "the same `n`"), allocates a new `(n,)` array, and runs one compiled loop: `out[i] = a[i] + b[i]` for every `i`. Neither input changes.

!!! warning "Common mistake: adding vectors of different dimensions"
    Vector addition is only defined for vectors of the same dimension. NumPy refuses, with an error that mentions broadcasting:

    ```python
    try:
        np.array([1.0, 2.0]) + np.array([1.0, 2.0, 3.0])
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: operands could not be broadcast together with shapes (2,) (3,)
    ```

    **How to recognise it:** the two shapes in the message differ. In real code this usually means two vectors came from different pipelines: a feature vector built with an extra column, or an embedding from a different model. **Fix:** find where the lengths diverged; never pad or trim to make the error go away.

    The opposite surprise: `a + 1` does *not* fail. NumPy adds the scalar 1 to every component (a broadcast, Day 9). In maths "vector + number" isn't defined, so if you meant to add a vector, check its shape.

## Scaling a vector

**What it is.** **Scalar multiplication** multiplies every component by the same number \( c \):

\[
c\,\mathbf{a} = [c\,a_1,\; c\,a_2,\; \dots,\; c\,a_n]
\]

**Why it exists.** Scaling changes how far an arrow reaches without (for positive \( c \)) changing its direction. You'll see it in every training loop: gradient descent (Month 2) updates weights with `w = w - learning_rate * gradient`, which is "scale the gradient vector down, then subtract it". It's also how you average vectors (add, then scale by \( 1/n \)) and how you normalise them (next section).

**Intuition first.** \( 2\mathbf{a} \) is the same walk done twice. \( 0.5\,\mathbf{a} \) is half the walk. \( -1\,\mathbf{a} \) is the same walk backwards: same length, opposite direction. \( 0\,\mathbf{a} \) is "don't move", the **zero vector** \( \mathbf{0} = [0, 0, \dots, 0] \).

**Worked by hand.** With \( \mathbf{a} = [3, 1] \): \( 2\mathbf{a} = [6, 2] \), \( 0.5\,\mathbf{a} = [1.5, 0.5] \), \( -1\,\mathbf{a} = [-3, -1] \).

```python
a = np.array([3.0, 1.0])
2 * a            # array([6., 2.])
0.5 * a          # array([1.5, 0.5])
-1 * a           # array([-3., -1.])   flipped
-a               # array([-3., -1.])   the same thing, shorter to write
0 * a            # array([0., 0.])     the zero vector

# One step of gradient descent: scale the gradient, then subtract it
w = np.array([0.5, -0.3])        # current weights
grad = np.array([0.2, -0.4])     # which way the error grows
learning_rate = 0.1
w - learning_rate * grad         # array([ 0.48, -0.26])
```

Scaling also **distributes** over addition, \( c(\mathbf{a} + \mathbf{b}) = c\,\mathbf{a} + c\,\mathbf{b} \), because each component is \( c(a_i + b_i) = c\,a_i + c\,b_i \). With \( \mathbf{a} = [3, 1] \), \( \mathbf{b} = [1, 3] \) and \( c = 2 \), both sides are \( [8, 8] \):

```python
b = np.array([1.0, 3.0])
2 * (a + b)                      # array([8., 8.])
2 * a + 2 * b                    # array([8., 8.])
```

Trace the gradient step above: `learning_rate * grad` is `[0.02, -0.04]`; subtracting it from `[0.5, -0.3]` gives `[0.48, -0.26]`. The weights moved a small step *against* the gradient, which is the whole idea of training.

**How it works underneath.** Python evaluates `0.1 * grad` by calling the array's multiply method with a scalar; NumPy treats the scalar as if it were repeated to the array's shape (Day 9) and runs one loop. A new array comes out; `grad` is unchanged.

!!! warning "Common mistake: scaling an integer array in place"
    If you create a vector from whole numbers, NumPy gives it an integer dtype. An in-place operation (`*=`, `/=`) must write its result back into that same integer memory, and a float result doesn't fit:

    ```python
    v = np.array([3, 1])          # an integer dtype, because every literal is an int
    try:
        v *= 0.5
    except TypeError as e:        # NumPy's UFuncTypeError is a kind of TypeError
        print(type(e).__name__ + ":", e)
    # UFuncTypeError: Cannot cast ufunc 'multiply' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
    ```

    (In the browser the message says `int32` instead of `int64`: browser NumPy's default integer is 32-bit, as Day 4 noted.) The non-in-place `v * 0.5` works (it makes a new float array), which is why this bug appears only when you switch to `*=` for speed. **Fix:** create vectors as floats from the start: `np.array([3.0, 1.0])`, or `np.array([3, 1], dtype=float)`, or `.astype(float)` (Day 4).

## Linear combinations

**What it is.** A **linear combination** of vectors \( \mathbf{a} \) and \( \mathbf{b} \) is any vector you can make by scaling each and adding: \( c_1\mathbf{a} + c_2\mathbf{b} \), where \( c_1 \) and \( c_2 \) are scalars called the **coefficients** (or weights).

**Why it exists.** It's the single most important idea in linear algebra, and you'll meet it under many names: a weighted average of embeddings, a mix of colours, the output of a layer (Day 10 shows that a matrix times a vector *is* a linear combination of the matrix's columns), and "span" and "basis" (Day 23). Today you only need to recognise one.

**Intuition first.** You may walk only along two kinds of street: the \( \mathbf{a} = [3, 1] \) direction and the \( \mathbf{b} = [1, 3] \) direction, any amount, forwards or backwards. Can you reach \( [5, 7] \)? Try one \( \mathbf{a} \) and two \( \mathbf{b} \)'s: \( [3, 1] + [2, 6] = [5, 7] \). Yes, with \( c_1 = 1 \) and \( c_2 = 2 \).

The simplest building blocks are the **standard basis vectors**, one per axis, each with a single 1: in 2-D, \( \mathbf{e}_1 = [1, 0] \) and \( \mathbf{e}_2 = [0, 1] \). Every vector is a linear combination of them, with its own components as the coefficients: \( [3, 2] = 3\,\mathbf{e}_1 + 2\,\mathbf{e}_2 \). That's what the components of a vector *mean*: how much of each axis direction it contains.

```python
a = np.array([3.0, 1.0])
b = np.array([1.0, 3.0])
1 * a + 2 * b        # array([5., 7.])

e1 = np.array([1.0, 0.0])
e2 = np.array([0.0, 1.0])
3 * e1 + 2 * e2      # array([3., 2.])   components = coefficients on the axes

# Average of three embeddings = a linear combination with coefficients 1/3
v1, v2, v3 = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([2.0, 2.0])
(v1 + v2 + v3) / 3   # array([1., 1.])
```

**How it works underneath.** `1 * a + 2 * b` is the Part 2 arithmetic, run in order: NumPy builds a temporary array for `1 * a`, another for `2 * b`, then a third for their sum. A linear combination of \( k \) vectors costs \( k \) scalings and \( k - 1 \) additions, each one pass over \( n \) numbers. (Day 10 shows how a matrix product does all of it in one call.)

Finding the coefficients for a target (like the 1 and 2 above) means solving equations; Day 11 does it with `np.linalg.solve`. For now, the skill is reading \( c_1\mathbf{a} + c_2\mathbf{b} \) and computing it.

!!! warning "Common mistake: `np.sum` over a list of vectors, without `axis`"
    Averaging embeddings is a linear combination, and it's easy to collapse everything into one number by accident. `np.sum` on a list of vectors first turns the list into a `(3, 2)` array, then (with no `axis`) adds **every** element:

    ```python
    np.sum([v1, v2, v3]) / 3                 # np.float64(2.0)   WRONG: one number, not a vector
    np.array([v1, v2, v3]).shape             # (3, 2)            one vector per row
    np.mean(np.array([v1, v2, v3]), axis=0)  # array([1., 1.])   right: average down the rows
    ```

    **How to recognise it:** the "average vector" has no shape, or comes out as a scalar. **Fix:** stack the vectors as rows and reduce with `axis=0` (Day 4), or write the sum out as `(v1 + v2 + v3) / 3`. A cousin of this bug divides by the wrong count, such as `/ 2` for three vectors: no error, a vector 1.5 times too long.

---

**Part 3: Length and distance**

## Length: the norm

**What it is.** The **length** of a vector, also called its **norm** or **magnitude**, is how long the arrow is: the straight-line distance from its tail to its tip. It's written with double bars, \( \|\mathbf{a}\| \), and it is always a single non-negative number.

**Why it exists.** Length tells you "how big" a vector is: how far a point is from the origin, how large a model's error vector is, how strong an accelerometer reading is. It's also the ingredient of distance, of unit vectors and of cosine similarity, all later today.

**Intuition first: Pythagoras.** The arrow \( [3, 4] \) goes 3 right and 4 up. Those two movements are the two short sides of a right-angled triangle, and the arrow is the long side (the hypotenuse). Pythagoras says the long side squared is the sum of the short sides squared: \( 3^2 + 4^2 = 25 \), so the length is \( \sqrt{25} = 5 \).

**The formula, one step at a time.**

- In 2-D: \( \|\mathbf{a}\| = \sqrt{a_1^2 + a_2^2} \).
- In 3-D, use Pythagoras twice. For \( [2, 3, 6] \): the "floor" part \( [2, 3] \) has length \( \sqrt{4 + 9} = \sqrt{13} \). The arrow climbs 6 straight up from there, which makes a second right-angled triangle with sides \( \sqrt{13} \) and 6. Its long side is \( \sqrt{13 + 36} = \sqrt{49} = 7 \). The square root and the square cancel in the middle, so this is \( \sqrt{2^2 + 3^2 + 6^2} \).
- In \( n \) dimensions, keep going: each new axis adds one more squared term.

\[
\|\mathbf{a}\| = \sqrt{a_1^2 + a_2^2 + \dots + a_n^2} = \sqrt{\sum_{i=1}^{n} a_i^2}
\]

The \( \sum_{i=1}^{n} \) (capital Greek sigma) means "add up the expression that follows, for \( i = 1, 2, \dots, n \)". It is a `for` loop with an accumulator, written in maths.

**Worked by hand.**

- \( \|[3, 4]\| = \sqrt{9 + 16} = \sqrt{25} = 5 \)
- \( \|[3, -4]\| = \sqrt{9 + 16} = 5 \) (squaring removes the sign: length doesn't care about direction)
- \( \|[2, 3, 6]\| = \sqrt{4 + 9 + 36} = \sqrt{49} = 7 \)
- \( \|[1, 1, \dots, 1]\| \) with 768 ones \( = \sqrt{768} \approx 27.7 \): in high dimensions, many small components add up to a long vector.

**Kotlin comparison.** In Kotlin you'd write the loop, or use `hypot` for two components:

```kotlin
import kotlin.math.hypot
import kotlin.math.sqrt

val v = doubleArrayOf(2.0, 3.0, 6.0)
val length = sqrt(v.sumOf { it * it })   // 7.0
val len2d = hypot(3.0, 4.0)              // 5.0, two components only
```

In Python, `math.hypot` takes any number of components, and NumPy has `np.linalg.norm`:

```python
import math

v = np.array([2.0, 3.0, 6.0])
np.sqrt(np.sum(v ** 2))       # np.float64(7.0)   the formula, step by step
np.linalg.norm(v)             # np.float64(7.0)   the library function
math.hypot(2.0, 3.0, 6.0)     # 7.0               plain Python, any number of arguments
np.linalg.norm(np.array([3.0, -4.0]))   # np.float64(5.0)
np.linalg.norm(np.ones(768))  # np.float64(27.712812921102035)
```

Read `np.sqrt(np.sum(v ** 2))` from the inside out, matching the formula: `v ** 2` squares each component (`[4., 9., 36.]`), `np.sum` adds them (`49.0`), `np.sqrt` takes the root (`7.0`). `np.linalg.norm` does the same in one call, and it also takes `axis=`, used below. It does *not* guard against overflow: `np.linalg.norm(np.array([1e200, 1e200]))` gives `inf` with a `RuntimeWarning`, while `math.hypot(1e200, 1e200)` gives `1.414213562373095e+200`. This matters only for components around \( 10^{154} \) and up, which real features never reach.

**How it works underneath: length and scaling.** Scale a vector by \( c \) and every squared component is multiplied by \( c^2 \), so the length is multiplied by \( \sqrt{c^2} = |c| \) (the **absolute value** of \( c \): \( c \) without its minus sign):

\[
\|c\,\mathbf{a}\| = |c|\,\|\mathbf{a}\|
\]

Doubling a vector doubles its length; flipping it with \( c = -1 \) leaves the length alone. Only the zero vector has length 0.

```python
a = np.array([3.0, 4.0])
np.linalg.norm(2 * a)        # np.float64(10.0)
np.linalg.norm(-3 * a)       # np.float64(15.0)   |−3| × 5
```

!!! warning "Common mistake: a misplaced parenthesis turns the norm into something else"
    The formula is "square, **sum**, then root". Put the `.sum()` outside the `sqrt` and you get "square, root, then sum", which undoes the squaring and adds up the absolute values instead:

    ```python
    v = np.array([3.0, -4.0])
    np.sqrt(v ** 2).sum()     # np.float64(7.0)   WRONG: sqrt(9) + sqrt(16) = 3 + 4
    np.sqrt((v ** 2).sum())   # np.float64(5.0)   right: sqrt(9 + 16)
    ```

    No error, a plausible number, and the wrong answer. **How to recognise it:** test on `[3, 4]`; the answer must be 5. (The 7 is a real quantity, the "L1 norm" or taxicab length, which Day 22 teaches on purpose.) Another variant of the same slip is forgetting the square root entirely and getting the **squared norm**, 25. That one is sometimes used deliberately because it's cheaper, but then you must compare squared values with squared values.

!!! android "Android bridge: the accelerometer reading is a vector"
    `SensorEvent.values` for `Sensor.TYPE_ACCELEROMETER` is a `FloatArray` of three components, the acceleration along the device's x, y and z axes in m/s². Lying still on a table, the phone measures gravity, so the **length** of that vector is close to 9.81 whichever way the phone is turned, even though the individual components change as you rotate it. That's why shake detectors compute `sqrt(x*x + y*y + z*z)` and compare it with gravity, instead of checking any single axis. The same reading in NumPy:

    ```python
    reading = np.array([0.3, 9.7, 1.2])   # an example accelerometer reading, m/s²
    np.linalg.norm(reading)               # np.float64(9.778547949465707)
    ```

## Distance between two points

**What it is.** The **distance** between two points \( \mathbf{a} \) and \( \mathbf{b} \) is the length of the arrow between them:

\[
d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\| = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}
\]

This straight-line distance is called the **Euclidean distance**.

**Why it exists.** "Nearest" is a distance question: the k-nearest-neighbours classifier (Phase 2) labels a new example by the labels of the closest stored ones; clustering groups points that are close; "find similar houses" means "small distance in feature space".

**Intuition first.** You already have the two pieces. Subtraction gives the walk from one point to the other (Part 2), and the norm gives the length of a walk. Distance is "how long is the walk?".

**Worked by hand.** From \( \mathbf{p} = [2, 1] \) to \( \mathbf{q} = [5, 5] \): the walk is \( \mathbf{q} - \mathbf{p} = [3, 4] \), and its length is \( \sqrt{9 + 16} = 5 \). From \( \mathbf{q} \) back to \( \mathbf{p} \) the walk is \( [-3, -4] \), and its length is also 5: distance is **symmetric**, because squaring removes the signs.

```python
p = np.array([2.0, 1.0])
q = np.array([5.0, 5.0])
np.linalg.norm(q - p)         # np.float64(5.0)
np.linalg.norm(p - q)         # np.float64(5.0)   symmetric
math.dist([2, 1], [5, 5])     # 5.0               plain Python's version

# Which of three stored houses is closest to a query? [bedrooms, bathrooms]
houses = np.array([[3.0, 2.0],
                   [1.0, 1.0],
                   [4.0, 3.0]])
query = np.array([3.0, 3.0])
dists = np.linalg.norm(houses - query, axis=1)   # one distance per row
dists                         # array([1.        , 2.82842712, 1.        ])
```

That last line packs two Day 4 and Day 9 ideas together, so step by step: `houses - query` subtracts the `(2,)` query from each row of the `(3, 2)` table (a broadcast), giving three walks `[[0, -1], [-2, -2], [1, 0]]`. `np.linalg.norm(..., axis=1)` takes the norm of each row, giving three distances. Houses 0 and 2 tie at distance 1. Day 22 builds this into a full nearest-neighbour search; here the point is that distance is subtraction followed by a norm.

!!! warning "Common mistake: subtracting the lengths instead of the vectors"
    "The distance between a and b" is the length of the *difference*, not the difference of the *lengths*:

    ```python
    np.linalg.norm(q) - np.linalg.norm(p)   # np.float64(4.834999834365686)   WRONG
    np.linalg.norm(q - p)                   # np.float64(5.0)                 right
    ```

    The wrong version only compares how far each point is from the origin. Two points on opposite sides of the origin at the same radius, like `[5, 0]` and `[-5, 0]`, would get "distance" 0. **Fix:** subtract first, then take one norm.

## Unit vectors and normalising

**What it is.** A **unit vector** is a vector of length exactly 1. **Normalising** a vector means dividing it by its own length, which keeps its direction and makes its length 1. The unit version of \( \mathbf{a} \) is written with a hat:

\[
\hat{\mathbf{a}} = \frac{\mathbf{a}}{\|\mathbf{a}\|}
\]

**Why it exists.** Sometimes only the *direction* of a vector carries meaning. For a text embedding, "which way it points" encodes the topic, while the length mostly reflects things like how long the text was. Normalising throws the length away so that comparisons depend on direction alone. Vector databases normalise embeddings once, when they're stored, so every later search can use the cheapest possible comparison (you'll see why in Part 5).

**Intuition first.** Normalising is like reducing every walk to a compass bearing: "north-east", without "how far". Every unit vector in 2-D ends on the circle of radius 1 around the origin, called the **unit circle**.

**Worked by hand.** \( \mathbf{a} = [3, 4] \) has length 5. Divide each component by 5: \( \hat{\mathbf{a}} = [0.6, 0.8] \). Check: \( \sqrt{0.36 + 0.64} = \sqrt{1} = 1 \). Why it always works: by the scaling rule, \( \|\mathbf{a} / \|\mathbf{a}\|\| = \|\mathbf{a}\| / \|\mathbf{a}\| = 1 \).

```python
a = np.array([3.0, 4.0])
a_hat = a / np.linalg.norm(a)
a_hat                        # array([0.6, 0.8])
np.linalg.norm(a_hat)        # np.float64(1.0)

d = np.array([1.0, 1.0])     # the diagonal direction
d / np.linalg.norm(d)        # array([0.70710678, 0.70710678])   1/√2 each
```

The standard basis vectors \( \mathbf{e}_1 \) and \( \mathbf{e}_2 \) are unit vectors already. Day 22 normalises a whole table of vectors at once, one row at a time.

**How it works underneath.** `a / np.linalg.norm(a)` computes one scalar (the length) and divides every component by it. The result is a new array; `a` keeps its length. Float rounding means a normalised vector's length can come out as `0.9999999999999999` instead of `1.0`; check unit length with `np.isclose(np.linalg.norm(v), 1.0)` (Day 4b), not `==`.

!!! warning "Common mistake: normalising the zero vector"
    The zero vector has length 0 and no direction, so it can't be normalised. NumPy divides 0 by 0 anyway, warns, and gives `nan` ("not a number", Day 4b), which then spreads through every calculation it touches:

    ```python
    z = np.zeros(3)
    z / np.linalg.norm(z)
    # <cell>:2: RuntimeWarning: invalid value encountered in divide
    # array([nan, nan, nan])
    ```

    (In Jupyter or a script the warning names your file and line instead of `<cell>`.) Zero vectors are common in real data: an empty review gives an all-zero word-count vector, and a blank image gives an all-zero pixel vector. **Fix:** decide what a zero vector should mean and handle it explicitly. Raising an error is the safest default:

    ```python
    def normalize(v: np.ndarray) -> np.ndarray:
        """Return v scaled to length 1. Raises ValueError for the zero vector."""
        length = np.linalg.norm(v)
        if length == 0:
            raise ValueError("cannot normalise the zero vector")
        return v / length

    normalize(np.array([3.0, 4.0]))   # array([0.6, 0.8])
    try:
        normalize(np.zeros(3))
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: cannot normalise the zero vector
    ```

    Comparing `length == 0` is safe here even though it's a float, because only an all-zero vector gives exactly 0.

---

**Part 4: The dot product**

## The dot product, algebraically

**What it is.** The **dot product** of two vectors of the same dimension multiplies matching components and adds up the results. It takes two vectors and gives back **one number**:

\[
\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \dots + a_n b_n = \sum_{i=1}^{n} a_i b_i
\]

**Why it exists.** It is the most-used operation in ML. A neuron's output is a dot product. A matrix multiplication (Day 10) is a grid of dot products. Attention in a transformer compares queries and keys with dot products. Semantic search ranks documents by dot products. In the next section you'll see what the number *means*: how much two vectors point the same way.

**Intuition first.** Think of a shopping basket and a price list. The basket is \( [2, 1, 3] \): 2 apples, 1 loaf, 3 cartons of milk. The prices are \( [0.5, 2.0, 1.2] \). The bill is \( 2 \cdot 0.5 + 1 \cdot 2.0 + 3 \cdot 1.2 = 1 + 2 + 3.6 = 6.6 \). That's a dot product: quantities times weights, summed. A neuron does exactly this with inputs and learned weights.

**Worked by hand.**

\[
[1, 2, 3] \cdot [4, 5, 6] = 1 \cdot 4 + 2 \cdot 5 + 3 \cdot 6 = 4 + 10 + 18 = 32
\]

**Kotlin comparison.** The loop you'd write in Kotlin:

```kotlin
fun dot(a: DoubleArray, b: DoubleArray): Double {
    require(a.size == b.size) { "sizes differ: ${a.size} vs ${b.size}" }
    var sum = 0.0
    for (i in a.indices) sum += a[i] * b[i]
    return sum
}
dot(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0))   // 32.0
```

The same in Python, three ways, from most explicit to most idiomatic:

```python
def dot_loop(a: list[float], b: list[float]) -> float:
    total = 0.0
    for x, y in zip(a, b, strict=True):   # pairs (a₁, b₁), (a₂, b₂), ...
        total += x * y
    return total

dot_loop([1, 2, 3], [4, 5, 6])            # 32.0
sum(x * y for x, y in zip([1, 2, 3], [4, 5, 6], strict=True))   # 32

u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])
u * v              # array([ 4., 10., 18.])   step 1: multiply matching components
(u * v).sum()      # np.float64(32.0)         step 2: add them up
u @ v              # np.float64(32.0)         the operator for it
np.dot(u, v)       # np.float64(32.0)         the older function name, same result
```

What's the same: the Kotlin loop, `dot_loop` and the generator expression all do one multiply and one add per component, like the formula. What's different: NumPy's `u @ v` runs that loop in compiled code, and for long vectors it uses the CPU's vector instructions to do several multiply-adds at once. `@` is Python's matrix-multiplication operator (added in Python 3.5 for NumPy's sake); on two 1-D arrays it means the dot product. You'll see `np.dot(u, v)` in older code; for 1-D arrays it's identical.

**How it works underneath.** A trace of `u @ v`, with an accumulator that starts at 0:

| step \( i \) | \( u_i \) | \( v_i \) | \( u_i v_i \) | running total |
|---|---|---|---|---|
| 1 | 1 | 4 | 4 | 4 |
| 2 | 2 | 5 | 10 | 14 |
| 3 | 3 | 6 | 18 | 32 |

Two vectors go in, one scalar comes out. It needs no temporary array, unlike `(u * v).sum()`, which first builds the `[4, 10, 18]` array and then sums it.

**The rules it follows.** Each one follows from the formula, because each term \( a_i b_i \) is an ordinary product:

- **Commutative:** \( \mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a} \).
- **Scalars pull out:** \( (c\,\mathbf{a}) \cdot \mathbf{b} = c\,(\mathbf{a} \cdot \mathbf{b}) \).
- **Distributes over addition:** \( \mathbf{a} \cdot (\mathbf{b} + \mathbf{c}) = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{c} \).
- **A vector dotted with itself is its squared length:** \( \mathbf{a} \cdot \mathbf{a} = a_1^2 + \dots + a_n^2 = \|\mathbf{a}\|^2 \). This links the dot product to the norm, and it's the key to the next section.

```python
u @ v, v @ u                  # (np.float64(32.0), np.float64(32.0))
(2 * u) @ v                   # np.float64(64.0)   = 2 × 32
a = np.array([3.0, 4.0])
a @ a                         # np.float64(25.0)   = ‖a‖² = 5²
np.sqrt(a @ a)                # np.float64(5.0)    the norm, from a dot product
```

!!! warning "Common mistake: `*` is not the dot product"
    `u * v` multiplies element by element and returns a **vector**; that operation has a name of its own, the **element-wise** (or **Hadamard**) **product**, written \( \mathbf{u} \odot \mathbf{v} \), and you'll meet the \( \odot \) symbol in Month 2's gradient formulas. `u @ v` returns **one number**. Both are legal, so the mix-up shows up later as a shape surprise:

    ```python
    (u * v).shape    # (3,)
    (u @ v).shape    # ()    a 0-dimensional result: a single number
    ```

    A NumPy scalar carries array attributes, which a Python `float` doesn't: its `.shape` is `()`, the shape of a 0-dimensional array, meaning "no axes at all". In PyTorch the same two operators mean the same two things. If a result you expected to be a number has a shape like `(3,)`, look for a `*` that should be `@`.

!!! warning "Common mistake: vectors of different lengths"
    NumPy refuses, with a message written for matrices, because `@` is the matrix-multiplication operator:

    ```python
    try:
        np.array([1.0, 2.0]) @ np.array([1.0, 2.0, 3.0])
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)
    ```

    Read it from the end: "size 3 is different from 2" is the useful part. Pure Python is more dangerous, because `zip` without `strict=True` stops at the shorter input **silently**:

    ```python
    sum(x * y for x, y in zip([1, 2, 3], [4, 5]))     # 14   no error, wrong answer
    try:
        sum(x * y for x, y in zip([1, 2, 3], [4, 5], strict=True))
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: zip() argument 2 is shorter than argument 1
    ```

    And `@` on two plain lists fails with `TypeError: unsupported operand type(s) for @: 'list' and 'list'`: convert with `np.array` first. In a real model, a length mismatch almost always means the input was preprocessed differently from the training data.

## What the dot product means: angles

**What it is.** The same number can be computed from the two lengths and the **angle** \( \theta \) (Greek "theta") between the arrows, when both are drawn from the same starting point:

\[
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta
\]

**Why it exists.** This is what turns a sum of products into geometry. It says the dot product measures **alignment**: how much two vectors point the same way, scaled by how long they are. Every use of the dot product in ML (similarity, attention, neurons as pattern detectors) rests on this formula.

**Intuition first: the shadow.** Shine a light straight down onto the line that \( \mathbf{a} \) lies on. \( \mathbf{b} \) casts a shadow on it. The dot product is "the length of \( \mathbf{a} \)" times "the signed length of \( \mathbf{b} \)'s shadow along \( \mathbf{a} \)". If \( \mathbf{b} \) leans along \( \mathbf{a} \), the shadow is long and the dot product is big. If \( \mathbf{b} \) stands at a right angle, the shadow has no length and the dot product is 0. If \( \mathbf{b} \) leans backwards, the shadow points the other way and the dot product is negative.

**The cosine, briefly.** For an angle \( \theta \) in a right-angled triangle, \( \cos\theta \) is "adjacent side ÷ hypotenuse": how much of a slanted length survives when you look only along one direction. The values you need: \( \cos 0° = 1 \) (fully along), \( \cos 45° \approx 0.707 \), \( \cos 90° = 0 \) (nothing along), \( \cos 135° \approx -0.707 \), \( \cos 180° = -1 \) (fully backwards). That side ratio only makes sense below 90°. The general definition uses the **unit circle** (the circle of radius 1 around the origin): start at \( [1, 0] \) and walk anticlockwise round the circle by the angle \( \theta \); \( \cos\theta \) is the x-coordinate where you land. Below 90° that's the side ratio again. Past 90° you're left of the y-axis, so \( \cos\theta \) is negative, and at 180° you're at \( [-1, 0] \). The cosine-similarity animation further down draws exactly this, with an orange dot on the x-coordinate. So the shadow of \( \mathbf{b} \) on \( \mathbf{a} \) has signed length \( \|\mathbf{b}\| \cos\theta \), and multiplying by \( \|\mathbf{a}\| \) gives the formula above.

<figure class="diagram" markdown>
![Animation: a fixed blue arrow a = [3, 0] lies along the x-axis. A green arrow b takes four positions in turn, [3, 1], [2, 2], [0, 2] and [-2, 2]. For each, a dashed drop line falls from b's tip to the x-axis, an orange bar marks b's shadow on a's line (3, 2, 0 and -2 long), a dashed red side labelled a minus b joins the two tips, and a panel shows a·b = 3 × shadow = 9, 6, 0 and -6](../../assets/img/phase-1/day05-dot-projection.svg)
<figcaption>Animated: the dot product is the length of a times the signed length of b's shadow on a. At a right angle the shadow vanishes; past it, the shadow points backwards and the dot product turns negative.</figcaption>
</figure>

**Worked by hand, both ways.** Take \( \mathbf{a} = [3, 0] \) and \( \mathbf{b} = [2, 2] \).

- Algebra: \( \mathbf{a} \cdot \mathbf{b} = 3 \cdot 2 + 0 \cdot 2 = 6 \).
- Geometry: \( \|\mathbf{a}\| = 3 \). \( \|\mathbf{b}\| = \sqrt{4 + 4} = \sqrt{8} \approx 2.828 \). \( \mathbf{b} \) points along the diagonal, so \( \theta = 45° \) and \( \cos\theta \approx 0.7071 \). The product is \( 3 \times 2.828 \times 0.7071 \approx 6.0 \). The shadow of \( \mathbf{b} \) is \( \|\mathbf{b}\|\cos\theta = 2 \): it reaches \( x = 2 \), exactly where the drop line from \( [2, 2] \) lands.

The two routes agree, and the code agrees with both:

```python
a = np.array([3.0, 0.0])
b = np.array([2.0, 2.0])
a @ b                                                        # np.float64(6.0)
np.linalg.norm(a) * np.linalg.norm(b) * np.cos(np.pi / 4)    # np.float64(6.000000000000001)
```

`np.cos` works in **radians**, not degrees: a full turn is \( 2\pi \) radians, so 180° is \( \pi \) and 45° is \( \pi/4 \). The `...001` at the end is float rounding in `np.cos(np.pi / 4)`, not a disagreement; Day 4b's `np.isclose` would call the two equal.

### Why it's true: a derivation from the law of cosines

You don't need to memorise this, but seeing it once makes the formula feel earned rather than magic. Draw \( \mathbf{a} \) and \( \mathbf{b} \) from the same point. The arrow from \( \mathbf{b} \)'s tip to \( \mathbf{a} \)'s tip is \( \mathbf{a} - \mathbf{b} \) (Part 2), so the three arrows form a triangle with sides of length \( \|\mathbf{a}\| \), \( \|\mathbf{b}\| \) and \( \|\mathbf{a} - \mathbf{b}\| \), and the angle \( \theta \) between the first two. (It's the dashed red side in the animation above.)

**Step 1: the law of cosines.** For any triangle with two sides \( p \) and \( q \) and the angle \( \theta \) between them, the third side \( r \) satisfies

\[
r^2 = p^2 + q^2 - 2pq\cos\theta
\]

Where it comes from, in two lines (shown for \( \theta \) below 90°; the other case is the same with a sign change): drop a perpendicular from the far end of \( q \) onto the line of \( p \). It lands \( q\cos\theta \) along \( p \) and stands \( q\sin\theta \) tall. Pythagoras on the right-angled triangle that contains \( r \) gives \( r^2 = (p - q\cos\theta)^2 + (q\sin\theta)^2 = p^2 - 2pq\cos\theta + q^2(\cos^2\theta + \sin^2\theta) \), and \( \cos^2\theta + \sin^2\theta = 1 \) (Pythagoras again, on the unit circle) leaves the formula. Khan Academy's [law of cosines proof](https://www.khanacademy.org/math/trigonometry) walks through it with pictures.

It's Pythagoras with a correction term. At \( \theta = 90° \), \( \cos\theta = 0 \) and it *is* Pythagoras. For a narrower angle the third side is shorter than Pythagoras would say, so something is subtracted; for a wider angle, \( \cos\theta \) is negative and the side is longer. For our triangle:

\[
\|\mathbf{a} - \mathbf{b}\|^2 = \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 - 2\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta
\]

**Step 2: the same length, by algebra.** A squared length is a vector dotted with itself, and the dot product distributes like ordinary multiplication:

\[
\|\mathbf{a} - \mathbf{b}\|^2 = (\mathbf{a} - \mathbf{b}) \cdot (\mathbf{a} - \mathbf{b}) = \mathbf{a}\cdot\mathbf{a} - 2\,\mathbf{a}\cdot\mathbf{b} + \mathbf{b}\cdot\mathbf{b} = \|\mathbf{a}\|^2 - 2\,\mathbf{a}\cdot\mathbf{b} + \|\mathbf{b}\|^2
\]

(The middle step is like expanding \( (x - y)^2 = x^2 - 2xy + y^2 \), using \( \mathbf{a}\cdot\mathbf{b} = \mathbf{b}\cdot\mathbf{a} \) to merge the two cross terms.)

**Step 3: compare.** Both lines equal \( \|\mathbf{a} - \mathbf{b}\|^2 \). Cancel \( \|\mathbf{a}\|^2 + \|\mathbf{b}\|^2 \) from both, then divide by \( -2 \):

\[
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\,\|\mathbf{b}\|\cos\theta
\]

Check with \( \mathbf{a} = [3, 0] \), \( \mathbf{b} = [2, 2] \): \( \mathbf{a} - \mathbf{b} = [1, -2] \), so \( \|\mathbf{a} - \mathbf{b}\|^2 = 5 \). The algebra side gives \( 9 - 2 \cdot 6 + 8 = 5 \). The geometry side gives \( 9 + 8 - 2 \cdot 6 = 5 \). Same number.

### The sign, and orthogonality

Lengths are never negative, so the sign of \( \mathbf{a} \cdot \mathbf{b} \) is the sign of \( \cos\theta \):

| \( \mathbf{a} \cdot \mathbf{b} \) | angle | meaning |
|---|---|---|
| positive | less than 90° | they point broadly the same way |
| zero | exactly 90° | they are at right angles: **orthogonal** |
| negative | more than 90° | they point broadly in opposite ways |

**Orthogonal** (from Greek for "right-angled") is the word for "dot product zero". In 2-D it means perpendicular; in 768-D it means the same thing, even though you can't draw it. A handy 2-D rule: \( [x, y] \) and \( [-y, x] \) are always orthogonal, because \( x \cdot (-y) + y \cdot x = 0 \).

```python
e = np.array([1.0, 0.0])
e @ np.array([1.0, 1.0])       # np.float64(1.0)    45°: positive
e @ np.array([0.0, 1.0])       # np.float64(0.0)    90°: orthogonal
e @ np.array([-1.0, 1.0])      # np.float64(-1.0)   135°: negative

np.array([2.0, 1.0]) @ np.array([-1.0, 2.0])             # np.float64(0.0)   the [x, y], [−y, x] rule
np.array([1.0, 2.0, 3.0]) @ np.array([3.0, 0.0, -1.0])   # np.float64(0.0)   orthogonal in 3-D
```

Why orthogonality matters in ML: two orthogonal embeddings share nothing; two word-count vectors with no words in common have a dot product of exactly 0; and Day 24 and Day 25 build whole coordinate systems out of mutually orthogonal unit vectors.

!!! warning "Common mistake: testing orthogonality with `== 0`"
    Vectors that come out of a calculation are rarely *exactly* orthogonal, because of float rounding. Two unit vectors at exactly 30° and 120° (90° apart on paper):

    ```python
    t = np.radians(30)                       # degrees → radians
    p = np.array([np.cos(t), np.sin(t)])
    r = np.array([-np.sin(t), np.cos(t)])    # the [−y, x] rule again
    p @ r                                    # np.float64(7.437084071668725e-18)
    p @ r == 0                               # np.False_
    np.isclose(p @ r, 0.0)                   # np.True_
    ```

    `np.radians` converts degrees to radians, the reverse of `np.degrees`; `np.sin` is the sine, the y-coordinate partner of the cosine on the unit circle (Day 4 lists both among the ufuncs). `7.4e-18` means \( 7.4 \times 10^{-18} \), a rounding crumb. (`np.False_` and `np.True_` are NumPy's booleans, displayed the same way as `np.float64`.) **Fix:** compare with a tolerance, `np.isclose(x, 0.0)` or `abs(x) < 1e-9`.

### Projection length: the shadow, in numbers

The shadow from the intuition has a name: the **scalar projection** of \( \mathbf{b} \) onto \( \mathbf{a} \), "how far \( \mathbf{b} \) reaches along \( \mathbf{a} \)'s direction". Rearranging the formula:

\[
\text{shadow of } \mathbf{b} \text{ on } \mathbf{a} = \|\mathbf{b}\|\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|} = \hat{\mathbf{a}} \cdot \mathbf{b}
\]

The last form is the useful one: **dotting with a unit vector measures how far you reach along that direction**. With \( \mathbf{a} = [3, 0] \) and \( \mathbf{b} = [2, 2] \): \( 6 / 3 = 2 \), as in the picture. The shadow as an arrow (the **vector projection**) is that length times the unit direction: \( 2 \times [1, 0] = [2, 0] \).

```python
a = np.array([3.0, 0.0])
b = np.array([2.0, 2.0])
a_hat = a / np.linalg.norm(a)
a_hat @ b                 # np.float64(2.0)          shadow length
(a_hat @ b) * a_hat       # array([2., 0.])          the shadow as a vector
```

!!! warning "Common mistake: the wrong denominator, or none"
    Both slips are silent:

    ```python
    a @ b                       # np.float64(6.0)                  WRONG: 3 × the shadow (‖a‖ is still in it)
    a @ b / np.linalg.norm(b)   # np.float64(2.1213203435596424)   WRONG: divided by the wrong length
    a_hat @ b                   # np.float64(2.0)                  right: the shadow of b on a
    ```

    The shadow of \( \mathbf{b} \) on \( \mathbf{a} \) divides by the length of \( \mathbf{a} \), the vector you're projecting *onto*. (Dividing by \( \|\mathbf{b}\| \) gives the shadow of \( \mathbf{a} \) on \( \mathbf{b} \), a different number.) **Check:** a shadow can never be longer than the thing casting it, so its size must be at most \( \|\mathbf{b}\| \), here \( 2.83 \). The raw 6 fails that check at once.

This is a preview. Day 25 builds projections into least squares and PCA; for today, remember "unit vector dot something = how much of that something lies along the unit vector".

## Recovering the angle with `arccos`

**What it is.** Turning the formula around gives the angle itself:

\[
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}
\qquad\Longrightarrow\qquad
\theta = \arccos\!\left(\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}\right)
\]

\( \arccos \) ("arc-cosine", the inverse of cosine) answers "which angle between 0° and 180° has this cosine?". `np.arccos` returns it in radians; `np.degrees` converts radians to degrees (multiplies by \( 180/\pi \)).

**Why it exists.** An angle is often easier to reason about than a raw cosine: "these two embeddings are 80° apart" says "nearly unrelated" at a glance. It also works in any dimension, where you can't draw the angle at all.

**Worked by hand.** \( \mathbf{a} = [3, 0] \), \( \mathbf{b} = [2, 2] \): \( \cos\theta = 6 / (3 \times 2.828) = 0.7071 \), and the angle with that cosine is 45°.

```python
def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between a and b in degrees, from 0 to 180."""
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    cos = np.clip(cos, -1.0, 1.0)     # guard against float error (see below)
    return float(np.degrees(np.arccos(cos)))

angle_deg(np.array([3.0, 0.0]), np.array([2.0, 2.0]))      # 45.00000000000001
angle_deg(np.array([1.0, 0.0]), np.array([0.0, 1.0]))      # 90.0
angle_deg(np.array([1.0, 0.0]), np.array([-3.0, 0.0]))     # 180.0
angle_deg(np.array([1.0, 1.0, 1.0]), np.array([1.0, 0.0, 0.0]))   # 54.735610317245346
```

The `45.00000000000001` is rounding in the last digit, as with `np.cos` above. The last one is the angle between a cube's long diagonal and one of its edges: a 3-D angle you could not easily measure with a protractor, computed in one line. `float(...)` converts the NumPy scalar to a plain Python float, so the function returns what its type hint promises.

**How it works underneath, and why the clip.** Mathematically the fraction is always between −1 and 1 (that's the **Cauchy–Schwarz inequality**: \( |\mathbf{a} \cdot \mathbf{b}| \le \|\mathbf{a}\|\|\mathbf{b}\| \), the shadow can never be longer than the thing casting it). But the computer works out the top and the bottom separately, each with its own rounding, and the fraction can land a hair outside the range. `arccos` of anything above 1 is undefined, so it returns `nan`.

!!! warning "Common mistake: `arccos` without clipping gives `nan` for parallel vectors"
    The angle between a vector and itself should be 0°. Without the clip:

    ```python
    v = np.array([1.0, 1.0, 1.0])
    cos = v @ v / (np.linalg.norm(v) * np.linalg.norm(v))
    cos                     # np.float64(1.0000000000000002)
    np.arccos(cos)
    # <cell>:4: RuntimeWarning: invalid value encountered in arccos
    # np.float64(nan)
    np.arccos(np.clip(cos, -1.0, 1.0))   # np.float64(0.0)
    ```

    Trace it: `v @ v` is exactly `3.0`. `np.linalg.norm(v)` is \( \sqrt{3} \) rounded to the nearest float, `1.7320508075688772`, and that times itself is `2.9999999999999996`, a tiny bit under 3. So the fraction is a tiny bit over 1. **Fix:** `np.clip(cos, -1.0, 1.0)` (Day 4b) forces the value back into range; it changes nothing for values already inside. Kotlin's `acos` returns `NaN` the same way, and its `coerceIn(-1.0, 1.0)` is the same fix.

---

**Part 5: Cosine similarity**

## Cosine similarity

**What it is.** **Cosine similarity** is the cosine of the angle between two vectors, computed from the formula above without taking the `arccos`:

\[
\operatorname{cos\_sim}(\mathbf{a}, \mathbf{b}) = \cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|\,\|\mathbf{b}\|}
\]

It's the dot product with both lengths divided out, so only **direction** is left. It always lies between −1 and 1.

**Why it exists.** For text, images and audio embeddings, direction carries the meaning and length mostly doesn't. Cosine similarity is the standard way to ask "how similar in meaning are these two things?". It is the scoring function behind **semantic search** and **RAG** (Retrieval-Augmented Generation, Phase 4): embed every document, embed the question, return the documents whose vectors have the highest cosine similarity to the question's vector, and hand those to the LLM.

**Intuition first.** Two people pointing at the same star from different distances are pointing in the same direction, however long their arms are. Cosine similarity compares where the arrows point and ignores how long they are.

<figure class="diagram" markdown>
![Animation on a unit circle: a short blue arrow a points right while a green arrow b of length 1 rotates through 0, 45, 90, 135 and 180 degrees; at each step a dashed line drops from b's tip to the x-axis and an orange dot marks the x-coordinate, 1.00, 0.71, 0.00, -0.71 and -1.00, which is the cosine similarity shown in the side panel](../../assets/img/phase-1/day05-cosine-angles.svg)
<figcaption>Animated: as b rotates away from a, the x-coordinate of its tip on the unit circle, cos θ, falls from 1 to −1.</figcaption>
</figure>

**Worked by hand.** \( \mathbf{a} = [1, 2] \), \( \mathbf{b} = [2, 4] \) (the same direction, twice as long): \( \mathbf{a} \cdot \mathbf{b} = 2 + 8 = 10 \); \( \|\mathbf{a}\| = \sqrt{5} \); \( \|\mathbf{b}\| = \sqrt{20} = 2\sqrt{5} \); so the cosine is \( 10 / (\sqrt{5} \cdot 2\sqrt{5}) = 10 / 10 = 1 \). Scaling a vector by any positive number never changes its cosine similarity with anything.

```python
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of a and b. Returns 0.0 if either is the zero vector."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0               # a zero vector has no direction: call it "unrelated"
    return float(a @ b / (na * nb))

a = np.array([1.0, 2.0])
cosine(a, 2 * a)                     # 0.9999999999999998   same direction: 1, up to rounding
cosine(a, 100 * a)                   # 1.0                  length is ignored
cosine(a, np.array([-2.0, 1.0]))     # 0.0                  orthogonal
cosine(a, -a)                        # -0.9999999999999998  opposite: −1, up to rounding
cosine(a, np.array([2.0, 1.0]))      # 0.7999999999999998   about 37° apart
cosine(a, np.zeros(2))               # 0.0                  the guard
```

The `0.9999999999999998` values are the same last-digit rounding you met with `arccos`: the true answers are exactly 1 and −1. For ranking documents that crumb never matters; before an `arccos`, clip.

Returning `0.0` for a zero vector is a choice, the same one Day 6's mini-project makes: an empty document is treated as unrelated to everything. Raising an error, as `normalize` did, is the other reasonable choice; returning `nan` silently is not.

**Interpreting the value.** Now that you've seen where the number comes from, here's a summary:

| cosine similarity | angle | reading |
|---|---|---|
| 1 | 0° | same direction (possibly different lengths) |
| around 0.7 to 0.9 | about 25° to 45° | closely related |
| 0 | 90° | orthogonal: nothing in common |
| negative | over 90° | pointing apart; opposite at −1 |

Two cautions. Vectors with no negative components, such as word counts or pixel brightnesses, can never be more than 90° apart, so their cosine lies between 0 and 1. And the thresholds for "related" depend on the embedding model: some models put almost every pair of sentences between 0.6 and 0.9, so compare scores *within* one model, never across models.

A toy with hand-made 3-number "embeddings" whose dimensions mean `[royalty, masculinity, food]`:

```python
king  = np.array([0.95, 0.90, 0.05])
queen = np.array([0.93, 0.10, 0.05])
apple = np.array([0.02, 0.40, 0.98])

cosine(king, queen)    # 0.7956442717964952    related: both royal
cosine(king, apple)    # 0.3087078305046096
cosine(queen, apple)   # 0.10850491102206603   barely related
```

Real embeddings have hundreds of dimensions, and the dimensions don't have human-readable meanings; a model learns them. The maths is identical.

**How it works underneath: cosine is the dot product of unit vectors.** Divide each vector by its length first, and the denominator becomes \( 1 \times 1 \):

\[
\cos\theta = \frac{\mathbf{a}}{\|\mathbf{a}\|} \cdot \frac{\mathbf{b}}{\|\mathbf{b}\|} = \hat{\mathbf{a}} \cdot \hat{\mathbf{b}}
\]

That's why vector databases normalise every stored embedding once: after that, cosine similarity costs one dot product per document, with no square roots at search time.

### Cosine versus Euclidean distance, on unit vectors

For unit vectors, the distance between the tips is a function of the angle alone. Take the Step 2 expansion from the derivation and use \( \|\hat{\mathbf{a}}\| = \|\hat{\mathbf{b}}\| = 1 \) and \( \hat{\mathbf{a}} \cdot \hat{\mathbf{b}} = \cos\theta \):

\[
\|\hat{\mathbf{a}} - \hat{\mathbf{b}}\|^2 = \|\hat{\mathbf{a}}\|^2 - 2\,\hat{\mathbf{a}}\cdot\hat{\mathbf{b}} + \|\hat{\mathbf{b}}\|^2 = 1 - 2\cos\theta + 1 = 2 - 2\cos\theta
\]

Read it at the extremes: same direction (\( \cos\theta = 1 \)) gives distance 0; orthogonal (\( \cos\theta = 0 \)) gives \( \sqrt{2} \); opposite (\( \cos\theta = -1 \)) gives \( \sqrt{4} = 2 \), the diameter of the unit circle. A bigger cosine always means a smaller distance, so **on normalised vectors, ranking by highest cosine and ranking by smallest Euclidean distance give the same order.** Worked check with \( \mathbf{a} = [1, 2] \), \( \mathbf{b} = [2, 1] \): the cosine is \( 4/5 = 0.8 \), so the squared distance between the unit vectors should be \( 2 - 1.6 = 0.4 \).

```python
a = np.array([1.0, 2.0])
b = np.array([2.0, 1.0])
a_hat, b_hat = normalize(a), normalize(b)
c = a_hat @ b_hat
c                                      # np.float64(0.7999999999999999)
np.linalg.norm(a_hat - b_hat) ** 2     # np.float64(0.4)
2 - 2 * c                              # np.float64(0.40000000000000013)
```

Both are 0.4, up to rounding in the last digit.

!!! warning "Common mistake: expecting the two rankings to agree on raw vectors"
    The equivalence needs unit vectors. On raw vectors, length gets into the distance, and the two measures can rank candidates in opposite orders:

    ```python
    query = np.array([1.0, 0.0])
    c1 = np.array([10.0, 1.0])     # almost the same direction, but long
    c2 = np.array([0.5, 0.5])      # 45° away, but short and nearby
    cosine(query, c1), cosine(query, c2)                     # (0.9950371902099892, 0.7071067811865475)
    np.linalg.norm(query - c1), np.linalg.norm(query - c2)   # (np.float64(9.055385138137417), np.float64(0.7071067811865476))
    ```

    Cosine says `c1` is the better match; Euclidean distance says `c2` is far closer. Neither is broken: they answer different questions. **Fix:** normalise first if direction is what matters, or choose one measure on purpose. Day 22 picks this up with the cosine *distance* \( 1 - \cos\theta \) and the other norms.

---

**Part 6: Vectors at work in ML**

## Toy semantic search

Here is the whole retrieval step of a RAG system, at toy size. Four "embedding" dimensions with readable meanings, `[battery, camera, payments, crash]`, and five support articles:

```python
docs = ["Battery drains fast overnight",
        "Camera app freezes on launch",
        "Refund for a double payment",
        "Phone gets hot and the battery dies",
        "Long FAQ page that mentions everything"]
E = np.array([[0.9, 0.0, 0.1, 0.2],
              [0.0, 0.8, 0.0, 0.6],
              [0.0, 0.0, 0.9, 0.1],
              [0.8, 0.1, 0.0, 0.3],
              [2.0, 2.0, 2.0, 2.0]])     # one row per document; the last one is long
q = np.array([0.85, 0.0, 0.0, 0.1])     # the question: "my battery dies quickly"
E.shape, q.shape                        # ((5, 4), (4,))
```

**Score with raw dot products first.** One dot product per document row. `E @ q` computes all five at once: a matrix times a vector is one dot product per row of the matrix (Day 10 explains why).

```python
[float(row @ q) for row in E]   # [0.785, 0.06, 0.010000000000000002, 0.7100000000000001, 1.9]
raw = E @ q
raw                             # array([0.785, 0.06 , 0.01 , 0.71 , 1.9  ])
```

The "mentions everything" page wins, with 1.9. It isn't about batteries; it's *long*, and the raw dot product rewards length (\( \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta \) has the lengths in it).

**Now score with cosine similarity.** Normalise every row and the query, then take dot products. `np.linalg.norm(E, axis=1, keepdims=True)` gives one length per row as a `(5, 1)` column (Day 4's `keepdims`). Dividing the `(5, 4)` table by that `(5, 1)` column divides every column of row \( i \) by length \( i \), so each row is divided by its own length (a broadcast; Day 9 gives the rule):

```python
E_unit = E / np.linalg.norm(E, axis=1, keepdims=True)
q_unit = q / np.linalg.norm(q)
scores = E_unit @ q_unit
scores.round(3)                 # array([0.989, 0.07 , 0.013, 0.964, 0.555])

order = np.argsort(-scores)     # indices from highest score to lowest
order                           # array([0, 3, 4, 1, 2])
for i in order[:2]:
    print(f"{scores[i]:.3f}  {docs[i]}")
# 0.989  Battery drains fast overnight
# 0.964  Phone gets hot and the battery dies
```

`np.argsort` (Day 4b) returns the indices that would sort an array in increasing order; sorting `-scores` instead gives decreasing order, so `order[:2]` is the top two. The two battery articles now come first and the long page drops to third. That is the reason semantic search uses cosine similarity, or normalises once and then uses the dot product.

!!! android "Android bridge: embeddings on the device"
    MediaPipe's **Text Embedder** and **Image Embedder** tasks run a small embedding model on the phone and return a vector for a piece of text or a photo. The task classes also provide a static `cosineSimilarity()` helper that compares two of those embeddings, and the embedder options include an L2-normalisation setting, which is the "normalise once at insert time" idea from this lesson. That's how an app can match a user's question to FAQ entries, or find "photos like this one", entirely offline. MediaPipe is covered in Phase 6 (module 6.7); the scoring it uses is the code above.

## A neuron is a dot product plus a number

**What it is.** An artificial **neuron** takes an input vector \( \mathbf{x} \) (the **features**), has a **weight** vector \( \mathbf{w} \) of the same dimension and one extra number \( b \) called the **bias**, and computes

\[
z = \mathbf{w} \cdot \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
\]

The weights and the bias are **learned** during training; \( z \) is the neuron's raw output, before an activation function (Day 12) bends it.

**Why it exists.** It's the smallest unit of every neural network. Each weight says how much its feature matters and in which direction (positive pushes \( z \) up, negative pushes it down); the bias shifts the result when every feature is 0. A **layer** is many neurons reading the same input, and Day 12 shows it turns into one matrix multiplication.

**Intuition first.** Back to the shopping bill: \( \mathbf{x} \) is the basket, \( \mathbf{w} \) the price list, \( b \) a fixed delivery charge. Training a neuron is adjusting the price list until the "bills" come out as the right answers.

**Worked by hand.** \( \mathbf{x} = [2, 1, 3] \), \( \mathbf{w} = [0.5, -0.2, 0.1] \), \( b = 0.4 \):

\[
z = 0.5 \cdot 2 + (-0.2) \cdot 1 + 0.1 \cdot 3 + 0.4 = 1.0 - 0.2 + 0.3 + 0.4 = 1.5
\]

```python
x = np.array([2.0, 1.0, 3.0])     # input features
w = np.array([0.5, -0.2, 0.1])    # learned weights
b = 0.4                           # learned bias
z = w @ x + b
z                                 # np.float64(1.5)
```

You'll also see this written \( \mathbf{w}^\top \mathbf{x} + b \) in papers and on Day 12: the same number, in matrix notation (Day 11 explains the \( \top \)).

**Underneath: a neuron is a pattern detector.** By the angle formula, \( \mathbf{w} \cdot \mathbf{x} = \|\mathbf{w}\|\|\mathbf{x}\|\cos\theta \). For inputs of the same length, \( z \) is largest when \( \mathbf{x} \) points the same way as \( \mathbf{w} \), equals \( b \) when \( \mathbf{x} \) is orthogonal to \( \mathbf{w} \), and is most negative when \( \mathbf{x} \) points the opposite way. The weight vector *is* the pattern the neuron responds to, and training means turning \( \mathbf{w} \) to point at useful patterns.

```python
w_hat = normalize(w)
aligned  = 2 * w_hat                     # length 2, pointing along w
opposite = -2 * w_hat                    # length 2, pointing against w
ortho    = 2 * normalize(np.array([0.2, 0.5, 0.0]))   # length 2, and w @ [0.2, 0.5, 0] = 0
float(w @ aligned + b), float(w @ ortho + b), float(w @ opposite + b)
# (1.495445115010332, 0.39999999999999997, -0.6954451150103321)
```

Same input length, three directions, three very different outputs: \( 2\|\mathbf{w}\| + b \), \( b \), and \( -2\|\mathbf{w}\| + b \), with \( \|\mathbf{w}\| = \sqrt{0.30} \approx 0.548 \).

!!! warning "Common mistake: feeding unscaled features to a neuron"
    The dot product adds up \( w_i x_i \) terms, so a feature measured in big units swamps the others before training has even started. A house as `[bedrooms, area_m2, age_years]` with every weight starting at 0.1:

    ```python
    house = np.array([3.0, 120.0, 15.0])
    w0 = np.array([0.1, 0.1, 0.1])
    w0 * house                        # array([ 0.3, 12. ,  1.5])   each feature's contribution
    (w0 * house) / (w0 @ house)       # array([0.02173913, 0.86956522, 0.10869565])
    ```

    The area alone is 87% of the output, only because square metres are bigger numbers than bedroom counts. Training would have to shrink the area weight a hundredfold before the other features could matter, which makes it slow and unstable. **Fix:** put features on comparable scales first. The crude version divides each feature by a typical size; Day 9 does it properly with **standardisation** (subtract each column's mean, divide by its standard deviation), and Day 22 shows the same problem wrecking distances.

    ```python
    typical = np.array([5.0, 200.0, 50.0])   # rough scale of each feature
    scaled = house / typical
    scaled                                    # array([0.6, 0.6, 0.3])
    w0 * scaled                               # array([0.06, 0.06, 0.03])   now comparable
    ```

## Many vectors at once: a preview

ML code almost never handles one vector at a time. A **batch** of inputs is stored as a 2-D array, one vector per row, and one operation handles all of them. You've already done it twice today: `np.linalg.norm(houses - query, axis=1)` and `E @ q`. For a neuron:

```python
X = np.array([[2.0, 1.0, 3.0],     # input 0 (the one worked by hand)
              [0.0, 1.0, 0.0],     # input 1
              [1.0, 0.0, 1.0]])    # input 2
X.shape                            # (3, 3): 3 inputs, 3 features each
[float(w @ row + b) for row in X]  # [1.5, 0.2, 1.0]   one dot product per row, by loop
X @ w + b                          # array([1.5, 0.2, 1. ])   all rows at once
```

`X @ w` is "dot `w` with every row of `X`", giving shape `(3,)`, and `+ b` adds the bias to each (a broadcast). Day 10 explains matrix multiplication properly, and Day 12 stacks many neurons' weight vectors into a matrix to make a whole layer.

## Try it: drag the vectors

Drag the dots at the tips of **a** and **b**. The panel shows both lengths, the dot product written out term by term, the angle, and the cosine similarity.

<div class="vector-playground"></div>

!!! example "Experiments to try"
    1. Make **a** and **b** point the same way, then make **b** twice as long. The dot product doubles; the cosine stays 1.00 and the angle 0°.
    2. Put them at right angles, for example `[2, 0]` and `[0, 3]`. The dot product is exactly 0.
    3. Point them in opposite directions. The dot product and the cosine both go negative; the angle is 180°.
    4. Find two vectors with a dot product of exactly 0 that are **not** on the axes. (Hint: the \( [x, y] \), \( [-y, x] \) rule. `[1, 2]` and `[-2, 1]` work.)
    5. Set **a** = `[3, 0]` and **b** = `[2, 2]`, the worked example. Check that the panel shows 6.00 and 45°.

## Going further

- **Other norms (L1, L∞) and cosine distance**: [Day 22](day-22-norms-distances.md) teaches the whole family of rulers, when to use which, and nearest-neighbour search.
- **Span, basis and linear independence**, which grow out of linear combinations: [Day 23](day-23-rank-span.md).
- **Projections, least squares and PCA**, which grow out of the shadow: [Day 25](day-25-projections-pca.md).
- **`np.vecdot(A, B)`** (NumPy 2.0+): one dot product per pair of matching rows, without writing a loop. See the [NumPy linear algebra reference](https://numpy.org/doc/stable/reference/routines.linalg.html).
- **The cross product** (`np.cross`): a 3-D-only operation that gives a vector perpendicular to two others; used in graphics and physics, rarely in ML. See the same reference.
- **Vector spaces in the abstract** (the axioms that make something "a vector"): [Mathematics for Machine Learning, chapter 2](https://mml-book.github.io/).
- **Approximate nearest-neighbour indexes** (FAISS, HNSW), which make cosine search fast over millions of vectors: covered with RAG in Phase 4; see the [FAISS wiki](https://github.com/facebookresearch/faiss/wiki).

---

## Exercises

**1. Warm-up: by hand, then in NumPy.** For \( \mathbf{a} = [2, -1, 3] \) and \( \mathbf{b} = [0, 4, 1] \), compute by hand \( \mathbf{a} + \mathbf{b} \), \( \mathbf{a} - \mathbf{b} \), \( 3\mathbf{a} \), \( \|\mathbf{a}\| \), the distance between \( \mathbf{a} \) and \( \mathbf{b} \), and \( \mathbf{a} \cdot \mathbf{b} \). Then check every answer in NumPy.

**2. Warm-up: cosine without NumPy.** Write `cosine_list(a: list[float], b: list[float]) -> float` in plain Python, using `math.sqrt`, `sum` and `zip(..., strict=True)`, returning `0.0` when either vector is all zeros. Check it against the lesson's NumPy `cosine` on `[1, 2, 3]` and `[3, 0, -1]`, on `[1, 2]` and `[2, 1]`, and on `[3, 4]` and `[0, 0]`, and show that it raises for vectors of different lengths.

**3. Core: normalise a whole batch.** Take the batch `B = np.array([[3., 4., 0.], [1., 1., 1.], [0., 2., 0.], [2., -1., 2.]])`, one vector per row. (a) First predict what `B / np.linalg.norm(B, axis=1)` does, then run it. (b) Normalise every row correctly in one line with `keepdims=True`, and show every row of the result `U` has length 1. (c) Compute the cosine of every row with row 0 using one `@`, and check it against `cosine(B[i], B[0])` for each row. (d) Check \( \|\hat{\mathbf{u}}_0 - \hat{\mathbf{u}}_1\|^2 = 2 - 2\cos\theta \) for rows 0 and 1. Use `np.allclose` / `np.isclose` for the comparisons.

**4. Core: mini word-similarity.** Represent these sentences as word-count vectors over the vocabulary `["model", "phone", "train", "battery", "data"]`, then compute the cosine similarity and the angle in degrees for every pair, and name the most similar pair:

- "train the model on data"
- "the phone battery died"
- "more data and more data helps the model"

**5. Core: angles in high dimensions.** The angle between a cube's diagonal \( [1, 1, 1] \) and an edge \( [1, 0, 0] \) is about 54.7°. Compute the angle between the all-ones vector and \( \mathbf{e}_1 = [1, 0, \dots, 0] \) in 2, 3, 10, 100 and 768 dimensions. What happens, and why? (Hint: compute \( \cos\theta \) by hand first.)

**6. Predict the output.** Write down what each line prints *before* running it:

```python
import numpy as np
a, b = [1, 2], [3, 4]
print(a + b)
print(np.array(a) + np.array(b))
print(np.array(a) * np.array(b))
print(np.array(a) @ np.array(b))
print(np.sqrt(np.array([3, -4]) ** 2).sum())
print(np.linalg.norm(-2 * np.array([3, -4])))
```

**7. Find and fix the bug.** This function should give the angle between two vectors in degrees. It returns `nan` for `[1, 0]` and `[1, 1]`, which are 45° apart. Find the bug, fix it, then test the fixed version on `v` and `v` for `v = [1, 1, 1]` (expect 0°) and fix the second bug you find.

```python
def angle_between(a, b):
    cos = a @ b / np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(cos)))

angle_between(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
# <cell>:3: RuntimeWarning: invalid value encountered in arccos
# nan
```

**8. Challenge: a tiny vector search.** Write `top_k(query, docs, k)` that returns the indices and cosine scores of the `k` rows of `docs` most similar to `query`, highest first. A row that is all zeros must get score 0, not `nan`. Test it on 8 random 16-dimensional "embeddings" from `rng = np.random.default_rng(5)`, with row 3 set to zeros and the query made as `docs[5] + 0.1 * rng.normal(size=16)`; row 5 should come first. Then show that ranking the normalised non-zero rows by *smallest Euclidean distance* to the normalised query gives the same order.

??? success "Solutions"
    **1.** By hand: \( \mathbf{a} + \mathbf{b} = [2, 3, 4] \); \( \mathbf{a} - \mathbf{b} = [2, -5, 2] \); \( 3\mathbf{a} = [6, -3, 9] \); \( \|\mathbf{a}\| = \sqrt{4 + 1 + 9} = \sqrt{14} \approx 3.742 \); distance \( = \|[2, -5, 2]\| = \sqrt{4 + 25 + 4} = \sqrt{33} \approx 5.745 \); \( \mathbf{a} \cdot \mathbf{b} = 0 - 4 + 3 = -1 \).

    ```python
    import numpy as np

    a = np.array([2.0, -1.0, 3.0])
    b = np.array([0.0, 4.0, 1.0])
    a + b                    # array([2., 3., 4.])
    a - b                    # array([ 2., -5.,  2.])
    3 * a                    # array([ 6., -3.,  9.])
    np.linalg.norm(a)        # np.float64(3.7416573867739413)
    np.linalg.norm(a - b)    # np.float64(5.744562646538029)
    a @ b                    # np.float64(-1.0)
    ```

    Every operation is component by component except the last two, which reduce a vector to one number. The negative dot product says the angle between them is a little over 90°.

    **2.**

    ```python
    import math
    import numpy as np

    def cosine_list(a: list[float], b: list[float]) -> float:
        """Cosine similarity of two plain lists. 0.0 if either is all zeros."""
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def cosine(a, b):                        # the lesson's NumPy version, for comparison
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(a @ b / (na * nb))

    cosine_list([1, 2, 3], [3, 0, -1])       # 0.0
    cosine_list([1, 2], [2, 1])              # 0.7999999999999998
    cosine(np.array([1.0, 2.0]), np.array([2.0, 1.0]))   # 0.7999999999999998
    cosine_list([3, 4], [0, 0])              # 0.0
    try:
        cosine_list([1, 2, 3], [1, 2])
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: zip() argument 2 is shorter than argument 1
    ```

    The three `sum(...)` calls are the formula's three sums: \( \sum a_i b_i \), \( \sum a_i^2 \) and \( \sum b_i^2 \). `zip(strict=True)` turns a length mismatch into an error instead of a silently truncated dot product. Both versions agree to the last digit on these small inputs; NumPy's version is the one to use on real embeddings, because its loops run in compiled code.

    **3.**

    ```python
    import numpy as np

    def cosine(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    B = np.array([[3., 4., 0.],
                  [1., 1., 1.],
                  [0., 2., 0.],
                  [2., -1., 2.]])

    # (a) without keepdims: norms has shape (4,), which lines up with the 3 COLUMNS
    try:
        B / np.linalg.norm(B, axis=1)
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: operands could not be broadcast together with shapes (4,3) (4,)

    # (b) keepdims makes the norms a (4, 1) column: one length per row
    U = B / np.linalg.norm(B, axis=1, keepdims=True)
    np.linalg.norm(U, axis=1)            # array([1., 1., 1., 1.])

    # (c) the cosine of every row with row 0, in one @
    scores = U @ U[0]
    scores.round(4)                      # array([1.    , 0.8083, 0.8   , 0.1333])
    np.allclose(scores, [cosine(row, B[0]) for row in B])   # True

    # (d) the unit-vector identity, for rows 0 and 1
    c = U[0] @ U[1]
    np.isclose(np.linalg.norm(U[0] - U[1]) ** 2, 2 - 2 * c)  # np.True_
    ```

    Without `keepdims`, the `(4,)` array of lengths is matched against the last axis of `B`, which has 3 columns, so NumPy refuses. (With a square batch it would *not* refuse: it would silently divide column \( j \) by row \( j \)'s length, which is worse.) With `keepdims=True` the lengths form a `(4, 1)` column, so row \( i \) is divided by its own length. `U @ U[0]` is one dot product per row, and for unit vectors a dot product *is* the cosine, which is why the scores match `cosine(row, B[0])`. This is exactly what a vector database does at insert time.

    **4.**

    ```python
    import numpy as np

    vocab = ["model", "phone", "train", "battery", "data"]

    def to_vec(sentence):
        words = sentence.split()
        return np.array([words.count(w) for w in vocab], dtype=float)

    def cosine(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    sentences = ["train the model on data",
                 "the phone battery died",
                 "more data and more data helps the model"]
    vecs = [to_vec(s) for s in sentences]
    vecs[0], vecs[1], vecs[2]
    # (array([1., 0., 1., 0., 1.]), array([0., 1., 0., 1., 0.]), array([1., 0., 0., 0., 2.]))
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        c = cosine(vecs[i], vecs[j])
        deg = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
        print(f"s{i} vs s{j}: cosine {c:.3f}, angle {deg:.1f}°")
    # s0 vs s1: cosine 0.000, angle 90.0°
    # s0 vs s2: cosine 0.775, angle 39.2°
    # s1 vs s2: cosine 0.000, angle 90.0°
    ```

    `words.count(w)` counts how often each vocabulary word appears, so each sentence becomes a 5-component count vector. Sentences 0 and 2 share "model" and "data", but sentence 2 says "data" twice and never says "train", so the vectors point in similar but not identical directions: by hand, \( (1 + 2) / (\sqrt{3}\sqrt{5}) = 3/\sqrt{15} \approx 0.775 \), about 39°. They are the most similar pair. Sentence 1 shares none, so its dot product with the others is 0: orthogonal. Notice the weakness: "the phone battery died" and "the battery ran out" would also score 0, despite meaning the same thing, because counting words knows nothing about meaning. Learned embeddings fix this; Day 6 builds this bag-of-words model properly, and Phase 4 replaces it.

    **5.** By hand: the all-ones vector in \( n \) dimensions has length \( \sqrt{n} \), \( \mathbf{e}_1 \) has length 1, and their dot product is 1 (only the first component overlaps). So \( \cos\theta = 1/\sqrt{n} \).

    ```python
    import numpy as np

    for n in [2, 3, 10, 100, 768]:
        ones = np.ones(n)
        e1 = np.zeros(n)
        e1[0] = 1.0
        cos = ones @ e1 / (np.linalg.norm(ones) * np.linalg.norm(e1))
        print(n, round(float(np.degrees(np.arccos(cos))), 2))
    # 2 45.0
    # 3 54.74
    # 10 71.57
    # 100 84.26
    # 768 87.93
    ```

    As \( n \) grows, \( 1/\sqrt{n} \) shrinks towards 0, so the angle creeps towards 90°. In high dimensions a vector spread evenly over every axis is nearly orthogonal to any single axis. This is the first glimpse of how strange high-dimensional geometry is; Day 22's "curse of dimensionality" experiment shows the same effect for distances.

    **6.** Predictions, then the real output:

    ```python
    import numpy as np
    a, b = [1, 2], [3, 4]
    print(a + b)                                      # [1, 2, 3, 4]
    print(np.array(a) + np.array(b))                  # [4 6]
    print(np.array(a) * np.array(b))                  # [3 8]
    print(np.array(a) @ np.array(b))                  # 11
    print(np.sqrt(np.array([3, -4]) ** 2).sum())      # 7.0
    print(np.linalg.norm(-2 * np.array([3, -4])))     # 10.0
    ```

    Line 1 is list concatenation, not addition. Lines 2 and 3 are element by element (`print` shows arrays without commas). Line 4 is the dot product \( 3 + 8 = 11 \), an integer because both arrays are integer arrays. Line 5 is the misplaced-parenthesis bug: square, root, then sum gives \( 3 + 4 = 7 \), not the length 5. Line 6 is the scaling rule: \( |-2| \times 5 = 10 \).

    **7.** Bug 1 is operator precedence. `/` and `*` have the same precedence and run left to right, so `a @ b / np.linalg.norm(a) * np.linalg.norm(b)` means `(a @ b / ‖a‖) * ‖b‖`: it multiplies by the second length instead of dividing. For `[1, 0]` and `[1, 1]` that's \( (1 / 1) \times 1.414 = 1.414 \), outside \( [-1, 1] \), so `arccos` gives `nan`. Bug 2 shows up after the fix: for parallel vectors, rounding can push the cosine to `1.0000000000000002`. The fix for both:

    ```python
    import numpy as np

    def angle_between(a, b):
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))   # bug 1: parentheses
        cos = np.clip(cos, -1.0, 1.0)                            # bug 2: float error
        return float(np.degrees(np.arccos(cos)))

    angle_between(np.array([1.0, 0.0]), np.array([1.0, 1.0]))   # 45.00000000000001
    v = np.array([1.0, 1.0, 1.0])
    angle_between(v, v)                                         # 0.0
    ```

    The `45.00000000000001` is rounding in the last digit. Without the clip, `angle_between(v, v)` returns `nan` with a `RuntimeWarning`, as in the Common mistake box in "Recovering the angle". Whenever you divide by a product, write the parentheses.

    **8.**

    ```python
    import numpy as np

    def top_k(query, docs, k):
        """Indices and cosine scores of the k rows of docs most similar to query."""
        doc_norms = np.linalg.norm(docs, axis=1)            # one length per row, shape (n,)
        q_norm = np.linalg.norm(query)
        dots = docs @ query                                  # one dot product per row
        safe = np.where(doc_norms == 0, 1.0, doc_norms)      # avoid dividing by 0
        scores = np.where(doc_norms == 0, 0.0, dots / (safe * q_norm))
        order = np.argsort(-scores)[:k]                      # highest first
        return order, scores[order]

    rng = np.random.default_rng(5)
    docs = rng.normal(size=(8, 16))
    docs[3] = 0.0                                            # a blank document
    query = docs[5] + 0.1 * rng.normal(size=16)              # a slightly noisy copy of row 5

    idx, sc = top_k(query, docs, 3)
    idx                       # array([5, 1, 7])
    sc.round(3)               # array([0.997, 0.472, 0.367])
    idx_all, sc_all = top_k(query, docs, 8)
    sc_all[idx_all == 3]      # array([0.])   the blank row scores 0, not nan

    # Same order by Euclidean distance on unit vectors (non-zero rows only)
    keep = np.linalg.norm(docs, axis=1) > 0
    rows = np.flatnonzero(keep)                              # original indices of the kept rows
    U = docs[keep] / np.linalg.norm(docs[keep], axis=1, keepdims=True)
    u = query / np.linalg.norm(query)
    by_cos = rows[np.argsort(-(U @ u))]
    by_dist = rows[np.argsort(np.linalg.norm(U - u, axis=1))]
    by_cos                    # array([5, 1, 7, 2, 6, 0, 4])
    by_dist                   # array([5, 1, 7, 2, 6, 0, 4])
    np.array_equal(by_cos, by_dist)   # True
    ```

    `docs @ query` computes every row's dot product at once. `np.where(condition, x, y)` (Day 4b) picks `x` where the condition holds and `y` elsewhere; the `safe` array replaces zero lengths by 1 so the division never sees 0, and the outer `np.where` then sets those rows' scores to 0. `np.flatnonzero(keep)` gives the original positions of the `True` entries, so both rankings report original row numbers. One caution for real systems: a score of 0 still beats every negative score, so the blank row 3 would rank above rows with negative cosines in a long result list. A real index skips zero rows entirely (or gives them `-inf`). And a zero *query* would still give `nan` scores here; check it before searching. The two rankings match because, for unit vectors, distance² is \( 2 - 2\cos\theta \): a strictly decreasing function of the cosine.

## Resources

- [3Blue1Brown: Essence of Linear Algebra, chapter 1, "Vectors, what even are they?"](https://www.youtube.com/watch?v=fNk_zzaMoSs) and [chapter 9, "Dot products and duality"](https://www.youtube.com/watch?v=LyGKycYT2v0): the best visual introduction there is. Watch the whole series over Month 1.
- [Khan Academy: Vectors and spaces](https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces): short videos and practice problems with hints.
- [Immersive Math, chapter 2, "Vectors"](http://immersivemath.com/ila/ch02_vectors/ch02.html) and [chapter 3, "The Dot Product"](http://immersivemath.com/ila/ch03_dotproduct/ch03.html): an interactive textbook whose figures you can drag.
- [Mathematics for Machine Learning (free book), chapters 2 and 3](https://mml-book.github.io/): the formal treatment, for reference when papers get dense.
- [NumPy linear algebra reference](https://numpy.org/doc/stable/reference/routines.linalg.html): `norm`, `dot`, `vecdot`, `matmul` and the rest.
- [MediaPipe Text Embedder guide for Android](https://ai.google.dev/edge/mediapipe/solutions/text/text_embedder/android): embeddings and cosine similarity on the device.

## Checklist

- [ ] I can describe a vector as a list, an arrow and a point, read \( \mathbf{a} \), \( a_i \) and \( \mathbb{R}^n \), and tell the two meanings of "dimension" apart.
- [ ] I know why a Python list isn't a vector, and I create vectors as 1-D float arrays.
- [ ] I can add, subtract and scale vectors by hand and in NumPy, and draw each operation.
- [ ] I can write a vector as a linear combination, including of the standard basis vectors.
- [ ] I can compute a norm by Pythagoras in any dimension, and the distance between two points.
- [ ] I can normalise a vector, and I guard against the zero vector.
- [ ] I can compute a dot product by hand, with a loop and with `@`, and say what its sign means.
- [ ] I can derive \( \mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta \) and compute a projection length.
- [ ] I can find the angle between two vectors with `np.arccos`, clipping first.
- [ ] I can compute and interpret cosine similarity, and explain \( \|\hat{\mathbf{a}} - \hat{\mathbf{b}}\|^2 = 2 - 2\cos\theta \).
- [ ] I ranked documents with toy embeddings and can explain why cosine beat the raw dot product.
- [ ] I can compute \( \mathbf{w} \cdot \mathbf{x} + b \) for one input and a batch, and explain why features are scaled first.
- [ ] I played with the vector playground and did the five experiments.

**Next:** [Day 6: Week 1 review and mini-project](day-06-week1-review.md)
