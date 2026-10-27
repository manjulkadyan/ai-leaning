# Day 2: Python's Core Types, Through Kotlin Eyes

**Phase 1 · Week 1 · ~140 minutes**

This is the first of three parts of Day 2. It covers names and objects, numbers, booleans, `None`, type conversion and control flow. [Day 2a](day-02a-strings.md) covers strings, text and formatting, and [Day 2b](day-02b-collections.md) covers lists, tuples, dicts and sets in depth. Do them on consecutive days.

## Objectives

- [ ] Explain the difference between a **name** and an **object** in Python, and use `id`, `is`, `==`, `type` and `isinstance` to inspect them.
- [ ] Predict the result of `/`, `//`, `%`, `**`, `divmod` and `round`, including with negative numbers, and say where each differs from Kotlin.
- [ ] Explain why `0.1 + 0.2 != 0.3`, compare floats safely with `math.isclose`, and handle `inf` and `nan`.
- [ ] Use truthiness, and the fact that `and` / `or` return one of their operands, without falling into the "zero is falsy" trap.
- [ ] Write the Python idioms that replace Kotlin's `?.`, `?:` and `!!` when a value may be `None`.
- [ ] Convert between `int`, `float`, `str` and `bool`, and recognise the `ValueError` and `TypeError` each conversion can raise.
- [ ] Format numbers in output with the four basic f-string specs: `{x}`, `:.2f`, `:>8` and `:.1%`.
- [ ] Write `if` / `elif` / `else`, conditional expressions and chained comparisons.
- [ ] Loop with `for` over `range`, strings and lists, using `enumerate`, `zip` and `reversed`; control loops with `while`, `while True`, `break`, `continue` and `for ... else`.
- [ ] Use `match` where you'd use Kotlin's `when`, and avoid its capture-pattern trap.
- [ ] Control `print` with `sep` and `end`, and explain why `input` always returns a string.

**Prerequisites.** [Day 1](day-01-setup.md): you can run Python in the REPL, in a script or a notebook, and with the ▶ Run button on this site, and you've read your first traceback (the error report Python prints when something fails).

---

## The big idea

In Kotlin, a variable has a type that the compiler knows before the program runs. `val count: Int = 3` means "`count` holds an `Int`, forever". If you write `count = "three"`, the build fails.

Python moves that knowledge from the variable to the **value**. Every value in memory knows its own type. A variable is only a **name** that refers to a value, and the same name can refer to an `int` now and a `str` a moment later. Think of names as **sticky notes** you put on objects, not boxes that contain them. Assignment moves a sticky note; it never copies the object.

That one change explains most of today's surprises for a Kotlin developer: why `b = a` doesn't copy a list, why `is` and `==` are different questions, why a type error shows up when the line *runs* instead of when you build, and why `None` can turn up anywhere.

The second big difference is that Python has fewer, bigger number types. There is one integer type, `int`, that never overflows, and one floating-point type, `float`, which is exactly Kotlin's `Double`. Most of the arithmetic surprises (division, rounding, `0.1 + 0.2`) come from the rules around those two types, and every one of them follows you into NumPy and PyTorch.

!!! info "How this page shows errors"
    Many examples on this page fail on purpose, so you can see the real error message. A line that fails stops the whole ▶ Run block, so failing lines are wrapped like this:

    ```python
    try:
        "3" + 4
    except TypeError as err:
        print("TypeError:", err)
    # TypeError: can only concatenate str (not "int") to str
    ```

    Read it as: "run the indented line under `try`; if it raises a `TypeError`, store the error in `err` and run the `except` part instead of stopping." The printed text is exactly the last line of the traceback you'd see without the wrapper. [Day 3b](day-03b-errors-files-json.md) teaches `try` / `except` properly. Mistakes that stop Python before it runs anything (syntax errors) are shown as plain text instead.

## Names and objects

### A name is a label, not a box

**What it is.** An **object** is a value in memory: the number `42`, the string `"hello"`, a list. Every object has a **type** (what kind of value it is), an **identity** (which object it is, unique while it exists) and a **value**. A **name** (what Kotlin calls a variable) is a label that refers to one object. The `=` sign **binds** a name to an object: it attaches the label.

**Why it matters.** Every rule about copying, comparing and mutating data in Python, NumPy and PyTorch follows from this. When a tensor changes "by itself", it's almost always two names on one object.

**Kotlin vs Python.** In Kotlin, you declare a name and its type once:

```kotlin
var x: Int = 42
x = "hello"        // compile error: type mismatch
```

In Python there's no declaration and no type on the name:

```python
x = 42
type(x)          # <class 'int'>
x = "hello"      # the label x now sits on a str; no error
type(x)          # <class 'str'>
```

`type(obj)` returns the object's type. The `<class 'int'>` display means "the type `int`" (in Python every type is a class).

What's the same: Kotlin variables that hold objects are also references, so `val b = a` on a `MutableList` shares one list, exactly as in Python. What's different: Python has no `val`, no declared types and no compile step that checks them. A name can be rebound to anything, at any time.

**Dynamic, but strong.** "Dynamic typing" means types are checked while the program runs, not before. It does *not* mean Python converts types for you. Python is **strongly typed**: it refuses to mix types in ways that don't make sense, where Kotlin would quietly help:

```python
try:
    "3" + 4
except TypeError as err:
    print("TypeError:", err)
# TypeError: can only concatenate str (not "int") to str
```

In Kotlin, `"3" + 4` compiles and gives `"34"`, because `String.plus` accepts `Any?`. Python treats it as a bug. You have to say which you meant: `"3" + str(4)` is `'34'`, and `int("3") + 4` is `7`.

The quotes in `'34'` show that the value is a string. When the REPL or the ▶ Run button echoes a value, it shows the developer form (called the **repr**), which puts strings in quotes so that `'7'` and `7` look different. `print` shows the plain form without quotes. [Day 2a](day-02a-strings.md) covers the pair, `repr` and `str`, in full.

### Identity: `id`, `is` and `==`

**What it is.** `id(obj)` returns an integer that identifies an object while it exists (in CPython, the standard interpreter, it's the memory address). `a is b` asks "are these two names on the **same object**?" `a == b` asks "do these two objects have **equal values**?"

**Why it matters.** You'll use `is` for two things. The everyday one is testing for `None` (`x is None`, later on this page). The other is debugging aliasing: when data changes "by itself", `a is b` tells you that two names you thought were separate are really on one object.

**Kotlin vs Python.** It's the same pair of questions Kotlin asks with `===` (referential equality) and `==` (structural equality, which calls `equals`):

| Question | Kotlin | Python |
|---|---|---|
| Same object? | `a === b` | `a is b` |
| Equal values? | `a == b` | `a == b` |

**Example.** Two lists with the same contents are equal but not identical. A second name on the same list is both:

```python
a = [1, 2]
b = a           # a second label on the same list
c = [1, 2]      # a different list that happens to hold equal values

a == c          # True   equal values
a is c          # False  two separate objects
a is b          # True   one object, two names
id(a) == id(b)  # True   that's what `is` compares
```

(A **list**, written `[1, 2]`, is Python's resizable array, like Kotlin's `MutableList`. You'll use a few in today's loops; Day 2b covers them fully.)

**Underneath.** Nothing is copied by `b = a`. Python makes the name `b` refer to the object `a` already refers to. If you then change the object through either name, both names see it, because there's only one object (the animation in the next section draws this):

```python
b.append(3)     # .append adds an item to the end of the list, in place
a               # [1, 2, 3]   changed "through" b
c               # [1, 2]      the separate list is untouched

d = a.copy()    # a NEW list with the same items
d is a          # False
```

### Mutable and immutable objects

**What it is.** A **mutable** object can be changed in place after it's created: lists are mutable (`append` changed the list above). An **immutable** object can never change: `int`, `float`, `bool`, `str` and `None` are all immutable. An operation that seems to change one actually creates a *new* object and moves the label.

**Why it matters.** It tells you when sharing is dangerous. Two names on one immutable object can never surprise each other, because neither can change it. Two names on one mutable object can.

**Kotlin vs Python.** Kotlin's `String` and `Int` are immutable too, and `x += 1` on a `var` also stores a new value. Two differences: Python has nothing like `val` to forbid rebinding a name, and it has no `++` operator.

**Example.** Watch the identity change when an `int` is "incremented":

```python
n = 5
m = n            # m and n label the same int object
before = id(n)
n += 1           # n = n + 1: build a NEW int 6, move the label n onto it
n                # 6
m                # 5    m still labels the old object
id(n) == before  # False  n is on a different object now
```

<figure class="diagram" markdown>
![Animation in two panels. Left: n = 5 and m = n put two labels on one int object 5; n += 1 creates a new object 6 and the n label's arrow swings over to it, while m stays on 5. Right: a = [1, 2] and b = a put two labels on one list; b.append(3) changes that list in place to [1, 2, 3], and both arrows stay where they are](../../assets/img/phase-1/day02-rebind-vs-mutate.svg)
<figcaption>Animated: on an immutable int, <code>n += 1</code> moves the label to a new object. On a mutable list, <code>append</code> changes the one object both labels point at.</figcaption>
</figure>

Python has no `++` or `--` operators. `n++` is a `SyntaxError`; write `n += 1`. The other arithmetic operators have the same **augmented assignment** form, which means "compute, then rebind the name to the result":

```python
steps = 100
steps -= 10      # steps = steps - 10
steps *= 2       # steps = steps * 2
steps //= 7      # floor division (below)
steps            # 25
steps %= 7       # remainder (below)
steps            # 4
steps **= 2      # power
steps            # 16
```

### What happens to an object when its last label moves

**What it is.** After `n += 1`, nothing refers to the old object any more (unless another name does). CPython, the standard interpreter, keeps a **reference count** on every object: how many names and containers point at it. When the count drops to 0, the object is freed immediately. A separate **cycle collector** runs now and then to free groups of objects that only point at each other.

**Kotlin vs Python.** On Android, ART's garbage collector finds unreachable objects in the background, at times you don't control. CPython frees most objects the moment the last reference goes. Either way, you never free memory by hand. (In the `n += 1` example, `5` isn't freed at all: CPython keeps small integers alive permanently, which the `is` box below comes back to.)

**`del` removes a name, not an object.** `del x` unbinds the name `x`. The object it pointed at loses one reference, and is freed only if that was the last one:

```python
weights = [0.1, 0.2]
alias = weights
del weights          # removes the name weights; the list still has the label alias
alias                # [0.1, 0.2]
try:
    weights
except NameError as err:
    print("NameError:", err)
# NameError: name 'weights' is not defined
```

The `NameError` is the "goes wrong": after `del`, or before a name is ever assigned, using it fails. In a notebook this also happens when you run cells out of order, which is why Day 1 recommends "restart and run all".

!!! warning "Common mistake: using `is` to compare numbers or strings"
    `is` checks identity, and Python is free to reuse or not reuse objects for equal values. CPython keeps one shared object for each small integer from -5 to 256, so `is` *seems* to work in quick tests, then fails for bigger numbers:

    ```python
    small_a = 256
    small_b = 256
    small_a is small_b    # True   CPython shares small ints
    big_a = int("1000")
    big_b = int("1000")
    big_a == big_b        # True   equal values
    big_a is big_b        # False  two separate int objects
    ```

    (`int("1000")` turns the text `"1000"` into the number 1000; type conversion is covered below.) **Fix:** use `==` to compare values. Use `is` only for `None`, `True` and `False`, which are guaranteed to be single objects.

### Checking types: `type` and `isinstance`

**What it is.** `type(x)` returns the exact type of an object. `isinstance(x, T)` asks "is `x` an instance of `T`, or of a subtype of `T`?" It also accepts a tuple of types: `isinstance(x, (int, float))` means "an `int` or a `float`". (A **tuple** is a fixed group of values in parentheses; Day 2b covers it.)

**Kotlin vs Python.** `isinstance(x, int)` is Kotlin's `x is Int`. Kotlin's smart cast then lets you use `x` as an `Int`; Python doesn't need one, because it never checked the name's type in the first place.

```python
isinstance(3, int)              # True
isinstance(3.0, int)            # False
isinstance(3, (int, float))     # True
type(3) == int                  # True
type(True)                      # <class 'bool'>
isinstance(True, int)           # True   bool is a subtype of int
```

**Why it matters.** Input checks ("is this a number?") and code that accepts several kinds of value both need a type test. You'll write them in config loaders and data validation.

**Underneath.** `bool` is defined as a subclass of `int`: `True` behaves as `1` and `False` as `0` in arithmetic. `isinstance` follows that family tree; `type(x) == T` doesn't. Prefer `isinstance` in general, and keep the `bool` surprise in mind when you validate numbers.

!!! warning "Common mistake: an exact type test for 'is it a number?'"
    Two opposite bugs. `type(x) == float` is too strict: an `int` is a perfectly good learning rate, but fails the test. `isinstance(x, (int, float))` is too loose: it lets `True` through, because `bool` is a subtype of `int`:

    ```python
    lr = 3
    type(lr) == float                  # False  rejects a valid int
    isinstance(lr, (int, float))       # True
    isinstance(True, (int, float))     # True   a bool slips through
    flag = True
    isinstance(flag, (int, float)) and not isinstance(flag, bool)   # False  the fix
    ```

    **Fix:** test with `isinstance(x, (int, float))`, and reject `bool` explicitly when it matters (Exercise 3 does this).

## Numbers

### `int`: integers that never overflow

**What it is.** `int` is Python's only integer type. It has **arbitrary precision**: it grows to as many digits as the value needs.

**Why it exists.** You never pick between `Int`, `Long` and `BigInteger`, and you never get a silent wraparound. Counting tokens, multiplying image dimensions, computing a factorial for a probability: the answer is always exact.

**Kotlin vs Python.** In Kotlin, `Int` is 32-bit and overflows silently:

```kotlin
val big = Int.MAX_VALUE + 1    // -2147483648, no error
val huge = 2.0.pow(100)        // a Double, not exact
```

In Python:

```python
2 ** 31 - 1 + 1          # 2147483648   no wraparound
2 ** 100                 # 1267650600228229401496703205376   exact
1_000_000                # 1000000   underscores in a literal are only for readability
```

(`**` is the power operator; it's covered with the other operators below.)

Integer literals can also be written in hexadecimal (`0x`), binary (`0b`) or octal (`0o`); they are still plain `int`s. `hex`, `bin` and `oct` go the other way and return text:

```python
0xFF                     # 255
0b1010                   # 10
0o17                     # 15
hex(255)                 # '0xff'
bin(5)                   # '0b101'
```

You've used `0xFF` on Android to mask a colour channel out of an ARGB pixel; it means exactly the same number in Python.

**Underneath.** A Kotlin `Int` is 4 bytes in a register. A Python `int` is a full object on the heap that stores its digits in an array that grows as needed, so `2 ** 100` uses more memory. That's also why a Python `int` is slow next to a Kotlin `Int`, and why NumPy (Day 4) stores numbers as fixed-size machine integers instead. Those **do** overflow, so the Kotlin intuition comes back there.

!!! warning "Common mistake: a big int loses digits when it becomes a float"
    `int` is exact, but `float` (next section) only keeps about 16 significant digits. Converting a large `int` to `float`, or dividing it with `/`, quietly rounds it:

    ```python
    exact = 2 ** 53 + 1
    exact                # 9007199254740993
    float(exact)         # 9007199254740992.0   the last digit changed
    ```

    **Fix:** keep large IDs and counts as `int`, and use `//` (integer division, below) when you need an integer result.

### `float`: Kotlin's `Double`, with the same rounding

**What it is.** `float` is a 64-bit **IEEE-754** floating-point number. IEEE-754 is the hardware standard almost every CPU and GPU uses; it stores a number as a sign, a 52-bit binary fraction and an exponent. A Python `float` is bit-for-bit a Kotlin `Double`. There is no 32-bit float in plain Python; NumPy and PyTorch have one (`float32`), and it's what models use.

**Literals.** A number with a decimal point or an exponent is a float. `1e-3` means \(1 \times 10^{-3}\); you'll write learning rates this way.

```python
3.0              # 3.0
1e-3             # 0.001
3e-4             # 0.0003
type(1e3)        # <class 'float'>   1e3 is 1000.0, a float, even though it's whole
2 * 3.0          # 6.0               an int mixed with a float gives a float
```

**Mixing types.** In arithmetic, an `int` meeting a `float` is converted to `float` first, so the result is a `float`. That's the only automatic number conversion Python does (Kotlin does the same for `2 * 3.0`).

### Why `0.1 + 0.2 != 0.3`

**What happens.** You already know this one from Kotlin, but it's worth seeing exactly why, because it decides how you compare every loss and accuracy you'll compute.

```python
0.1 + 0.2              # 0.30000000000000004
0.1 + 0.2 == 0.3       # False
```

**Underneath.** A float is a binary fraction. Just as 1/3 has no finite decimal expansion (0.333…), 1/10 has no finite *binary* expansion. So the literal `0.1` is stored as the nearest representable double, which is very slightly more than 0.1. The format spec `.20f` below means "show 20 digits after the decimal point" (the f-string primer before Control flow introduces format specs, and Day 2a covers them in full):

```python
f"{0.1:.20f}"          # '0.10000000000000000555'
f"{0.2:.20f}"          # '0.20000000000000001110'
f"{0.3:.20f}"          # '0.29999999999999998890'
f"{0.1 + 0.2:.20f}"    # '0.30000000000000004441'
```

**Why does `0.1` display as `0.1`, then?** When Python shows a float, it prints the *shortest* decimal that converts back to exactly the same double. `0.1` is the shortest text that maps to the stored 0.1000000000000000055…, so that's what you see. No decimal shorter than 17 digits maps to the double that `0.1 + 0.2` produces, which is why that one shows all its digits:

```python
repr(0.1)              # '0.1'
repr(0.1 + 0.2)        # '0.30000000000000004'
```

Doubles near 0.3 are spaced about \(5.55 \times 10^{-17}\) apart. The literal `0.3` becomes the double slightly *below* 0.3. The exact sum of the stored 0.1 and 0.2 falls exactly halfway between that double and the next one up (a tie), and IEEE-754's tie rule, round to the neighbour whose last binary digit is even, picks the upper one. So the two sides are neighbouring doubles: different by one step, which `==` reports as "not equal".

<figure class="diagram" markdown>
![A number line zoomed in around 0.3. Tick marks show three neighbouring doubles: 0.29999999999999993 on the left, 0.29999999999999998890 in the middle labelled "the literal 0.3 is stored here", and 0.30000000000000004441 on the right labelled "0.1 + 0.2 ends up here". The true value 0.3 sits slightly right of the middle tick, and the exact sum of the stored 0.1 and 0.2 sits halfway between the middle and right ticks, with an arrow showing it rounds to the right tick. The gap between ticks is labelled 5.55e-17](../../assets/img/phase-1/day02-float-gap.svg)
<figcaption>Floats are a grid of points, not a continuous line. <code>0.3</code> and <code>0.1 + 0.2</code> round to neighbouring points.</figcaption>
</figure>

**The fix: compare with a tolerance.** `math.isclose(a, b)` returns `True` when `a` and `b` differ by less than a relative tolerance, `rel_tol=1e-09` by default, meaning "agree to about 9 significant digits". Import the `math` module first: `import math` loads Python's standard maths library and makes it available as the name `math`.

```python
import math

math.isclose(0.1 + 0.2, 0.3)                 # True
math.isclose(1e-10, 0.0)                     # False  allowed gap is 1e-9 * 1e-10 = 1e-19, but the gap is 1e-10
math.isclose(1e-10, 0.0, abs_tol=1e-9)       # True   near zero, give an absolute tolerance
```

**Underneath `isclose`.** The test is `abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)`. "Relative" means the allowed gap scales with the size of the numbers, which is what you want for most values. Near zero, the relative gap is tiny (in the example above, \(10^{-9} \times 10^{-10} = 10^{-19}\)), so pass `abs_tol` when one side can be `0.0`. NumPy has the same idea for arrays, `np.allclose`, on Day 4b.

!!! warning "Common mistake: testing a float with `==`"
    A loop that adds `0.1` until it reaches `1.0` never sees exactly `1.0`:

    ```python
    total = 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1 + 0.1
    total            # 0.9999999999999999
    total == 1.0     # False
    ```

    In a `while total != 1.0:` loop this never stops (you'll see it, safely, in the `while` section). **Fix:** compare with `math.isclose`, or use `<` / `>=`, or count in integers and divide at the end.

### Arithmetic operators: `/`, `//`, `%`, `**`

**What they are.**

- `/` is **true division**: it always returns a `float`, even for two `int`s.
- `//` is **floor division**: it divides and rounds *down*, towards minus infinity.
- `%` is the **remainder** (modulo) that goes with `//`.
- `**` is **power**. `divmod(a, b)` returns both `a // b` and `a % b` at once.

**Why it matters.** Batch counts, array indices, train/test split sizes and circular buffers are all integer division and remainders. Getting a `float` where you need an `int` is the most common first-week bug.

**Kotlin vs Python.** This is where Kotlin habits break:

| Expression | Kotlin | Python |
|---|---|---|
| `7 / 2` | `3` (Int / Int truncates) | `3.5` (`/` always gives a float) |
| integer division of 7 by 2 | `7 / 2` → `3` | `7 // 2` → `3` |
| `-7 / 2` (integers) | `-3` (truncates towards zero) | `-7 // 2` → `-4` (floors) |
| `-7 % 2` | `-1` (sign of the left side) | `1` (sign of the right side) |
| floor division / mod in Kotlin | `(-7).floorDiv(2)` → `-4`, `(-7).mod(2)` → `1` | `//` and `%` |
| `2 ^ 3` | `2 xor 3` is bitwise; power is `2.0.pow(3)` | `2 ** 3` → `8` (`^` is bitwise XOR too) |

**Example.**

```python
7 / 2            # 3.5
6 / 2            # 3.0   still a float
7 // 2           # 3
7 % 2            # 1
divmod(7, 2)     # (3, 1)

-7 // 2          # -4    floor of -3.5 is -4, not -3
-7 % 2           # 1
divmod(-7, 2)    # (-4, 1)
7 // -2          # -4
7 % -2           # -1    the remainder takes the sign of the divisor
-7.5 // 2        # -4.0  // on floats floors too, and returns a float

2 ** 10          # 1024
2 ** -1          # 0.5   a negative power gives a float
2 ** 0.5         # 1.4142135623730951   square root
-2 ** 2          # -4    ** binds tighter than the minus sign: -(2 ** 2)
(-2) ** 2        # 4
```

**Underneath.** Python guarantees one identity for `//` and `%`, for every sign combination:

\[
a = (a \mathbin{//} b) \cdot b + (a \bmod b)
\]

Check it for \(a = -7, b = 2\): \((-4) \cdot 2 + 1 = -7\). Because `//` rounds down, the remainder always has the same sign as `b` (or is 0). That's what makes `%` useful for wrapping an index: `i % n` is always in `0 .. n-1` for positive `n`, even when `i` is negative.

```python
n = 5
-1 % n           # 4   one step "before" 0 wraps to the last slot
12 % n           # 2
```

!!! android "Android bridge"
    A ring buffer of the last `n` audio samples or sensor readings (the kind you write for `SensorEventListener`) needs `(index - 1 + n) % n` in Kotlin, because Kotlin's `%` can return a negative number. In Python, `(index - 1) % n` is enough. Kotlin's `Int.mod(n)` (Kotlin 1.5+) behaves like Python's `%`, and `floorDiv` like `//`.

**A realistic example.** How many mini-batches does a dataset of 1,000 samples make with a batch size of 64?

```python
n_samples, batch_size = 1000, 64        # two names bound on one line
full, leftover = divmod(n_samples, batch_size)
full, leftover                          # (15, 40)  15 full batches, then 40 samples
math.ceil(n_samples / batch_size)       # 16   batches including the partial one
-(-n_samples // batch_size)             # 16   the same, in pure integer maths
```

`math.ceil(x)` rounds up to the nearest integer and returns an `int`. The last line is a common integer-only trick: floor-dividing the negated value rounds *up*, because flooring \(-15.625\) gives \(-16\).

!!! warning "Common mistake: `/` where an integer is needed"
    Splitting data at "80% of the way through" with `/` gives a float, and you can't index with a float:

    ```python
    samples = [10, 20, 30, 40, 50]
    cut = len(samples) * 0.8        # 4.0, a float
    try:
        samples[cut]
    except TypeError as err:
        print("TypeError:", err)
    # TypeError: list indices must be integers or slices, not float
    samples[int(cut)]               # 50
    samples[len(samples) * 4 // 5]  # 50   integer maths all the way
    ```

    (`len(x)` returns the number of items in `x`; `samples[i]` reads the item at position `i`, counting from 0.) **Fix:** use `//`, or convert with `int(...)` deliberately.

### Rounding: `round` uses banker's rounding

**What it is.** `round(x)` returns the nearest integer as an `int`. `round(x, n)` rounds to `n` digits after the decimal point and returns a `float`. When `x` is exactly halfway, Python rounds to the **even** neighbour. This is called **banker's rounding** (or "round half to even").

**Why it exists.** Always rounding halves up pushes sums upward on average. Rounding to even is unbiased: half the ties go up, half go down. It's the IEEE-754 default and what NumPy does too.

**Kotlin vs Python.** Kotlin has both behaviours, under different names:

```kotlin
kotlin.math.round(2.5)   // 2.0  ties to even, like Python
2.5.roundToInt()         // 3    ties towards positive infinity
Math.round(2.5)          // 3    Java's: half up
```

**Example.**

```python
round(0.5), round(1.5), round(2.5), round(3.5)   # (0, 2, 2, 4)
round(-0.5)          # 0
round(3.14159, 2)    # 3.14
round(1234, -2)      # 1200   a negative n rounds to tens, hundreds, ...
type(round(2.5))     # <class 'int'>
type(round(2.5, 0))  # <class 'float'>
```

**Underneath.** `round(2.675, 2)` looks like a tie, but `2.675` is stored as `2.67499999999999982236431605997495353221893310546875`, which is slightly below the tie, so it goes down:

```python
round(2.675, 2)      # 2.67
f"{2.675:.20f}"      # '2.67499999999999982236'
```

!!! warning "Common mistake: expecting `round(2.5)` to be 3"
    Code ported from Kotlin's `roundToInt()` or Java's `Math.round` gives different answers on exact halves. A pixel value computed as `round(127.5)` is `128` (128 is even), but `round(126.5)` is `126`, not `127`. **Fix:** if you need "half up" for positive numbers, use `math.floor(x + 0.5)`. For display, don't round at all; format the number (`f"{x:.2f}"`, below). For money, use the `decimal` module (see Going further).

### The `math` module, `inf` and `nan`

**What it is.** `math` is the standard library module for scalar maths on single numbers. `inf` is a float that's larger than every other float; `nan` ("not a number") is a float that means "undefined result", such as infinity minus infinity.

**Why it matters.** A diverging training run shows up as a loss of `inf` or `nan`. You'll check for both in training loops, and NumPy's versions (Day 4b) follow the same rules.

```python
math.sqrt(16)        # 4.0
math.floor(-2.5)     # -3    largest integer <= x
math.ceil(-2.5)      # -2    smallest integer >= x
math.trunc(-2.5)     # -2    drop the fraction (towards zero)
int(-2.5)            # -2    int() of a float also truncates
math.log(math.e)     # 1.0   natural log
math.log10(1000)     # 3.0
math.log2(8)         # 3.0
math.exp(1)          # 2.718281828459045
math.pi              # 3.141592653589793
math.factorial(5)    # 120
abs(-3)              # 3     abs is built in, not in math
```

`inf` and `nan` come from `math.inf` / `math.nan` or from the strings `float("inf")` / `float("nan")`:

```python
big = math.inf
big > 1e308              # True
-big                     # -inf
1e308 * 10               # inf    overflow in float arithmetic gives inf
big - big                # nan
nan = float("nan")
nan == nan               # False  nan is not equal to anything, even itself
math.isnan(nan)          # True   the right way to test for nan
math.isinf(-big)         # True
nan + 1                  # nan    nan spreads through arithmetic
```

**Kotlin vs Python.** The values are the same as Kotlin's `Double.POSITIVE_INFINITY` and `Double.NaN`, and `Double.NaN == Double.NaN` is `false` in Kotlin too. The big difference is **division by zero**. Kotlin's `1.0 / 0` quietly gives `Infinity`. Python raises an error, for floats as well as ints:

```python
try:
    1.0 / 0
except ZeroDivisionError as err:
    print("ZeroDivisionError:", err)
# ZeroDivisionError: division by zero
```

(Python 3.13 and older word the float case as `float division by zero`.) The `math` functions also refuse inputs outside their domain instead of returning `nan`:

```python
try:
    math.sqrt(-1)
except ValueError as err:
    print("ValueError:", err)
# ValueError: expected a nonnegative input, got -1.0
```

(Python 3.13 and older print `math domain error` here.) NumPy is the opposite: `np.sqrt(-1.0)` returns `nan` with a warning. Day 4b covers that.

**Overflow: sometimes `inf`, sometimes an error.** Plain multiplication that overflows gives `inf` quietly (`1e308 * 10` above). But `math.exp` and `**` on floats raise `OverflowError` instead. This is the one you'll hit first in ML, because the sigmoid and softmax functions call `exp` on scores that can be large:

```python
try:
    math.exp(1000)
except OverflowError as err:
    print("OverflowError:", err)
# OverflowError: math range error
math.exp(-1000)          # 0.0   the other direction underflows quietly to zero
```

`2.0 ** 10000` raises `OverflowError` too (its message text depends on the operating system). NumPy's `np.exp(1000.0)` returns `inf` with a warning instead (Day 4b), and real softmax code subtracts the largest score first so `exp` never sees a big number.

!!! warning "Common mistake: `x == nan` is always `False`"
    Checking `if loss == float("nan"):` never fires, because `nan` is unequal to everything:

    ```python
    loss = float("inf") - float("inf")   # a diverged loss: nan
    loss == float("nan")                 # False  the check silently fails
    math.isnan(loss)                     # True   the fix
    math.isfinite(loss)                  # False  catches nan, inf and -inf in one test
    ```

## Booleans and truthiness

### `bool`, comparisons and logic words

**What it is.** `bool` has two values, `True` and `False` (capitalised). Comparisons (`==`, `!=`, `<`, `<=`, `>`, `>=`) return a `bool`. The logical operators are words: `and`, `or`, `not`, instead of Kotlin's `&&`, `||`, `!`.

```python
3 > 2                # True
3 > 2 and 2 > 5      # False
not 3 > 2            # False
True + True          # 2    bool is an int underneath
```

Writing `true` (lower case) fails with `NameError: name 'true' is not defined`, because Python looks for a variable called `true`.

**Precedence: which part runs first.** Arithmetic binds tighter than comparisons, comparisons bind tighter than `not`, `not` tighter than `and`, and `and` tighter than `or`. So `not 3 > 2` is `not (3 > 2)`, and `a or b and c` is `a or (b and c)`. This is the same order as Kotlin's `!`, `&&`, `||`, except that Kotlin's `!` binds tighter than comparisons, while Python's `not` binds looser:

```python
not 3 > 2                  # False  not (3 > 2)
True or False and False    # True   True or (False and False)
(True or False) and False  # False  parentheses change it
x, y = 1, 2
not x == y                 # True   not (x == y)
(not x) == y               # False  (not 1) is False, and False == 2 is False
```

The last two lines are the trap for a Kotlin reader, where `!x == y` means `(!x) == y`. When a condition mixes `not`, `and` and `or`, add parentheses; they cost nothing and the next reader doesn't have to remember the table.

**Comparing different types.** `==` between unrelated types is allowed and returns `False`; ordering comparisons (`<`, `>`) between unrelated types raise an error:

```python
"5" == 5             # False  no conversion, no error
try:
    1 < "a"
except TypeError as err:
    print("TypeError:", err)
# TypeError: '<' not supported between instances of 'int' and 'str'
```

The silent `False` is the dangerous one: a number read from a file as the text `"5"` is never equal to the number `5`.

Numbers are the exception to "different types are unequal": `int`, `float` and `bool` compare by numeric value across types:

```python
1 == 1.0             # True
True == 1            # True
0.5 < 1              # True
```

Kotlin is stricter here: `1 == 1.0` doesn't compile, because `Int` and `Double` can't be compared with `==`.

### Truthiness: every object is true or false

**What it is.** In an `if` or `while`, Python accepts *any* object, not only a `bool`. It asks the object whether it's "truthy" or "falsy". `bool(x)` shows the answer. The **falsy** values are:

- `None` and `False`;
- zero of every number type: `0`, `0.0`, and `0j` (complex zero; see Going further);
- every empty collection or sequence: `""` (empty string), `[]` (empty list), `()` (empty tuple), `{}` (empty **dict**, not an empty set), `set()` (empty set) and `range(0)` (an empty range, covered with `for` loops below). Day 2b covers tuples, dicts and sets.

Everything else is **truthy**, including `"False"`, `"0"`, `[0]` and `-1`.

**Why it exists.** It makes the most common checks short: `if items:` means "if the list isn't empty", `if name:` means "if the string isn't empty".

**Kotlin vs Python.** Kotlin's `if` only takes a `Boolean`. `if (list)` doesn't compile; you write `if (list.isNotEmpty())`. Python lets you write the short form, and you'll see it everywhere in Python code.

```python
bool(0), bool(0.0), bool(""), bool([]), bool(None)   # (False, False, False, False, False)
bool("False"), bool("0"), bool([0]), bool(-1)        # (True, True, True, True)

items = []
if not items:
    print("no items yet")
# no items yet
```

**Underneath.** `bool(x)` first calls `x.__bool__()` if the type defines it (numbers do: nonzero is true). Otherwise it calls `x.__len__()` and checks for zero (strings and collections do). Objects that define neither are always truthy. Methods with double underscores like these are called **dunder methods**; Day 3 shows how to write your own.

!!! warning "Common mistake: `if not x:` when 0 is a valid value"
    Truthiness can't tell "missing" from "zero". A threshold of `0.0` is a real setting, but it's falsy:

    ```python
    threshold = 0.0                  # the user really asked for 0.0
    if not threshold:
        threshold = 0.5              # meant as "if no threshold was given"
    threshold                        # 0.5   the user's 0.0 was thrown away
    ```

    **Fix:** when "missing" is represented by `None`, test for `None` explicitly: `if threshold is None:`. Keep `if not x:` for "empty or zero, I don't care which".

    The opposite surprise matters in training code: `nan` is a nonzero float, so it's **truthy**. A check like `if not loss:` never catches a diverged loss:

    ```python
    loss = float("nan")
    bool(loss)                       # True   nan is truthy
    math.isnan(loss)                 # True   test for it explicitly
    ```

### `and` and `or` return one of their operands

**What it is.** `and` and `or` don't return `True` or `False`. They return **one of the two operands**, and they stop as soon as the answer is known (**short-circuit** evaluation):

- `a or b`: if `a` is truthy, return `a` (and never evaluate `b`); otherwise return `b`.
- `a and b`: if `a` is falsy, return `a`; otherwise return `b`.
- `not a` always returns a real `bool`.

**Why it exists.** It gives a one-line default: `name or "anonymous"`. It's also how guard checks like `x is not None and x > 0` avoid evaluating the second part when the first fails.

**Trace.** `"" or "anonymous"`: `""` is falsy, so `or` moves on and returns `"anonymous"`. `"Ana" or "anonymous"`: `"Ana"` is truthy, so `or` returns `"Ana"` without looking at the right side.

```python
"" or "anonymous"        # 'anonymous'
"Ana" or "anonymous"     # 'Ana'
0 or 10                  # 10
[] and "never seen"      # []    falsy left side is returned as is
3 and "seen"             # 'seen'
None or 0 or ""          # ''    every operand falsy: the last one is returned
not "text"               # False
```

<figure class="diagram" markdown>
![Animation of two expressions evaluated left to right. For "" or "anonymous": the left operand "" is tested, found falsy, and crossed out, then the right operand "anonymous" is boxed as the result. For "Ana" or "anonymous": the left operand is found truthy and boxed as the result, while the right operand fades out, labelled "never evaluated". A note says and works the same way with truthy and falsy swapped](../../assets/img/phase-1/day02-short-circuit.svg)
<figcaption>Animated: <code>or</code> returns the first truthy operand and never looks at the rest.</figcaption>
</figure>

**Kotlin comparison.** `name ?: "anonymous"` is Kotlin's Elvis operator. It substitutes the default only for `null`. Python's `or` substitutes it for *every* falsy value, which isn't the same thing (next box).

!!! warning "Common mistake: `x == 1 or 2` and `or` as a null default"
    Two classic bugs, both without an error:

    ```python
    device_count = 5
    if device_count == 1 or 2:          # reads like "1 or 2", means (count == 1) or (2)
        print("treated as 1 or 2 devices!")
    # treated as 1 or 2 devices!

    epochs = 0                          # the user asked for zero epochs (evaluate only)
    epochs = epochs or 10               # meant: "default to 10 if not given"
    epochs                              # 10    the user's 0 was replaced
    ```

    The first condition is always true because `2` is truthy. **Fix:** `device_count == 1 or device_count == 2`, or `device_count in (1, 2)` (`in` checks membership). For the second, use a `None` check: `epochs = 10 if epochs is None else epochs` (the conditional expression, explained in the control flow section).

## `None`: Python's `null`

### What `None` is

**What it is.** `None` is the single object of type `NoneType`. It means "no value here". Functions that don't return anything return `None`, and it's the usual stand-in for a missing optional value.

**Why it matters.** It's the Python version of Kotlin's `null`, but **without null-safety**. Any name can be `None` at any time, and nothing checks it before the program runs.

**Kotlin vs Python.** Kotlin separates `String` from `String?`, and the compiler forces you to handle the `null` case. Python has no such split. You can write a **type hint** that says a value may be missing, `name: str | None = None`, and an editor or a type checker such as mypy will warn you, but the Python interpreter ignores hints completely. Day 3 covers type hints.

```python
result = print("hello")      # print shows text and returns None
# hello
result is None               # True
type(None)                   # <class 'NoneType'>
```

**Underneath.** There's exactly one `None` object in a running program, so `is None` is a fast, exact identity test. Always use `is None` / `is not None`, never `== None`: `==` can be redefined by a type (NumPy arrays compare element by element, for example), but `is` can't.

### Replacing `?.`, `?:` and `!!`

Python has no `?.`, `?:` or `!!` operators. Here's what Python code uses instead. Each line assumes `name` may be a `str` or `None`:

| Kotlin | Python idiom |
|---|---|
| `name?.uppercase()` | `name.upper() if name is not None else None` |
| `name ?: "anon"` | `name if name is not None else "anon"` (or `name or "anon"` when empty strings should be replaced too) |
| `name?.let { show(it) }` | `if name is not None:` then `show(name)` on the next line |
| `name!!` | `assert name is not None` (fails loudly; Day 3b) |

These use the conditional expression `A if condition else B`, which evaluates to `A` when the condition is true and `B` otherwise. It's Python's version of Kotlin's `if (condition) A else B` expression, and the control flow section covers it.

```python
name = None
shout = name.upper() if name is not None else None
shout                                        # None   (the echo shows nothing for None)
display = name if name is not None else "anon"
display                                      # 'anon'

name = "pixel"
name.upper() if name is not None else None   # 'PIXEL'
```

`.upper()` is a string method that returns an upper-case copy; Day 2a covers string methods.

**`?.let { }` and `!!`.** Kotlin's `name?.let { show(it) }` runs a block only when the value isn't `null`. Python uses a plain `if`:

```python
name = "pixel"
if name is not None:
    print(name.upper())
# PIXEL
```

Kotlin's `name!!` says "I'm sure this isn't null; crash if it is". The closest Python spelling is `assert`. `assert condition, message` does nothing when the condition is true, and raises `AssertionError` with the message when it's false:

```python
name = None
try:
    assert name is not None, "name is required"
except AssertionError as err:
    print("AssertionError:", err)
# AssertionError: name is required
```

There's a catch: `!!` always checks, but Python **removes every `assert`** when it's started with the `-O` ("optimise") flag, so `python -O train.py` would sail past the check and fail later with a confusing `AttributeError`. Use `assert` for "this can't happen" checks while developing. For a check that must always run, such as validating a config value, test and **raise** an error yourself:

```python
name = None
try:
    if name is None:
        raise ValueError("name is required")
except ValueError as err:
    print("ValueError:", err)
# ValueError: name is required
```

`raise` creates an error and stops the normal flow, exactly like Kotlin's `throw`. Day 3b covers `raise` and `assert` properly.

!!! warning "Common mistake: calling a method on `None`"
    This is the Python twin of a `NullPointerException`, and you'll meet it weekly. It happens when a value you expected to be set wasn't:

    ```python
    model_name = None                 # e.g. a config value that was never filled in
    try:
        model_name.upper()
    except AttributeError as err:
        print("AttributeError:", err)
    # AttributeError: 'NoneType' object has no attribute 'upper'
    ```

    **How to recognise it:** the message names `'NoneType'`. The bug is not on this line; it's wherever the value should have been set. Trace back to where `model_name` was assigned. **Fix:** set it, or guard with `if model_name is not None:`.

## Type conversion

### `int()`, `float()`, `str()`, `bool()`

**What it is.** Calling a type creates a new object of that type from another value: `int("42")` makes the integer 42 from the text `"42"`. Python does no implicit conversion between strings and numbers, ever.

**Why it matters.** Everything that arrives from outside your program (a CSV cell, a command-line flag, JSON from an API, `input()`) arrives as text. You convert it before doing maths.

**Kotlin vs Python.**

| Kotlin | Python | Note |
|---|---|---|
| `"42".toInt()` | `int("42")` | |
| `"3.5".toDouble()` | `float("3.5")` | |
| `x.toString()` | `str(x)` | |
| `3.99.toInt()` | `int(3.99)` | both truncate towards zero → 3 |
| `"42".toIntOrNull()` | no built-in; `try` / `except ValueError` (Day 3b) | |
| `" 42 ".toInt()` throws | `int(" 42 ")` → `42` | Python strips surrounding whitespace |

**Example.**

```python
int("42")            # 42
int(" 42\n")         # 42     surrounding whitespace and newlines are ignored
int(3.99)            # 3      truncates, doesn't round
int(-3.99)           # -3     towards zero
int("ff", 16)        # 255    second argument: the number base
float("3.5")         # 3.5
float("1e-3")        # 0.001
float("nan")         # nan
str(3.0)             # '3.0'
str(1e-5)            # '1e-05'
str(None)            # 'None'
bool("False")        # True   any non-empty string is truthy!
bool(0.0)            # False
```

**Underneath.** `int(x)` asks the object `x` how to become an integer. A `float` knows (truncate). A `str` is parsed: optional whitespace, an optional sign, then digits only. There's no "best effort": `int("12abc")` doesn't return 12. That strictness is the point; it catches dirty data early.

!!! warning "Common mistake: parsing text that isn't a clean integer"
    Data files contain decimals, thousands separators and placeholders. Each fails with a `ValueError` that quotes the bad text:

    ```python
    for text in ["3.5", "1,000", "", "n/a"]:
        try:
            int(text)
        except ValueError as err:
            print("ValueError:", err)
    # ValueError: invalid literal for int() with base 10: '3.5'
    # ValueError: invalid literal for int() with base 10: '1,000'
    # ValueError: invalid literal for int() with base 10: ''
    # ValueError: invalid literal for int() with base 10: 'n/a'

    int(float("3.5"))              # 3      parse as float first, then truncate
    int("1,000".replace(",", ""))  # 1000   remove the separator first
    ```

    (`for text in [...]:` repeats the indented lines once per item; loops are covered in detail below. `"1,000".replace(",", "")` returns a copy of the text with every comma replaced by nothing; Day 2a covers string methods.) The message always ends with the exact text in quotes, which tells you which value in your file is dirty. The `None` case is a different error, because `None` isn't text at all: `int(None)` raises `TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'`.

    Floats that aren't finite can't become integers either, because no integer is infinite:

    ```python
    try:
        int(math.inf)
    except OverflowError as err:
        print("OverflowError:", err)
    # OverflowError: cannot convert float infinity to integer
    try:
        int(math.nan)
    except ValueError as err:
        print("ValueError:", err)
    # ValueError: cannot convert float NaN to integer
    ```

    You'll see these when a diverged loss or a division gone wrong reaches code that computes an index or a count.

## A first look at f-strings

The loops below print results, so you need a way to put numbers into text and control how they look. This section is a short primer with the four forms you'll use today. [Day 2a](day-02a-strings.md) teaches strings and the full formatting language.

**What it is.** An **f-string** is a string literal with an `f` before the opening quote. Any expression inside `{...}` is evaluated and its value inserted into the text. After the expression you can write `:` and a **format spec**, a short code that says how to show the value.

**Why it exists.** Every training log and metrics printout needs numbers shown with a fixed number of decimals, lined up in columns, or as percentages.

**Kotlin vs Python.** `{x}` is Kotlin's `"$x"` / `"${x}"` template. The format spec replaces Kotlin's separate `"%.2f".format(x)` call, inside the same string:

```kotlin
val loss = 0.4521
println("loss ${"%.2f".format(loss)}")   // loss 0.45
```

```python
loss = 0.4521
acc = 0.7523
print(f"loss {loss}")          # loss 0.4521      {x}: insert the value
print(f"loss {loss:.2f}")      # loss 0.45        .2f: fixed-point, 2 decimals
print(f"[{loss:>8.2f}]")       # [    0.45]       >8: right-align in 8 characters
print(f"[{'cat':<6}]")         # [cat   ]         <6: left-align in 6 characters
print(f"acc {acc:.1%}")        # acc 75.2%        .1%: times 100, 1 decimal, add %
print(f"{2 + 3} items")        # 5 items          any expression works inside {}
```

Read a spec from left to right: the optional alignment and width (`>8`, `<6`), then a dot and the number of decimals, then a letter for the style (`f` for fixed decimals, `%` for a percentage). `>8.2f` means "right-aligned in 8 characters, 2 decimals".

**Underneath.** `f"{loss:.2f}"` calls the built-in `format(loss, ".2f")`, which asks the float to format itself. Rounding for display follows the same round-half-to-even rule as `round`. The value itself is not changed; only the text is.

```python
format(0.4521, ".2f")          # '0.45'
f"{0.125:.2f}"                 # '0.12'   a tie, rounded to even
```

!!! warning "Common mistake: `%` on a value that's already a percentage"
    `.1%` multiplies by 100. If you computed `75.23` (already in percent), it comes out 100 times too big, with no error:

    ```python
    acc_percent = 75.23
    f"{acc_percent:.1%}"           # '7523.0%'   wrong
    f"{acc_percent:.1f}%"          # '75.2%'     .1f, then a literal % sign
    ```

    And a spec like `.2f` only works on numbers. A number read from a file is still text, and `f"{'0.75':.2f}"` raises `ValueError: Unknown format code 'f' for object of type 'str'`; convert it with `float(...)` first.

## Control flow

### Blocks are indentation

**What it is.** Python has no braces. A line ending in `:` opens a **block**, and the block is every following line indented further. The block ends when the indentation goes back. The standard is 4 spaces.

**Kotlin vs Python.**

```kotlin
if (score > 0.9) {
    println("great")
}
```

```python
score = 0.95
if score > 0.9:
    print("great")
# great
```

No parentheses are needed around the condition, and no braces around the body. An empty block isn't allowed; write `pass`, a statement that does nothing, where Kotlin would write `{}`. You'll use it for "nothing to do here yet":

```python
use_gpu = False
if use_gpu:
    pass                 # TODO: set up the GPU later
else:
    print("running on CPU")
# running on CPU
```

**Long lines.** A statement normally ends at the end of the line. Inside any brackets, `( )`, `[ ]` or `{ }`, Python keeps reading onto the next lines until the bracket closes. This is called **implicit line joining**, and it's how you split a long expression. A backslash at the very end of a line also continues it, but it breaks if any character, even a space, follows it, so prefer brackets:

```python
total_params = (
    3_400_000        # backbone
    + 2_000_000      # classifier head
)
total_params             # 5400000
```

**How it goes wrong.** Indentation errors are found before anything runs. Running a script with a missing indent prints:

```text
  File "train.py", line 2
    print("great")
    ^^^^^
IndentationError: expected an indented block after 'if' statement on line 1
```

and forgetting the colon prints `SyntaxError: expected ':'`. **Fix:** indent with 4 spaces (set your editor to insert spaces for Tab), and never mix tabs and spaces.

### `if`, `elif`, `else` and the conditional expression

**What it is.** `if` / `elif` / `else` chooses one block to run. `elif` is "else if" (Python rejects `else if` with `SyntaxError: expected ':'`). The **conditional expression** `A if condition else B` is an expression, so it produces a value you can assign.

**Kotlin vs Python.** In Kotlin, `if` is itself an expression: `val grade = if (score > 0.9) "great" else "ok"`. In Python, the `if` *statement* produces no value, so there's a separate expression form with the condition in the middle:

```python
score = 0.83
if score > 0.9:
    grade = "great"
elif score > 0.7:
    grade = "good"
else:
    grade = "needs work"
grade                                        # 'good'

label = "pass" if score >= 0.8 else "fail"   # read: "pass" if (score >= 0.8) else "fail"
label                                        # 'pass'
```

**Underneath.** The conditions are tested top to bottom, and the first truthy one wins; the rest are never evaluated. Any object can be a condition, by truthiness.

### Chained comparisons

**What it is.** Python lets you chain comparisons: `0 < x < 1` means `0 < x and x < 1`, with `x` evaluated only once.

**Kotlin vs Python.** Kotlin writes `x > 0 && x < 1`, or `x in 0.0..1.0` for an inclusive range. In Python, `0 <= x <= 1` is the inclusive version.

```python
p = 0.42
0 <= p <= 1          # True   a valid probability
lo, hi = 10, 20
lo < 15 < hi         # True
1 < 3 > 2            # True   legal: (1 < 3) and (3 > 2), but confusing; avoid
```

**How it goes wrong.** A chain is always "each neighbouring pair, joined with `and`", even when that isn't what you meant. A typo that flips one sign still runs and gives a plausible answer:

```python
x = 5
1 < x < 3            # False  the intended range check: is x between 1 and 3?
1 < x > 3            # True   a flipped sign: (1 < x) and (x > 3), no error
```

`in` and `is` are comparison operators too, so they chain, which produces the famous puzzle:

```python
False == False in [False]     # True
(False == False) in [False]   # False
```

The first line is not `(False == False) in [False]`. It's the chain `(False == False) and (False in [False])`, which is `True and True`. **Fix:** keep chains to the form `low < x < high`, and use parentheses or `and` for anything else.

### `for` loops, `range`, `enumerate` and `zip`

**What it is.** Python's `for` is always a **for-each** loop: `for item in iterable:` binds the name `item` to each element in turn. An **iterable** is any object that can hand out its elements one at a time: strings, lists, ranges, files and many more. There's no C-style `for (i = 0; i < n; i++)`.

**Why it matters.** Every training loop is a `for` over epochs and batches. You'll write them daily, mostly over `range` and over data loaders.

**Kotlin vs Python.** Kotlin's `for (x in items)` is the same idea. The index-based Kotlin loops map onto `range`:

| Kotlin | Python | Values |
|---|---|---|
| `for (i in 0 until 5)` (or `0..<5`) | `for i in range(5):` | 0, 1, 2, 3, 4 |
| `for (i in 2..7)` | `for i in range(2, 8):` | 2 … 7 (Python's stop is excluded) |
| `for (i in 0 until 10 step 3)` | `for i in range(0, 10, 3):` | 0, 3, 6, 9 |
| `for (i in 5 downTo 1)` | `for i in range(5, 0, -1):` | 5, 4, 3, 2, 1 |
| `items.forEachIndexed { i, x -> }` | `for i, x in enumerate(items):` | (0, first), (1, second), … |
| `a.zip(b)` | `zip(a, b)` | pairs, stops at the shorter |

**`range`.** `range(stop)`, `range(start, stop)` and `range(start, stop, step)` describe arithmetic sequences. The stop value is never included, the same "fences" rule as slicing. A `range` is **lazy**: it stores only start, stop and step, and produces numbers on demand, so `range(10**9)` costs no memory. `list(...)` turns it into a list so you can see the values:

```python
for i in range(3):
    print(i)
# 0
# 1
# 2
list(range(2, 8))            # [2, 3, 4, 5, 6, 7]
list(range(0, 10, 3))        # [0, 3, 6, 9]
list(range(5, 0, -1))        # [5, 4, 3, 2, 1]
list(range(5, 0))            # []   counting up from 5 to 0 gives nothing
range(10)                    # range(0, 10)   lazy: no list is built
len(range(0, 10, 3))         # 4
```

**Looping over strings and lists** gives you the elements directly:

```python
for ch in "abc":
    print(ch, end=" ")       # end=" " prints a space instead of a newline (see print, below)
print()                      # finish the line
# a b c
losses = [0.92, 0.61, 0.48]
for loss in losses:
    print(f"{loss:.1f}")
# 0.9
# 0.6
# 0.5
```

**`enumerate`** gives you the position and the element together, as a pair. `for i, x in ...` **unpacks** each pair into two names (Day 2b covers unpacking). `start=1` makes the count start at 1:

```python
for epoch, loss in enumerate(losses, start=1):
    print(f"epoch {epoch}: loss {loss}")
# epoch 1: loss 0.92
# epoch 2: loss 0.61
# epoch 3: loss 0.48
```

**`zip`** walks two (or more) sequences in step, giving one tuple per position:

```python
names = ["cat", "dog", "bird"]
scores = [0.91, 0.07, 0.02]
for name, score in zip(names, scores):
    print(f"{name:<5}{score:>6.1%}")
# cat   91.0%
# dog    7.0%
# bird   2.0%
list(zip("abc", [1, 2]))     # [('a', 1), ('b', 2)]   stops at the shorter one
```

!!! warning "Common mistake: `zip` silently drops the extra items"
    If the lists have different lengths (a label is missing, say), `zip` stops at the shorter one without any error, and your data no longer lines up. Pass `strict=True` (Python 3.10+) to turn that into an error:

    ```python
    try:
        list(zip(["cat", "dog", "bird"], [0.91, 0.07], strict=True))
    except ValueError as err:
        print("ValueError:", err)
    # ValueError: zip() argument 2 is shorter than argument 1
    ```

**`reversed`** walks a sequence backwards, like Kotlin's `reversed()` or `downTo`. It's lazy too:

```python
for i in reversed(range(3)):
    print(i, end=" ")
print()
# 2 1 0
for loss in reversed(losses):
    print(loss, end=" ")
print()
# 0.48 0.61 0.92
```

**Underneath.** Each time round, the loop asks the iterable for its next element and binds the loop name to it. The loop name is an ordinary name, not a copy and not a new scope. Two consequences follow, shown in the next two boxes. [Day 3d](day-03d-iterators-generators.md) shows exactly what `for` does step by step (`iter` and `next`).

!!! warning "Common mistake: assigning to the loop variable doesn't change the list"
    The loop name is a label on each element in turn. Rebinding it moves the label; the list is untouched:

    ```python
    values = [1, 2, 3]
    for v in values:
        v = v * 10           # rebinds v; the list still holds 1, 2, 3
    values                   # [1, 2, 3]
    v                        # 30   the loop variable survives the loop (unlike Kotlin)
    ```

    **Fix:** build a new list, `scaled = []` then `scaled.append(v * 10)` inside the loop, or write a list comprehension (Day 2b). And notice the second surprise: in Kotlin, `v` would not exist after the loop; in Python it keeps its last value.

!!! warning "Common mistake: removing items from a list while looping over it"
    The loop walks positions 0, 1, 2, … in the list as it is *now*. Removing an item shifts the rest left, so the next one is skipped:

    ```python
    scores = [0.2, 0.1, 0.9, 0.3]
    for s in scores:
        if s < 0.5:
            scores.remove(s)     # remove deletes the first matching item
    scores                       # [0.1, 0.9]   0.1 survived: it was skipped
    ```

    Trace: position 0 is `0.2`, removed, so the list becomes `[0.1, 0.9, 0.3]`. The loop moves to position 1, which is now `0.9`; `0.1` slid into position 0 and is never checked. Position 2 is `0.3`, removed. **Fix:** build a new list of the items to keep instead of removing from the one you're looping over.

### `while`, `break`, `continue` and `else` on loops

**What it is.** `while condition:` repeats its block as long as the condition is truthy. Inside any loop, `break` leaves the loop immediately and `continue` skips to the next round. A loop can also have an `else:` block, which runs only if the loop finished **without** hitting `break`.

**Why it matters.** Early stopping, retry loops, scanning for the first bad value and polling are all `while` / `break` patterns.

**Kotlin vs Python.** `while`, `break` and `continue` work as in Kotlin. Python has no `do { } while` (use `while True:` with a `break`), and no labelled breaks like `break@outer`: to leave two nested loops, set a flag or move the loops into a function and `return` (Day 3). `else` on a loop has no Kotlin equivalent.

**Example: `continue` and `break`.**

```python
readings = [12, -1, 15, 9, 999, 14]   # -1 means "sensor off", 999 means "sensor broken"
total = 0
for r in readings:
    if r == -1:
        continue             # skip this reading, go on with the next
    if r == 999:
        break                # stop reading entirely
    total += r
total                        # 36   12 + 15 + 9; the loop stopped before 14
```

**`while True:` with `break`: Python's do-while.** Kotlin's `do { ... } while (cond)` runs the body at least once. Python writes it as an endless loop that breaks out from inside. A retry loop is the typical case. Here a list of status codes stands in for a flaky download that fails twice:

```python
responses = [503, 503, 200]      # what each attempt "returns"
attempt = 0
while True:
    status = responses[attempt]
    attempt += 1
    print(f"attempt {attempt}: status {status}")
    if status == 200 or attempt == 5:   # success, or give up after 5 tries
        break
# attempt 1: status 503
# attempt 2: status 503
# attempt 3: status 200
```

Always give a `while True:` a way out that is guaranteed to happen, like the attempt limit here.

**`for ... else`: "searched and didn't find".** Read the `else` as "no break":

```python
losses = [0.9, 0.7, 0.62, 0.55]
for epoch, loss in enumerate(losses):
    if loss != loss:         # only nan is unequal to itself (math.isnan says it better)
        print("diverged at epoch", epoch)
        break
else:
    print("no nan found")
# no nan found
```

**Underneath: a trace of an early-stopping loop.** Training stops when the validation loss hasn't improved for `patience` epochs in a row:

```python
val_losses = [0.80, 0.62, 0.55, 0.57, 0.56, 0.58, 0.51]
patience = 3
best = math.inf
bad_epochs = 0
epoch = 0
while epoch < len(val_losses):
    loss = val_losses[epoch]
    if loss < best:
        best = loss
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs == patience:
            print(f"stopping at epoch {epoch}, best {best}")
            break
    epoch += 1
# stopping at epoch 5, best 0.55
```

| epoch | loss | better than best? | best | bad_epochs |
|---|---|---|---|---|
| 0 | 0.80 | yes (vs inf) | 0.80 | 0 |
| 1 | 0.62 | yes | 0.62 | 0 |
| 2 | 0.55 | yes | 0.55 | 0 |
| 3 | 0.57 | no | 0.55 | 1 |
| 4 | 0.56 | no | 0.55 | 2 |
| 5 | 0.58 | no | 0.55 | 3 → `break` |

**Leaving two nested loops.** `break` only leaves the innermost loop. Kotlin would use a label (`break@outer`); Python has none. The plain fix is a flag that the outer loop checks:

```python
grid = [[0.1, 0.4], [0.7, 0.95], [0.99, 0.2]]   # e.g. scores per (row, column)
found = None
for r, row in enumerate(grid):
    for c, score in enumerate(row):
        if score > 0.9:
            found = (r, c)       # a pair of numbers in parentheses: a tuple (Day 2b)
            break                # leaves the inner loop only
    if found is not None:
        break                    # leaves the outer loop
found                            # (1, 1)
```

Without the second `break`, the outer loop would go on to row 2 and overwrite `found` with `(2, 0)`. Day 3 shows the cleaner version: put the loops in a function and `return` from the middle.

Epoch 6 (loss 0.51, which would have been better) is never looked at. That's the trade-off early stopping makes, and why `patience` is a setting you tune.

!!! warning "Common mistake: a `while` condition that can never become false"
    Here's the float trap from earlier, inside a loop. The condition `x != 1.0` stays true forever, because `x` steps from `0.9999999999999999` straight past `1.0`. A safety counter stops this demo; without it, the loop would never end:

    ```python
    x = 0.0
    steps = 0
    while x != 1.0 and steps < 15:   # the steps check is only a safety net for this demo
        x += 0.1
        steps += 1
    steps, x                          # (15, 1.5000000000000002)  it went straight past 1.0
    ```

    **Fix:** loop on a count (`for i in range(10):`), or compare with `<` (`while x < 1.0:`). A notebook cell stuck in an endless loop is stopped with the "interrupt kernel" button (the ■ square); in a terminal, press Ctrl+C.

### `match`: Python's `when`

**What it is.** `match value:` compares a value against a series of `case` **patterns** and runs the first one that fits. Patterns can be literal values, alternatives with `|`, a wildcard `_`, a name that captures the value, a sequence with a fixed shape, and more. Added in Python 3.10.

**Kotlin vs Python.**

```kotlin
val delegate = when (device) {
    "gpu" -> "GpuDelegate"
    "npu", "dsp" -> "NnApiDelegate"
    else -> "CPU"
}
```

```python
device = "dsp"
match device:
    case "gpu":
        delegate = "GpuDelegate"
    case "npu" | "dsp":
        delegate = "NnApiDelegate"
    case _:
        delegate = "CPU"
delegate                     # 'NnApiDelegate'
```

What's the same: first match wins; `case _:` is Kotlin's `else ->`; `|` is Kotlin's comma. What's different: Python's `match` is a **statement**, not an expression, so each branch assigns the result itself. And there's no exhaustiveness check: without `case _:`, an unmatched value silently does nothing.

**Guards and capture.** `case n if n > 100:` binds the matched value to the name `n`, then checks the extra condition (the **guard**):

```python
batch = 512
match batch:
    case 0:
        size_note = "empty"
    case n if n > 256:
        size_note = f"large ({n})"
    case _:
        size_note = "normal"
size_note                    # 'large (512)'
```

**Sequence patterns** check the length and shape of a list or tuple and unpack it at the same time. This is where `match` beats `when`: here it checks an image's shape:

```python
shape = (224, 224, 3)
match shape:
    case (h, w, 3):
        layout = f"HWC colour image, {h}x{w}"
    case (3, h, w):
        layout = f"CHW colour image, {h}x{w}"
    case (h, w):
        layout = f"greyscale, {h}x{w}"
    case _:
        layout = "unknown"
layout                       # 'HWC colour image, 224x224'
```

Trace: `(224, 224, 3)` has three items, so it's tested against `(h, w, 3)`: the first two items are captured into `h` and `w`, and the third must equal the literal `3`. It does, so that branch runs and the others are skipped.

!!! warning "Common mistake: a bare name in a `case` captures; it doesn't compare"
    A plain name in a pattern means "match anything and bind it to this name", never "compare with this variable":

    ```python
    GPU = "gpu"
    chosen = "cpu"
    match chosen:
        case GPU:                # captures: binds GPU = "cpu" and always matches
            print("using the GPU:", GPU)
    # using the GPU: cpu
    ```

    With another `case` after it, Python refuses to run the file at all: `SyntaxError: name capture 'GPU' makes remaining patterns unreachable`. **Fix:** use the literal (`case "gpu":`), a guard (`case d if d == GPU:`), or a dotted name such as an enum member (`case Device.GPU:`), which Python treats as a value to compare. (An **enum** is a class of named constants, like Kotlin's `enum class`; Day 3 introduces classes.)

## `print` and `input`

### `print` options: `sep` and `end`

**What it is.** `print(a, b, c)` converts each argument with `str()`, joins them with `sep` (default: one space) and writes `end` afterwards (default: a newline). It returns `None`.

**Kotlin vs Python.** `println(x)` is `print(x)`; Kotlin's `print(x)` (no newline) is `print(x, end="")`. Kotlin takes one argument; Python takes any number.

```python
print("loss", 0.25, "acc", 0.91)
# loss 0.25 acc 0.91
print("2026", "10", "22", sep="-")
# 2026-10-22
print("a", "b", "c", sep="")
# abc
print("loading", end="... ")
print("done")
# loading... done
print(1, 2.5, None, True, [1, "a"])
# 1 2.5 None True [1, 'a']
```

Notice the list prints its strings with quotes: a list's `str()` uses the repr of its elements.

**How it goes wrong.** `print` is the only output a script gives you, and it's easy to print something that *looks* right. `print("7")` and `print(7)` both show `7`. When the type matters, print the repr: `print(repr(x))` shows `'7'` for the string and `7` for the number. Day 2a explains `repr`.

### `input` always returns a string

**What it is.** `input(prompt)` shows the prompt, waits for the user to type a line and press Enter, and returns that line as a `str`, without the newline. It can't run in the browser on this site, because nothing can type into it; try it in the REPL.

**Kotlin vs Python.** It's Kotlin's `readln()` (or `readLine()`), which also returns a `String`.

Here is a session in the terminal REPL (`>>>` is the prompt; the text after `?` is what you type):

```text
>>> name = input("Model name? ")
Model name? mobilenet
>>> epochs = int(input("Epochs? "))
Epochs? 10
>>> print(f"training {name} for {epochs} epochs")
training mobilenet for 10 epochs
```

**How it goes wrong.** Forgetting `int(...)` leaves a string, and arithmetic on it does something else without any error. This block stands in for `input()` with the string it would return:

```python
typed = "10"           # what input() returns when you type 10
typed * 2              # '1010'   string repetition, not 20
int(typed) * 2         # 20
try:
    int("ten")         # what happens if the user types ten
except ValueError as err:
    print("ValueError:", err)
# ValueError: invalid literal for int() with base 10: 'ten'
```

## Summary tables

Now that each piece has been taught, here are the comparisons in one place.

| Kotlin | Python | Watch out for |
|---|---|---|
| `Int`, `Long`, `BigInteger` | `int` | never overflows (NumPy ints do) |
| `Double` | `float` | same IEEE-754 double; no 32-bit float without NumPy |
| `Boolean` (`true`) | `bool` (`True`) | a subtype of `int`; `and` / `or` return operands |
| `null`, `T?` | `None` | no null-safety; test with `is None` |
| `String`, `Char` | `str` | no char type; strings are Day 2a |
| `a / b` on `Int` | `a // b` | `//` floors: `-7 // 2` is `-4` |
| `a % b` | `a % b` | sign follows the divisor: `-7 % 2` is `1` |
| `===` / `==` | `is` / `==` | use `is` only for `None`, `True`, `False` |
| `if (...) a else b` | `a if ... else b` | |
| `when` | `match` | a statement; bare names capture |
| `0 until n` | `range(n)` | |
| `println` / `print` | `print()` / `print(end="")` | |
| `"%.2f".format(x)` | `f"{x:.2f}"` | `.1%` multiplies by 100 |
| `do { } while` | `while True:` … `break` | give it a guaranteed exit |
| `break@outer` | a flag, or `return` from a function | |

Mutable vs immutable, for the types so far: `int`, `float`, `bool`, `str` and `None` are immutable; `list` is mutable. Day 2b adds tuples (immutable), dicts and sets (mutable).

## Going further

These are real and occasionally useful, but you won't need them in the next month:

- **Complex numbers** (`3 + 4j`, the `cmath` module): [Numeric types](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex).
- **Bitwise operators** (`&`, `|`, `^`, `~`, `<<`, `>>`), the same ones you use to unpack ARGB pixels on Android: [Bitwise operations on integer types](https://docs.python.org/3/library/stdtypes.html#bitwise-operations-on-integer-types).
- **Exact decimal and fraction arithmetic** for money and exact ratios: [`decimal`](https://docs.python.org/3/library/decimal.html) and [`fractions`](https://docs.python.org/3/library/fractions.html).
- **The walrus operator** `:=` (assignment inside an expression, as in `while (line := f.readline()):`): [Assignment expressions](https://docs.python.org/3/reference/expressions.html#assignment-expressions).
- **More `match` patterns** (class patterns, mapping patterns, `as`): [PEP 636, the `match` tutorial](https://peps.python.org/pep-0636/).
- **Float internals**: `sys.float_info`, `math.ulp`, `math.nextafter`, `math.fsum`: [Floating-point arithmetic: issues and limitations](https://docs.python.org/3/tutorial/floatingpoint.html).
- **Reference counting and the garbage collector** in detail: [`gc` module](https://docs.python.org/3/library/gc.html) and [`sys.getrefcount`](https://docs.python.org/3/library/sys.html#sys.getrefcount).

---

## Exercises

Try each one before opening the solutions. Every solution block re-creates its own inputs, so you can run them in any order.

**1. Warm-up: batches.** A dataset has 50,000 images and a batch size of 128. Using `divmod`, print how many full batches there are and how many images are left over. Then compute the total number of batches including the partial one, twice: with `math.ceil` and with integer maths only.

**2. Warm-up: compare floats properly.** For each value in `values = [0.1 + 0.2, 0.3, float("nan"), 0.3000001, float("inf")]`, print the value, then whether it's `== 0.3`, whether it's close to 0.3 by `math.isclose`, whether it's `nan`, and whether it's finite. Explain the first row.

**3. Core: validate a learning rate.** Write a loop that checks each value in `candidates = [0.001, 0, -0.1, 1.5, True, None, float("nan"), 3e-4]` and prints either `ok` or the reason it's rejected. Rules: must not be `None`; must be an `int` or `float` but **not** a `bool`; must not be `nan`; must satisfy `0 < lr <= 1`. Use `continue` to move on after the first failed rule.

**4. Core: a command dispatcher with `match`.** A tiny training tool receives commands as tuples: `commands = [("train", 10), ("train", 0), ("eval",), ("export", "tflite"), ("export", "onnx"), ("sleep", 5), ()]`. Loop over them and use one `match` statement to print:

- `training for N epochs` for `("train", N)` when `N > 0`, and `epochs must be positive` otherwise;
- `evaluating` for `("eval",)` (a one-item tuple: the comma makes it a tuple);
- `exporting to tflite` for `("export", "tflite")`, and `can't export to X` for any other format;
- `empty command` for `()`, and `unknown command: ...` with the tuple for anything else.

**5. Core: `while` and `for ... else`.** (a) A batch of 512 samples needs `512 * 3_000_000` bytes of GPU memory, and the budget is `400_000_000` bytes. With a `while` loop, halve the batch size (using `//=`) until it fits, printing each size you try. (b) Given `accs = [0.61, 0.72, 0.80, 0.84, 0.86]` (validation accuracy per epoch), loop over the targets `[0.8, 0.9]` and, for each, use `for ... else` to print the first epoch (counting from 1) that reaches the target, or that the target wasn't reached.

**6. Predict the output.** Write down what each line prints, then run it:

```python
print(-9 // 4, -9 % 4)
print(round(4.5), round(5.5))
print(0 or None or "x")
print(1 == 1.0, True + True)
print(1 < 2 < 3 > 2)
print(not 0 == 1)
print(bool("0"), bool(0), bool(float("nan")))
```

**7. Find and fix the bug.** This code should average the valid sensor readings, skipping missing values (`""` or `"n/a"`). The expected answer is `6.25` (the readings 12, 0, 7 and 6). It runs without an error but prints `8`. There are two bugs; find both:

```python
# Exercise 7: this version has two bugs. Fix them in the solution.
readings = ["12", "0", "", "7", "n/a", "6"]
total = 0
count = 0
for r in readings:
    if r == "" or r == "n/a":
        continue
    value = int(r)
    if not value:
        continue
    total += value
    count += 1
print(total // count)
```

**8. Challenge: square root by bisection.** Find \(\sqrt{2}\) without `math.sqrt`. Start with `lo, hi = 1.0, 2.0` (the root is between them). Repeat at most 100 times: take `mid = (lo + hi) / 2`; stop if `mid * mid` is close to 2 by `math.isclose` with `rel_tol=1e-12`; otherwise move `lo` or `hi` to `mid`, keeping the root between them. Use `for ... else` to print either how many steps it took or that it didn't converge. Then run it again with the stopping test `mid * mid == 2.0` and explain what happens.

??? success "Solutions"
    **1. Batches.**

    ```python
    import math

    n_images, batch_size = 50_000, 128
    full, leftover = divmod(n_images, batch_size)
    print(full, leftover)
    print(math.ceil(n_images / batch_size), -(-n_images // batch_size))
    # 390 80
    # 391 391
    ```

    `divmod` gives `50000 // 128 = 390` and `50000 % 128 = 80` in one call. Rounding up gives 391 batches. The integer-only version works because floor-dividing a negative number rounds towards minus infinity: \(-50000 // 128 = -391\), and negating that gives 391, with no float involved.

    **2. Compare floats properly.**

    ```python
    import math

    values = [0.1 + 0.2, 0.3, float("nan"), 0.3000001, float("inf")]
    for v in values:
        print(v, v == 0.3, math.isclose(v, 0.3), math.isnan(v), math.isfinite(v))
    # 0.30000000000000004 False True False True
    # 0.3 True True False True
    # nan False False True False
    # 0.3000001 False False False True
    # inf False False False False
    ```

    The first row is the neighbouring-double problem: `0.1 + 0.2` is one step away from the double that `0.3` becomes, so `==` says `False` while `isclose` (agreement to about 9 significant digits) says `True`. `0.3000001` differs in the 7th digit, which is outside the default tolerance. `nan` is unequal to everything and close to nothing, and neither `nan` nor `inf` is finite.

    **3. Validate a learning rate.**

    ```python
    import math

    candidates = [0.001, 0, -0.1, 1.5, True, None, float("nan"), 3e-4]
    for lr in candidates:
        if lr is None:
            print(f"{str(lr):>6}: missing")
            continue
        if isinstance(lr, bool) or not isinstance(lr, (int, float)):
            print(f"{str(lr):>6}: not a number")
            continue
        if math.isnan(lr):
            print(f"{str(lr):>6}: nan")
            continue
        if not 0 < lr <= 1:
            print(f"{str(lr):>6}: out of range")
            continue
        print(f"{str(lr):>6}: ok")
    #  0.001: ok
    #      0: out of range
    #   -0.1: out of range
    #    1.5: out of range
    #   True: not a number
    #   None: missing
    #    nan: nan
    # 0.0003: ok
    ```

    `None` must be tested first: every later test would crash or mislead on it. `bool` is checked before the number test because `isinstance(True, int)` is `True`, and `True` would otherwise pass as the number 1. `nan` must be caught explicitly, because `0 < nan` is `False` and would only report "out of range". `0 < lr <= 1` is a chained comparison. `str(lr)` turns each value into text so `>6` can right-align it, `None` included.

    **4. A command dispatcher with `match`.**

    ```python
    commands = [("train", 10), ("train", 0), ("eval",), ("export", "tflite"),
                ("export", "onnx"), ("sleep", 5), ()]
    for cmd in commands:
        match cmd:
            case ("train", epochs) if epochs > 0:
                msg = f"training for {epochs} epochs"
            case ("train", _):
                msg = "epochs must be positive"
            case ("eval",):
                msg = "evaluating"
            case ("export", "tflite"):
                msg = "exporting to tflite"
            case ("export", fmt):
                msg = f"can't export to {fmt}"
            case ():
                msg = "empty command"
            case _:
                msg = f"unknown command: {cmd}"
        print(msg)
    # training for 10 epochs
    # epochs must be positive
    # evaluating
    # exporting to tflite
    # can't export to onnx
    # unknown command: ('sleep', 5)
    # empty command
    ```

    Order matters twice. The guarded `("train", epochs)` case must come before the catch-all `("train", _)`, or every train command would be "must be positive". And the literal `("export", "tflite")` must come before the capturing `("export", fmt)`, which matches any format. `_` inside a pattern matches one item without binding it. `()` only matches an empty sequence, and `("sleep", 5)` falls through to `case _:`. The `commands` list is split over two lines inside its brackets (implicit line joining).

    **5. `while` and `for ... else`.**

    ```python
    batch = 512
    bytes_per_sample = 3_000_000
    budget = 400_000_000
    while batch * bytes_per_sample > budget:
        print("too big:", batch)
        batch //= 2
    print("fits:", batch)

    accs = [0.61, 0.72, 0.80, 0.84, 0.86]
    for target in [0.8, 0.9]:
        for epoch, acc in enumerate(accs, start=1):
            if acc >= target:
                print(f"target {target:.0%} reached at epoch {epoch}")
                break
        else:
            print(f"target {target:.0%} not reached in {len(accs)} epochs")
    # too big: 512
    # too big: 256
    # fits: 128
    # target 80% reached at epoch 3
    # target 90% not reached in 5 epochs
    ```

    The `while` condition is checked before each round, so the loop stops as soon as `128 * 3_000_000 = 384_000_000` fits. In part (b) the `else` belongs to the inner `for`: it runs only when that loop finished without `break`, which is exactly "no epoch reached the target". `:.0%` shows a percentage with no decimals. The `break` leaves only the inner loop, so the outer loop goes on to the next target.

    **6. Predict the output.**

    ```python
    print(-9 // 4, -9 % 4)
    print(round(4.5), round(5.5))
    print(0 or None or "x")
    print(1 == 1.0, True + True)
    print(1 < 2 < 3 > 2)
    print(not 0 == 1)
    print(bool("0"), bool(0), bool(float("nan")))
    # -3 3
    # 4 6
    # x
    # True 2
    # True
    # True
    # True False True
    ```

    `-9 / 4` is `-2.25`, which floors to `-3`, and \(-3 \cdot 4 + 3 = -9\) gives the remainder 3. Banker's rounding sends both ties to the even neighbour (4 and 6). `or` returns the first truthy operand, `"x"`. Numbers compare by value across `int` and `float`, and `True` counts as 1. The chain means `1 < 2 and 2 < 3 and 3 > 2`. `not` binds looser than `==`, so the sixth line is `not (0 == 1)`. `"0"` is a non-empty string and `nan` a nonzero float, so both are truthy.

    **7. Find and fix the bug.**

    ```python
    readings = ["12", "0", "", "7", "n/a", "6"]
    total = 0
    count = 0
    for r in readings:
        if r == "" or r == "n/a":
            continue
        value = int(r)              # fix 1: no truthiness check here, so 0 is counted
        total += value
        count += 1
    print(total / count)            # fix 2: true division, not floor division
    # 6.25
    ```

    Bug 1: `if not value: continue` treats the valid reading `0` as missing, because `0` is falsy. Missing values were already skipped at the string level, so the number check had to go. With only that bug, the sum is 25 over 3 readings. Bug 2: `//` floors, so `25 // 3` gives `8` instead of `8.333...`; after fixing bug 1 it would still print `6` instead of `6.25`. An average needs `/`.

    **8. Square root by bisection.**

    ```python
    import math

    for exact in [False, True]:
        lo, hi = 1.0, 2.0
        for step in range(1, 101):
            mid = (lo + hi) / 2
            if exact:
                done = mid * mid == 2.0
            else:
                done = math.isclose(mid * mid, 2.0, rel_tol=1e-12)
            if done:
                print(f"converged after {step} steps: {mid}")
                break
            if mid * mid < 2.0:
                lo = mid
            else:
                hi = mid
        else:
            print(f"no convergence in 100 steps; stuck at {mid}")
    print(math.sqrt(2))
    # converged after 38 steps: 1.4142135623733338
    # no convergence in 100 steps; stuck at 1.414213562373095
    # 1.4142135623730951
    ```

    Each step halves the interval, so after about 38 halvings `mid` agrees with \(\sqrt{2}\) to 12 significant digits and `isclose` stops the loop. With `==`, the loop never stops: no double squares to exactly `2.0` (\(\sqrt{2}\) is irrational), so after about 52 steps `lo` and `hi` are neighbouring doubles, `mid` can't move any more, and the `else` branch reports the failure after 100 steps. That's the float lesson and the `for ... else` lesson in one: never stop a numerical loop on `==`, and always cap it.

## Resources

- [Python tutorial, §3 An Informal Introduction to Python](https://docs.python.org/3/tutorial/introduction.html): numbers and strings, in the official tutorial's words.
- [Python tutorial, §4 More Control Flow Tools](https://docs.python.org/3/tutorial/controlflow.html): `if`, `for`, `range`, `break`, `continue`, loop `else` and `match`.
- [Built-in Types](https://docs.python.org/3/library/stdtypes.html): the reference for truth testing, comparisons and numeric operations.
- [Floating-Point Arithmetic: Issues and Limitations](https://docs.python.org/3/tutorial/floatingpoint.html): the official explanation of `0.1 + 0.2`.
- [Ned Batchelder: Facts and myths about Python names and values](https://nedbatchelder.com/text/names.html): the best explanation of the sticky-note model, as an article and a talk.
- [Python Tutor](https://pythontutor.com/): paste code and watch names and objects being drawn step by step.

## Checklist

- [ ] I can explain names vs objects, and when to use `is` vs `==`.
- [ ] I can predict `/`, `//`, `%`, `**`, `divmod` and `round` results, including for negative numbers.
- [ ] I compare floats with `math.isclose` and test for `nan` with `math.isnan`.
- [ ] I know the falsy values, and that `and` / `or` return an operand.
- [ ] I can write the Python versions of `?.`, `?:` and `!!`.
- [ ] I can convert between `int`, `float`, `str` and `bool`, and read the errors they raise.
- [ ] I can format numbers with `{x}`, `:.2f`, `:>8` and `:.1%`.
- [ ] I can write `if` / `elif` / `else`, conditional expressions and chained comparisons.
- [ ] I can loop with `for`, `range`, `enumerate`, `zip` and `reversed`, and control loops with `while`, `while True`, `break`, `continue` and `else`.
- [ ] I can use `match` instead of `when`, and avoid the capture-pattern trap.
- [ ] I can use `print` with `sep` and `end`, and I know `input` returns a string.

**Next:** [Day 2a: Strings, text and formatting](day-02a-strings.md)
