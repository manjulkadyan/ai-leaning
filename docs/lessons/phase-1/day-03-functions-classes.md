# Day 3: Functions, Scope and Type Hints

**Phase 1 · Week 1 · ~150 minutes**

## Objectives

- [ ] Define functions with `def`, return nothing, one value or several values, and document them with a docstring.
- [ ] Predict what a function can and can't change in the caller's objects: mutating an argument versus rebinding the parameter.
- [ ] Write functions with positional, keyword and default parameters, `*args`, `**kwargs`, keyword-only and positional-only parameters, and call them with `*` and `**` unpacking.
- [ ] Read the `TypeError` messages Python gives for a call that doesn't match a function's parameters.
- [ ] Explain why `def f(x=[])` shares one list between calls, trace it step by step, and fix it with `None`.
- [ ] Predict which object a name refers to using the LEGB rule, and use `global` and `nonlocal`.
- [ ] Write closures (functions that remember their settings) and avoid the late-binding loop trap.
- [ ] Pass functions as values: lambdas, `key=` for `sorted` / `max` / `min`, `map` / `filter` versus comprehensions, and `functools.partial`.
- [ ] Add type hints (`list[int]`, `dict[str, float]`, `X | None`, `Callable`) and say who checks them (mypy, your editor) and who doesn't (Python itself).

**Prerequisites.**

- [Day 1](day-01-setup.md): running a script, the REPL, virtual environments and `pip` (for the mypy example, which runs on your computer).
- [Day 2](day-02-python-types.md): names and objects (`is`, `id`), `None`, truthiness, f-strings and their format specs, `if` and `for`.
- [Day 2b](day-02b-collections.md): lists, tuples and unpacking, dicts, `sorted(...)`, and comprehensions.

Classes, decorators and modules come next, in [Day 3a](day-03a-classes-modules.md); they build directly on today's closures and `*args, **kwargs`. Errors are [Day 3b](day-03b-errors-files-json.md), files and JSON are [Day 3c](day-03c-files-json.md), and iterators and generators are [Day 3d](day-03d-iterators-generators.md).

---

## The big idea

In Kotlin, the compiler reads your whole file before anything runs. A `fun` is a *declaration*: it describes structure, and the compiler checks every call against it.

In Python, `def` is a **statement that runs**, top to bottom, like any other line. It builds a function object and sticks a name on it. Calls are matched to parameters only when they run, and names are looked up only when they're used.

**Analogy.** A Kotlin file is a blueprint the compiler reads in full before building starts. A Python file is a list of instructions read aloud to a builder, one at a time: "build a function and label it `accuracy`". The builder can't look ahead, and a label can be moved to a different object later.

Keep this picture in mind. It explains the surprises in this lesson: why a typo inside a function only fails when the function is called, why a default list is shared between calls (it was built once, when `def` ran), why an assignment inside a function creates a new local variable, why a closure in a loop sees the loop's last value, and why type hints don't stop a wrong-type call. Tomorrow's lesson ([Day 3a](day-03a-classes-modules.md)) applies the same picture to `class`, `@decorator` and `import`.

---

## Defining and calling functions

**What it is.** A function is a named, reusable block of code. You **define** it with `def`, list its **parameters** (the names it receives) in brackets, and **call** it with **arguments** (the values you pass). `return` sends a value back to the caller.

**Why it exists.** Every piece of ML code you'll write is a pile of small functions: load the data, clean a column, compute accuracy, run one training step. Good functions are what make that pile readable.

**Kotlin comparison.**

```kotlin
/** Fraction of correct predictions. */
fun accuracy(correct: Int, total: Int): Double = correct.toDouble() / total
```

```python
def accuracy(correct: int, total: int) -> float:
    """Fraction of correct predictions."""
    return correct / total

accuracy(87, 100)        # 0.87
accuracy.__doc__         # 'Fraction of correct predictions.'
```

What's the same: a name, typed parameters, a return type, a documentation comment.

What's different:

- `def` instead of `fun`, a colon at the end of the line, and the body is **indented** instead of wrapped in `{}`.
- `: int` and `-> float` are **type hints**. Python doesn't check them when the code runs (the "Type hints" section shows what does).
- `/` always gives a `float` in Python (Day 2), so you don't need `toDouble()`.
- The string on the first line of the body is a **docstring**, Python's KDoc. Unlike a KDoc comment, it's kept at run time in the function's `__doc__` attribute. (Names wrapped in double underscores, like `__doc__`, are attributes that Python itself sets and uses; Day 2 called this naming style "dunder".) Your editor shows the docstring on hover, and `help(accuracy)` in a REPL or Jupyter shows it with the signature. The "Docstrings" subsection below shows the full layout.
- There's no expression-body form like Kotlin's `= ...`. A `def` function always uses `return`; the one-expression form is `lambda`, later in this lesson.

### Returning nothing, or several values

A function with no `return` statement (or a bare `return`) gives back `None`, Python's "no value" object from Day 2. It's the equivalent of Kotlin's `Unit`. One difference: Python has no separate "no value" type for `null`, so the same `None` also plays the role of Kotlin's `null` (Day 2).

```python
def log_epoch(epoch: int, loss: float) -> None:
    print(f"epoch {epoch}: loss {loss:.3f}")      # no return statement

result = log_epoch(1, 0.4321)
# epoch 1: loss 0.432
result is None           # True
```

To return several values, separate them with commas. Python packs them into one **tuple** (Day 2b), and the caller usually unpacks it straight away:

```python
def min_max(values: list[float]) -> tuple[float, float]:
    return min(values), max(values)        # really: return (min(values), max(values))

pair = min_max([0.3, 0.9, 0.1])
pair                     # (0.1, 0.9)
type(pair)               # <class 'tuple'>

lo, hi = min_max([0.3, 0.9, 0.1])
lo, hi                   # (0.1, 0.9)
```

In Kotlin you'd return a `Pair` or a small `data class` and destructure it with `val (lo, hi) = ...`. Python skips the wrapper type: the comma *is* the tuple.

**How it goes wrong.** Unpacking needs exactly as many names as the tuple has values. Unpack a 3-value result into two names and you get a `ValueError`:

```python
def summarize(values):
    return min(values), max(values), sum(values) / len(values)

try:
    lo, hi = summarize([0.2, 0.8, 0.5])
except ValueError as e:
    print("ValueError:", e)
# ValueError: too many values to unpack (expected 2, got 3)
```

That's the ▶ Run button's Python 3.14; Python 3.13 and older stop at `(expected 2)`. Too many names gives `not enough values to unpack (expected 4, got 3)`. Fix: match the count, or collect the rest with a starred name (Day 2b): `lo, *rest = summarize(...)` gives `lo = 0.2` and `rest = [0.8, 0.5]`.

### Docstrings

**What it is.** A **docstring** is a string literal placed as the **first statement** of a function body. Python stores it in the function's `__doc__` attribute.

**Why it exists.** It's the documentation your editor shows on hover and `help()` prints, so it's where you say what the function expects and returns. Library code you'll read (NumPy, pandas, PyTorch) documents every public function this way, in one of two common layouts. The **Google style** below uses `Args:` and `Returns:` sections; the **NumPy style** uses `Parameters` and `Returns` headings underlined with dashes.

**Kotlin comparison.** KDoc's `/** ... @param correct ... @return ... */` is a comment the compiler and Dokka read; Python's docstring is a real string object kept in the running program.

```python
def accuracy(correct: int, total: int) -> float:
    """Fraction of correct predictions.

    Args:
        correct: number of predictions that matched the label.
        total: number of predictions made. Must be greater than 0.

    Returns:
        A float between 0.0 and 1.0.
    """
    return correct / total

accuracy.__doc__.splitlines()[0]     # 'Fraction of correct predictions.'
```

The layout: one summary line, a blank line, then the details. Triple quotes (Day 2) let the string span lines.

**Underneath.** `__doc__` holds the string exactly as written, indentation included. `inspect.getdoc` (from `inspect`, the standard library's module for looking inside live objects) returns it cleaned up, which is what `help()` displays:

```python
import inspect
print(inspect.getdoc(accuracy))
# Fraction of correct predictions.
#
# Args:
#     correct: number of predictions that matched the label.
#     total: number of predictions made. Must be greater than 0.
#
# Returns:
#     A float between 0.0 and 1.0.
```

**How it goes wrong.** Only a string that is the first statement counts. A `#` comment, or a string placed after other code, leaves `__doc__` as `None`, and the editor shows nothing:

```python
def no_doc(x):
    # Fraction of correct predictions.   (a comment, not a docstring)
    return x

def late_doc(x):
    y = x
    """This string is not the first statement."""
    return y

no_doc.__doc__, late_doc.__doc__     # (None, None)
```

### Underneath: `def` is a statement that builds an object

When Python reaches a `def` line, it does two things, and only two:

1. It builds a **function object**: the compiled body, the parameter names, the default values, the docstring, and a link back to the module it lives in.
2. It binds the function's name to that object, exactly like `x = 5` binds `x`.

It does **not** run the body. That only happens on a call. A function object is a value like any other:

```python
type(accuracy)           # <class 'function'>
accuracy.__name__        # 'accuracy'

score = accuracy         # a second name for the same object (no call: no brackets)
score(9, 10)             # 0.9
score is accuracy        # True
```

Because the body doesn't run until the call, a typo inside a function is invisible until you call it:

```python
def broken():
    return undefined_name + 1     # no error yet: the body hasn't run

try:
    broken()
except NameError as e:
    print("NameError:", e)
# NameError: name 'undefined_name' is not defined
```

That `try:` / `except` wrapper catches the error so the rest of this page keeps running, and prints the message Python would show. You'll see it on every "goes wrong" example in this lesson and the next; [Day 3b](day-03b-errors-files-json.md) teaches it properly. Kotlin would have refused to compile `broken`. Python finds the problem only on the path that actually runs, which is why tests (Day 3b) matter more in Python.

!!! warning "Common mistake: `print` instead of `return`"
    In a notebook, a function that *prints* its result looks like it works. But the caller gets `None`:

    ```python
    def mean_loss(losses):
        print(sum(losses) / len(losses))       # shows the value, returns None

    avg = mean_loss([0.5, 0.25])
    # 0.375
    try:
        avg * 100
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'
    ```

    When you see `'NoneType'` in an error, look for a function that forgot its `return`. Fix: `return sum(losses) / len(losses)`, and print at the call site if you want to see it.

## How arguments are passed: shared objects

**What it is.** When you call `f(data)`, Python binds the parameter name inside `f` to **the same object** the caller passed. Nothing is copied. This is called **call by sharing** (or "pass by object reference"). Two consequences follow:

- **mutating** the object through the parameter (`xs.append(...)`, `xs[0] = ...`, `d["k"] = ...`) changes the caller's object, because there is only one object;
- **rebinding** the parameter (`xs = something_else`) only moves the function's own name; the caller's name still points at the original.

**Why it matters.** ML helpers take lists, dicts and (from Day 4) NumPy arrays of training data. A helper that changes its argument in place silently changes the caller's data, and the bug shows up later, far away, as a model trained on the wrong numbers. You need to know, for every function you write, whether it changes its input or returns a new object.

**Kotlin comparison.** Kotlin passes object references the same way: a `fun addBias(xs: MutableList<Double>) { xs.add(1.0) }` changes the caller's list too. The difference is that Kotlin parameters are `val`: `xs = mutableListOf()` inside the function is a compile error. Python lets you rebind a parameter, and the rebinding doesn't reach the caller, which surprises people who expect it to.

```python
def add_bias(xs):
    xs.append(1.0)               # mutates the one shared list

def add_bias2(xs):
    xs = xs + [1.0]              # builds a NEW list and rebinds the local name
    return xs

features = [0.5, 0.2]
add_bias(features)
features                 # [0.5, 0.2, 1.0]   the caller's list changed

features2 = [0.5, 0.2]
out = add_bias2(features2)
features2, out           # ([0.5, 0.2], [0.5, 0.2, 1.0])   the caller's list didn't
```

**Underneath.** Use `is` (Day 2) to watch the names. On entry, the parameter and the caller's variable are two names for one object; after a rebinding, they're not:

```python
data = [3, 1, 2]

def look(xs):
    print("same object as data on entry:", xs is data)
    xs = sorted(xs)              # sorted() returns a new list (Day 2b)
    print("same object after xs = sorted(xs):", xs is data)
    return xs

look(data)
# same object as data on entry: True
# same object after xs = sorted(xs): False
# [1, 2, 3]
data                     # [3, 1, 2]   untouched
```

Step by step: the call binds `xs` to the list `data` points at. `sorted(xs)` builds a second list, and `xs = ...` moves only the name `xs` to it. `data` still points at the first list, which nothing changed. Immutable arguments (numbers, strings, tuples) can't be mutated at all, so for them only rebinding exists: `def bump(n): n += 1` returns a new number and leaves the caller's variable alone.

!!! warning "Common mistake: a helper that changes the training data"
    ```python
    def clip_scores(values, hi=1.0):
        for i, v in enumerate(values):
            if v > hi:
                values[i] = hi           # BUG: writes into the caller's list
        return values

    train = [0.4, 1.7, 0.9]
    clipped = clip_scores(train)
    train                    # [0.4, 1.0, 0.9]   the original data is gone
    clipped is train         # True              "two" results, one object
    ```

    Nothing errors; the raw value 1.7 has quietly been lost. Fix: build a new list and leave the input alone, `return [min(v, hi) for v in values]`, or copy first with `values = list(values)` (shallow copies are in Day 2b). If a function does change its input on purpose, say so in its name or docstring and return `None`, the way `list.sort()` does (Day 2b).

## Parameters: positional, keyword and default

**What it is.** When you call a function, each argument is matched to a parameter in one of two ways:

- a **positional argument** is matched by its position: first argument to first parameter;
- a **keyword argument** is written `name=value` and matched by name, in any order.

A parameter with a **default value** (`epochs=10`) can be left out; it then takes the default.

**Why it exists.** ML functions take many settings. A training function might have 20 parameters, almost all with sensible defaults, and you override two of them by name: `train(model, lr=0.01, epochs=5)`. Keyword arguments make those calls readable.

**Kotlin comparison.** This is the same idea as Kotlin's default and named arguments:

```kotlin
fun train(modelName: String, epochs: Int = 10, lr: Double = 0.001, verbose: Boolean = false): String =
    "$modelName: $epochs epochs, lr=$lr, verbose=$verbose"

train("mlp", lr = 0.01, verbose = true)
```

```python
def train(model_name, epochs=10, lr=0.001, verbose=False):
    return f"{model_name}: {epochs} epochs, lr={lr}, verbose={verbose}"

train("mlp")                           # 'mlp: 10 epochs, lr=0.001, verbose=False'
train("mlp", 5)                        # 'mlp: 5 epochs, lr=0.001, verbose=False'
train("mlp", lr=0.01, verbose=True)    # 'mlp: 10 epochs, lr=0.01, verbose=True'
train(verbose=True, model_name="cnn")  # 'cnn: 10 epochs, lr=0.001, verbose=True'
```

The difference: Kotlin matches arguments at compile time, and a wrong name is a compile error. Python matches them **when the call runs**, so a wrong name is a run-time `TypeError`.

**Underneath: how Python binds a call.** For `train("mlp", 5, verbose=True)`, Python fills an empty slot for each parameter:

1. Positional arguments fill the parameters from the left: `model_name = "mlp"`, `epochs = 5`.
2. Keyword arguments fill slots by name: `verbose = True`.
3. Any slot still empty takes its default: `lr = 0.001`.
4. If a slot without a default is still empty, or a slot was filled twice, or a keyword names no parameter, Python raises `TypeError`.

In a `def`, parameters with defaults must come after the ones without: `def f(a=1, b)` is a `SyntaxError: parameter without a default follows parameter with a default`. In a call, positional arguments must come before keyword ones: `train(lr=0.1, "mlp")` is a `SyntaxError: positional argument follows keyword argument`. (Syntax errors stop the whole file before anything runs, so they're shown here as text rather than as a runnable block.)

**How it goes wrong.** Each failed rule from step 4 has its own message. Learn to read them; you'll see them weekly:

```python
try:
    train()
except TypeError as e:
    print("TypeError:", e)
# TypeError: train() missing 1 required positional argument: 'model_name'

try:
    train("mlp", model_name="cnn")
except TypeError as e:
    print("TypeError:", e)
# TypeError: train() got multiple values for argument 'model_name'

try:
    train("mlp", epoch=3)
except TypeError as e:
    print("TypeError:", e)
# TypeError: train() got an unexpected keyword argument 'epoch'. Did you mean 'epochs'?
```

The second one is common with wrappers: `"mlp"` filled `model_name` by position, then the keyword tried to fill it again. The third is a typo; recent Python versions even suggest the right name.

## The mutable default trap

**What it is.** A default value is computed **once, when the `def` statement runs**, and stored on the function object. Every call that leaves the argument out gets *that same object*. If the default is mutable (a list, a dict, a set) and the function changes it, the change is still there on the next call.

**Why it matters.** This is the most famous Python bug. In ML code it shows up as data that leaks between calls: a list of layers that keeps growing, a cache dict shared by two experiments, a batch that contains last call's samples.

**Kotlin comparison.** Kotlin evaluates a default expression **on every call**:

```kotlin
fun addLayer(name: String, layers: MutableList<String> = mutableListOf()): MutableList<String> {
    layers.add(name)
    return layers
}
addLayer("conv")   // [conv]
addLayer("relu")   // [relu]   a fresh list each call
```

So your Kotlin instinct is exactly wrong here. The same code in Python:

```python
def add_layer(name, layers=[]):        # BUG: one list, built when def runs
    layers.append(name)
    return layers

add_layer("conv")        # ['conv']
add_layer("relu")        # ['conv', 'relu']           !!
add_layer("pool")        # ['conv', 'relu', 'pool']   !!!
add_layer.__defaults__   # (['conv', 'relu', 'pool'],)
```

**Underneath: the trace.** `__defaults__` is where the function object keeps its default values: a tuple with one entry per defaulted parameter.

<figure class="diagram" markdown>
![Animation: the def statement builds a function object add_layer whose defaults slot points to one empty list. Call 1 binds the parameter layers to that list and appends conv; call 2 binds layers to the same list and appends relu, so it now holds conv and relu; call 3 adds pool. A second panel shows the fixed version, where each call creates a fresh list](../../assets/img/phase-1/day03-mutable-default.svg)
<figcaption>Animated: every call that leaves out `layers` gets the one list stored on the function object.</figcaption>
</figure>

- `def` runs: Python evaluates `[]` once, gets a new empty list (call it L), and stores it in `add_layer.__defaults__`.
- Call 1, `add_layer("conv")`: no `layers` argument, so `layers` is bound to L. `append` changes L to `['conv']`. The function returns L.
- Call 2, `add_layer("relu")`: `layers` is bound to **the same L**, which already holds `'conv'`. After `append`, L is `['conv', 'relu']`.
- Call 3: L becomes `['conv', 'relu', 'pool']`. Every result you got back is the same object, so all three now show three items.

You can watch the "once" happen by making the default come from a function that prints:

```python
def default_layers():
    print("building the default list")
    return []

def add_layer2(name, layers=default_layers()):    # the print happens HERE, at def time
    layers.append(name)
    return layers
# building the default list

add_layer2("conv")       # ['conv']           (nothing printed: no new list)
add_layer2("relu")       # ['conv', 'relu']
```

**The fix: use `None` as a sentinel.** A **sentinel** is a special value that means "nothing was passed". `None` is immutable, so sharing it is harmless, and the function builds a fresh list on every call:

```python
def add_layer_fixed(name, layers=None):
    if layers is None:           # the caller passed nothing
        layers = []              # a NEW list, built during this call
    layers.append(name)
    return layers

add_layer_fixed("conv")          # ['conv']
add_layer_fixed("relu")          # ['relu']

mine = ["input"]
add_layer_fixed("conv", mine)    # ['input', 'conv']   the caller's list is still used
```

!!! warning "Common mistake: `if not layers` instead of `if layers is None`"
    `if not layers:` looks shorter, but an empty list is falsy (Day 2), so it also replaces a list the caller passed on purpose:

    ```python
    def add_layer_bad(name, layers=None):
        if not layers:               # True for None AND for []
            layers = []
        layers.append(name)
        return layers

    shared = []
    add_layer_bad("conv", shared)    # ['conv']
    shared                           # []   the caller's list was ignored
    ```

    Test for the sentinel exactly: `if layers is None`.

Immutable defaults (`10`, `0.001`, `"adam"`, `None`, a tuple) are always safe, because nothing can change them in place.

## `*args` and `**kwargs`: any number of arguments

**What it is.** Two special parameters collect "everything else":

- `*args` collects extra **positional** arguments into a tuple;
- `**kwargs` collects extra **keyword** arguments into a dict, from parameter name to value.

The names `args` and `kwargs` ("keyword arguments") are only a convention; the `*` and `**` are the syntax.

**Why it exists.** Two jobs. First, functions that take any number of values, like `print` or `max`. Second, and more important for you, **forwarding**: a function that takes whatever it's given and passes it on to another function, unchanged. Wrappers, decorators and subclasses do this constantly.

**Kotlin comparison.** `*args` is Kotlin's `vararg`. Kotlin has no equivalent of `**kwargs`; the nearest is passing a `Map<String, Any>`.

```kotlin
fun logMetrics(runName: String, vararg values: Double) = println("$runName ${values.toList()}")
logMetrics("run1", 0.5, 0.25)
```

```python
def log_metrics(run_name, *values, **tags):
    print(run_name, values, tags)

log_metrics("run1", 0.5, 0.25, lr=0.01, epochs=3)
# run1 (0.5, 0.25) {'lr': 0.01, 'epochs': 3}
log_metrics("run2")
# run2 () {}
```

**Underneath.** For `log_metrics("run1", 0.5, 0.25, lr=0.01, epochs=3)`: `"run1"` fills `run_name` by position. There are no more named parameters, so the leftover positionals `0.5, 0.25` go into the tuple `values`. The keywords `lr` and `epochs` match no named parameter, so they go into the dict `tags`. With nothing left over, you get an empty tuple and an empty dict, never `None`.

### Unpacking at the call site

The same stars work the other way round in a **call**: `*seq` spreads a list or tuple into positional arguments, and `**d` spreads a dict into keyword arguments. Kotlin has the first half as the spread operator, `logMetrics("run1", *arr)`.

```python
config = {"epochs": 3, "lr": 0.1}
train("mlp", **config)        # same as train("mlp", epochs=3, lr=0.1)
# 'mlp: 3 epochs, lr=0.1, verbose=False'

args = ["cnn", 7]
train(*args)                  # same as train("cnn", 7)
# 'cnn: 7 epochs, lr=0.001, verbose=False'
```

`*` and `**` unpacking can appear anywhere in a call, even several times and after keyword arguments: Python collects all the positional values in order, then all the keyword ones. So `f(1, *[2, 3], x=4, *[5])` passes the positionals 1, 2, 3, 5 and the keyword `x=4`.

You'll use `**config` constantly when hyperparameters come from a JSON or YAML file (Day 3c): load a dict, splat it into the function.

The forwarding pattern combines both directions. This wrapper accepts any call and passes it on untouched:

```python
def verbose_train(*args, **kwargs):
    print("calling train with", args, kwargs)
    return train(*args, **kwargs)        # unpack exactly what came in

verbose_train("mlp", lr=0.5)
# calling train with ('mlp',) {'lr': 0.5}
# 'mlp: 10 epochs, lr=0.5, verbose=False'
```

Remember this shape; it's the core of every decorator in [Day 3a](day-03a-classes-modules.md).

**How it goes wrong.** Two classic slips.

A config dict with a key the function doesn't know:

```python
config = {"epochs": 3, "lr": 0.1, "momentum": 0.9}
try:
    train("mlp", **config)
except TypeError as e:
    print("TypeError:", e)
# TypeError: train() got an unexpected keyword argument 'momentum'
```

Fix: remove the key, or give the function a `**kwargs` parameter to absorb extras on purpose.

Forgetting the star passes the whole list as **one** argument:

```python
def total(*values):
    return values

total(1, 2, 3)           # (1, 2, 3)
nums = [1, 2, 3]
total(nums)              # ([1, 2, 3],)   one argument: a tuple holding the list
total(*nums)             # (1, 2, 3)      unpacked: three arguments
```

## Keyword-only and positional-only parameters

**What it is.** Two markers in a parameter list control *how* arguments may be passed:

- everything after a bare `*` is **keyword-only**: callers must write the name;
- everything before a `/` is **positional-only**: callers must not write the name.

```text
def f(a, b, /, c, d=1, *, e, f=2):
      └─┬─┘    └──┬──┘    └──┬──┘
 positional-only  either   keyword-only
```

A `*args` parameter also makes everything after it keyword-only, since `*args` swallows all the remaining positionals.

**Why it exists.** Keyword-only parameters make calls self-explaining. Compare `resize(img, 224, 224, True)` with `resize(img, width=224, height=224, keep_aspect=True)`: four months later, only the second one tells you what `True` means. Positional-only parameters let a library rename a parameter later without breaking anyone's code, because no caller can have used the name.

**Kotlin comparison.** Kotlin lets the *caller* choose named arguments, and its style guide asks you to name Boolean arguments, but the function can't demand it. Python's `*` lets the function enforce it.

```python
def resize(image, /, *, width, height, keep_aspect=True):
    return f"{image} -> {width}x{height}, keep_aspect={keep_aspect}"

resize("cat.jpg", width=224, height=224)
# 'cat.jpg -> 224x224, keep_aspect=True'
```

**Underneath.** The binding steps from earlier still apply, with two extra checks: positional arguments may only fill slots before the `*`, and keyword arguments may only fill slots after the `/`. Built-in functions use `/` a lot. `help(len)` shows the signature `len(obj, /)`, which is why `len(obj=[1])` fails with `TypeError: len() takes no keyword arguments`.

**How it goes wrong.**

```python
try:
    resize("cat.jpg", 224, 224)
except TypeError as e:
    print("TypeError:", e)
# TypeError: resize() takes 1 positional argument but 3 were given

try:
    resize(image="cat.jpg", width=224, height=224)
except TypeError as e:
    print("TypeError:", e)
# TypeError: resize() got some positional-only arguments passed as keyword arguments: 'image'

try:
    resize("cat.jpg", width=224)
except TypeError as e:
    print("TypeError:", e)
# TypeError: resize() missing 1 required keyword-only argument: 'height'
```

"takes 1 positional argument but 3 were given" is the message to recognise: when you see it on a function you know takes more parameters, check for a `*` in its signature.

## Scope: where Python looks up a name

**What it is.** A **namespace** is a mapping from names to objects; a module, a function call and a class each have their own. The **scope** of a name is the part of the code where it can be seen. When Python meets a bare name like `scale`, it searches four namespaces in a fixed order, called **LEGB**:

1. **L**ocal: names assigned inside the current function call (parameters included).
2. **E**nclosing: names in any function that *contains* the current one, from the inside out.
3. **G**lobal: names at the top level of the current module (file).
4. **B**uilt-in: Python's own names, like `len`, `print`, `range` and `list`.

The first match wins. If nothing matches, you get a `NameError`.

**Why it matters.** Most "why does this variable have the wrong value?" bugs in scripts and notebooks are scope bugs. And closures (later in this lesson) and decorators ([Day 3a](day-03a-classes-modules.md)) are built on the "Enclosing" step.

**Kotlin comparison.** Kotlin has **block scope**: a `val` inside an `if` or a `for` body disappears at the closing brace. Python has no block scope. Only functions (lambdas included), comprehensions, classes and modules create scopes, so a variable assigned inside a loop or an `if` is still there afterwards:

```kotlin
for (i in 0 until 3) { val squared = i * i }
// println(squared)   // compile error: unresolved reference
```

```python
for i in range(3):
    squared = i * i

i, squared               # (2, 4)   both survive the loop
```

A comprehension is different: it runs in its own small scope, so its loop variable does **not** leak out:

```python
[sq for sq in range(3)]  # [0, 1, 2]
try:
    sq
except NameError as e:
    print("NameError:", e)
# NameError: name 'sq' is not defined
```

**Example: LEGB in action.**

```python
scale = 10                       # global

def outer():
    scale = 2                    # local to outer, "enclosing" for inner
    def inner(x):
        return x * scale         # L: no 'scale' here → E: found in outer → 2
    return inner(5)

outer()                          # 10   (5 * 2, not 5 * 10)
scale                            # 10   the global was never touched
```

<figure class="diagram" markdown>
![Animation: four nested boxes labelled Built-in, Global, Enclosing (outer) and Local (inner). A search marker for the name scale starts in the Local box, finds nothing, moves out to Enclosing and finds scale = 2. A second search for len moves from Local through Enclosing and Global and is found in Built-in](../../assets/img/phase-1/day03-legb.svg)
<figcaption>Animated: a name is looked up from the inside out (Local, Enclosing, Global, Built-in), and the first match wins.</figcaption>
</figure>

**Underneath: the "assignment makes it local" rule.** Python decides which names are local when it *compiles* the function, before any call: **if a function assigns to a name anywhere in its body, that name is local for the whole body.** Reading a global is fine. Assigning to one creates a new local instead, and `+=` counts as an assignment.

```python
count = 0

def increment():
    count += 1               # count is assigned here, so it's local everywhere in increment

try:
    increment()
except UnboundLocalError as e:
    print("UnboundLocalError:", e)
# UnboundLocalError: cannot access local variable 'count' where it is not associated with a value
```

`count += 1` means `count = count + 1`. The right-hand side reads the *local* `count`, which has no value yet. `UnboundLocalError` is a kind of `NameError` that says exactly this.

There are two fixes. The `global` statement says "this name means the module-level variable":

```python
def increment_global():
    global count
    count += 1

increment_global()
increment_global()
count                    # 2
```

`global` works, but hidden state that any function can change makes code hard to test. The better fix is usually to pass the value in and return the new one: `count = add_one(count)`.

`nonlocal` is the same idea for the **enclosing** function's variable; you'll need it for closures in the next section.

Note the difference between **rebinding** a name and **mutating** an object. This needs no `global`, because `history.append` changes the list without assigning to the name `history`:

```python
history = []

def record(loss):
    history.append(loss)     # reads the global name, mutates the list: fine

record(0.9)
record(0.7)
history                  # [0.9, 0.7]
```

!!! warning "Common mistake: shadowing a built-in"
    Built-ins come last in LEGB, so any global with the same name hides them. `list`, `dict`, `sum`, `max`, `input`, `id` and `type` are tempting variable names:

    ```python
    list = ["cat", "dog"]          # a global named 'list' now hides the built-in
    try:
        list("abc")
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: 'list' object is not callable

    del list                       # remove the global; the built-in is visible again
    list("abc")                    # ['a', 'b', 'c']
    ```

    In a notebook, the bad assignment may be in a cell you ran an hour ago. If a built-in suddenly "is not callable", search for where you assigned to its name. Use names like `labels`, `total` and `best` instead.

## Closures: functions that remember

**What it is.** A **closure** is an inner function that keeps using variables from the enclosing function, even after the enclosing function has returned. Those remembered variables are called **free variables** of the inner function.

**Why it exists.** It lets you build a function from settings, once, and use it many times. ML is full of these: learning-rate schedules, data transforms configured with a mean and a standard deviation, callbacks. Closures are also how decorators work.

**Kotlin comparison.** Kotlin lambdas capture variables too, and a captured `var` can be changed directly. In Python, an inner function can *read* an enclosing variable freely, but to *rebind* it, you must declare it `nonlocal` (the scope rule from the last section).

```kotlin
fun makeCounter(): () -> Int {
    var n = 0
    return { n += 1; n }      // Kotlin may change the captured var directly
}
```

```python
def make_counter():
    n = 0
    def tick():
        nonlocal n               # rebind the enclosing n, not a new local
        n += 1
        return n
    return tick                  # return the function itself (no brackets)

counter = make_counter()
counter()                # 1
counter()                # 2
other = make_counter()   # a separate n
other()                  # 1
counter()                # 3
```

**How it goes wrong.** Leave out `nonlocal` and `n += 1` makes `n` local to `tick` (the scope rule from the last section), so the first call fails:

```python
def make_counter_broken():
    n = 0
    def tick():
        n += 1                   # assignment: n is local to tick, and has no value yet
        return n
    return tick

try:
    make_counter_broken()()      # build a counter, then call it
except UnboundLocalError as e:
    print("UnboundLocalError:", e)
# UnboundLocalError: cannot access local variable 'n' where it is not associated with a value
```

Fix: add `nonlocal n` as the first line of `tick`. Reading `n` alone (`return n + 1`) would work without it; only rebinding needs the declaration.

**A realistic example: a learning-rate schedule.** "Step decay" multiplies the learning rate by `drop` every `every` epochs. You configure it once and the training loop calls it with an epoch number:

```python
def make_step_decay(initial: float, drop: float, every: int):
    def lr_at(epoch: int) -> float:
        return initial * drop ** (epoch // every)     # // is floor division (Day 2)
    return lr_at

schedule = make_step_decay(initial=0.1, drop=0.5, every=10)
schedule(0), schedule(9), schedule(10), schedule(25)
# (0.1, 0.1, 0.05, 0.025)
```

**Underneath.** When `make_step_decay` returns, its local namespace would normally be thrown away. But `lr_at` uses `initial`, `drop` and `every`, so Python keeps those three variables alive in small boxes called **cells**, attached to the function object:

```python
schedule.__code__.co_freevars                         # ('drop', 'every', 'initial')
[cell.cell_contents for cell in schedule.__closure__] # [0.5, 10, 0.1]
```

`__code__` is the function's compiled body (the bytecode from Day 1), and its `co_freevars` lists the remembered variables in alphabetical order, which is why it isn't the parameter order; `__closure__` holds one cell per name, in that same order. The cells hold the *variables*, not copies of their values at the time. That detail causes the next bug.

!!! android "Android bridge"
    You've written closures every time you did `var taps = 0; button.setOnClickListener { taps++ }`. The listener lambda outlives the function that created it, and it still changes `taps`. To make that work, the Kotlin compiler moves a captured, mutable local into a small heap box (`kotlin.jvm.internal.Ref.IntRef` on the JVM) that both the function and the lambda point to. Python's cells are the same box, made visible as `__closure__`. The one difference is the rebinding rule: Kotlin lets the lambda assign to the captured `var` directly, while Python needs `nonlocal`.

!!! warning "Common mistake: closures in a loop see the last value"
    ```python
    funcs = []
    for k in [1, 2, 3]:
        def times_k(x):
            return x * k          # k is looked up when times_k is CALLED
        funcs.append(times_k)

    [f(10) for f in funcs]        # [30, 30, 30]   not [10, 20, 30]
    ```

    All three functions share one variable `k`, and by the time any of them is called, the loop has finished with `k = 3`. Fix: freeze the current value with a default argument, which is evaluated when `def` runs (the same rule that caused the mutable-default bug, used on purpose):

    ```python
    funcs = []
    for k in [1, 2, 3]:
        def times_k(x, k=k):      # default captured NOW, once per loop pass
            return x * k
        funcs.append(times_k)

    [f(10) for f in funcs]        # [10, 20, 30]
    ```

## Functions as values: lambdas and key functions

**What it is.** In Python, functions are objects, so you can store them in variables, lists and dicts, pass them to other functions, and return them. A **lambda** is a small anonymous function written as one expression: `lambda x: x * x`. A **higher-order function** is a function that takes or returns other functions.

**Why it exists.** You'll pass functions all the time: an activation to a layer, a loss to a trainer, a "sort by this" rule to `sorted`, a transform to a dataset.

**Kotlin comparison.**

```kotlin
val square: (Double) -> Double = { x -> x * x }
val activations = mapOf("relu" to ::relu, "square" to square)
listOf("banana", "fig", "kiwi").sortedBy { it.length }
```

```python
def relu(x: float) -> float:
    return max(0.0, x)

square = lambda x: x * x
activations = {"relu": relu, "square": square}

activations["relu"](-2.0)        # 0.0
activations["square"](3.0)       # 9.0
```

What's the same: function references (`relu` is Kotlin's `::relu`), lambdas, maps of functions.

What's different:

- A Python lambda is **one expression**, with an implicit `return`. No statements, no assignments, no multiple lines. Anything longer gets a `def`.
- There's no implicit `it`; you name the parameter: `lambda s: len(s)`.
- `relu` (no brackets) is the function object; `relu(2.0)` (with brackets) calls it. Kotlin needs `::relu` to get the reference; in Python, the bare name is the reference.
- PEP 8, Python's official style guide, says don't assign a lambda to a name, as `square = lambda ...` does above; use `def`, which gives the function a real `__name__` for error messages. Lambdas are for passing inline.

**Underneath.** A lambda builds exactly the same kind of function object as `def`; the only differences are the one-expression body and the missing name:

```python
def square_def(x):
    return x * x

type(square) is type(square_def)     # True   both are <class 'function'>
square.__name__, square_def.__name__ # ('<lambda>', 'square_def')
```

That `'<lambda>'` is what appears in tracebacks, which is why named functions are easier to debug.

**How it goes wrong.** A statement inside a lambda is a syntax error, so the whole file fails before anything runs (shown as text for that reason). The line `f = lambda x: y = x` is rejected with:

```text
SyntaxError: cannot assign to lambda
```

Python reads `lambda x: y` as the expression and then sees `= x`, an attempt to assign to the lambda itself. If you need an assignment, a loop or a `try`, write a `def`.

### Key functions: `sorted`, `max` and `min`

`sorted`, `max`, `min` and `list.sort` accept a `key=` argument: a function that turns each element into the value to compare. You used `key=` on Day 2b; here's what it does.

```python
words = ["banana", "fig", "cherry", "kiwi"]
sorted(words, key=len)                     # ['fig', 'kiwi', 'banana', 'cherry']

preds = [("cat", 0.71), ("dog", 0.18), ("fox", 0.93)]
max(preds, key=lambda p: p[1])             # ('fox', 0.93)
sorted(preds, key=lambda p: p[1], reverse=True)
# [('fox', 0.93), ('cat', 0.71), ('dog', 0.18)]

scores = {"alice": 0.81, "bob": 0.92, "cara": 0.77}
max(scores, key=scores.get)                # 'bob'   key = the dict lookup
```

**Underneath.** `sorted(words, key=len)` calls `len` once per element and sorts the elements by those results:

```text
element   key = len(element)
"banana"  6
"fig"     3
"cherry"  6
"kiwi"    4
sorted by key: 3, 4, 6, 6  →  fig, kiwi, banana, cherry
```

`banana` and `cherry` tie on 6, and they keep their original order. Python's sort is **stable**: equal keys never swap. That lets you sort by several things in passes (see the example below). `max(scores, key=scores.get)` works because iterating a dict gives its keys, and `scores.get` is a bound method (a function already attached to that dict) that maps each key to its value.

**Sorting by two keys.** To sort predictions by confidence, highest first, and break ties by label alphabetically, you have two options. Two stable passes, the *secondary* key first; or one pass with a tuple key, since tuples compare element by element. `-p[1]` negates the score so that ascending order puts high scores first:

```python
preds4 = [("cat", 0.9), ("ant", 0.7), ("dog", 0.9), ("bee", 0.7)]
by_label = sorted(preds4, key=lambda p: p[0])                # pass 1: secondary key
sorted(by_label, key=lambda p: p[1], reverse=True)          # pass 2: primary key
# [('cat', 0.9), ('dog', 0.9), ('ant', 0.7), ('bee', 0.7)]
sorted(preds4, key=lambda p: (-p[1], p[0]))                 # one pass, tuple key
# [('cat', 0.9), ('dog', 0.9), ('ant', 0.7), ('bee', 0.7)]
```

In pass 2, `cat` and `dog` tie on 0.9, and stability keeps them in pass 1's alphabetical order. The tuple key does the same in one sort; negation only works for numbers, so for two text keys in opposite directions use the two-pass form.

### `map` and `filter` versus comprehensions

`map(f, items)` applies `f` to each item; `filter(pred, items)` keeps the items where `pred` is truthy. Both return **lazy iterators**: objects that produce values one at a time only when asked (the full story is [Day 3d](day-03d-iterators-generators.md)). Wrap them in `list(...)` to see the values.

```python
values = [-2.0, -0.5, 0.0, 1.5]
list(map(relu, values))                    # [0.0, 0.0, 0.0, 1.5]
list(filter(lambda v: v > 0, values))      # [1.5]

[relu(v) for v in values]                  # [0.0, 0.0, 0.0, 1.5]   same, as a comprehension
[v for v in values if v > 0]               # [1.5]
```

In Python, comprehensions are usually preferred: they read left to right and need no lambda. `map` is fine when you already have a named function, as in `map(relu, values)`. Kotlin's `list.map { }` is eager and gives a list; Python's `map` behaves like Kotlin's `asSequence().map { }`.

!!! warning "Common mistake: printing or reusing a `map`"
    A `map` object isn't a list. Printing it shows only its type and memory address (`<map object at 0x...>`, a different address each run), and it can be read **once**: the second pass finds it empty.

    ```python
    m = map(relu, values)
    repr(m).startswith("<map object at 0x")   # True   not the values
    list(m), list(m)                          # ([0.0, 0.0, 0.0, 1.5], [])
    ```

    Fix: wrap it in `list(...)` once and keep the list, or use a comprehension. [Day 3d](day-03d-iterators-generators.md) explains why iterators run out.

### Fixing some arguments: `functools.partial`

`partial` builds a new function with some arguments already filled in:

```python
from functools import partial

def clip(x, lo, hi):
    return max(lo, min(x, hi))

clip01 = partial(clip, lo=0.0, hi=1.0)     # a new function: clip with lo and hi fixed
[clip01(v) for v in [-0.3, 0.4, 1.7]]      # [0.0, 0.4, 1.0]
```

It's a closure you don't have to write by hand. Kotlin has no built-in partial application; you'd write the lambda yourself, `val clip01 = { x: Double -> clip(x, 0.0, 1.0) }`.

**Underneath.** A `partial` object stores the original function and the arguments you fixed. Calling it passes the stored positional arguments first, then yours, and merges the keyword arguments (yours win on a clash):

```python
clip01.func is clip, clip01.args, clip01.keywords   # (True, (), {'lo': 0.0, 'hi': 1.0})
clip01(5, hi=10.0)                                   # 5     hi=10.0 overrides the stored 1.0
```

**How it goes wrong.** Fixed *positional* arguments fill parameters from the left, by the binding rules from earlier, not the ones you meant:

```python
bad = partial(clip, 0.0, 1.0)      # meant lo=0.0, hi=1.0
bad.args                           # (0.0, 1.0)
bad(0.5)                           # 1.0   not 0.5: this is clip(0.0, 1.0, 0.5), so x=0.0, lo=1.0, hi=0.5
```

Fix: fix the later parameters by keyword, `partial(clip, lo=0.0, hi=1.0)`.

!!! warning "Common mistake: calling the function instead of passing it"
    `key=` wants a function. With brackets, you call it on the spot and pass its result:

    ```python
    try:
        sorted(words, key=len())
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: len() takes exactly one argument (0 given)
    ```

    Fix: `key=len`. The same applies anywhere a function is expected: `activations={"relu": relu}`, not `relu()`.

## Type hints

**What it is.** A **type hint** (or annotation) is a note in the code saying what type a parameter, return value or variable should have. Python stores hints but **does not check them when the code runs**. Separate tools do: a **type checker** such as mypy or pyright, and your editor (VS Code and PyCharm run one in the background).

**Why it exists.** Hints bring back much of the safety you're used to from Kotlin: wrong-type calls flagged before running, autocomplete that knows what `x` is, and documentation that can't drift away from the code. Most major libraries (NumPy, PyTorch, pandas) ship type hints, though coverage varies.

**Kotlin comparison.** The syntax maps almost one to one, but the enforcement doesn't:

| Kotlin | Python hint |
|---|---|
| `Int`, `Double`, `String`, `Boolean` | `int`, `float`, `str`, `bool` |
| `List<Int>` | `list[int]` |
| `Map<String, Double>` | `dict[str, float]` |
| `Pair<Double, Double>` | `tuple[float, float]` |
| `String?` | <code>str &#124; None</code> |
| `(Double) -> Double` | `Callable[[float], float]` |
| `Unit` | `None` |

```python
from collections.abc import Callable

def mean(values: list[float]) -> float:
    return sum(values) / len(values)

def find_label(labels: dict[int, str], idx: int) -> str | None:
    return labels.get(idx)                 # None when idx is missing

def apply_all(fn: Callable[[float], float], values: list[float]) -> list[float]:
    return [fn(v) for v in values]

mean([0.5, 0.9])                           # 0.7
find_label({0: "cat", 1: "dog"}, 1)        # 'dog'
find_label({0: "cat"}, 5) is None          # True
apply_all(relu, [-1.0, 2.0])               # [0.0, 2.0]
```

Read the new pieces this way:

- `list[float]` is "a list whose elements are floats". The brackets give the element type, like Kotlin's `<>`.
- `str | None` is "a str or None", Kotlin's `String?`. Older code writes it `Optional[str]` (from `typing`); it means the same.
- `Callable[[float], float]` is "a function taking one float and returning a float". The first bracket lists the parameter types.

A few more forms you'll meet constantly:

```python
from typing import Any, List, Dict, Optional, Union

threshold: float = 0.5                   # a variable annotation: same idea, on a plain name
shape: tuple[int, ...] = (32, 3, 224, 224)   # tuple[int, ...] = a tuple of ints, any length

def first(xs: list[Any]) -> Any:         # Any = "any type at all": the checker stops checking
    return xs[0]

# The pre-Python-3.9 spellings, still all over older ML code:
def old_style(xs: List[int], d: Dict[str, float], v: Union[int, str], o: Optional[str]) -> None:
    pass

first(["a", 2]), shape   # ('a', (32, 3, 224, 224))
```

`List[int]` and `Dict[str, float]` (capitalised, imported from `typing`) mean the same as `list[int]` and `dict[str, float]` to a type checker; they're the old spelling, from before Python 3.9 allowed brackets on the built-in types. `Union[int, str]` is the old `int | str`, and `Optional[str]` the old `str | None`. `tuple[float, float]` (fixed length, one type per position) differs from `tuple[float, ...]` (any length, all one type).

**Underneath: hints are stored, not enforced.** Python keeps them in a dict on the function, and that's all:

```python
mean.__annotations__     # {'values': list[float], 'return': <class 'float'>}

def double(x: int) -> int:
    return x * 2

double("ab")             # 'abab'   no error: the hint is ignored at run time
```

A type checker reads the source without running it. Here's a file with two mistakes, checked by mypy on my machine. To try it, save the block exactly as shown (first line included) as `hints.py`, then run `pip install mypy` and `mypy hints.py` in a terminal:

```python
# Run this on your computer (or in Colab), not in the browser.
def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def find_label(labels: dict[int, str], idx: int) -> str | None:
    return labels.get(idx)


mean(["0.5", "0.9"])                 # wrong element type
label = find_label({0: "cat"}, 1)
print(label.upper())                 # label might be None
```

```text
$ mypy hints.py
hints.py:10: error: List item 0 has incompatible type "str"; expected "float"  [list-item]
hints.py:10: error: List item 1 has incompatible type "str"; expected "float"  [list-item]
hints.py:12: error: Item "None" of "str | None" has no attribute "upper"  [union-attr]
Found 3 errors in 1 file (checked 1 source file)

$ python hints.py
Traceback (most recent call last):
  File "/home/you/demo/hints.py", line 10, in <module>
    mean(["0.5", "0.9"])                 # wrong element type
    ~~~~^^^^^^^^^^^^^^^^
  File "/home/you/demo/hints.py", line 3, in mean
    return sum(values) / len(values)
           ~~~^^^^^^^^
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

mypy found both bugs without running anything, including the `None` case that Python would only hit on an unlucky input. Python found the first bug only by crashing, deep inside `sum`.

!!! warning "Common mistake: trusting `X | None` without checking"
    A hint of `str | None` is a promise that the value is *sometimes* `None`. Kotlin forces you to handle that with `?.` or `?:`; Python doesn't, and the run-time error appears far from the cause:

    ```python
    label = find_label({0: "cat"}, 1)
    try:
        label.upper()
    except AttributeError as e:
        print("AttributeError:", e)
    # AttributeError: 'NoneType' object has no attribute 'upper'
    ```

    Fix: check first, `label.upper() if label is not None else "unknown"`, and run a type checker so it reminds you.

## Going further

These are real and useful, but not needed yet:

- **Generic** type hints (`def first[T](xs: list[T]) -> T`), `TypedDict`, `Literal` and type aliases: [typing module docs](https://docs.python.org/3/library/typing.html).
- More of `functools`: `reduce`, `cache` and `singledispatch` (one function, different code per argument type): [functools docs](https://docs.python.org/3/library/functools.html). `lru_cache` and `wraps` are in [Day 3a](day-03a-classes-modules.md).
- The `operator` module (`operator.itemgetter(1)` as a ready-made `key=` function): [operator docs](https://docs.python.org/3/library/operator.html).
- Exactly how a call is matched to parameters, and `inspect.signature` for reading signatures from code: [inspect docs](https://docs.python.org/3/library/inspect.html).
- Scopes and namespaces in full, including the class-body scope: [Python tutorial §9.2](https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces).
- Recursion and Python's recursion limit (`sys.getrecursionlimit()`, 1000 by default): [sys docs](https://docs.python.org/3/library/sys.html#sys.getrecursionlimit).

---

## Exercises

**Warm-ups**

1. **Several return values.** Write `summarize(values: list[float]) -> tuple[float, float, float]` that returns the minimum, maximum and mean, with a docstring. Call it on `[0.2, 0.8, 0.5]`, unpack the result into three names, and print them.
2. **Sorting by a key.** Given `preds = [("cat", 0.70), ("dog", 0.20), ("fox", 0.90), ("cow", 0.40), ("owl", 0.05)]`, use `sorted(..., key=..., reverse=True)` to get the top three by confidence, and `min(..., key=...)` to get the least confident one.

**Core**

3. **A scaler that remembers its training data.** In ML you compute scaling statistics on the training set once, then apply the same scaling to new data. Write `make_min_max_scaler(train_values)` that returns a function `scale(x)` mapping the training minimum to 0 and the maximum to 1, using a closure. If every training value is equal, `scale` should return `0.0`. Show which variables the closure keeps.
4. **A pipeline of functions.** Write `apply_pipeline(values, *steps)` that passes each value through every step function in order and returns the list of results. Use it with `relu`, a `partial` of `clip(x, lo, hi)` and a lambda that turns 0–1 into a whole percentage. Then call it with no steps, and with a list of steps unpacked with `*`.
5. **Keyword-only flags and hints.** Write `top_k(scores: dict[str, float], k: int = 3, *, smallest: bool = False) -> list[tuple[str, float]]` that returns the `k` best `(name, score)` pairs, highest first, or lowest first when `smallest=True`. Show that passing `smallest` by position fails, and read the error.

**Predict the output**

6. Write down what each `print` shows **before** running it. Then run it and explain any surprise.

    ```python
    def add(item, bag=[]):
        bag.append(item)
        return len(bag)

    total = 10
    def bump(n):
        total = n + 1
        return total

    def show(*args, **kwargs):
        return len(args), sorted(kwargs)

    fs = [lambda: i * 2 for i in range(3)]

    print(add("a"), add("b"), add("c", []))
    print(bump(5), total)
    print(show(1, 2, x=3, *[4, 5], y=6))
    print([f() for f in fs])
    ```

**Find and fix the bug**

7. Each of these three functions has one bug, and none of them raises an error. The intended output is `[2, 20, 200]`, then `[1, 2] [3]`, then `['hi', '', '']`. Run it, see what it prints instead, and fix all three.

    ```python
    def make_scalers(factors):
        scalers = []
        for f in factors:
            def scale(x):
                return x * f
            scalers.append(scale)
        return scalers

    def collect(values, out=[]):
        for v in values:
            out.append(v)
        return out

    def pad(tokens, length, fill=None):
        if not fill:
            fill = "<pad>"
        return tokens + [fill] * (length - len(tokens))

    print([s(2) for s in make_scalers([1, 10, 100])])
    print(collect([1, 2]), collect([3]))
    print(pad(["hi"], 3, fill=""))
    ```

**Challenge**

8. **A learning-rate schedule factory.** Write `make_schedule(kind, /, **params)` that returns a function `lr_at(epoch)`. Support four kinds, each built by its own inner factory function with **keyword-only** parameters:
    - `"constant"` (`lr`): always `lr`;
    - `"step"` (`lr`, `drop`, `every`): `lr * drop ** (epoch // every)`;
    - `"exponential"` (`lr`, `gamma`): `lr * gamma ** epoch`;
    - `"warmup"` (`lr`, `warmup_epochs`): rises linearly to `lr` over the first `warmup_epochs` epochs (`lr * (epoch + 1) / warmup_epochs`), then stays at `lr`.

    Pick the factory from a dict of functions, and pass the parameters on with `**params`. Print each schedule for epochs 0 to 5 with 4 decimals. Then show what happens with an unknown kind, a missing parameter and a misspelt one.

??? success "Solutions"
    **1. Several return values.**

    ```python
    def summarize(values: list[float]) -> tuple[float, float, float]:
        """Return (minimum, maximum, mean) of a non-empty list."""
        return min(values), max(values), sum(values) / len(values)

    lo, hi, avg = summarize([0.2, 0.8, 0.5])
    print(lo, hi, avg)
    # 0.2 0.8 0.5
    summarize.__doc__
    # 'Return (minimum, maximum, mean) of a non-empty list.'
    ```

    `return a, b, c` builds one tuple, and `lo, hi, avg = ...` unpacks it by position, so the order in the `return` is the order of the names. The hint `tuple[float, float, float]` documents that order for the reader and the type checker.

    **2. Sorting by a key.**

    ```python
    preds = [("cat", 0.70), ("dog", 0.20), ("fox", 0.90), ("cow", 0.40), ("owl", 0.05)]
    sorted(preds, key=lambda p: p[1], reverse=True)[:3]
    # [('fox', 0.9), ('cat', 0.7), ('cow', 0.4)]
    min(preds, key=lambda p: p[1])
    # ('owl', 0.05)
    ```

    The lambda turns each `(label, confidence)` tuple into the number to compare, `p[1]`. `reverse=True` sorts from high to low, and `[:3]` slices the first three. Without `key=`, tuples would be compared by their first element, the label, alphabetically.

    **3. A scaler that remembers its training data.**

    ```python
    def make_min_max_scaler(train_values: list[float]):
        lo, hi = min(train_values), max(train_values)
        span = hi - lo
        def scale(x: float) -> float:
            if span == 0:                  # every training value equal: avoid dividing by zero
                return 0.0
            return (x - lo) / span
        return scale

    scale = make_min_max_scaler([10.0, 20.0, 50.0])
    [scale(v) for v in [10.0, 30.0, 50.0, 60.0]]
    # [0.0, 0.5, 1.0, 1.25]
    flat = make_min_max_scaler([3.0, 3.0])
    flat(3.0), flat(7.0)
    # (0.0, 0.0)
    scale.__code__.co_freevars, [c.cell_contents for c in scale.__closure__]
    # (('lo', 'span'), [10.0, 40.0])
    ```

    `min` and `max` run once, inside `make_min_max_scaler`; each call to `scale` only does one subtraction and one division. The closure keeps only the variables `scale` actually uses, `lo` and `span`, not `hi` or the training list. A new value outside the training range, like 60, maps outside 0–1 (1.25), which is the honest result: the statistics come from the training data only.

    **4. A pipeline of functions.**

    ```python
    from functools import partial

    def relu(x):
        return max(0.0, x)

    def clip(x, lo, hi):
        return max(lo, min(x, hi))

    def apply_pipeline(values, *steps):
        out = []
        for v in values:
            for step in steps:             # each step's output is the next step's input
                v = step(v)
            out.append(v)
        return out

    apply_pipeline([-1.0, 0.25, 3.0], relu, partial(clip, lo=0.0, hi=1.0), lambda v: round(v * 100))
    # [0, 25, 100]
    apply_pipeline([-1.0, 0.25, 3.0])
    # [-1.0, 0.25, 3.0]
    steps = [relu, partial(clip, lo=0.0, hi=1.0)]
    apply_pipeline([-1.0, 0.25, 3.0], *steps)
    # [0.0, 0.25, 1.0]
    ```

    `*steps` collects any number of functions into a tuple; with none, the inner loop doesn't run and each value comes back unchanged. Trace 3.0: `relu` gives 3.0, the clip gives 1.0, the lambda gives 100. In the last call, `*steps` at the call site unpacks the list into two separate arguments.

    **5. Keyword-only flags and hints.**

    ```python
    def top_k(scores: dict[str, float], k: int = 3, *, smallest: bool = False) -> list[tuple[str, float]]:
        """Return the k best (name, score) pairs, highest first unless smallest=True."""
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=not smallest)
        return ranked[:k]

    losses = {"mlp": 0.42, "cnn": 0.31, "rnn": 0.55, "linear": 0.61}
    top_k(losses, 2, smallest=True)
    # [('cnn', 0.31), ('mlp', 0.42)]
    top_k(losses)
    # [('linear', 0.61), ('rnn', 0.55), ('mlp', 0.42)]
    try:
        top_k(losses, 2, True)
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: top_k() takes from 1 to 2 positional arguments but 3 were given
    ```

    `scores.items()` gives `(name, score)` pairs, and the key picks the score. `reverse=not smallest` flips the order with one Boolean. The bare `*` makes `smallest` keyword-only, so `top_k(losses, 2, True)` is rejected: a reader of that call couldn't tell what `True` means. The error counts positional slots: `scores` and `k`, "from 1 to 2", because `k` has a default.

    **6. Predict the output.**

    ```python
    def add(item, bag=[]):
        bag.append(item)
        return len(bag)

    total = 10
    def bump(n):
        total = n + 1
        return total

    def show(*args, **kwargs):
        return len(args), sorted(kwargs)

    fs = [lambda: i * 2 for i in range(3)]

    print(add("a"), add("b"), add("c", []))
    # 1 2 1
    print(bump(5), total)
    # 6 10
    print(show(1, 2, x=3, *[4, 5], y=6))
    # (4, ['x', 'y'])
    print([f() for f in fs])
    # [4, 4, 4]
    ```

    Line 1: the first two calls share the default list, so it holds 1 item, then 2. The third call passes its own new list, so its length is 1. Line 2: the assignment makes `total` local to `bump`, so it returns 6 and the global stays 10. Line 3: `*[4, 5]` may appear after a keyword argument, and it still adds positional arguments, so `args` is `(1, 2, 4, 5)`; `sorted` on a dict gives its sorted keys. Line 4: every lambda looks up `i` when called. That `i` is the comprehension's own loop variable (invisible outside the comprehension, which has its own scope), and the lambdas still share it after the loop has finished with `i = 2`, so all give 4; `lambda i=i: i * 2` would give `[0, 2, 4]`.

    **7. Find and fix the bug.**

    ```python
    def make_scalers(factors):
        scalers = []
        for f in factors:
            def scale(x, f=f):             # fix 1: freeze this pass's f as a default
                return x * f
            scalers.append(scale)
        return scalers

    def collect(values, out=None):         # fix 2: None sentinel, not a shared list
        if out is None:
            out = []
        for v in values:
            out.append(v)
        return out

    def pad(tokens, length, fill=None):
        if fill is None:                   # fix 3: "" is falsy, but it's a real fill value
            fill = "<pad>"
        return tokens + [fill] * (length - len(tokens))

    print([s(2) for s in make_scalers([1, 10, 100])])
    # [2, 20, 200]
    print(collect([1, 2]), collect([3]))
    # [1, 2] [3]
    print(pad(["hi"], 3, fill=""), pad(["hi"], 3))
    # ['hi', '', ''] ['hi', '<pad>', '<pad>']
    ```

    The buggy version printed `[200, 200, 200]`, then `[1, 2, 3] [1, 2, 3]`, then `['hi', '<pad>', '<pad>']`. Bug 1 is late binding: all three closures shared the loop variable `f`, which ended at 100. Bug 2 is the mutable default: both calls returned the same list, and `print` showed it twice after both calls had appended to it. Bug 3 is truthiness: `not ""` is `True`, so the caller's empty-string fill was replaced; test for the sentinel with `is None`.

    **8. Challenge: a learning-rate schedule factory.**

    ```python
    from collections.abc import Callable

    def make_schedule(kind: str, /, **params) -> Callable[[int], float]:
        def constant(*, lr):
            return lambda epoch: lr
        def step(*, lr, drop, every):
            return lambda epoch: lr * drop ** (epoch // every)
        def exponential(*, lr, gamma):
            return lambda epoch: lr * gamma ** epoch
        def warmup(*, lr, warmup_epochs):
            def lr_at(epoch):
                return lr * min(1.0, (epoch + 1) / warmup_epochs)
            return lr_at
        factories = {"constant": constant, "step": step,
                     "exponential": exponential, "warmup": warmup}
        return factories[kind](**params)

    schedules = {
        "constant": make_schedule("constant", lr=0.1),
        "step": make_schedule("step", lr=0.1, drop=0.5, every=2),
        "exponential": make_schedule("exponential", lr=0.1, gamma=0.5),
        "warmup": make_schedule("warmup", lr=0.1, warmup_epochs=4),
    }
    for name, sched in schedules.items():
        print(f"{name:>11}: " + "  ".join(f"{sched(e):.4f}" for e in range(6)))
    #    constant: 0.1000  0.1000  0.1000  0.1000  0.1000  0.1000
    #        step: 0.1000  0.1000  0.0500  0.0500  0.0250  0.0250
    # exponential: 0.1000  0.0500  0.0250  0.0125  0.0063  0.0031
    #      warmup: 0.0250  0.0500  0.0750  0.1000  0.1000  0.1000

    try:
        make_schedule("cosine", lr=0.1)
    except KeyError as e:
        print("KeyError:", e)
    # KeyError: 'cosine'
    try:
        make_schedule("step", lr=0.1, drop=0.5)
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: make_schedule.<locals>.step() missing 1 required keyword-only argument: 'every'
    try:
        make_schedule("step", lr=0.1, drop=0.5, evry=2)
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: make_schedule.<locals>.step() got an unexpected keyword argument 'evry'. Did you mean 'every'?
    ```

    `**params` collects the settings as a dict, and `factories[kind](**params)` unpacks them into the chosen factory, so each factory's own keyword-only signature checks the names; that's where the two `TypeError`s come from. `make_schedule.<locals>.step` is that function's **qualified name**: "the `step` defined inside `make_schedule`". Each factory returns a closure over its settings. The `/` makes `kind` positional-only, so a schedule parameter could even be called `kind` without a clash. `{name:>11}` right-aligns the name in 11 characters (Day 2's format specs), and `min(1.0, ...)` caps the warmup at the full rate. In the warmup row, `0.1 * 3 / 4` is stored as 0.07500000000000001, which is why the output is formatted to 4 decimals. The `"  ".join(...)` builds one string from the formatted values, with the two-space string placed between items as the separator (Day 2's `join`); the `(... for e in range(6))` inside it is a generator expression, a lazy comprehension that [Day 3d](day-03d-iterators-generators.md) explains.

## Resources

- [Python tutorial §4.8–4.9: Defining functions, and more on defining functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions): defaults, keyword arguments, special parameters (`/` and `*`), `*args`, unpacking, lambdas, docstrings and annotations.
- [Python tutorial §9.2: Python scopes and namespaces](https://docs.python.org/3/tutorial/classes.html#python-scopes-and-namespaces): the official description of LEGB, `global` and `nonlocal`.
- [Python FAQ: Why are default values shared between objects?](https://docs.python.org/3/faq/programming.html#why-are-default-values-shared-between-objects): the mutable-default trap, from the source.
- [Python docs: functools](https://docs.python.org/3/library/functools.html): `partial` and the rest of the toolbox.
- [mypy: Type hints cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html): the hint syntax on one page.
- [Real Python: Python Scope and the LEGB Rule](https://realpython.com/python-scope-legb-rule/): more examples of scope, `global` and `nonlocal`.

## Checklist

- [ ] I can define a function that returns nothing, one value or a tuple, unpack the result correctly, and document it with a multi-line docstring.
- [ ] I can predict whether a function changes the caller's list: mutating an argument changes it, rebinding the parameter doesn't.
- [ ] I can write and call functions with positional, keyword, default, `*args`, `**kwargs`, keyword-only and positional-only parameters, including `*` and `**` unpacking.
- [ ] I can read "missing required argument", "got multiple values", "unexpected keyword argument" and "takes N positional arguments" errors.
- [ ] I can explain why `def f(x=[])` shares one list between calls, and fix it with `None`.
- [ ] I can predict which variable a name refers to with LEGB, and I know when `global` and `nonlocal` are needed.
- [ ] I can write a closure, and I know why closures created in a loop all see the last value.
- [ ] I can pass a function as a value, write a lambda, use `key=` with `sorted`, `max` and `min`, and use `partial`.
- [ ] I can add type hints, including `list[int]`, `dict[str, float]`, `X | None` and `Callable`, and I know mypy checks them and Python doesn't.

**Next:** [Day 3a: Classes, decorators and modules](day-03a-classes-modules.md)
