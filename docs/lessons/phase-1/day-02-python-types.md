# Day 2: Python's Core Types, Through Kotlin Eyes

**Phase 1 · Week 1 · ~90 minutes**

## Objectives

- [ ] Use Python's numbers, strings, booleans and `None` and know how they differ from Kotlin's.
- [ ] Choose correctly between `list`, `tuple`, `dict` and `set`.
- [ ] Explain why `b = a` doesn't copy a list, and how to copy one when you need to.
- [ ] Write list and dict comprehensions, the Python equivalent of `map` and `filter`.
- [ ] Use slicing to grab parts of a sequence, a skill you'll use on every NumPy array.

**Prerequisites.** [Day 1](day-01-setup.md): a working venv and Jupyter.

---

## The big idea

Kotlin makes you declare what a variable *is*. Python lets a name point at anything, and the **value** carries its type. Think of Python variables as **sticky-note labels** you stick on objects, not boxes that hold values. Most surprises for Kotlin developers come from forgetting this.

```python
x = 42          # x labels an int
x = "hello"     # now the same label is on a str; no error
type(x)         # <class 'str'>
```

This is called **dynamic typing**. Python is still *strongly* typed: `"3" + 4` is an error, not `"34"`.

## Scalars

| Python | Kotlin | Notes |
|---|---|---|
| `int` | `Int`, `Long`, `BigInteger` | Unlimited size. `2 ** 100` just works. |
| `float` | `Double` | 64-bit. There's no 32-bit float in plain Python (NumPy has one). |
| `bool` (`True`, `False`) | `Boolean` | Capitalized. Logic uses words: `and`, `or`, `not`. |
| `str` | `String` | Immutable. Single or double quotes are the same. |
| `None` | `null` | The "no value" object. Test with `x is None`. |

```python
7 / 2        # 3.5   true division always gives a float
7 // 2       # 3     floor division, like Kotlin's Int / Int
7 % 2        # 1
2 ** 3       # 8     power
0.1 + 0.2    # 0.30000000000000004  same IEEE-754 floats as Kotlin's Double
```

!!! warning "Common mistake"
    In Kotlin, `7 / 2` is `3`. In Python it's `3.5`. Use `//` when you want integer division. This shows up when computing array indices or batch counts.

**Strings** work as you'd expect, with f-strings for templates:

```python
model = "MobileNetV3"
acc = 0.7523
print(f"{model} top-1 accuracy: {acc:.1%}")   # MobileNetV3 top-1 accuracy: 75.2%
"mobile" in "MobileNet".lower()               # True
"a,b,c".split(",")                            # ['a', 'b', 'c']
```

**Truthiness.** Any value can be used in an `if`. Empty things and zero are "falsy": `0`, `0.0`, `""`, `[]`, `{}`, `None`. Everything else is "truthy".

```python
items = []
if not items:
    print("empty")      # prints
```

## Collections

Four built-ins cover almost everything.

| Python | Literal | Kotlin analog | Mutable? | Use it for |
|---|---|---|---|---|
| `list` | `[1, 2, 3]` | `MutableList` | yes | Ordered sequence you'll add to |
| `tuple` | `(1, 2, 3)` | `Pair`/`Triple`, data class | no | Fixed bundles, e.g. an image shape `(224, 224, 3)` |
| `dict` | `{"lr": 0.01}` | `MutableMap` | yes | Key → value, e.g. configs, label maps |
| `set` | `{1, 2, 3}` | `MutableSet` | yes | Uniqueness, fast membership |

```python
layers = ["conv", "relu", "pool"]
layers.append("dense")
layers[0]          # 'conv'
layers[-1]         # 'dense'   negative index counts from the end
len(layers)        # 4

shape = (224, 224, 3)
h, w, c = shape    # unpacking, like Kotlin's destructuring

config = {"lr": 0.001, "epochs": 10}
config["batch_size"] = 32
config.get("dropout", 0.0)   # 0.0  default if missing, like getOrDefault

labels = {"cat", "dog", "cat"}
labels             # {'cat', 'dog'}
```

!!! android "Android bridge"
    A tuple is how Python returns "multiple values", the way you'd return a `Pair` or a small data class. A `dict` is everywhere in ML: model configs, JSON from an API, and the mapping from a model's output index to a human label (`{0: "cat", 1: "dog"}`), just like the `labels.txt` you load next to a `.tflite` model.

## Names and objects: the one rule to remember

Assignment never copies. It attaches another label to the same object. Watch it happen:

<figure class="diagram" markdown>
![Animation: a = [1, 2] creates a list labelled a; b = a attaches a second label b to the same list; b.append(3) changes the single list, so a sees [1, 2, 3] too](../../assets/img/phase-1/names-and-objects.svg)
<figcaption>Animated: <code>b = a</code> adds a second label to the same list. There is still only one list.</figcaption>
</figure>

```python
a = [1, 2]
b = a
b.append(3)
print(a)        # [1, 2, 3]
a is b          # True   same object

c = a.copy()    # or list(a), or a[:]
c.append(4)
print(a)        # [1, 2, 3]  unaffected
```

It's the same as Kotlin with a `MutableList`: `val b = a` shares the reference. The difference is that Python has no `val`, so nothing stops you mutating shared data by accident. This rule carries straight over to NumPy arrays and PyTorch tensors, where an accidental shared view can silently corrupt training data.

## Slicing

`seq[start:stop:step]` takes a sub-sequence. `stop` is excluded, like Kotlin's `until`.

```python
nums = [0, 10, 20, 30, 40, 50]
nums[1:4]     # [10, 20, 30]
nums[:3]      # [0, 10, 20]      first three
nums[-2:]     # [40, 50]         last two
nums[::2]     # [0, 20, 40]      every second item
nums[::-1]    # [50, 40, 30, 20, 10, 0]   reversed
```

!!! tip "Why this matters for ML"
    Splitting data into training and test sets is a slice: `train, test = data[:800], data[800:]`. Cropping an image is a 2-D slice: `img[0:224, 0:224]`. You'll write slices every day.

## Comprehensions: Python's `map` and `filter`

Kotlin:

```kotlin
val squares = (1..10).filter { it % 2 == 0 }.map { it * it }
```

Python:

```python
squares = [n * n for n in range(1, 11) if n % 2 == 0]
# [4, 16, 36, 64, 100]
```

Read it left to right as "**n squared, for each n in 1..10, if n is even**". Dicts and sets have them too:

```python
words = ["tensor", "graph", "model"]
lengths = {w: len(w) for w in words}      # {'tensor': 6, 'graph': 5, 'model': 5}
first_letters = {w[0] for w in words}     # {'t', 'g', 'm'}
```

Useful built-ins that pair with them:

```python
sum([1, 2, 3])                       # 6
max(lengths, key=lengths.get)        # 'tensor'
sorted(words, key=len)               # ['graph', 'model', 'tensor']
list(zip(["a", "b"], [1, 2]))        # [('a', 1), ('b', 2)]
for i, w in enumerate(words):        # like forEachIndexed
    print(i, w)
```

## Control flow, quickly

```python
score = 0.83
if score > 0.9:
    grade = "great"
elif score > 0.7:
    grade = "good"
else:
    grade = "needs work"

while score < 1.0:
    score += 0.05
```

Python 3.10+ also has `match`, similar to Kotlin's `when`:

```python
match device:
    case "gpu":
        delegate = "GpuDelegate"
    case "npu" | "dsp":
        delegate = "VendorDelegate"
    case _:
        delegate = "CPU"
```

## Common mistakes

!!! warning "Mutable default arguments"
    You'll meet this tomorrow with functions, but note it now: `def f(items=[])` shares **one** list across every call. Use `items=None` and create the list inside.

!!! warning "Modifying a list while looping over it"
    `for x in items: items.remove(x)` skips elements. Build a new list with a comprehension instead.

---

## Exercises

1. Given `temps = [22.5, 19.0, 25.3, 30.1, 18.7]`, use one comprehension to produce the temperatures above 20, converted to Fahrenheit (`f = c * 9/5 + 32`).
2. Count how often each word appears in `"the model saw the cat and the dog"` using a `dict`. Then do it again with `collections.Counter`.
3. Predict, then check: `a = [[0] * 3] * 3; a[0][0] = 1; print(a)`. Explain the result with the sticky-note picture.
4. Given `pixels = list(range(100))`, use slicing to get the first 80 items as "train" and the last 20 as "test". Then take every 10th item.

??? success "Solutions"
    ```python
    # 1
    hot_f = [c * 9 / 5 + 32 for c in temps if c > 20]

    # 2
    counts = {}
    for w in "the model saw the cat and the dog".split():
        counts[w] = counts.get(w, 0) + 1

    from collections import Counter
    Counter("the model saw the cat and the dog".split())

    # 3 prints [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    # [[0] * 3] * 3 puts three labels on ONE inner list.
    # Fix: a = [[0] * 3 for _ in range(3)]

    # 4
    train, test = pixels[:80], pixels[80:]
    every_tenth = pixels[::10]
    ```

## Resources

- [Python tutorial, §3 An Informal Introduction](https://docs.python.org/3/tutorial/introduction.html) and [§5 Data Structures](https://docs.python.org/3/tutorial/datastructures.html)
- [Ned Batchelder: Facts and myths about Python names and values](https://nedbatchelder.com/text/names.html): the best explanation of the sticky-note model, as an article and a talk.
- [Python Tutor](https://pythontutor.com/): paste code and watch the names-and-objects diagram draw itself step by step.
- [Kaggle Learn Python: Lists and Dictionaries lessons](https://www.kaggle.com/learn/python)
- [Real Python: List comprehensions](https://realpython.com/list-comprehension-python/)

## Checklist

- [ ] I know the difference between `/` and `//`.
- [ ] I can pick list vs tuple vs dict vs set for a situation.
- [ ] I can explain why `b = a` doesn't copy, and how to copy.
- [ ] I can slice the first, last and every n-th items.
- [ ] I rewrote a Kotlin `filter { }.map { }` chain as a comprehension.
- [ ] I tried exercise 3 in Python Tutor and saw the shared list.

**Next:** Day 3: Functions, classes and modules (coming soon)
