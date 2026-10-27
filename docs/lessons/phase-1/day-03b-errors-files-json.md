# Day 3b: Errors and Exceptions

**Phase 1 · Week 1 · ~140 minutes**

Days 2 to 3a used `try:` / `except` on many "goes wrong" examples without explaining it. Today explains it properly: how to read what Python tells you when something fails, and how to handle, raise and design errors yourself. [Day 3c](day-03c-files-json.md) then puts all of this to work on files, JSON, logging and tests.

## Objectives

- [ ] Read a multi-frame traceback from the bottom up and find the line in *your* code that caused it.
- [ ] Recognise the common built-in exceptions (`TypeError`, `ValueError`, `KeyError`, `IndexError`, `AttributeError`, `NameError`, `ZeroDivisionError`, `FileNotFoundError`, `ImportError`) from their messages, and tell a `SyntaxError` from a runtime error.
- [ ] Use the exception hierarchy to know what an `except` clause will catch.
- [ ] Handle exceptions with `try`, `except`, `else` and `finally`, catching specific types in the right order.
- [ ] Explain why a bare `except:` (or a careless `except Exception:`) is a bug, and when catching broadly is acceptable.
- [ ] Raise exceptions with `raise`, re-raise with a bare `raise`, chain with `raise ... from ...`, and write custom exception classes.
- [ ] Compare Python's error handling with Kotlin's (unchecked exceptions, `runCatching`, `toIntOrNull`) and choose between the EAFP and LBYL styles.
- [ ] Use `assert` for internal checks, and explain why it disappears under `python -O`.

**Prerequisites.**

- [Day 1](day-01-setup.md): running a script from a terminal, and your first look at a traceback.
- [Day 2](day-02-python-types.md): `int()` / `float()` conversions, f-strings and format specs, `None` and truthiness.
- [Day 2b](day-02b-collections.md): lists and dicts (`[]` versus `get`), comprehensions.
- [Day 3](day-03-functions-classes.md): functions, closures and scope, and the `TypeError` you get from a wrong call (`missing 1 required positional argument`).
- [Day 3a](day-03a-classes-modules.md): classes, inheritance and `super().__init__()`, `__mro__`, `__getitem__`, `forward` and `NotImplementedError`, decorators with arguments and `functools.wraps`, modules, and the "naming your file after a library" mistake.

!!! info "How to read the code on this page"
    Every Python block has a **▶ Run** button, and blocks share their variables from top to bottom like notebook cells, so run them in order. A bare expression shows its value when run, as the `# comment` beside it says.

    Two other kinds of block have no Run button:

    - Blocks starting with `>>>` are transcripts from the Python 3.14 REPL that **fail on purpose**, so you can see the full, real traceback. Today the traceback is the subject, so you'll see many of them. Type them into a `python3` prompt to see the error yourself. `<python-input-1>` in a traceback or a warning is the name the REPL gives to what you typed, numbered from 0 for each statement you enter; in a file, you'd see the file's path instead.
    - Blocks starting with `$` are terminal sessions: a command and what it printed.

    Blocks labelled `# Run this on your computer (or in Colab), not in the browser.` are files of a small multi-file project; they need a real terminal. Their first two comment lines (`# Run this...` and `# filename.py`) are page labels: save only the code below them, so the line numbers in the tracebacks match.

    Runnable blocks that show an error catch it with `try` / `except` and print the exception's type and message, so the page keeps running. Paths in tracebacks are shortened: your own files to `/home/you/my_project/...`, and Python's own files to `.../lib/python3.14/...`.

---

## The big idea

Your code is about to meet the outside world. Configs with typos, CSV files with a missing column, a JSON file cut off halfway through a download, a dataset folder someone renamed. Day 3 was about code that *you* control. Today is about what happens when something goes wrong, for reasons that are bugs in your code and for reasons that aren't, and how to tell the two apart.

Python reports every runtime problem the same way: it creates an **exception** object and throws it up through the functions that are running. If some function on the way says "I know how to deal with this kind of problem", the exception stops there. If nothing does, the program stops and prints a **traceback**: the list of calls the exception climbed through.

**Analogy.** An exception is a fire alarm in a tall office building. It goes off on one floor (the function where the problem happened). If that floor has the right extinguisher for that kind of fire (an `except` for that exception type), the fire is put out there and work carries on. If not, the alarm moves up a floor, and up again. If it reaches the roof with nobody handling it, the building is evacuated (the program exits) and the fire brigade writes a report of every floor the fire passed through (the traceback).

Kotlin works the same way: `throw`, `try` / `catch` / `finally`, and exceptions climbing the call stack. What's different in Python is the *style*. Python code uses exceptions for ordinary situations ("this key isn't there", "this string isn't a number") far more often than Kotlin does, and Python's compiler checks only the grammar of your code, not names or types, so a typo in a variable name is found only when that line runs. So reading and handling exceptions is a skill you'll use every day.

One import is used on this page (for an exception class from the `json` module), so run this block first:

```python
import json
```

---

## What an exception is

**What it is.** An **exception** is an object that describes a problem found while the program runs. Its **type** (a class, such as `ValueError`) says what kind of problem it is, and its **message** says what exactly went wrong. **Raising** an exception means creating that object and throwing it, which stops the current line. **Handling** (or **catching**) it means a piece of code further up says "I'll deal with this", and the program carries on from there.

**Why it exists.** A function that can't do its job needs a way to say so that the caller can't ignore by accident. Returning a special value (`-1`, `None`) is easy to forget to check; an exception stops everything until someone deals with it. In ML code you'll meet exceptions every time a file is missing, a shape doesn't match, a config value has the wrong type, or a downloaded file is corrupt.

**Kotlin comparison.**

```kotlin
val lr = "0.0o1".toDouble()        // throws NumberFormatException
```

```python
try:
    lr = float("0.0o1")
except ValueError as e:
    print(type(e).__name__, "-", e)
# ValueError - could not convert string to float: '0.0o1'
```

Kotlin throws `NumberFormatException`; Python throws `ValueError`. The machinery is the same: an object is thrown, the rest of the line doesn't run, and the nearest matching handler catches it. Here `except ValueError as e` catches it and binds the exception object to the name `e`. `type(e).__name__` is the class's name as a string, and `print(e)` shows the message. The next sections take this apart piece by piece.

An exception is an ordinary object. You can create one without raising it, look at it, and raise it later:

```python
err = ValueError("lr must be positive")
type(err)                  # <class 'ValueError'>
err.args                   # ('lr must be positive',)
str(err)                   # 'lr must be positive'
isinstance(err, Exception) # True
```

`args` is the tuple of arguments the exception was created with, and `str(err)` turns it into the message.

**How it works underneath: unwinding the stack.** Each function call that's running has a **frame**: a record of its local variables and of the line it's on. The frames form a **call stack**: `train.py` called `load_config`, which called `parse_lr`, which called `float`. When `float` raises, Python looks at the innermost frame for a `try` whose `except` matches the exception's type. If there isn't one, Python stops running that frame and looks at the frame that called it. This is called **unwinding the stack**. The abandoned frame isn't freed yet: the exception's traceback keeps a reference to it, which is how a debugger can show its variables after the crash. Unwinding continues until a handler matches, or until the outermost frame is left, at which point Python prints the traceback and exits with status code 1 (the "Exit codes" section shows why that number matters).

**How it goes wrong.** An exception nobody handles stops the program at that line, and nothing after it runs, as the next section shows on a real script. The fix is not "catch everything": it's to read the traceback, find the cause, and decide whether the problem should be fixed in the code (a bug) or handled (a situation the program should expect, like a missing file).

## Reading a traceback

**What it is.** A **traceback** is the report Python prints when an exception escapes: one entry per frame the exception climbed through, then the exception itself. **Why it exists:** it tells you *what* went wrong and *how the program got there*, which is usually enough to find the bug without a debugger.

Here is a small project with two files. `config.py` parses `key=value` lines:

```python
# Run this on your computer (or in Colab), not in the browser.
# config.py
def parse_lr(text):
    return float(text)


def load_config(lines):
    config = {}
    for line in lines:
        key, value = line.split("=")
        if key == "lr":
            config["lr"] = parse_lr(value)
    return config
```

`train.py` uses it, with a typo in the data (the letter `o` instead of a zero):

```python
# Run this on your computer (or in Colab), not in the browser.
# train.py
from config import load_config

lines = ["epochs=10", "lr=0.0o1"]
config = load_config(lines)
print("training with", config)
```

Running it prints this, and `training with` never appears:

```text
$ python train.py
Traceback (most recent call last):
  File "/home/you/my_project/train.py", line 4, in <module>
    config = load_config(lines)
  File "/home/you/my_project/config.py", line 10, in load_config
    config["lr"] = parse_lr(value)
                   ~~~~~~~~^^^^^^^
  File "/home/you/my_project/config.py", line 2, in parse_lr
    return float(text)
ValueError: could not convert string to float: '0.0o1'
```

**Read it bottom-up, in three steps:**

1. **The last line** is the exception: its type (`ValueError`) and its message (`could not convert string to float: '0.0o1'`). Often this alone tells you the problem. The message even quotes the bad value.
2. **The frame directly above it** is where the exception was raised: `config.py`, line 2, in the function `parse_lr`, on the line `return float(text)`.
3. **The frames higher up** show how the program got there. `parse_lr` was called by `load_config` (`config.py`, line 10), which was called from `train.py`, line 4. `<module>` means "the top level of a file, outside any function".

Each frame has the same two-line shape: `File "<path>", line <number>, in <function>`, then the source line itself. The marks under a line point at the part that failed: `~` marks what is being called (`parse_lr`) and `^` the call itself (`(value)`). Python leaves the marks out when they'd underline the whole line, which is why `return float(text)` and `config = load_config(lines)` have none.

"Most recent call last" means the order runs from the outermost call at the top to the innermost at the bottom. Kotlin and Java stack traces run the other way round, with the throwing line first and `at ...` lines below it. That's the main adjustment for an Android developer: in Python, **the answer is at the bottom.**

<figure class="diagram" markdown>
![Animation: on the left, a call stack of four boxes, train.py line 4, config.py line 10 in load_config, config.py line 2 in parse_lr, and float('0.0o1') which raises ValueError. A red dot climbs from the bottom box to the top, showing the exception unwinding through every frame. On the right, the printed traceback, with three highlights appearing in order: the last line with the exception type and message, the parse_lr frame above it, and the two outer frames showing how the program got there](../../assets/img/phase-1/day03b-traceback.svg)
<figcaption>Animated: the exception climbs out of every frame; the traceback lists those frames outermost first, so you read it from the bottom.</figcaption>
</figure>

**Where's the bug?** Not in `float`, which did its job, and probably not in `parse_lr`. The bad *value* came from the data in `train.py`. The traceback shows where the program noticed the problem, which isn't always where the problem started. Follow the bad value (`'0.0o1'`) up the frames until you find where it came from.

**When the traceback goes through library code.** Real tracebacks often pass through code you didn't write: numpy, pandas, PyTorch, or the standard library. Here a script loads a JSON file that was cut off halfway (the `logger.exception` line that printed it is explained in [Day 3c](day-03c-files-json.md)):

```text
$ python summarize.py
...
2026-10-26 19:08:37,656 ERROR __main__: skipping b.json
Traceback (most recent call last):
  File "/home/you/my_project/summarize.py", line 11, in main
    record = json.loads(path.read_text(encoding="utf-8"))
  File ".../lib/python3.14/json/__init__.py", line 352, in loads
    return _default_decoder.decode(s)
           ~~~~~~~~~~~~~~~~~~~~~~~^^^
  File ".../lib/python3.14/json/decoder.py", line 345, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File ".../lib/python3.14/json/decoder.py", line 361, in raw_decode
    obj, end = self.scan_once(s, idx)
               ~~~~~~~~~~~~~~^^^^^^^^
json.decoder.JSONDecodeError: Expecting ',' delimiter: line 2 column 1 (char 30)
```

Read the last line first, as always. "line 2 column 1" counts lines and columns from 1, the way an editor does; "char 30" counts characters from 0, the way Python indexes a string. Both point just past the end of the file's text, where JSON still expected a `,` or a `}`. The class is printed as `json.decoder.JSONDecodeError` because tracebacks show the module that *defines* a class; your code can name it `json.JSONDecodeError`, because the `json` package re-exports it. Then go up and **skip every frame whose path is inside the library** (`lib/python3.14/...`, or `site-packages/...` for things you installed with pip). The lowest frame in *your* files is the line to look at: `summarize.py`, line 11. Bugs in numpy or the standard library are rare; a bad argument from your code is common.

Two more things differ from the first traceback. The `...` line at the top stands for the script's earlier output, and the `ERROR ... skipping b.json` line was printed by the logging call that also printed this traceback. And it starts at `main`, with no `<module>` frame above it, because this traceback was *caught and logged*: a caught exception's traceback runs only from the frame that caught it down to the one that raised it.

**A preview: NumPy shape errors.** From Day 4 on, the error you'll meet most is NumPy refusing to combine arrays whose shapes don't fit. Read it bottom-up the same way:

```pycon
>>> import numpy as np
>>> pixels = np.ones((2, 3))
>>> weights = np.ones(2)
>>> pixels * weights
Traceback (most recent call last):
  File "<python-input-3>", line 1, in <module>
    pixels * weights
    ~~~~~~~^~~~~~~~~
ValueError: operands could not be broadcast together with shapes (2,3) (2,)
```

The last line names the two shapes, `(2,3)` and `(2,)`, and the frame above it marks the `*` that failed. That's all you need for now: Day 9 explains which shapes fit together. PyTorch reports the same kind of mistake as a `RuntimeError`, for example `mat1 and mat2 shapes cannot be multiplied`.

### Syntax errors happen before anything runs

A **`SyntaxError`** is different from every other exception on this page. Python first **compiles** a whole file to bytecode (Day 1), and if the text isn't valid Python, compiling fails and *no line of the file runs*, not even the ones before the mistake. The report points at the line and column instead of listing frames. Here `report.py` is two lines, `print("starting")` and `print("done"`, and `starting` is never printed:

```text
$ python report.py
  File "/home/you/my_project/report.py", line 2
    print("done"
         ^
SyntaxError: '(' was never closed
```

In the REPL, the same mistakes look like this:

```pycon
>>> if True
  File "<python-input-0>", line 1
    if True
           ^
SyntaxError: expected ':'
```

```pycon
>>> def f():
... return 1
...
  File "<python-input-0>", line 2
    return 1
    ^^^^^^
IndentationError: expected an indented block after function definition on line 1
```

`IndentationError` is a kind of `SyntaxError` about indentation, the thing that replaces Kotlin's `{}`. (An unclosed bracket in the REPL gives no error: the REPL waits with `...` for you to close it.) This is the one class of mistake that Python, like the Kotlin compiler, catches before the program starts. Your editor underlines these as you type. Everything else in this lesson is found only when the line actually runs.

"No line runs" is true for the file you start. A syntax error in a module you *import* is found when the `import` line runs, so the lines before it have already run, and a normal frame for the `import` line sits above the syntax error. Here `main.py` prints `main starting`, then imports `helpers.py`, whose line 4 is broken:

```text
$ python main.py
main starting
Traceback (most recent call last):
  File "/home/you/my_project/main.py", line 2, in <module>
    import helpers
  File "/home/you/my_project/helpers.py", line 4
    print("helpers loaded"
         ^
SyntaxError: '(' was never closed
```

## The common built-in exceptions

These are the exceptions you'll see most, each with the real message Python 3.14 prints. Learn to recognise them by the last line; each message is written to tell you the fix.

**`TypeError`: an operation got a value of the wrong type.** Python is strongly typed (Day 2): it won't turn an `int` into a `str` for you.

```pycon
>>> epochs = 5
>>> "epochs: " + epochs
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    "epochs: " + epochs
    ~~~~~~~~~~~^~~~~~~~
TypeError: can only concatenate str (not "int") to str
```

Fix: `f"epochs: {epochs}"` or `"epochs: " + str(epochs)`. Calling a function with the wrong number or names of arguments is also a `TypeError` (Day 3 showed `missing 1 required positional argument`), and so is `len(5)` (`object of type 'int' has no len()`).

**`ValueError`: the type is right, the value isn't.** `int` accepts strings, but not this string:

```pycon
>>> int("3.5")
Traceback (most recent call last):
  File "<python-input-0>", line 1, in <module>
    int("3.5")
    ~~~^^^^^^^
ValueError: invalid literal for int() with base 10: '3.5'
```

Fix: `int(float("3.5"))` if you really want 3, or check the data. `math.sqrt(-1)` is a `ValueError` too (`expected a nonnegative input, got -1.0`).

**`KeyError`: a dict doesn't have that key.** The message is only the key, in quotes, which surprises people the first time. The quotes are there because a `KeyError` shows its argument with `repr()`, so you can see whether the key was the string `'1'` or the number `1`:

```pycon
>>> config = {"lr": 0.01}
>>> config["learning_rate"]
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    config["learning_rate"]
    ~~~~~~^^^^^^^^^^^^^^^^^
KeyError: 'learning_rate'
```

Fix: use the right key, or `config.get("learning_rate", default)` when a missing key is normal (Day 2b).

**`IndexError`: a sequence position doesn't exist.**

```pycon
>>> labels = ["cat", "dog"]
>>> labels[2]
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    labels[2]
    ~~~~~~^^^
IndexError: list index out of range
```

A 2-item list has indexes 0 and 1. Off-by-one errors and empty lists are the usual causes.

**`AttributeError`: that object has no attribute or method with that name.** The most common version mentions `'NoneType'`, which means the variable holds `None`, usually because a function returned nothing:

```pycon
>>> result = None
>>> result.shape
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    result.shape
AttributeError: 'NoneType' object has no attribute 'shape'
```

This is Python's version of Kotlin's `NullPointerException`, except that Kotlin's compiler stops most of them before they happen and Python's doesn't. Find where the `None` came from: a function without `return`, an in-place method like `list.sort()` whose result was assigned (Day 2b), or a lookup that found nothing.

**`NameError`: that name isn't defined.** Usually a typo, and Python suggests the fix:

```pycon
>>> learning_rate = 0.1
>>> print(learning_rat)
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    print(learning_rat)
          ^^^^^^^^^^^^
NameError: name 'learning_rat' is not defined. Did you mean: 'learning_rate'?
```

On this site, a `NameError` also happens when you run a block before the one that defines its variables. Its subclass `UnboundLocalError` is the scope mistake from Day 3: assigning to a name inside a function makes it local for the whole function.

**`ZeroDivisionError`: dividing by zero.** Integer and float division both raise it, unlike Kotlin's `Double` division, which quietly gives `Infinity` or `NaN`:

```pycon
>>> correct, total = 0, 0
>>> correct / total
Traceback (most recent call last):
  File "<python-input-1>", line 1, in <module>
    correct / total
    ~~~~~~~~^~~~~~~
ZeroDivisionError: division by zero
```

An empty batch or an empty class in an accuracy computation is the usual cause. (NumPy arrays behave differently: they warn and give `inf` or `nan`, which Day 4b covers.)

**`FileNotFoundError`: no file at that path.**

```pycon
>>> open("data/train.csv")
Traceback (most recent call last):
  File "<python-input-0>", line 1, in <module>
    open("data/train.csv")
    ~~~~^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'data/train.csv'
```

`[Errno 2]` is the operating system's error number for "no such file". A relative path like `data/train.csv` is looked up from the **current working directory**, the folder the program was started from, not the folder the script is in. That's the cause more often than a missing file: you ran `python scripts/train.py` from a different folder than you thought. Day 3c shows how to build paths with `pathlib` that don't depend on it.

**`ModuleNotFoundError` and `ImportError`: an import failed.** `ModuleNotFoundError` (a subclass of `ImportError`) means the module couldn't be found at all, usually because it isn't installed in the environment you're running (Day 1), or because of a typo:

```pycon
>>> import torchh
Traceback (most recent call last):
  File "<python-input-0>", line 1, in <module>
    import torchh
ModuleNotFoundError: No module named 'torchh'
```

A plain `ImportError` means the module was found but doesn't contain the name you asked for:

```pycon
>>> from json import load_file
Traceback (most recent call last):
  File "<python-input-0>", line 1, in <module>
    from json import load_file
ImportError: cannot import name 'load_file' from 'json' (.../lib/python3.14/json/__init__.py)
```

The path in brackets tells you *which* `json` was imported, which is how you spot a file of your own shadowing a library (Day 3a's "naming your file after a library" mistake).

### The exception hierarchy

Exception types are classes, and they inherit from each other. That matters because `except LookupError:` catches every subclass of `LookupError` too. Each class's `__mro__` (method resolution order, Day 3a) lists its parents:

```python
for cls in (KeyError, IndexError, FileNotFoundError, ZeroDivisionError,
            ValueError, json.JSONDecodeError, KeyboardInterrupt):
    parents = " -> ".join(c.__name__ for c in cls.__mro__[1:])
    print(f"{cls.__name__:18} -> {parents}")
# KeyError           -> LookupError -> Exception -> BaseException -> object
# IndexError         -> LookupError -> Exception -> BaseException -> object
# FileNotFoundError  -> OSError -> Exception -> BaseException -> object
# ZeroDivisionError  -> ArithmeticError -> Exception -> BaseException -> object
# ValueError         -> Exception -> BaseException -> object
# JSONDecodeError    -> ValueError -> Exception -> BaseException -> object
# KeyboardInterrupt  -> BaseException -> object
```

(`{cls.__name__:18}` pads the name to 18 characters so the arrows line up; Day 2 covered format specs.) Three families are worth remembering:

- **`LookupError`** is the parent of `KeyError` and `IndexError`: "you asked a container for something it doesn't have".
- **`OSError`** is the parent of every error the operating system reports: `FileNotFoundError`, `FileExistsError`, `PermissionError`, `IsADirectoryError`, and network errors such as `TimeoutError` and `ConnectionError`.
- **`Exception`** is the parent of every "normal" error. Above it sits **`BaseException`**, which also has `KeyboardInterrupt` (you pressed Ctrl+C) and `SystemExit` (the program called `sys.exit()`) as children. Those two aren't errors, they're requests to stop, and that's why they sit outside `Exception`.

`json.JSONDecodeError` being a `ValueError` is typical: library exceptions usually subclass the closest built-in one, so `except ValueError:` also catches bad JSON.

**Kotlin comparison.** The two trees have the same shape:

| Python | Kotlin / JVM | Meaning |
|---|---|---|
| `BaseException` | `Throwable` | the root of everything that can be raised |
| `Exception` | `Exception` | normal errors; what you catch |
| `KeyboardInterrupt`, `SystemExit` | `Error`, and coroutines' `CancellationException` | "stop now" signals you shouldn't swallow |
| `ValueError` | `IllegalArgumentException` (`NumberFormatException` is one) | right type, bad value |
| `LookupError`, `KeyError`, `IndexError` | `NoSuchElementException`, `IndexOutOfBoundsException` | asked a container for something it lacks |
| `RuntimeError` | `IllegalStateException`, `RuntimeException` | wrong state, or "something went wrong" |
| `OSError` and subclasses | `IOException` and subclasses | the operating system said no |

<figure class="diagram" markdown>
![A tree with BaseException at the top. Its children are KeyboardInterrupt and SystemExit, marked as stop signals, and Exception. Under Exception are LookupError with KeyError and IndexError, OSError with FileNotFoundError, PermissionError and ConnectionError, ArithmeticError with ZeroDivisionError, ValueError with JSONDecodeError, TypeError, and RuntimeError with RecursionError. A bracket shows that except Exception catches the whole Exception subtree but not the stop signals](../../assets/img/phase-1/day03b-hierarchy.svg)
<figcaption>Catching a class catches everything below it; except Exception covers every normal error and leaves the stop signals alone.</figcaption>
</figure>

This table is a summary of what you have seen, to come back to:

| Exception | Typical cause | First thing to check |
|---|---|---|
| `SyntaxError`, `IndentationError` | text isn't valid Python | the line and column shown; brackets, colons, indentation |
| `TypeError` | wrong type, or wrong arguments to a call | the types of the values on that line |
| `ValueError` | right type, unacceptable value | the value quoted in the message |
| `KeyError` | dict key missing | spelling of the key; use `get` if missing is normal |
| `IndexError` | position past the end | the length of the sequence; empty lists |
| `AttributeError` | no such attribute; often `'NoneType'` | where the `None` came from |
| `NameError` | name not defined | typos; running blocks out of order |
| `ZeroDivisionError` | division by zero | empty batches, empty classes |
| `RuntimeError` | wrong state; PyTorch shape and device mismatches | the message; tensor shapes and devices |
| `RecursionError` | a function called itself too many times | the missing stopping case |
| `FileNotFoundError` | path doesn't exist | the current working directory |
| `ModuleNotFoundError`, `ImportError` | module not installed, or name not in it | which environment is running; the path shown |

## Handling exceptions: `try` and `except`

**What it is.** A `try` statement runs a block of code, its **`try` block**. If an exception is raised inside it, Python checks the `except` clauses below, in order. The first one whose type matches (the exception is an instance of that class or a subclass) runs, and the program carries on after the whole `try` statement. If none matches, the exception carries on up the stack as if the `try` weren't there.

**Why it exists.** Some failures are expected and have a sensible response: use a default learning rate, skip a corrupt image, ask for a different file. `try` lets you write that response once, next to the code that can fail.

**Kotlin comparison.**

```kotlin
fun parseLr(text: String): Double =
    try {
        text.toDouble()
    } catch (e: NumberFormatException) {
        println("bad lr '$text', using default")
        0.001
    }
```

```python
def parse_lr(text):
    try:
        return float(text)
    except ValueError:
        print(f"bad lr {text!r}, using default")
        return 0.001

parse_lr("0.01")         # 0.01
parse_lr("0.0o1")
# bad lr '0.0o1', using default
# 0.001
```

What's the same: the shape, and the rule that only matching exception types are caught. What's different:

- `except` instead of `catch`, and the type comes first: `except ValueError as e:` versus `catch (e: NumberFormatException)`. The `as e` part is optional; leave it out when you don't need the object.
- `try` is a statement in Python, not an expression, so it can't produce a value the way Kotlin's `val x = try { ... } catch ...` does. Return from inside it, or assign to a name in each branch.
- `{text!r}` in the f-string formats the value with `repr()`, so a string shows with its quotes. That makes stray spaces visible in error messages.

**Several `except` clauses, and a tuple of types.** One `try` can have several handlers, and one handler can name several types in a tuple:

```python
def first_label(config):
    try:
        return config["labels"][0].upper()
    except (KeyError, IndexError) as e:
        return f"bad structure: {type(e).__name__} {e}"
    except AttributeError:
        return "labels must be strings"

first_label({"labels": ["cat", "dog"]})   # 'CAT'
first_label({"labels": []})               # 'bad structure: IndexError list index out of range'
first_label({"classes": ["cat"]})         # "bad structure: KeyError 'labels'"
first_label({"labels": [7]})              # 'labels must be strings'
```

The function returns a different answer for each kind of problem. A missing key and an empty list share one handler through the tuple, and `type(e).__name__` tells them apart in the message. `7` has no `upper` method, so that call raises `AttributeError`, which has its own handler. Day 3c reads this kind of structure from a JSON file, where all three problems happen for real.

**How it works underneath: matching is `isinstance`, top to bottom.** When an exception reaches a `try`, Python takes each `except` clause in order and asks "is this exception an instance of that class?", which includes subclasses. The first yes wins and the others are skipped. Code in the `try` block after the line that raised **never runs**: the exception jumped out of it. Variables assigned before the failing line keep their values.

```python
steps = []
try:
    steps.append("before")
    value = int("seven")
    steps.append("after")          # skipped: the line above raised
except ValueError:
    steps.append("handled")
steps.append("carried on")
steps                              # ['before', 'handled', 'carried on']
```

One detail with no Kotlin equivalent: the name bound by `as e` is **deleted** at the end of the `except` block (Python does this to free the traceback the exception holds). Copy it to another name if you need it later.

```python
try:
    int("seven")
except ValueError as e:
    saved = e
try:
    e
except NameError as missing:
    print("NameError:", missing)
# NameError: name 'e' is not defined
saved                              # ValueError("invalid literal for int() with base 10: 'seven'")
```

!!! warning "Common mistake: a general `except` before a specific one"
    Because the first match wins, put the most specific types first. Here every `KeyError` is a `LookupError`, so the second handler can never run:

    ```python
    def read_key(d, key):
        try:
            return d[key]
        except LookupError:
            return "lookup failed"
        except KeyError:                 # unreachable: KeyError is a LookupError
            return f"missing key {key!r}"

    read_key({}, "lr")                   # 'lookup failed'
    ```

    Python doesn't warn you. Swap the two clauses and the call returns `"missing key 'lr'"`. Kotlin's `catch` blocks follow the same first-match rule.

**A more realistic example: keep going past bad records.** Real data has a few broken rows. Skip them, count them, and keep the good ones:

```python
raw_ratings = ["4.5", "3.9", "N/A", "4.1", "", "5.0"]
ratings, bad = [], []
for text in raw_ratings:
    try:
        ratings.append(float(text))
    except ValueError:
        bad.append(text)
ratings                  # [4.5, 3.9, 4.1, 5.0]
bad                      # ['N/A', '']
```

Note what the `try` contains: only the one line that can fail. The next section shows why that matters.

## `else` and `finally`

**What they are.** A `try` statement can have two more clauses:

- **`else:`** runs only if the `try` block finished **without** an exception. It goes after the `except` clauses.
- **`finally:`** runs **always**, last: after the `try` succeeded, after an `except` handled an exception, after an exception that nothing here caught, and even after a `return` inside the `try`.

**Why they exist.** `else` lets you keep the `try` block small, so that it only covers the line you expect to fail. Code in `else` that raises the same exception type won't be silently caught by the wrong handler. `finally` is for clean-up that must happen no matter what: closing a file, releasing a lock, stopping a timer, deleting a temporary file.

**Kotlin comparison.** Kotlin has `finally`, and it means exactly the same. Kotlin has no `else` on `try`; you'd put that code at the end of the `try` block.

```python
def parse_and_report(text):
    try:
        lr = float(text)
    except ValueError:
        print("  except: bad value, using 0.001")
        lr = 0.001
    else:
        print("  else: parsed", lr)
    finally:
        print("  finally: done with", repr(text))
    return lr

print("run 1"); parse_and_report("0.01")
print("run 2"); parse_and_report("abc")
# run 1
#   else: parsed 0.01
#   finally: done with '0.01'
# 0.01
# run 2
#   except: bad value, using 0.001
#   finally: done with 'abc'
# 0.001
```

The `0.01` and `0.001` lines are the return values, echoed because each call is a bare expression.

**How it goes wrong: a `try` block that's too wide.** Here a config gives a learning rate and a scale factor for it, both as text. The first version puts both conversions in the `try`:

```python
def scaled_lr_bad(config):
    try:
        lr = float(config["lr"])
        lr = lr * float(config["scale"])     # this line can raise ValueError too
    except ValueError:
        print("bad lr, using 0.001")
        lr = 0.001
    return lr

scaled_lr_bad({"lr": "0.01", "scale": "x2"})
# bad lr, using 0.001
# 0.001
```

The learning rate was fine; the *scale* was broken. The handler can't tell the two `ValueError`s apart, so it prints a wrong message and quietly drops the scale. With `else`, the `try` covers only the line the handler is about, and a bad scale raises its own error:

```python
def scaled_lr(config):
    try:
        lr = float(config["lr"])
    except ValueError:
        print("bad lr, using 0.001")
        return 0.001
    else:
        return lr * float(config["scale"])   # not covered by the except above

scaled_lr({"lr": "abc", "scale": "2"})       # prints "bad lr, using 0.001", returns 0.001
scaled_lr({"lr": "0.01", "scale": "2"})      # 0.02
try:
    scaled_lr({"lr": "0.01", "scale": "x2"})
except ValueError as e:
    print("ValueError:", e)
# bad lr, using 0.001
# 0.001
# 0.02
# ValueError: could not convert string to float: 'x2'
```

The traceback for the last call would point at the `scale` line, which is where the problem is.

<figure class="diagram" markdown>
![Animation: a flowchart with a try box at the top, else on the left, except ValueError on the right, finally below both, then the next statement. In run 1, with text "0.01", a green dot goes from try through else to finally and on. In run 2, with text "abc", a red dot goes from try through except to finally and on](../../assets/img/phase-1/day03b-try-flow.svg)
<figcaption>Animated: with no exception, try leads to else; with one, to the matching except. Both paths go through finally.</figcaption>
</figure>

**How it works underneath: `finally` runs even on the way out.** If the `try` block returns, or an exception that no `except` matches is passing through, Python pauses that exit, runs `finally`, then carries on leaving:

```python
def risky(x):
    try:
        return 10 / x
    finally:
        print("finally runs before the function returns or raises")

risky(4)
# finally runs before the function returns or raises
# 2.5
try:
    risky(0)
except ZeroDivisionError as e:
    print("caught by the caller:", e)
# finally runs before the function returns or raises
# caught by the caller: division by zero
```

`risky` has no `except` at all. A `try` with only `finally` is common: "I don't handle this error, but I clean up on the way out".

!!! warning "Common mistake: `return` inside `finally`"
    A `return` in `finally` replaces whatever was happening, including an exception that was on its way up. The error disappears:

    ```pycon
    >>> def f():
    ...     try:
    ...         return 1/0
    ...     finally:
    ...         return "finally wins"
    ...
    <python-input-0>:5: SyntaxWarning: 'return' in a 'finally' block
    >>> f()
    'finally wins'
    ```

    The `ZeroDivisionError` is gone without a trace. Since Python 3.14 you get a `SyntaxWarning` for this (and for `break` or `continue` in `finally`); older versions say nothing. Never return from `finally`; use it only for clean-up.

## Catching too much: bare `except` and `except Exception`

**What it is.** An `except:` with no type is a **bare except**. It catches everything, including `KeyboardInterrupt` and `SystemExit`. `except Exception:` catches every normal error, but lets those two through.

**Kotlin comparison.** A bare `except:` is Kotlin's `catch (e: Throwable)`, which also catches things you should never catch there (`OutOfMemoryError`, or a coroutine's `CancellationException`). `except Exception:` is closer to `catch (e: Exception)`.

**Why it's a bug.** You write a `try` because you expect one particular failure. A bare or broad `except` also catches every failure you *didn't* expect, including your own typos, and turns them into the "expected" response. The program carries on with wrong results, and the traceback that would have shown you the bug is thrown away.

```python
def mean_rating(ratings):
    try:
        return sum(ratings) / len(ratinsg)      # typo: ratinsg
    except:                                     # meant for empty lists
        return 0.0

mean_rating([4.5, 3.9, 4.1])                    # 0.0
```

The answer is `0.0` for *every* input, because the typo raises `NameError` on every call and the bare `except` hides it. Catch the one type you meant, and the typo shows up the first time you run the code:

```python
def mean_rating(ratings):
    try:
        return sum(ratings) / len(ratinsg)
    except ZeroDivisionError:                  # only the empty-list case
        return 0.0

try:
    mean_rating([4.5, 3.9, 4.1])
except NameError as e:
    print("NameError:", e)
# NameError: name 'ratinsg' is not defined
```

(Uncaught, the traceback would also add `Did you mean: 'ratings'?`. Python adds that hint when it prints a traceback, so it isn't part of the message `e` holds.)

With the typo fixed, `ZeroDivisionError` is the only thing caught, and only an empty list gives `0.0`. (Better still for an empty list: `if not ratings: return 0.0`, which says what you mean without an exception.)

A bare `except` also catches **Ctrl+C**. Put one inside a long training loop and pressing Ctrl+C only moves on to the next iteration: you can't stop the program.

**When broad catching is right.** At the **top of a loop over many independent items** (files, images, API calls), one bad item shouldn't kill a two-hour job. There, `except Exception as e:` is acceptable *if* you log the error with its traceback, count it, and carry on. [Day 3c](day-03c-files-json.md) shows `logger.exception` for exactly this. Never write `except: pass`, which hides everything and records nothing.

!!! warning "Common mistake: `except Exception: pass` around a whole function"
    ```python
    try:
        train_model()
    except Exception:
        pass
    ```

    Press Run: nothing happens. `train_model` doesn't even exist on this page, so the first line raises `NameError`, and `except Exception: pass` swallows it. That's the whole problem in one block. In a real script, if training fails on step 1, this prints nothing, and you find out an hour later that no model file was saved. If you don't know what to do with an error, don't catch it: the traceback is more useful than silence.

## Exit codes, `sys.exit` and Ctrl+C

**What it is.** When a Python program ends, it hands the operating system a number, its **exit code** (or exit status). `0` means success; anything else means failure. An uncaught exception gives `1`. `sys.exit(...)` ends the program on purpose with a code you choose.

**Why it matters.** Other programs read that number, not your output. A shell script with `set -e`, a CI job, a job scheduler on a GPU cluster, or a `train.py && evaluate.py` command line all treat non-zero as "this step failed, stop here". A script that prints an error and then ends normally exits with `0`, and the next step runs on missing results.

You see the code with `echo $?` right after the command (macOS and Linux), or `$LASTEXITCODE` in Windows PowerShell:

```text
$ cat fail.py
raise ValueError("boom")

$ python fail.py
Traceback (most recent call last):
  File "/home/you/my_project/fail.py", line 1, in <module>
    raise ValueError("boom")
ValueError: boom
$ echo $?
1
```

**Ending on purpose: `sys.exit`.** For a problem you've already diagnosed, such as a bad command-line argument or a missing config field, a full traceback is noise for the user. `sys.exit("message")` prints the message to stderr and exits with code `1`; `sys.exit(0)` (or reaching the end of the file) exits with `0`; `sys.exit(2)` exits with `2`.

```text
$ cat check_config.py
import sys
sys.exit("bad config: lr is missing")

$ python check_config.py
bad config: lr is missing
$ echo $?
1
```

**How it works underneath.** `sys.exit` doesn't stop Python on the spot: it raises `SystemExit`, which climbs the stack like any exception (so `finally` blocks still run). When it reaches the top, Python exits with its `code` instead of printing a traceback. That's why it's outside `Exception` in the hierarchy: an `except Exception:` must not stop a program from exiting.

```python
import sys
try:
    sys.exit("bad config: lr is missing")
except SystemExit as e:
    print("caught SystemExit, code =", repr(e.code))
# caught SystemExit, code = 'bad config: lr is missing'
```

**The one good reason to catch a `BaseException` child: save before stopping.** Pressing Ctrl+C raises `KeyboardInterrupt` wherever the program is. A long training loop can catch it, save a checkpoint, and then re-raise so the program still stops:

```python
# Run this on your computer (or in Colab), not in the browser.
# train_loop.py
import time

def save_checkpoint(step):
    print(f"saved checkpoint at step {step}")

step = 0
try:
    while True:
        step += 1
        time.sleep(0.1)          # stands in for one training step
except KeyboardInterrupt:
    save_checkpoint(step)
    raise                        # then stop as usual
```

Run it and press Ctrl+C after a second (your step number will differ):

```text
$ python train_loop.py
^Csaved checkpoint at step 11
Traceback (most recent call last):
  File "/home/you/my_project/train_loop.py", line 10, in <module>
    time.sleep(0.1)          # stands in for one training step
    ~~~~~~~~~~^^^^^
KeyboardInterrupt
$ echo $?
130
```

`130` is the shell's code for "stopped by Ctrl+C". The handler names `KeyboardInterrupt` exactly and re-raises it; it never swallows it.

## Raising exceptions: `raise`

**What it is.** `raise SomeError("message")` creates an exception and throws it. It's Kotlin's `throw`.

**Why it exists.** Your own functions find problems too: a config value out of range, a list of predictions with a different length than the labels. Raising stops the computation at the point where the problem is detected, with a message that says what's wrong, instead of letting a bad value travel on and fail somewhere confusing.

**Kotlin comparison.**

```kotlin
fun setLr(lr: Double) {
    require(lr > 0) { "lr must be positive, got $lr" }      // throws IllegalArgumentException
}
```

```python
def check_lr(lr):
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")
    return lr

check_lr(0.01)            # 0.01
try:
    check_lr(-0.1)
except ValueError as e:
    print("ValueError:", e)
# ValueError: lr must be positive, got -0.1
```

Python has no `require` or `check`; the `if ...: raise ...` pair is the idiom. Choose the type the way the built-ins do:

- `ValueError` when the type is right but the value isn't (Kotlin's `IllegalArgumentException`);
- `TypeError` when the type is wrong;
- `KeyError` / `IndexError` when your class behaves like a dict or a list (Day 3a's `__getitem__`);
- `RuntimeError` when the object is in the wrong state for this call (Kotlin's `IllegalStateException`, what `check()` throws);
- `NotImplementedError` for a method a subclass must override (Day 3a's `forward`);
- your own class (below) when callers need to catch *your* errors specifically.

**How it works underneath.** `raise` takes an exception *object*: `raise ValueError("...")` creates the object and raises it. `raise ValueError` without brackets is allowed too; Python creates the object for you, with no message. When the exception object is raised, Python records the current frame in its `__traceback__` attribute, and more frames are added as it climbs the stack. That attribute is what gets printed as the traceback.

```python
try:
    raise ValueError
except ValueError as e:
    print(repr(e), "| message:", repr(str(e)))
# ValueError() | message: ''
```

**How it goes wrong.**

- **No message.** `raise ValueError` gives a traceback that ends `ValueError` with nothing after it. Always say what was wrong and include the bad value: `f"lr must be positive, got {lr}"`.
- **Raising something that isn't an exception.** Only instances of `BaseException` subclasses can be raised:

    ```python
    try:
        raise "lr must be positive"
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: exceptions must derive from BaseException
    ```

## Re-raising and chaining: `raise` and `raise ... from ...`

**What it is.** Inside an `except` block, a bare **`raise`** (no argument) re-raises the exception being handled, with its original traceback. **`raise NewError(...) from e`** raises a new exception and records `e` as its **cause**, so the traceback shows both.

**Why it exists.** Two common situations:

- you want to *do* something about an error (log it, clean up, add a count) but still let it fail, because you can't fix it here: re-raise;
- a low-level error (`KeyError: 'lr'`) doesn't mean much to the person who wrote the config. You translate it into a high-level one ("config is missing the learning rate") while keeping the original for debugging: chain.

**Kotlin comparison.** A bare `raise` is Kotlin's `throw e` inside the `catch`. `raise ... from e` is Kotlin's `throw ConfigException("...", e)`, passing `e` as the `cause`; the stack trace then shows `Caused by: ...`.

**Example: re-raise after recording.**

```python
failures = []

def load_value(config, key):
    try:
        return config[key]
    except KeyError:
        failures.append(key)       # record it...
        raise                      # ...and let it carry on up

try:
    load_value({"epochs": 10}, "lr")
except KeyError as e:
    print("caller saw KeyError:", e)
# caller saw KeyError: 'lr'
failures                           # ['lr']
```

**Bare `raise` or `raise e`?** Coming from Kotlin's `throw e`, you'll want to write `raise e`. It works, and the traceback is kept, but the `raise e` line is added to it as an extra frame:

```text
$ python reraise.py                  # the except block says: raise e
Traceback (most recent call last):
  File "/home/you/my_project/reraise.py", line 7, in <module>
    load({})
    ~~~~^^^^
  File "/home/you/my_project/reraise.py", line 5, in load
    raise e
  File "/home/you/my_project/reraise.py", line 3, in load
    return config["lr"]
           ~~~~~~^^^^^^
KeyError: 'lr'

$ python bare.py                     # the except block says: raise
Traceback (most recent call last):
  File "/home/you/my_project/bare.py", line 7, in <module>
    load({})
    ~~~~^^^^
  File "/home/you/my_project/bare.py", line 3, in load
    return config["lr"]
           ~~~~~~^^^^^^
KeyError: 'lr'
```

Prefer bare `raise`: it says "the same exception, unchanged", and the traceback only shows where the error really happened.

**Example: chaining with `from`.** The `ConfigError` class is a one-line custom exception; the next section explains how they're defined.

```pycon
>>> class ConfigError(Exception):
...     pass
...
>>> config = {"epochs": 10}
>>> try:
...     lr = config["lr"]
... except KeyError as e:
...     raise ConfigError("config is missing the learning rate") from e
...
Traceback (most recent call last):
  File "<python-input-2>", line 2, in <module>
    lr = config["lr"]
         ~~~~~~^^^^^^
KeyError: 'lr'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<python-input-2>", line 4, in <module>
    raise ConfigError("config is missing the learning rate") from e
ConfigError: config is missing the learning rate
```

There are now **two** tracebacks. Read the last one first, as usual: that's the error your caller sees. The line between them, "The above exception was the direct cause of the following exception", tells you the first one is its cause. The line numbers count from the start of each statement typed into the REPL; in a file they're real line numbers.

**How it works underneath: three attributes.** `from e` sets the new exception's `__cause__` to `e`. Separately, Python *always* records the exception that was being handled when a new one was raised, in `__context__`:

```python
class ConfigError(Exception):
    pass

try:
    try:
        {"epochs": 10}["lr"]
    except KeyError as e:
        raise ConfigError("config is missing the learning rate") from e
except ConfigError as err:
    print("cause:  ", repr(err.__cause__))
    print("context:", repr(err.__context__))
# cause:   KeyError('lr')
# context: KeyError('lr')
```

When the traceback is printed, Python follows these links and prints each exception in the chain.

**How it goes wrong: an accidental chain.** If the *handler itself* has a bug, Python shows both errors, joined by a different sentence:

```pycon
>>> config = {"epochs": 10}
>>> try:
...     lr = config["lr"]
... except KeyError:
...     print("using default " + 0.001)
...
Traceback (most recent call last):
  File "<python-input-1>", line 2, in <module>
    lr = config["lr"]
         ~~~~~~^^^^^^
KeyError: 'lr'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<python-input-1>", line 4, in <module>
    print("using default " + 0.001)
          ~~~~~~~~~~~~~~~~~^~~~~~~
TypeError: can only concatenate str (not "float") to str
```

Learn the two sentences apart:

- "**The above exception was the direct cause**" means someone wrote `raise ... from ...` on purpose.
- "**During handling of the above exception, another exception occurred**" usually means a bug *inside an `except` block*. Fix the last error (here, use an f-string); the first one was being handled correctly.

To hide the original exception on purpose (when it's noise for the reader), write `raise ConfigError(...) from None`. Only the new traceback is printed:

```pycon
>>> class ConfigError(Exception):
...     pass
...
>>> config = {"epochs": 10}
>>> try:
...     lr = config["lr"]
... except KeyError:
...     raise ConfigError("config is missing the learning rate") from None
...
Traceback (most recent call last):
  File "<python-input-2>", line 4, in <module>
    raise ConfigError("config is missing the learning rate") from None
ConfigError: config is missing the learning rate
```

Compare it with the two-traceback version above. Use `from None` sparingly: the original is often what you need when debugging.

## Custom exception classes

**What it is.** A custom exception is a class that inherits from `Exception` (or from a more specific built-in, such as `ValueError`). Often the body is only `pass` or a docstring.

**Why it exists.** Callers can catch *your* errors without catching everything else. A library typically defines one base class (`MlkitError`) and a few subclasses; a caller can then write `except MlkitError:` to handle all of them, or `except ConfigError:` for one. The `requests` HTTP library works this way: `requests.exceptions.RequestException` is the base, with `ConnectionError`, `HTTPError` and `Timeout` under it. pandas groups its exceptions in the `pandas.errors` module instead, and many of them subclass the closest built-in (`pandas.errors.ParserError` is a `ValueError`), so `except ValueError:` still catches them.

**Kotlin comparison.**

```kotlin
open class MlkitException(message: String) : Exception(message)
class ConfigException(val key: String, message: String) : MlkitException("$key: $message")
```

```python
class MlkitError(Exception):
    """Base class for every error this package raises."""


class ConfigError(MlkitError):
    """A config file is missing a field or has a bad value."""

    def __init__(self, key, message):
        super().__init__(f"{key}: {message}")   # sets the message
        self.key = key                          # extra data for callers


def validate(config):
    if "lr" not in config:
        raise ConfigError("lr", "missing")
    if config["lr"] <= 0:
        raise ConfigError("lr", f"must be positive, got {config['lr']}")
    return config

try:
    validate({"lr": -1})
except MlkitError as e:                       # the base class catches the subclass
    print(type(e).__name__, "|", e, "| key =", e.key)
# ConfigError | lr: must be positive, got -1 | key = lr
```

**How it works underneath.** Nothing special: an exception class is a normal class (Day 3a), and `except` matches it with `isinstance`, so the inheritance tree *is* the catching tree. `super().__init__(message)` passes the message to `Exception`, which stores it in `args`, and `str(e)` reads `args`. Extra attributes like `key` are ordinary instance attributes.

**How it goes wrong.**

- **Forgetting `super().__init__(...)`.** Python still stores the constructor's arguments in `args`, so the message becomes a tuple:

    ```python
    class BadConfigError(Exception):
        def __init__(self, key, message):
            self.key = key                     # no super().__init__(...)

    str(BadConfigError("lr", "must be positive"))   # "('lr', 'must be positive')"
    ```

- **Inheriting from `BaseException`.** Then `except Exception:` doesn't catch it, and it behaves like Ctrl+C:

    ```python
    class StopTraining(BaseException):
        pass

    try:
        try:
            raise StopTraining("loss is nan")
        except Exception:
            print("caught by except Exception")
    except BaseException as e:
        print("missed by except Exception:", type(e).__name__)
    # missed by except Exception: StopTraining
    ```

    Always inherit from `Exception` or one of its subclasses. Name your classes with an `Error` suffix, as the built-ins do.

## Python versus Kotlin: EAFP, LBYL and `runCatching`

**No checked exceptions, in either language.** Java forces you to declare or catch "checked" exceptions like `IOException`. Kotlin dropped that, and so does Python: any function can raise anything, and nothing forces the caller to handle it. The docstring is where you say what a function raises (`Raises: ConfigError if ...`).

**Two styles.** Python's documentation names them:

- **LBYL**, "look before you leap": check first, then act. `if key in config: value = config[key]`. This is the usual Kotlin style, helped by null-safety: `config[key] ?: default`.
- **EAFP**, "easier to ask forgiveness than permission": act, and handle the exception if it fails. `try: value = config[key] except KeyError: ...`. This is common Python style.

```python
config = {"epochs": 10}

# LBYL
if "lr" in config:
    lr = config["lr"]
else:
    lr = 0.001

# EAFP
try:
    lr = config["lr"]
except KeyError:
    lr = 0.001

# For this particular case, neither: dict.get exists for it
lr = config.get("lr", 0.001)
lr                         # 0.001
```

Neither style is always right. Use LBYL when the check is cheap and clear, and when failure is common. Use EAFP when the check would duplicate the work, or when the world can change between checking and acting. Files are the classic case: `if path.exists(): open(path)` can still fail, because the file can be deleted, or be unreadable, between the two lines. Opening it inside a `try` has no such gap.

**How EAFP goes wrong: the exception came from somewhere else.** An `except KeyError:` around a function call catches a `KeyError` from *anywhere* inside that call, not only the one you had in mind:

```python
LABEL_NAMES = {0: "cat", 1: "dog"}

def describe(sample):
    return f"{sample['id']}: {LABEL_NAMES[sample['label']]}"

sample = {"id": "img_7", "label": 2}       # it has a label, but label 2 is unknown
try:
    text = describe(sample)
except KeyError:
    text = "img_7 has no label"            # meant for a missing 'label' key
text                                       # 'img_7 has no label'
```

The message is wrong: the sample has a label. The `KeyError` was for the key `2` in `LABEL_NAMES`, a real problem that is now hidden. The fix is the same as for `else` above: keep the `try` around only the lookup you mean (`label = sample["label"]`), and call `describe` outside it, or check with `"label" in sample` first.

**`runCatching`, `toIntOrNull` and friends.** Kotlin has helpers that turn an exception into a value: `"abc".toIntOrNull()` gives `null`, and `runCatching { ... }` gives a `Result`. Python's standard library has no such helpers, but you can write one in a few lines:

```python
def to_float_or_none(text):
    """Like Kotlin's toDoubleOrNull()."""
    try:
        return float(text)
    except ValueError:
        return None

[to_float_or_none(t) for t in ["4.5", "N/A", "3"]]    # [4.5, None, 3.0]
```

Catch the specific type, as here. Don't write a general "run this and return `None` on any error" helper: that's a bare `except` with extra steps.

!!! android "Android bridge"
    Kotlin's `runCatching { }` catches **every** `Throwable`, and that includes `CancellationException`, which coroutines use to stop work. Wrapping a suspending call in `runCatching` can therefore swallow a cancellation and keep a coroutine running after its scope was cancelled, a well-known Android pitfall. Python's bare `except:` makes the same mistake with `KeyboardInterrupt`. The lesson carries over in both directions: catch the narrowest type you can handle, and let "please stop" signals through. In Python, that means catching `Exception` or narrower, never bare `except:` or `BaseException`.

## Assertions: `assert`

**What it is.** `assert condition, message` checks that `condition` is true. If it isn't, it raises `AssertionError` with the message. It's a one-line internal sanity check.

**Why it exists.** It states an assumption your code depends on, such as "these two arrays have the same length" or "this probability is between 0 and 1", and fails loudly at that point if the assumption is wrong. It's also the whole of how tests check results in pytest (Day 3c's "Testing basics").

**Kotlin comparison.** Kotlin's `assert(cond) { "message" }` is the same idea, and is also switchable: JVM assertions only run with the `-ea` flag. In Android app code, assertions are effectively never enabled, so Android code relies on `require` and `check` for checks that must run. Kotlin's `require` and `check` always run; Python's `if ...: raise` is their equivalent.

```python
def accuracy(preds, labels):
    assert len(preds) == len(labels), f"{len(preds)} preds but {len(labels)} labels"
    return sum(p == y for p, y in zip(preds, labels)) / len(labels)

accuracy([1, 0, 1], [1, 1, 1])        # 0.6666666666666666
try:
    accuracy([1, 0], [1, 1, 1])
except AssertionError as e:
    print("AssertionError:", e)
# AssertionError: 2 preds but 3 labels
```

(`sum(p == y for ...)` counts the `True`s, because `True` counts as 1 in arithmetic, as Day 2 showed.)

**How it works underneath: `-O` removes asserts.** `assert cond, msg` is shorthand for `if __debug__: if not cond: raise AssertionError(msg)`. `__debug__` is `True` normally, and `False` when Python runs with the `-O` ("optimise") flag, in which case the compiler leaves every `assert` out of the bytecode completely:

```text
$ cat check.py
batch_size = 0
assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
print("training with batch_size", batch_size)

$ python check.py
Traceback (most recent call last):
  File "/home/you/my_project/check.py", line 2, in <module>
    assert batch_size > 0, f"batch_size must be positive, got {batch_size}"
           ^^^^^^^^^^^^^^
AssertionError: batch_size must be positive, got 0

$ python -O check.py
training with batch_size 0
```

So: **use `assert` for "this can't happen unless my code has a bug"; use `if ...: raise ValueError(...)` for checking input** (config values, file contents, function arguments from callers). Input checks must still run when someone uses `-O`.

!!! warning "Common mistake: brackets around an assert"
    `assert` is a statement, not a function. With brackets, the condition and the message become one tuple, and a non-empty tuple is always true:

    ```pycon
    >>> x = -1
    >>> assert (x > 0, "x must be positive")
    <python-input-1>:1: SyntaxWarning: assertion is always true, perhaps remove parentheses?
    ```

    The assertion never fails. Python warns you; read the warning. Write `assert x > 0, "x must be positive"`.

---

## Going further

These are real and worth knowing exist, but aren't needed yet:

- **Exception groups and `except*`**, for raising several unrelated errors at once (used by `asyncio` task groups): [Python tutorial §8.9](https://docs.python.org/3/tutorial/errors.html#raising-and-handling-multiple-unrelated-exceptions).
- **Exception notes** with `e.add_note(...)`, to add context to an exception before re-raising it: [Python tutorial §8.10](https://docs.python.org/3/tutorial/errors.html#enriching-exceptions-with-notes).
- The full **built-in exception hierarchy**, including warnings: [Built-in exceptions](https://docs.python.org/3/library/exceptions.html#exception-hierarchy).
- The **`warnings`** module, for "this works but probably isn't what you meant" messages such as deprecations: [warnings docs](https://docs.python.org/3/library/warnings.html).
- The **`traceback`** module, to format or save a traceback yourself: [traceback docs](https://docs.python.org/3/library/traceback.html).
- **Debugging with `pdb`** and `breakpoint()`, to stop at the line that raised and look at the variables: [pdb docs](https://docs.python.org/3/library/pdb.html). Day 1 showed the same idea with your editor's debugger.
- **`contextlib.suppress`**, the standard library's "ignore this one exception type" helper: `with contextlib.suppress(FileNotFoundError): path.unlink()`. It uses `with`, which [Day 3c](day-03c-files-json.md) explains: [contextlib docs](https://docs.python.org/3/library/contextlib.html#contextlib.suppress).
- Covered next, in [Day 3c](day-03c-files-json.md): `with` and context managers, file errors in practice, `json.JSONDecodeError`, `logger.exception` for logging a caught error with its traceback, and pytest's `pytest.raises` for testing that code raises.

---

## Exercises

**Warm-ups**

1. **A safe conversion.** Write `safe_float(text, default=None)` that returns `float(text)`, or `default` when the text isn't a number. Test it on `"0.5"`, `" 3 "`, `"1e-3"`, `"N/A"` and `""`. Then call it on `None` and explain what happens.
2. **Raising with a good message.** Write `check_probability(p)` that returns `p` if it's a number between 0 and 1 inclusive. It raises `TypeError` if `p` isn't an `int` or `float` (a `bool` doesn't count), and `ValueError` if it's out of range; both messages must include the bad value. Call it on `0.3`, `1`, `1.5`, `"0.5"` and `True`, catching both exception types in one `except` and printing the type name and message.

**Core**

3. **Parse a config with clear errors.** Write `parse_config(lines)` for lines like `"lr=0.01"`, which returns a dict with `lr` (a `float`) and `epochs` (an `int`). Define `ConfigError(Exception)` with a `line_no` attribute. Raise `ConfigError` with the 1-based line number and a clear message when a line has no `=`, a key is unknown, or a value doesn't convert (chained with `from` to the original `ValueError`); and, with `line_no=None` and a message starting `end of config`, when `lr` or `epochs` is missing at the end. Test it on one good config and four broken ones, printing each error and the type of its `__cause__`.
4. **Keep going, and count what failed.** Each record below should give `clicks / views`. Process all of them, collecting the click rates of the good ones, and count failures by exception type name in a dict. Use specific `except` clauses only, so that a typo in your own code would still crash.

    ```python
    records = [
        {"app": "Lens", "clicks": 30, "views": 120},
        {"app": "Keep", "clicks": 5, "views": 0},
        {"app": "Files", "clicks": "12", "views": 48},
        {"app": "Maps", "views": 10},
        {"app": "Clock", "clicks": 9, "views": 36},
    ]
    ```

**Predict the output**

5. Write down everything this prints, in order, **before** running it. Don't press Run until you've written your answer.

    ```python
    def step(x):
        try:
            print("try", x)
            result = 10 / x
        except ZeroDivisionError:
            print("except")
            return "default"
        else:
            print("else")
            return result
        finally:
            print("finally")

    print(step(2))
    print(step(0))
    try:
        print(step("a"))
    except TypeError:
        print("caller caught TypeError")
    ```

**Find and fix the bug**

6. This function should average one metric over several training runs, and raise a clear `KeyError` naming the metric when a run doesn't have it. It returns `0.0` for every input. Find the two bugs, explain why you see `0.0` instead of an error, and fix both.

    ```python
    def average_metric(runs, metric):
        try:
            values = [run[metric] for run in runs]
            return sum(values) / len(value)
        except:
            return 0.0

    runs = [{"acc": 0.81, "loss": 0.52}, {"acc": 0.89, "loss": 0.41}]
    print(average_metric(runs, "acc"))       # expected: 0.85
    print(average_metric(runs, "f1"))        # expected: a KeyError that names 'f1'
    ```

**Challenge**

7. **A retry decorator.** Network downloads fail now and then. Write a decorator with arguments, `@retry(times=3, on=(ConnectionError,))`, that calls the function up to `times` times (and raises `ValueError` if `times` is less than 1). It retries only when the exception is one of the types in `on`, printing `attempt N failed: <message>`; any other exception goes straight to the caller. If every attempt fails, it raises `RetryError("gave up after N attempts")` chained with `from` to the last exception. Use `functools.wraps`. Test it with a fake `download` that fails a set number of times before it succeeds, with one that always fails, and with one that raises `ValueError`.

??? success "Solutions"
    **1. A safe conversion.**

    ```python
    def safe_float(text, default=None):
        try:
            return float(text)
        except ValueError:
            return default

    for t in ["0.5", " 3 ", "1e-3", "N/A", ""]:
        print(repr(t), "->", safe_float(t))
    # '0.5' -> 0.5
    # ' 3 ' -> 3.0
    # '1e-3' -> 0.001
    # 'N/A' -> None
    # '' -> None

    try:
        safe_float(None)
    except TypeError as e:
        print("TypeError:", e)
    # TypeError: float() argument must be a string or a real number, not 'NoneType'
    ```

    `float` accepts surrounding spaces and scientific notation, and raises `ValueError` for text that isn't a number, which the function turns into the default. `None` isn't text at all, so `float` raises `TypeError` instead, and that's deliberately *not* caught: passing `None` is a bug in the caller, and hiding it would turn a clear error into a silent `None`. Catch only the failure you expect.

    **2. Raising with a good message.**

    ```python
    def check_probability(p):
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise TypeError(f"probability must be a number, got {type(p).__name__} {p!r}")
        if not 0 <= p <= 1:
            raise ValueError(f"probability must be between 0 and 1, got {p}")
        return p

    for p in [0.3, 1, 1.5, "0.5", True]:
        try:
            print("ok:", check_probability(p))
        except (TypeError, ValueError) as e:
            print(f"{type(e).__name__}: {e}")
    # ok: 0.3
    # ok: 1
    # ValueError: probability must be between 0 and 1, got 1.5
    # TypeError: probability must be a number, got str '0.5'
    # TypeError: probability must be a number, got bool True
    ```

    The type check comes first, because comparing a string with `0 <= p` would itself raise a less helpful `TypeError`. `bool` is a subclass of `int` in Python (Day 2), so `isinstance(True, int)` is `True`, and it has to be excluded explicitly. `0 <= p <= 1` is a chained comparison (Day 2). One `except` with a tuple handles both types, and `type(e).__name__` tells them apart in the output.

    **3. Parse a config with clear errors.**

    ```python
    class ConfigError(Exception):
        def __init__(self, line_no, message):
            where = "end of config" if line_no is None else f"line {line_no}"
            super().__init__(f"{where}: {message}")
            self.line_no = line_no

    CONVERTERS = {"lr": float, "epochs": int}

    def parse_config(lines):
        config = {}
        for line_no, line in enumerate(lines, start=1):
            if "=" not in line:
                raise ConfigError(line_no, f"expected key=value, got {line!r}")
            key, value = line.split("=", 1)
            if key not in CONVERTERS:
                raise ConfigError(line_no, f"unknown key {key!r}")
            try:
                config[key] = CONVERTERS[key](value)
            except ValueError as e:
                raise ConfigError(line_no, f"bad value for {key}: {value!r}") from e
        for key in CONVERTERS:
            if key not in config:
                raise ConfigError(None, f"missing {key!r}")
        return config

    print(parse_config(["lr=0.01", "epochs=5"]))
    # {'lr': 0.01, 'epochs': 5}

    broken = [
        ["lr=0.01", "epochs 5"],
        ["lr=0.01", "epoch=5"],
        ["lr=0.0o1", "epochs=5"],
        ["lr=0.01"],
    ]
    for lines in broken:
        try:
            parse_config(lines)
        except ConfigError as e:
            print(f"ConfigError: {e}  (cause: {type(e.__cause__).__name__})")
    # ConfigError: line 2: expected key=value, got 'epochs 5'  (cause: NoneType)
    # ConfigError: line 2: unknown key 'epoch'  (cause: NoneType)
    # ConfigError: line 1: bad value for lr: '0.0o1'  (cause: ValueError)
    # ConfigError: end of config: missing 'epochs'  (cause: NoneType)
    ```

    `CONVERTERS` maps each allowed key to the function that converts its value, so one line, `CONVERTERS[key](value)`, calls `float` or `int` (functions are values, Day 3). `enumerate(lines, start=1)` gives human line numbers. `split("=", 1)` splits at the first `=` only. Only the conversion is inside a `try`, and its `ValueError` is chained with `from e`; the other errors are found by checks, so they have no cause. A missing key isn't on any line, so it gets `line_no=None` rather than blaming a valid line (or "line 0" for an empty config). One exception type for every config problem means a caller needs one `except ConfigError`, and `line_no` lets an editor jump to the line.

    **4. Keep going, and count what failed.**

    ```python
    records = [
        {"app": "Lens", "clicks": 30, "views": 120},
        {"app": "Keep", "clicks": 5, "views": 0},
        {"app": "Files", "clicks": "12", "views": 48},
        {"app": "Maps", "views": 10},
        {"app": "Clock", "clicks": 9, "views": 36},
    ]

    rates, failures = {}, {}
    for record in records:
        try:
            rates[record["app"]] = record["clicks"] / record["views"]
        except (KeyError, ZeroDivisionError, TypeError) as e:
            name = type(e).__name__
            failures[name] = failures.get(name, 0) + 1
            print(f"skipped {record['app']}: {name} {e}")
    print(rates)
    print(failures)
    # skipped Keep: ZeroDivisionError division by zero
    # skipped Files: TypeError unsupported operand type(s) for /: 'str' and 'int'
    # skipped Maps: KeyError 'clicks'
    # {'Lens': 0.25, 'Clock': 0.25}
    # {'ZeroDivisionError': 1, 'TypeError': 1, 'KeyError': 1}
    ```

    Each record fails in a different way: zero views, a number stored as text, and a missing key. The tuple names exactly those three types, so they're counted and skipped, while a `NameError` from a typo in the loop would still stop the program. `failures.get(name, 0) + 1` is Day 2b's counting idiom. The `try` holds only the line that can fail on bad data. Catching `TypeError` is a judgement call: here it means "bad data", but it would also hide a type bug in that same line, which is why the line is kept to one expression.

    **5. Predict the output.**

    ```python
    def step(x):
        try:
            print("try", x)
            result = 10 / x
        except ZeroDivisionError:
            print("except")
            return "default"
        else:
            print("else")
            return result
        finally:
            print("finally")

    print(step(2))
    print(step(0))
    try:
        print(step("a"))
    except TypeError:
        print("caller caught TypeError")
    # try 2
    # else
    # finally
    # 5.0
    # try 0
    # except
    # finally
    # default
    # try a
    # finally
    # caller caught TypeError
    ```

    `step(2)`: no exception, so `else` runs and reaches `return result`, but `finally` runs before the function actually returns, so `finally` is printed before the outer `print` shows `5.0`. `step(0)`: `ZeroDivisionError` is caught, `except` returns `"default"`, and again `finally` runs first. `step("a")`: `10 / "a"` raises `TypeError`, which no `except` here matches, so neither `except` nor `else` runs; `finally` still runs on the way out, and the caller's handler catches the error. The outer `print` never runs for that call, so nothing is printed for its result.

    **6. Find and fix the bug.**

    ```python
    def average_metric(runs, metric):
        try:
            values = [run[metric] for run in runs]
            return sum(values) / len(value)
        except:
            return 0.0

    runs = [{"acc": 0.81, "loss": 0.52}, {"acc": 0.89, "loss": 0.41}]
    print(average_metric(runs, "acc"))       # 0.0
    print(average_metric(runs, "f1"))        # 0.0

    def average_metric(runs, metric):
        if not runs:
            raise ValueError("no runs to average")
        try:
            values = [run[metric] for run in runs]
        except KeyError:
            raise KeyError(f"metric {metric!r} missing from at least one run") from None
        return sum(values) / len(values)

    print(round(average_metric(runs, "acc"), 4))
    try:
        average_metric(runs, "f1")
    except KeyError as e:
        print("KeyError:", e)
    # 0.85
    # KeyError: "metric 'f1' missing from at least one run"
    ```

    Bug 1 is the typo `len(value)`, which raises `NameError` on every call. Bug 2 is the bare `except:`, which catches that `NameError` (and the `KeyError` for `"f1"`, and anything else) and returns `0.0`, so every call looks like it worked. The fix catches only `KeyError`, only around the line that can raise it, and re-raises it with a clearer message (`from None`, because the original `KeyError: 'f1'` adds nothing). The empty list gets an explicit check instead of a hidden `ZeroDivisionError`. `round(..., 4)` is there because `(0.81 + 0.89) / 2` is `0.8500000000000001` in floating point (Day 2). The double quotes in the output come from `KeyError` itself: its message is the `repr()` of its argument, because the argument is normally a bare key. That's why many libraries raise a `ValueError` or their own `LookupError` subclass when the message is a sentence.

    **7. A retry decorator.**

    ```python
    import functools

    class RetryError(Exception):
        pass

    def retry(times=3, on=(ConnectionError, TimeoutError)):
        if times < 1:
            raise ValueError(f"times must be at least 1, got {times}")
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                for attempt in range(1, times + 1):
                    try:
                        return fn(*args, **kwargs)
                    except on as e:
                        print(f"attempt {attempt} failed: {e}")
                        last = e
                raise RetryError(f"gave up after {times} attempts") from last
            return wrapper
        return decorator

    def make_download(fail_times, error=ConnectionError):
        calls = []
        @retry(times=3, on=(ConnectionError, TimeoutError))
        def download(url):
            calls.append(url)
            if len(calls) <= fail_times:
                raise error(f"network down (call {len(calls)})")
            return f"downloaded {url}"
        return download

    print(make_download(2)("labels.json"))
    # attempt 1 failed: network down (call 1)
    # attempt 2 failed: network down (call 2)
    # downloaded labels.json

    try:
        make_download(5)("labels.json")
    except RetryError as e:
        print("RetryError:", e, "| cause:", repr(e.__cause__))
    # attempt 1 failed: network down (call 1)
    # attempt 2 failed: network down (call 2)
    # attempt 3 failed: network down (call 3)
    # RetryError: gave up after 3 attempts | cause: ConnectionError('network down (call 3)')

    try:
        make_download(1, error=ValueError)("labels.json")
    except ValueError as e:
        print("ValueError, not retried:", e)
    # ValueError, not retried: network down (call 1)

    print(make_download(0).__name__)
    # download
    ```

    Three layers, as in Day 3a's decorators with arguments: `retry(...)` returns `decorator`, which wraps `fn` in `wrapper`. `except on as e` works because `except` accepts a tuple of classes held in a variable. `last = e` copies the exception, because `e` is deleted at the end of the `except` block. A `return` inside the `try` ends the loop on success; if the loop runs out, the `raise ... from last` line runs. `ValueError` isn't in `on`, so it passes straight through on the first call. The default for `on` is the network errors, not `(Exception,)`: retrying on every exception would retry your own bugs too. The `times < 1` check matters because with `times=0` the loop never runs and `last` is never assigned, so the `raise` line would fail with a confusing `UnboundLocalError`. `make_download` uses a closure (Day 3) so each test gets its own `calls` list. `functools.wraps` keeps the name `download`. Real code would also wait between attempts (`time.sleep`, often doubling each time); it's left out here so the page runs instantly.

## Resources

- [Python tutorial §8: Errors and exceptions](https://docs.python.org/3/tutorial/errors.html): syntax errors, handling, raising, chaining, user-defined exceptions, clean-up actions.
- [Built-in exceptions](https://docs.python.org/3/library/exceptions.html): every built-in exception, what raises it, and the full hierarchy.
- [The `raise` statement](https://docs.python.org/3/reference/simple_stmts.html#the-raise-statement) and [the `assert` statement](https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement): the exact rules for chaining and for `-O`.
- [Python glossary: EAFP](https://docs.python.org/3/glossary.html#term-EAFP): the official one-paragraph description of the style, next to LBYL.
- [Kotlin: Exceptions](https://kotlinlang.org/docs/exceptions.html): the Kotlin side, to compare `throw`, `try` as an expression, `require` and `check`.
- [Real Python: Python exceptions, an introduction](https://realpython.com/python-exceptions/): the same ground with more examples.

## Checklist

- [ ] I can read a multi-frame traceback from the bottom up, skip library frames, and find the line in my own code to look at.
- [ ] I recognise `TypeError`, `ValueError`, `KeyError`, `IndexError`, `AttributeError`, `NameError`, `ZeroDivisionError`, `FileNotFoundError` and `ImportError` from their messages, and I know a `SyntaxError` stops the file before anything runs.
- [ ] I know which exceptions an `except` clause catches, from the hierarchy (`LookupError`, `OSError`, `Exception`, `BaseException`).
- [ ] I can use `try`, `except`, `else` and `finally`, with specific exception types in the right order.
- [ ] I can explain why a bare `except:` is a bug, and when catching broadly is acceptable.
- [ ] I can raise exceptions with a useful message, re-raise with `raise`, chain with `raise ... from ...`, and write my own exception classes.
- [ ] I can compare Python's exceptions with Kotlin's and choose between EAFP and LBYL.
- [ ] I use `assert` only for internal checks, because `python -O` removes it.

**Next:** [Day 3c: Files, JSON, logging and tests](day-03c-files-json.md)
