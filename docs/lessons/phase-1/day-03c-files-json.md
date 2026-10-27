# Day 3c: Files, JSON, Logging and Tests

**Phase 1 · Week 1 · ~170 minutes**

[Day 3b](day-03b-errors-files-json.md) taught you to read, handle and raise exceptions. Today puts that to work where errors happen most: reading and writing files, the CSV and JSON formats, logging what a long-running script is doing, and testing small pieces of code with pytest.

## Objectives

- [ ] Explain what `with` does through `__enter__` and `__exit__`, write a context manager as a class and with `@contextmanager`, and compare it with Kotlin's `use {}`.
- [ ] Open files with the right mode (`r`, `w`, `a`, `x`, `b`) and `encoding="utf-8"`, and recognise the errors and garbled text a wrong mode or encoding gives.
- [ ] Read text files whole, as a list of lines, line by line or with `readline`, move the position with `seek`, and read and write binary files.
- [ ] Work with paths through `pathlib.Path`: `/`, `name`, `stem`, `suffix`, `exists`, `mkdir(parents=True)`, `glob`, `read_text` and `write_text`; build paths from your script's folder with `__file__`; and copy and delete with `shutil`.
- [ ] Read and write CSV with the `csv` module, including `DictReader` and `DictWriter`, and convert its strings to numbers.
- [ ] Save and load JSON (`dumps`, `loads`, `dump`, `load`), predict what changes in a round trip, and fix "not JSON serializable" and `JSONDecodeError` problems.
- [ ] Load a config file into a dataclass with clear errors, and read and write JSON Lines.
- [ ] Replace debugging `print`s with `logging`, at the right level, send it to a file, and log caught exceptions with their tracebacks.
- [ ] Write a first pytest test file with `assert`, `pytest.approx` and `pytest.raises`, run it, and read a failure.

**Prerequisites.**

- [Day 1](day-01-setup.md): running a script from a terminal, `pip install` inside a virtual environment.
- [Day 2](day-02-python-types.md): numbers and why `0.1 + 0.2 != 0.3`, `True` being an `int`, `for` loops with `continue`.
- [Day 2a](day-02a-strings.md): strings and their methods (`strip`, `split`, `join`, `splitlines`), f-strings and `print(..., end="")`, `bytes` with `encode` / `decode`.
- [Day 2b](day-02b-collections.md): lists, dicts (`get`, `|` merge), `defaultdict` and `Counter`, nested JSON-like data, comprehensions, `*` unpacking.
- [Day 3](day-03-functions-classes.md): functions, `**kwargs` unpacking, lambdas and `key=`, closures, scope, and type hints (which Python doesn't check when the code runs).
- [Day 3a](day-03a-classes-modules.md): classes and dunder methods, dataclasses with `asdict` and `__post_init__`, decorators with arguments, `if __name__ == "__main__"`, and `sys.path`.
- [Day 3b](day-03b-errors-files-json.md): reading tracebacks, `try` / `except` / `else` / `finally`, `raise ... from ...`, custom exception classes, and `assert`.

!!! info "How to read the code on this page"
    Every Python block has a **▶ Run** button, and blocks share their variables from top to bottom like notebook cells, so run them in order, starting with the imports block below. A bare expression shows its value when run, as the `# comment` beside it says.

    Three other kinds of block have no Run button:

    - Blocks starting with `>>>` are transcripts from the Python 3.14 REPL that fail on purpose, as on Day 3b.
    - Blocks starting with `$` are terminal sessions: a command and what it printed.
    - Blocks labelled `# Run this on your computer (or in Colab), not in the browser.` need a real terminal or a package the browser doesn't have, such as pytest.

    In those blocks, the first two comment lines (`# Run this on your computer...` and `# filename.py`) are page labels: leave them out when you save the file, so the line numbers in tracebacks and test reports match. Paths in tracebacks are shortened: your own files to `/home/you/my_project/...`, and Python's own files to `.../lib/python3.14/...`. Test timings like `in 0.01s` vary by machine.

---

## The big idea

Almost every ML project starts and ends with files. A dataset arrives as a folder of images and a CSV of labels; hyperparameters live in a JSON config; predictions go out as JSON Lines; the trained model is saved as a checkpoint. Files are also where Day 3b's exceptions happen most: a path that doesn't exist, a file cut off halfway, a config with a typo, a number stored as text.

Three ideas carry the whole lesson:

- **Resources must be released.** An open file holds an operating-system handle and a buffer of unwritten data. `with` guarantees it's closed, however the block ends.
- **A file is bytes; text is your interpretation of them.** The mode (`"r"` versus `"rb"`) and the encoding decide how bytes become strings, and getting them wrong gives errors or garbled text.
- **Formats lose information.** CSV has no types, so every number comes back as text. JSON has no tuples and only string keys. Knowing what changes in a round trip saves hours of debugging.

**Analogy.** A `with` block is a library checkout desk. You borrow the book at the desk (`__enter__`), read it wherever you like (the body), and the desk takes it back at the exit (`__exit__`), whether you finished the book, fell asleep, or left in a hurry because the fire alarm went off. You can't walk out with it.

All the code on this page uses these imports, so run this block first:

```python
import csv
import io
import json
import logging
import tempfile
from pathlib import Path
```

---

## Where files live on this page

The ▶ Run button runs Python in your browser (Pyodide, Day 1), which can't see your computer's disk. Instead it has a **virtual filesystem** that lives in memory: `open`, `pathlib` and the `csv` and `json` modules all work on it normally, and everything disappears when you reload the page.

So that the examples don't depend on what's in the current folder, every file today goes into a fresh temporary folder. `tempfile.mkdtemp()` creates an empty folder with a random, unused name in the system's temporary directory and returns its path as a string; `Path(...)` turns that into a path object (explained in the "pathlib" section):

```python
work = Path(tempfile.mkdtemp())
work.is_dir()              # True
list(work.iterdir())       # []
```

On your own computer, the same code creates a real folder under `/tmp` (or your user's `Temp` folder on Windows). The name is random, so the page never prints it. `mkdtemp` leaves clean-up to you: the folder stays until you delete it (or the system clears its temporary directory). The "Context managers" section shows `TemporaryDirectory`, which deletes itself. In your own projects you'll use real paths like `Path("data/train.csv")`.

## Context managers and `with`

**What it is.** A **context manager** is an object that sets something up when a block starts and tears it down when the block ends. The **`with`** statement uses one:

```python
with open(work / "hello.txt", "w", encoding="utf-8") as f:
    f.write("hello\n")
```

`open(...)` returns a file object, which is a context manager. `with` calls its **`__enter__`** method at the start, binds the result to `f`, runs the indented block, and calls its **`__exit__`** method at the end, which closes the file. `__exit__` runs however the block ends: normally, with `return`, or with an exception.

**Why it exists.** Files, network connections, locks and GPU memory are limited resources that must be released. Without `with`, you'd need a `try` / `finally` around every use, and forgetting it leaks the resource on the first exception. For files that matters more than it looks: written data may sit in a buffer in memory until the file is closed, so a crash before `close()` can leave a half-written file.

**Kotlin comparison.** Kotlin's `use {}` does exactly this for anything `Closeable`:

```kotlin
File("hello.txt").bufferedWriter().use { w ->
    w.write("hello\n")
}                              // w.close() is called here, even after an exception
```

```python
with open(work / "hello.txt", "w", encoding="utf-8") as f:
    f.write("hello\n")
# f.close() was called here, even after an exception
f.closed                   # True
```

What's the same: setup, body, guaranteed clean-up. What's different: `use` is an extension function that takes a lambda and works on `Closeable`; `with` is syntax, and works on any object with `__enter__` and `__exit__` methods. Notice also that `f` still exists after the block (Python has no block scope, Day 3), but it's a closed file.

**Example: a context manager you can watch.** Here's a class with both methods, which prints when each runs:

```python
class Traced:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print(f"enter {self.name}")
        return self                    # this is what `as` binds

    def __exit__(self, exc_type, exc, tb):
        print(f"exit {self.name}: exc_type={exc_type}")
        return False                   # False: don't swallow the exception

with Traced("block A") as t:
    print("inside, t.name =", t.name)
# enter block A
# inside, t.name = block A
# exit block A: exc_type=None

try:
    with Traced("block B"):
        print("about to fail")
        1 / 0
        print("never printed")
except ZeroDivisionError as e:
    print("caught outside:", e)
# enter block B
# about to fail
# exit block B: exc_type=<class 'ZeroDivisionError'>
# caught outside: division by zero
```

**How it works underneath.** `with` is shorthand for a `try` / `finally`:

1. evaluate the expression after `with` (`Traced("block B")`), giving the **manager**;
2. call `manager.__enter__()`, and bind its return value to the `as` name, if there is one;
3. run the body inside a `try`;
4. whatever happens, call `manager.__exit__(exc_type, exc, tb)`. If the body finished normally, all three are `None`. If it raised, they're the exception's class, the exception object and its traceback;
5. if `__exit__` returns a true value, the exception is **swallowed**; otherwise it carries on up, as with block B.

<figure class="diagram" markdown>
![Animation: on the left, a with open(path) as f block reading and parsing JSON, and below it the try/finally code Python roughly turns it into, with __enter__ before the try and __exit__ in the finally. On the right, two runs: with valid JSON, a green dot passes through __enter__, the body and __exit__, which closes the file; with broken JSON, a red dot passes through __enter__, a failing body raising JSONDecodeError, __exit__ which still closes the file, and then on to the caller](../../assets/img/phase-1/day03c-with-exit.svg)
<figcaption>Animated: __exit__ runs on both paths, so the file is always closed; the exception still reaches your code afterwards.</figcaption>
</figure>

A file object's `__enter__` returns the file itself, which is why `with open(...) as f` gives you the file. Its `__exit__` calls `close()` and returns `False`, so exceptions are never hidden.

**`@contextmanager`: a context manager from a function.** Writing a class with two methods is a lot for a small setup and teardown. `contextlib.contextmanager` is a decorator (Day 3a) that builds a context manager from a function with one `yield` in it. Everything before `yield` is the setup (`__enter__`); the value yielded is what `as` binds; everything after `yield` is the teardown (`__exit__`). `yield` is the keyword that makes a **generator**, a function that can pause and resume; Day 3d teaches it fully. Here all you need is "the `with` body runs at the point of the `yield`":

```python
from contextlib import contextmanager
import time

@contextmanager
def timed(label):
    start = time.perf_counter()
    try:
        yield                                  # the with-body runs here
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: done ({'fast' if elapsed < 1 else 'slow'})")

with timed("sum of squares"):
    total = sum(i * i for i in range(10_000))
# sum of squares: done (fast)
total                                          # 333283335000
```

`time.perf_counter()` is a clock for measuring durations, in seconds. (The example prints "fast" or "slow" instead of the time itself, because the exact time depends on your machine.) The `try` / `finally` around `yield` matters: if the body raises, the exception appears *at the `yield`*, and without `finally` the teardown would be skipped.

**Several managers in one `with`.** One `with` can manage several objects, separated by commas. Since Python 3.10 you can wrap them in brackets and put one per line, which is the readable form. They're entered left to right and exited in **reverse** order, like nested blocks:

```python
with (
    Traced("outer") as a,
    Traced("inner") as b,
):
    print("body")
# enter outer
# enter inner
# body
# exit inner: exc_type=None
# exit outer: exc_type=None

with (
    open(work / "hello.txt", encoding="utf-8") as src,
    open(work / "copy.txt", "w", encoding="utf-8") as dst,
):
    dst.write(src.read())
(work / "copy.txt").read_text(encoding="utf-8")    # 'hello\n'
```

Reverse order matters when one resource depends on another: the inner one is released first. Older code writes the same thing on one line, or splits it with a `\` at the end of the line, which continues a statement on the next line. (`read_text` reads a whole file in one call; it's in the "pathlib" section.)

**`suppress`: ignoring one exception on purpose.** `contextlib.suppress` is a ready-made manager that swallows the exception types you name, a tidy replacement for `try: ... except X: pass` where ignoring really is right. `unlink()` deletes a file, and raises `FileNotFoundError` if it isn't there:

```python
from contextlib import suppress

with suppress(FileNotFoundError):
    (work / "old_cache.txt").unlink()          # delete it if it's there
(work / "old_cache.txt").unlink(missing_ok=True)   # the same, for files specifically
(work / "old_cache.txt").exists()              # False
```

For deleting files, prefer `unlink(missing_ok=True)`: it says exactly what you mean, and you'll see it everywhere. `suppress` is the general tool for any other exception you've decided to ignore.

**`TemporaryDirectory`: a manager that deletes things.** `tempfile.TemporaryDirectory()` creates a temporary folder on entry, gives you its path as a string, and deletes the folder and everything in it on exit. Its `__exit__` does real work:

```python
with tempfile.TemporaryDirectory() as d:
    scratch = Path(d)
    (scratch / "batch_0.txt").write_text("tmp", encoding="utf-8")
    print(sorted(p.name for p in scratch.iterdir()))
# ['batch_0.txt']
scratch.exists()                               # False
```

Use it for intermediate files a step needs only while it runs. (This page keeps using `work` from `mkdtemp`, because later blocks need its files.)

!!! warning "Common mistake: using the file after the `with` block"
    The file is closed when the block ends, so reading it afterwards fails:

    ```python
    with open(work / "hello.txt", encoding="utf-8") as f:
        first = f.read()
    try:
        rest = f.read()                     # outside the with block
    except ValueError as e:
        print("ValueError:", e)
    # ValueError: I/O operation on closed file.
    ```

    Do all the reading inside the block, or read everything into a variable there (`text = f.read()`) and work with the variable afterwards.

## Opening files: `open`, modes and encodings

**What it is.** `open(path, mode, encoding=...)` opens a file and returns a file object. The **mode** is a short string saying what you'll do with it:

- `"r"`: read (the default). The file must exist.
- `"w"`: write. Creates the file, or **empties it** if it exists.
- `"a"`: append. Creates the file, or adds to its end.
- `"x"`: exclusive create. Creates the file, and fails if it already exists.
- add `"b"` for **binary** (`"rb"`, `"wb"`): you read and write `bytes` instead of `str`, with no encoding. For images, model weights, anything that isn't text.
- add `"+"` to read and write the same file (`"r+"`); rarely needed.

The **encoding** is how characters are turned into bytes on disk (Day 2a's `encode` / `decode`). Text mode decodes on read and encodes on write.

**Why it exists.** Every dataset, config, log and checkpoint is a file. Choosing the mode is choosing whether you can destroy data, so it's worth being deliberate.

**Kotlin comparison.**

```kotlin
File("train_log.txt").writeText("epoch 1: loss 0.90\n")             // overwrite
File("train_log.txt").appendText("epoch 2: loss 0.70\n")            // append
val text = File("train_log.txt").readText()                          // UTF-8 by default
```

```python
log_path = work / "train_log.txt"
with open(log_path, "w", encoding="utf-8") as f:
    n = f.write("epoch 1: loss 0.90\n")
    f.write("epoch 2: loss 0.70\n")
n                          # 19
with open(log_path, "a", encoding="utf-8") as f:
    print("epoch 3: loss 0.65", file=f)
with open(log_path, encoding="utf-8") as f:
    text = f.read()
text                       # 'epoch 1: loss 0.90\nepoch 2: loss 0.70\nepoch 3: loss 0.65\n'
```

Line by line:

- `f.write(s)` writes the string and returns how many characters it wrote (19 here, counting the `\n`). Unlike `print`, it **doesn't add a newline**: you write `\n` yourself.
- `print(..., file=f)` sends `print`'s output to the file instead of the screen, newline included.
- `open(log_path, encoding="utf-8")` uses the default mode, `"r"`. `f.read()` returns the whole file as one string.

Kotlin's `readText()` always uses UTF-8. Python's `open` uses the operating system's **default encoding** if you don't give one. On Linux and macOS that's UTF-8; on Windows it has traditionally been a legacy code page such as `cp1252`. So code without `encoding=` can work on your laptop and break on a colleague's. **Always pass `encoding="utf-8"` in text mode.**

**`"x"` mode: don't overwrite.** For outputs you never want to overwrite by accident, like a results file from a long run:

```python
try:
    with open(log_path, "x", encoding="utf-8") as f:
        f.write("new run\n")
except FileExistsError as e:
    print(type(e).__name__, "-", e.strerror)
# FileExistsError - File exists
```

(`e.strerror` is the operating system's description of the error. The page prints that instead of the whole message, because the message contains the random temporary path.)

**How it works underneath: buffering, position and newlines.** A file object keeps a **buffer**: writes collect in memory and are handed over in larger chunks, which is much faster than one system call per write. `close()` (or the end of the `with` block) **flushes** the buffer: it hands the data to the operating system. That's why a crashed program without `with` can leave a file with its last lines missing. `f.flush()` forces a flush without closing. The operating system then writes to the physical disk when it chooses; `os.fsync(f.fileno())` forces that too, which only matters for files that must survive a power cut (the challenge exercise uses it).

The file object also keeps a **position**, like a cursor. `read()` reads from the position to the end and leaves the position at the end, so a second `read()` returns `''`. `f.seek(0)` moves the position back to the start, and `f.tell()` reports where it is:

```python
with open(log_path, encoding="utf-8") as f:
    first = f.read()
    second = f.read()                # the position is at the end now
    end = f.tell()
    f.seek(0)                        # back to the start
    again = f.read()
len(first), repr(second), end, again == first     # (57, "''", 57, True)
```

In text mode, treat the number from `tell()` as an opaque bookmark: something to pass back to `seek()`, not a character count (here they happen to match because the text is ASCII). In text mode, Python also translates line endings: it writes `\n` as the platform's line ending (`\r\n` on Windows) and turns any line ending back into `\n` when reading, so your code only ever sees `\n`.

!!! warning "Common mistake: opening with `\"w\"` to read"
    `"w"` empties the file the moment it opens, before you've read or written anything:

    ```python
    with open(log_path, "w", encoding="utf-8") as f:     # meant "r"
        try:
            f.read()
        except io.UnsupportedOperation as e:
            print("UnsupportedOperation:", e)
    # UnsupportedOperation: not readable
    log_path.read_text(encoding="utf-8")                 # ''
    ```

    The error tells you the file wasn't opened for reading, but the damage is already done: the three log lines are gone. `io.UnsupportedOperation` is the error file objects raise for an operation their mode doesn't allow. Use `"r"` (or no mode) to read, and think twice before every `"w"`.

**The wrong encoding.** Text written as UTF-8 and read with a different encoding either fails or gives garbled text (called **mojibake**):

```python
fr_path = work / "labels_fr.txt"
fr_path.write_text("café\nthé\n", encoding="utf-8")    # 9 (the number of characters written)
fr_path.read_bytes()                           # b'caf\xc3\xa9\nth\xc3\xa9\n'
fr_path.read_text(encoding="latin-1")          # 'cafÃ©\nthÃ©\n'
try:
    fr_path.read_text(encoding="ascii")
except UnicodeDecodeError as e:
    print("UnicodeDecodeError:", e)
# UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 in position 3: ordinal not in range(128)
```

`read_bytes()` shows what's really on disk: `é` is the two bytes `\xc3\xa9` in UTF-8. Latin-1 (like Windows' `cp1252`) reads each byte as its own character, so `é` turns into `Ã©`. ASCII only knows bytes up to 127 and fails at the first byte above that. If you see `Ã©` or similar in your data, the file was UTF-8 and something read it with the wrong encoding.

## Reading: whole files, line by line, and binary

There are three ways to read a text file. Start with a small file of labels:

```python
labels_path = work / "labels.txt"
with open(labels_path, "w", encoding="utf-8") as f:
    f.write("cat\ndog\nfox\n")

with open(labels_path, encoding="utf-8") as f:
    whole = f.read()               # one string
whole                              # 'cat\ndog\nfox\n'

with open(labels_path, encoding="utf-8") as f:
    lines = f.readlines()          # a list of lines, each ending in \n
lines                              # ['cat\n', 'dog\n', 'fox\n']

with open(labels_path, encoding="utf-8") as f:
    for line in f:                 # one line at a time
        print(repr(line), "->", repr(line.rstrip("\n")))
# 'cat\n' -> 'cat'
# 'dog\n' -> 'dog'
# 'fox\n' -> 'fox'
```

- `f.read()` returns everything as one string. Then `whole.splitlines()` gives the lines without their `\n`.
- `f.readlines()` returns a list of lines, **each still ending in `\n`**.
- `for line in f:` reads one line per iteration. A file object is an **iterator** over its lines (Day 3d explains iterators), so only one line is in memory at a time. This is how you read a 10 GB log file.

There's a fourth, lower-level way: `f.readline()` reads **one** line, including its `\n`, each time you call it. At the end of the file it returns `''`, an empty string. A blank line in the middle comes back as `'\n'`, not `''`, which is how `readline` tells "end of file" from "blank line":

```python
with open(work / "two_lines.txt", "w", encoding="utf-8") as f:
    f.write("header\n\nrow 1\n")
with open(work / "two_lines.txt", encoding="utf-8") as f:
    print([f.readline(), f.readline(), f.readline(), f.readline()])
# ['header\n', '\n', 'row 1\n', '']
```

Use it to read a header line before looping over the rest; for everything else, `for line in f` is simpler.

`line.rstrip("\n")` removes the newline from the end only. `line.strip()` would also remove spaces and tabs at both ends, which is usually what you want for data but can damage text where leading spaces matter.

**Why it matters.** Reading a whole file is simplest and fine for configs and small datasets. Line by line is for anything large, or for streaming formats like JSON Lines (in the JSON section).

!!! warning "Common mistake: forgetting that lines end in `\n`"
    ```python
    with open(labels_path, encoding="utf-8") as f:
        labels = f.readlines()
    "dog" in labels                                  # False
    labels[1]                                        # 'dog\n'
    labels = labels_path.read_text(encoding="utf-8").splitlines()
    "dog" in labels                                  # True
    ```

    The `\n` is invisible when you `print` a line, so this one is hard to spot. Print with `repr()` (or look at the list) and the `\n` shows. `splitlines()` or a `line.strip()` in a comprehension fixes it.

**Binary mode.** `"rb"` and `"wb"` read and write `bytes`, unchanged. Every PNG file starts with the same 8 bytes, its **signature**, which is how image libraries recognise the format:

```python
png_path = work / "fake.png"
with open(png_path, "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n" + bytes(8))      # signature + 8 zero bytes
with open(png_path, "rb") as f:
    header = f.read(8)                            # read only the first 8 bytes
header                                            # b'\x89PNG\r\n\x1a\n'
header == b"\x89PNG\r\n\x1a\n"                    # True
png_path.stat().st_size                           # 16
```

`f.read(8)` reads at most 8 bytes (or characters, in text mode). `bytes(8)` is 8 zero bytes. `stat().st_size` is the file's size in bytes. Binary mode takes no `encoding`. You'll rarely read image bytes yourself (Pillow and OpenCV do it), but checkpoints and `.npy` files (Day 4b) are binary, and it's good to know what `"b"` means.

## pathlib: paths as objects

**What it is.** `pathlib.Path` represents a file system path as an object, with methods and properties for its parts and for the file it points to. It replaces gluing strings together and most of the older `os.path` functions.

**Why it exists.** Paths built from strings break: a missing `/`, a Windows `\`, a file extension sliced off wrongly. Every ML project has `data/raw/`, `data/processed/`, `checkpoints/epoch_03.pt`; `Path` makes building and inspecting those safe and readable.

**Kotlin comparison.** Kotlin on the JVM has `java.io.File` and `java.nio.file.Path`, with Kotlin extensions such as `Path.div` (so `/` works) and `readText()`:

```kotlin
val p = Path("data") / "images" / "cat_001.jpg"     // kotlin.io.path
p.name              // "cat_001.jpg"
p.nameWithoutExtension   // "cat_001"
p.extension         // "jpg"
```

```python
p = Path("data") / "images" / "cat_001.jpg"
p                          # PosixPath('data/images/cat_001.jpg')
p.name                     # 'cat_001.jpg'
p.stem                     # 'cat_001'
p.suffix                   # '.jpg'
p.parent                   # PosixPath('data/images')
p.with_suffix(".png")      # PosixPath('data/images/cat_001.png')
p.parts                    # ('data', 'images', 'cat_001.jpg')
str(p)                     # 'data/images/cat_001.jpg'
```

The `/` operator joins paths (Python's `__truediv__` dunder, Day 3a, does the work). On Linux and macOS (and in the browser) a path displays as `PosixPath`; on Windows it's `WindowsPath`, with `\` separators. You write the same code either way. Note `suffix` includes the dot (`'.jpg'`) where Kotlin's `extension` doesn't. `str(p)` gives the plain string, for the rare function that wants one; `open`, `json`, numpy, pandas and PyTorch all accept `Path` objects directly.

**Working with real files.** Build a small dataset folder in the temporary directory:

```python
images = work / "data" / "images"
images.mkdir(parents=True, exist_ok=True)
for name in ["cat_001.jpg", "cat_002.jpg", "dog_001.jpg", "notes.txt"]:
    (images / name).write_text("fake image", encoding="utf-8")
(work / "data" / "labels.csv").write_text("file,label\n", encoding="utf-8")   # 11

images.exists(), images.is_dir()                     # (True, True)
(images / "cat_001.jpg").is_file()                   # True
(images / "zebra.jpg").exists()                      # False
sorted(f.name for f in images.iterdir())             # ['cat_001.jpg', 'cat_002.jpg', 'dog_001.jpg', 'notes.txt']
sorted(f.name for f in images.glob("*.jpg"))         # ['cat_001.jpg', 'cat_002.jpg', 'dog_001.jpg']
sorted(f.name for f in images.glob("cat_*"))         # ['cat_001.jpg', 'cat_002.jpg']
sorted(f.relative_to(work).as_posix() for f in work.rglob("*.csv"))   # ['data/labels.csv']
(images / "cat_001.jpg").read_text(encoding="utf-8") # 'fake image'
```

What each method does:

- `mkdir(parents=True, exist_ok=True)` creates the folder, **and any missing parent folders** (`parents=True`), and doesn't complain if it already exists (`exist_ok=True`). Like `mkdir -p` in a shell, or Kotlin's `createDirectories()`.
- `write_text(s, encoding=...)` and `read_text(encoding=...)` open, write or read, and close the file in one call. `write_text` returns the number of characters written, like `f.write`, which is the `11` echoed above (and the `9` in the encoding example). `read_bytes` and `write_bytes` are the binary versions. They're the shortest way to handle small files; use `open` with `with` when you need to read line by line or write in pieces.
- `exists()`, `is_file()` and `is_dir()` ask the file system; they return `False` rather than raising.
- `iterdir()` gives every entry in a folder; `glob(pattern)` gives the entries matching a pattern, where `*` matches any run of characters within a name; `rglob(pattern)` searches all subfolders too ("recursive glob").
- These return paths in whatever order the file system gives, which can differ between machines. Wrap them in `sorted(...)` when the order matters, which in ML is always: an unsorted file list gives a different train/test split on a different machine.
- `relative_to(work)` cuts the `work` prefix off a path; `as_posix()` writes it with `/` on every operating system.

**Paths relative to your script.** A relative path such as `Path("data")` is resolved from the **current working directory** (Day 3b's `FileNotFoundError` explains why that bites). `Path.cwd()` tells you what that is, and `resolve()` turns a relative path into the absolute one it really means:

```python
Path.cwd().is_absolute()                       # True
Path("data").resolve() == Path.cwd() / "data"  # True
print(Path.cwd())
```

The last line prints a different folder on every machine: wherever Python was started. In the browser it's `/home/pyodide`, Pyodide's home folder.

To find files next to your script, wherever it's launched from, build the path from the script's own location. `__file__` is a variable Python sets in every file it runs or imports: the path of that file. `Path(__file__).resolve().parent` is the folder the script is in:

```python
# Run this on your computer (or in Colab), not in the browser.
# scripts/show.py
from pathlib import Path

script_dir = Path(__file__).resolve().parent
config = script_dir / "config.json"
print("cwd:   ", Path.cwd())
print("config:", config, config.exists())
```

```text
$ cd /home/you/my_project
$ python scripts/show.py
cwd:    /home/you/my_project
config: /home/you/my_project/scripts/config.json True

$ cd /home/you
$ python my_project/scripts/show.py
cwd:    /home/you
config: /home/you/my_project/scripts/config.json True
```

The working directory changed; the config path didn't. `Path("config.json")` would have found the file only in the first run. `__file__` doesn't exist in a notebook or in this browser page, which is why the script isn't runnable here; in a notebook, paths are relative to the notebook's folder.

**Copying and deleting.** `path.unlink()` deletes a file, and `path.rmdir()` deletes an **empty** folder. For whole folders and copies, use the `shutil` ("shell utilities") module: `shutil.copy(src, dst)` copies a file, and `shutil.rmtree(folder)` deletes a folder with everything in it:

```python
import shutil

backup = work / "backup"
backup.mkdir(exist_ok=True)
copied = shutil.copy(images / "cat_001.jpg", backup / "cat_001.jpg")
copied.name                                    # 'cat_001.jpg'
sorted(p.name for p in backup.iterdir())       # ['cat_001.jpg']
try:
    backup.rmdir()                             # not empty
except OSError as e:
    print(type(e).__name__, "-", e.strerror)
# OSError - Directory not empty
shutil.rmtree(backup)
backup.exists()                                # False
```

`shutil.copy` returns the destination path; the page shows only its `name`, because the full path contains the random folder name. `rmdir` refuses a folder with files in it (`OSError`, the parent class of the file errors from Day 3b), so `rmtree` is the tool for that. Treat `rmtree` with respect: there's no recycle bin.

The older `os.path` module does the same jobs with functions on strings: `os.path.join("data", "images")`, `os.path.exists(p)`, `os.path.splitext(name)`. You'll see it in older code and tutorials; `pathlib` is the modern choice.

!!! warning "Common mistake: building paths with `+`"
    ```python
    folder = "data"
    folder + "train.csv"                       # 'datatrain.csv'
    Path(folder) / "train.csv"                 # PosixPath('data/train.csv')
    ```

    The string version silently loses the separator, and the `FileNotFoundError` comes much later. Use `/` on a `Path`. A related trap: `Path("~/datasets")` does **not** mean your home folder; `~` is a shell feature. Use `Path("~/datasets").expanduser()` or `Path.home() / "datasets"`.

`mkdir` without `parents=True` fails when a parent is missing:

```python
try:
    (work / "runs" / "exp1").mkdir()           # runs/ doesn't exist yet
except FileNotFoundError as e:
    print(type(e).__name__, "-", e.strerror)
# FileNotFoundError - No such file or directory
```

## CSV with the `csv` module

**What it is.** **CSV** (comma-separated values) is a text format for tables: one row per line, fields separated by commas, the first line usually a header with the column names. A field that contains a comma or a quote is wrapped in double quotes. Python's `csv` module reads and writes it correctly, quotes included.

**Why it exists.** Most small datasets arrive as CSV, including the Play Store data this course uses. From Week 3 you'll load CSVs with pandas (Day 15), which is what you'll use for analysis. The `csv` module is still useful for small scripts, for streaming files row by row, and for understanding what pandas does for you.

**Kotlin comparison.** Kotlin's standard library has no CSV parser; Android projects use a library such as OpenCSV or kotlin-csv, or write a fragile `line.split(",")`. Python ships one.

**The problem with `split(",")`.** An app name containing a comma is quoted in the file, and splitting on commas breaks it:

```python
csv_path = work / "apps.csv"
rows = [
    ["name", "category", "rating", "installs"],
    ["Snapseed", "PHOTOGRAPHY", "4.5", "50000000"],
    ["Notes, Lists & Reminders", "PRODUCTIVITY", "4.2", "1000000"],
    ["Pocket Casts", "MUSIC_AND_AUDIO", "4.6", "5000000"],
]
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(csv_path.read_text(encoding="utf-8"))
# name,category,rating,installs
# Snapseed,PHOTOGRAPHY,4.5,50000000
# "Notes, Lists & Reminders",PRODUCTIVITY,4.2,1000000
# Pocket Casts,MUSIC_AND_AUDIO,4.6,5000000

third_line = csv_path.read_text(encoding="utf-8").splitlines()[2]
third_line.split(",")      # ['"Notes', ' Lists & Reminders"', 'PRODUCTIVITY', '4.2', '1000000']
```

`csv.writer(f)` wraps a file; `writerows(list_of_lists)` writes each inner list as one row (`writerow` writes one). The writer added the quotes around the name with a comma by itself. Splitting that line on commas gives five fields instead of four.

**Reading it properly.**

```python
with open(csv_path, encoding="utf-8", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)              # the first row
    data = list(reader)                # the remaining rows
header                                 # ['name', 'category', 'rating', 'installs']
data[1]                                # ['Notes, Lists & Reminders', 'PRODUCTIVITY', '4.2', '1000000']
```

`csv.reader(f)` gives one list of strings per row, with the quotes handled. For tab-separated files (`.tsv`, common for ML datasets), pass `delimiter="\t"`: `csv.reader(f, delimiter="\t")`, and the same for `csv.writer`. `next(reader)` takes one row from it (an iterator again, Day 3d), and `list(reader)` takes the rest.

**`DictReader` and `DictWriter`: rows as dicts.** Positions like `row[2]` break when someone adds a column. `csv.DictReader` uses the header row as keys:

```python
with open(csv_path, encoding="utf-8", newline="") as f:
    apps = list(csv.DictReader(f))
apps[0]                                # {'name': 'Snapseed', 'category': 'PHOTOGRAPHY', 'rating': '4.5', 'installs': '50000000'}
best = max(apps, key=lambda row: float(row["rating"]))
best["name"]                           # 'Pocket Casts'

summary_path = work / "summary.csv"
with open(summary_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "rating"])
    writer.writeheader()
    for row in apps:
        writer.writerow({"name": row["name"], "rating": row["rating"]})
print(summary_path.read_text(encoding="utf-8"), end="")
# name,rating
# Snapseed,4.5
# "Notes, Lists & Reminders",4.2
# Pocket Casts,4.6
```

`DictWriter` needs `fieldnames`, the column order; `writeheader()` writes that row. (`print(..., end="")` stops `print` from adding a second newline after the file's own last one, as Day 2a showed.)

**How it works underneath.** The `csv` module reads character by character, tracking whether it's inside quotes, so a comma inside quotes is data and a comma outside is a separator. Two consequences:

- **Every field is a string.** CSV has no types: `'4.5'` and `'50000000'` are text until you convert them with `float()` and `int()`. pandas guesses types for you; the `csv` module doesn't.
- **Open CSV files with `newline=""`.** A quoted field may contain a newline, and the module needs to see the file's raw line endings to handle it. The `csv` documentation asks for `newline=""` on both reading and writing; without it, on Windows, written files get an extra blank line between rows.

!!! warning "Common mistake: comparing numbers as strings"
    Since every field is a string, comparisons are alphabetical, character by character, not numeric:

    ```python
    "9" > "10"                                   # True
    sorted(["100", "20", "3"])                   # ['100', '20', '3']
    sorted(["100", "20", "3"], key=int)          # ['3', '20', '100']
    ```

    `"9" > "10"` because the first characters already differ, and `'9'` comes after `'1'`. A "top apps by installs" list sorted on the raw strings is quietly wrong in the same way. Convert with `int()` or `float()` (here, `key=int`) before comparing or sorting.

## JSON

**What it is.** **JSON** (JavaScript Object Notation) is a text format for nested data: objects `{...}` with string keys, arrays `[...]`, strings in double quotes, numbers, `true`, `false` and `null`. Python's `json` module converts between JSON text and Python's dicts, lists, strings, numbers, booleans and `None`.

**Why it exists.** It's the format of nearly every web API response, of most config files you'll meet (a Hugging Face model's `config.json`, a COCO dataset's annotations), of label maps (`labels.json`), and of Jupyter notebooks themselves (an `.ipynb` file is JSON). In ML you'll write experiment configs and results as JSON, and load label maps that an Android app also reads.

**Kotlin comparison.** On Android you'd use `org.json.JSONObject`, Gson, Moshi or kotlinx.serialization, usually mapping JSON onto a data class:

```kotlin
@Serializable data class Config(val model: String, val lr: Double, val epochs: Int)
val text = Json.encodeToString(Config("mobilenet_v3", 0.001, 10))
val config = Json.decodeFromString<Config>(text)
```

Python's `json` maps onto plain dicts and lists, with no class and no schema. That's less safe (a typo in a key isn't caught when loading) and much less ceremony. Turning a dict into a dataclass is a separate, explicit step, shown below.

**Four functions.** The ones with an `s` work on **s**trings; the ones without work on files:

| | to JSON | from JSON |
|---|---|---|
| string | `json.dumps(obj)` | `json.loads(text)` |
| file | `json.dump(obj, f)` | `json.load(f)` |

```python
config = {
    "model": "mobilenet_v3",
    "lr": 0.001,
    "epochs": 10,
    "pretrained": True,
    "weights": None,
    "layers": [64, 32],
}
text = json.dumps(config)
text           # '{"model": "mobilenet_v3", "lr": 0.001, "epochs": 10, "pretrained": true, "weights": null, "layers": [64, 32]}'
type(text)     # <class 'str'>
print(json.dumps(config, indent=2))
# {
#   "model": "mobilenet_v3",
#   "lr": 0.001,
#   "epochs": 10,
#   "pretrained": true,
#   "weights": null,
#   "layers": [
#     64,
#     32
#   ]
# }
back = json.loads(text)
back == config             # True
back is config             # False
```

`dumps` ("dump string") gives one string of JSON. `True` became `true` and `None` became `null`. `indent=2` adds newlines and two-space indentation for humans; use it for any file a person might read. `loads` parses the text back into a **new** dict, equal to the original but a different object.

**Files.** `json.dump(obj, f)` writes to an open file; `json.load(f)` reads from one:

```python
config_path = work / "config.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
with open(config_path, encoding="utf-8") as f:
    loaded = json.load(f)
loaded["layers"]           # [64, 32]
```

`json.loads(path.read_text(encoding="utf-8"))` and `path.write_text(json.dumps(obj, indent=2), encoding="utf-8")` do the same in one line each, and you'll see both styles.

Day 3b's `first_label` example, now reading from a file. Each kind of failure gets its own handler: a missing file, and a JSON structure without the expected key or with an empty list. Test it on three real files and one missing one:

```python
def first_label(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["labels"][0]
    except FileNotFoundError:
        return "no file"
    except (KeyError, IndexError) as e:
        return f"bad structure: {type(e).__name__} {e}"

(work / "good.json").write_text('{"labels": ["cat", "dog"]}', encoding="utf-8")    # 26
(work / "empty.json").write_text('{"labels": []}', encoding="utf-8")               # 14
(work / "other.json").write_text('{"classes": ["cat"]}', encoding="utf-8")         # 20
first_label(work / "good.json")    # 'cat'
first_label(work / "empty.json")   # 'bad structure: IndexError list index out of range'
first_label(work / "other.json")   # "bad structure: KeyError 'labels'"
first_label(work / "missing.json") # 'no file'
```

**How it works underneath: the type mapping, and what a round trip changes.** `json` converts each Python type to one JSON type, and each JSON type back to one Python type:

| Python | JSON | back to Python |
|---|---|---|
| `dict` | object `{}` | `dict` |
| `list`, `tuple` | array `[]` | `list` |
| `str` | string | `str` |
| `int`, `float` | number | `int` or `float` |
| `True`, `False` | `true`, `false` | `True`, `False` |
| `None` | `null` | `None` |

Two rows lose information. JSON has no tuple, so a tuple comes back as a list. And JSON object keys are always strings, so `dumps` turns `int` (and `float`, `bool`, `None`) keys into strings, and `loads` leaves them as strings:

```python
original = {"input_shape": (1, 224, 224, 3), 0: "cat", 1: "dog"}
restored = json.loads(json.dumps(original))
restored                   # {'input_shape': [1, 224, 224, 3], '0': 'cat', '1': 'dog'}
restored == original       # False
```

<figure class="diagram" markdown>
![Three columns: a Python dict before, the JSON text, and the Python dict after loading. lr 0.001, pretrained True and weights None come back unchanged (True and None are written as true and null in the JSON). The tuple (1, 224, 3) is written as a JSON array and comes back as the list [1, 224, 3], and the integer key 0 is written as the string "0" and comes back as the string '0'; both changed values are highlighted in red](../../assets/img/phase-1/day03c-json-roundtrip.svg)
<figcaption>JSON has no tuples and only string keys, so those two things come back changed after a round trip.</figcaption>
</figure>

The int-key change bites in ML all the time. A label map `{0: "cat", 1: "dog"}` saved as JSON and loaded back has keys `'0'` and `'1'`, and `labels[0]` raises `KeyError: 0`. On disk, a Hugging Face model's `config.json` stores `id2label` exactly like this, with string keys. The `transformers` library converts them back for you when it loads the config, but reading the file with `json` yourself (or from your Android app) gives you the strings. Convert after loading:

```python
id2label = json.loads(json.dumps({0: "cat", 1: "dog", 2: "fox"}))
id2label                                               # {'0': 'cat', '1': 'dog', '2': 'fox'}
id2label = {int(k): v for k, v in id2label.items()}
id2label[0]                                            # 'cat'
```

**How it goes wrong: types JSON can't hold.** `dumps` raises `TypeError` for any type not in the table. Sets, dataclasses, `Path`s, `datetime`s and NumPy numbers are the usual ones:

```python
import numpy as np

for value in [{"cat", "dog"}, np.float32(0.91), Path("data")]:
    try:
        json.dumps({"x": value})
    except TypeError as e:
        print("TypeError:", e)
# TypeError: Object of type set is not JSON serializable
# TypeError: Object of type float32 is not JSON serializable
# TypeError: Object of type PosixPath is not JSON serializable
```

The NumPy one is the classic ML case: `accuracy = (preds == labels).mean()` is a NumPy number, not a Python `float`, and saving `{"accuracy": accuracy}` fails. Two fixes:

```python
results = {"accuracy": np.float32(0.91), "classes": {"cat", "dog"}}

# 1. Convert explicitly before saving (clearest):
clean = {"accuracy": float(results["accuracy"]), "classes": sorted(results["classes"])}
json.dumps(clean)          # '{"accuracy": 0.9100000262260437, "classes": ["cat", "dog"]}'

# 2. Pass default=: a function json calls for any object it can't handle
def to_jsonable(obj):
    if isinstance(obj, np.generic):        # any NumPy scalar
        return obj.item()                  # -> the plain Python number
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"can't save {type(obj).__name__} as JSON")

json.dumps(results, default=to_jsonable)   # '{"accuracy": 0.9100000262260437, "classes": ["cat", "dog"]}'
```

`0.9100000262260437` isn't a bug: `float32` stores 0.91 with only about 7 significant digits, and turning it into a 64-bit Python `float` shows the stored value exactly (Day 4 covers float precision). `obj.item()` converts any NumPy scalar to the matching Python type. Raising `TypeError` at the end of `default` is what `json` expects for anything you still can't handle.

**How it goes wrong: NaN and infinity.** Python's `json` writes `float("nan")` and `float("inf")` as `NaN` and `Infinity`. Those aren't valid JSON, and strict parsers (browsers' `JSON.parse`, Moshi, kotlinx.serialization) reject the file. A training run whose loss blew up is exactly how a `nan` ends up in your results:

```python
json.dumps({"loss": float("nan"), "best": float("inf")})    # '{"loss": NaN, "best": Infinity}'
try:
    json.dumps({"loss": float("nan")}, allow_nan=False)
except ValueError as e:
    print("ValueError:", e)
# ValueError: Out of range float values are not JSON compliant: nan
import math
loss = float("nan")
json.dumps({"loss": None if math.isnan(loss) else loss})    # '{"loss": null}'
```

`allow_nan=False` makes `dumps` refuse instead of writing invalid JSON. The usual fix is to write `None` (JSON `null`) for a missing or invalid number. `math.isnan` (Day 2) checks for NaN, because `nan == nan` is `False`.

**How it goes wrong: text that isn't valid JSON.** `loads` raises `json.JSONDecodeError` (a `ValueError`) with a line and column. The usual causes are Python syntax pasted where JSON was expected:

```python
for bad in ["{'lr': 0.1}", '{"lr": 0.1,}', '{"pretrained": True}', ""]:
    try:
        json.loads(bad)
    except json.JSONDecodeError as e:
        print(f"{bad!r:24} -> {e}")
# "{'lr': 0.1}"            -> Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
# '{"lr": 0.1,}'           -> Illegal trailing comma before end of object: line 1 column 11 (char 10)
# '{"pretrained": True}'   -> Expecting value: line 1 column 16 (char 15)
# ''                       -> Expecting value: line 1 column 1 (char 0)
```

JSON needs double quotes, allows no trailing commas, and spells booleans in lower case. An empty string (an empty file, or a failed download) gives "Expecting value ... char 0". `str(config)` produces Python syntax, not JSON; use `json.dumps(config)`.

**Two more options you'll use.** `ensure_ascii=False` keeps non-ASCII characters readable instead of escaping them, and `sort_keys=True` sorts keys, so two files with the same content are byte-for-byte identical (good for diffs and git):

```python
json.dumps({"label": "café", "id": 3})                                    # '{"label": "caf\\u00e9", "id": 3}'
json.dumps({"label": "café", "id": 3}, ensure_ascii=False, sort_keys=True)  # '{"id": 3, "label": "café"}'
```

(In the first result, `\\u00e9` is how the repr shows the six characters `é`, JSON's escape for `é`. Both forms load back to the same string.)

### A realistic example: a training config as a dataclass

Dataclasses (Day 3a) and JSON fit together: `asdict` turns a dataclass into a dict for saving, and `Config(**d)` builds one back from a loaded dict, with `**` unpacking the dict into keyword arguments (Day 3). Wrap the loading in a function that turns low-level errors into one clear `ConfigError`, the custom exception class from Day 3b:

```python
from dataclasses import dataclass, asdict, field

class ConfigError(Exception):
    def __init__(self, key, message):
        super().__init__(f"{key}: {message}")
        self.key = key


@dataclass
class TrainConfig:
    model: str = "mobilenet_v3"
    lr: float = 0.001
    epochs: int = 10
    layers: list[int] = field(default_factory=lambda: [64, 32])

    def __post_init__(self):
        if isinstance(self.lr, bool) or not isinstance(self.lr, (int, float)):
            raise ConfigError("lr", f"must be a number, got {self.lr!r}")
        if self.lr <= 0:
            raise ConfigError("lr", f"must be positive, got {self.lr}")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise ConfigError("epochs", f"must be an int, got {self.epochs!r}")


def save_config(cfg, path):
    path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def load_config(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TrainConfig(**data)
    except FileNotFoundError as e:
        raise ConfigError(str(path.name), "file not found") from e
    except json.JSONDecodeError as e:
        raise ConfigError(str(path.name), f"not valid JSON ({e})") from e
    except TypeError as e:                 # unknown key: TrainConfig() got an unexpected keyword
        raise ConfigError(str(path.name), str(e)) from e


cfg_path = work / "train_config.json"
save_config(TrainConfig(lr=0.01, epochs=3), cfg_path)
print(cfg_path.read_text(encoding="utf-8"))
# {
#   "model": "mobilenet_v3",
#   "lr": 0.01,
#   "epochs": 3,
#   "layers": [
#     64,
#     32
#   ]
# }
load_config(cfg_path)      # TrainConfig(model='mobilenet_v3', lr=0.01, epochs=3, layers=[64, 32])

broken = {
    "typo.json": '{"learning_rate": 0.01}',
    "cut.json": '{"lr": 0.01, "epo',
    "neg.json": '{"lr": -1}',
    "text.json": '{"lr": "0.01"}',
    "words.json": '{"epochs": "ten"}',
}
for name, content in broken.items():
    (work / name).write_text(content, encoding="utf-8")
for name in [*broken, "missing.json"]:
    try:
        load_config(work / name)
    except ConfigError as e:
        print("ConfigError:", e)
# ConfigError: typo.json: TrainConfig.__init__() got an unexpected keyword argument 'learning_rate'
# ConfigError: cut.json: not valid JSON (Unterminated string starting at: line 1 column 14 (char 13))
# ConfigError: lr: must be positive, got -1
# ConfigError: lr: must be a number, got '0.01'
# ConfigError: epochs: must be an int, got 'ten'
# ConfigError: missing.json: file not found
```

Walk through it:

- `__post_init__` (Day 3a) runs after the generated `__init__`, so a bad value raises `ConfigError` the moment the object is built, wherever it came from.
- The `isinstance` checks are needed because **type hints aren't checked when the code runs** (Day 3): `lr: float` doesn't stop `TrainConfig(lr="0.01")`. Without the first check, `"0.01" <= 0` would raise `TypeError: '<=' not supported between instances of 'str' and 'int'`, which `load_config`'s `except TypeError` (meant for unknown keys) would report as a confusing message. Without the `epochs` check, `"ten"` would load without complaint and fail an hour later inside the training loop. Libraries such as pydantic do this checking for you (see "Going further").
- `load_config` has one `try` with three handlers, one per kind of failure, each turning it into a `ConfigError` that names the file, and chained with `from e` so the original is kept for debugging.
- The `ConfigError` raised inside `__post_init__` isn't one of the three handled types, so it passes straight through `load_config` unchanged.
- `[*broken, "missing.json"]` unpacks the dict's keys into a new list and adds one more name (Day 2b's `*` unpacking).

A caller now has to handle one exception type, with a message that says which file and what's wrong.

### JSON Lines: one record per line

Large ML datasets and prediction logs often use **JSON Lines** (`.jsonl`): one complete JSON object per line, no surrounding brackets. You can append a record without rewriting the file, and read it line by line without loading everything:

```python
preds_path = work / "predictions.jsonl"
records = [{"file": "cat_001.jpg", "label": "cat", "score": 0.97},
           {"file": "dog_001.jpg", "label": "dog", "score": 0.88}]
with open(preds_path, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")
with open(preds_path, "a", encoding="utf-8") as f:
    f.write(json.dumps({"file": "cat_002.jpg", "label": "dog", "score": 0.51}) + "\n")

with open(preds_path, encoding="utf-8") as f:
    loaded = [json.loads(line) for line in f if line.strip()]
len(loaded)                                # 3
[r["label"] for r in loaded]               # ['cat', 'dog', 'dog']
```

`if line.strip()` skips blank lines, such as one left at the end of a file.

## Logging instead of `print`

**What it is.** The `logging` module sends messages through a **logger**, which gives each message a **level** (how serious it is) and passes it to **handlers** (where it goes: the screen, a file) with a **format** (time, level, source). The five standard levels, from least to most serious, are `DEBUG`, `INFO`, `WARNING`, `ERROR` and `CRITICAL`.

**Why it exists.** `print` is fine while you explore. A training script that runs for hours needs more: timestamps, the ability to turn the chatty messages off without deleting them, messages saved to a file, and full tracebacks for the errors it skipped. Every ML library uses `logging`, which is why you can quiet them all with one setting.

**Kotlin comparison.** Android's `Log.d(TAG, "...")`, `Log.i`, `Log.w`, `Log.e`, or Timber. The tag is the logger's name, and the level filters what Logcat shows. Python's logger name plays the role of the `TAG`.

```python
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
logger = logging.getLogger("trainer")

logger.debug("batch 17 shapes: %s", (32, 3, 224, 224))     # below INFO: not shown
logger.info("epoch %d: loss %.3f", 1, 0.9123)
logger.warning("lr %.0e looks high", 0.5)
logger.error("could not save checkpoint to %s", "ckpt/epoch_01.pt")
# INFO trainer: epoch 1: loss 0.912
# WARNING trainer: lr 5e-01 looks high
# ERROR trainer: could not save checkpoint to ckpt/epoch_01.pt
```

Line by line:

- `logging.basicConfig(...)` sets up the **root logger**, the parent of all loggers: `level=logging.INFO` shows `INFO` and above, and `format` says what each line looks like. `%(levelname)s`, `%(name)s` and `%(message)s` are filled in for each message. `force=True` replaces any setup that already exists (explained below).
- `logging.getLogger("trainer")` gets (or creates) the logger with that name. In a module, write `logger = logging.getLogger(__name__)`, so the name is the module's import path (`mlkit.train`) and you can see where every message came from.
- `logger.info("epoch %d: loss %.3f", 1, 0.9123)` passes the format string and the values **separately**. `%d` is an integer, `%.3f` a float with 3 decimals, `%s` anything via `str()`, and `%.0e` scientific notation with no decimals: the older `%` formatting, which predates f-strings. Logging uses it so the string is only built if the message will actually be shown. `logger.info(f"epoch {epoch}: loss {loss:.3f}")` also works, but the f-string is built on every call, even when the message is filtered out, and linters flag it. Pass the values separately.
- `debug` printed nothing, because its level is below `INFO`.

**How it works underneath.** A call like `logger.info(...)` goes through three steps:

1. **Level check**: is `INFO` at or above the logger's effective level? If not, stop here. This is cheap, which is why the values are passed separately: no string is built for a skipped message.
2. **Record**: Python builds a `LogRecord` holding the message, the level, the logger's name, the time, the file and line number.
3. **Handlers**: the record goes to the logger's own handlers, then travels up the logger tree to its parents' handlers, ending at the root logger (this is called **propagation**). The parents' *levels* aren't checked again on the way; each handler on the way checks its *own* level, formats the record with its **formatter**, and writes it out. `basicConfig` added one handler that writes to the screen (the standard error stream, `sys.stderr`).

<figure class="diagram" markdown>
![A pipeline from left to right: the call logger.info in a logger named trainer.data goes to a level check against the logger's level, where DEBUG messages stop; a passing message becomes a LogRecord with message, level, name, time and line; the record propagates up to the root logger, which has two handlers, a console handler at level INFO and a file handler at level DEBUG, each with its own formatter, writing to the screen and to train.log](../../assets/img/phase-1/day03c-logging-flow.svg)
<figcaption>A message passes the logger's level, becomes a LogRecord, goes to its logger's handlers and up to the root's, and each handler filters and formats it again.</figcaption>
</figure>

So there are two levels to set. `logger.setLevel(...)` decides which messages are created at all; `handler.setLevel(...)` decides which of those one destination writes. Loggers form a tree by name (`mlkit` is the parent of `mlkit.train`), so setting the level of one library's logger quiets that library only:

```python
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").info("found 312 fonts")   # below WARNING: dropped
logging.getLogger("mlkit.train").info("epoch 1 done")
# INFO mlkit.train: epoch 1 done
```

`matplotlib.font_manager` has no level of its own, so it uses its parent's, `WARNING`, and its `INFO` message is dropped. `mlkit.train` has none either, and nor does `mlkit`, so it falls back to the root's `INFO`.

**Logging to a file.** A training run should keep its log after the terminal closes. You can attach your own handlers to a logger, each with its own level and formatter. Here the screen gets `INFO` and above, and a file gets everything:

```python
log = logging.getLogger("trainer")
log.setLevel(logging.DEBUG)                        # create DEBUG messages too
log.propagate = False                              # don't also pass them to the root's handlers

console = logging.StreamHandler()                  # writes to the screen (stderr)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

file_handler = logging.FileHandler(work / "train.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)               # the file gets everything
file_handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)s %(message)s"))

log.addHandler(console)
log.addHandler(file_handler)
log.debug("batch 0: 32 images")
log.info("epoch 1: loss %.3f", 0.912)
# INFO: epoch 1: loss 0.912

for handler in (console, file_handler):            # detach them when done
    log.removeHandler(handler)
    handler.close()
log.propagate = True
print((work / "train.log").read_text(encoding="utf-8"), end="")
# DEBUG   trainer batch 0: 32 images
# INFO    trainer epoch 1: loss 0.912
```

Line by line:

- `log.setLevel(logging.DEBUG)` lets the logger create `DEBUG` records at all; then each handler filters with its own `setLevel`. The screen showed one line and the file got two.
- `logging.Formatter(...)` sets each handler's format. `%(levelname)-7s` pads the level name to 7 characters, left-aligned.
- `log.propagate = False` stops records travelling up to the root logger. Without it, the root's handler from the earlier `basicConfig` would print every record a second time, `DEBUG` included: the logger's level is only checked once, where the message is created, not again on the way up.
- The loop at the end detaches and closes both handlers (and restores propagation), so later blocks on this page start clean. A script that runs once can leave that to the end of the program.

When a file is the only destination, `logging.basicConfig(filename="train.log", level=logging.INFO, format=...)` does all of this in one line.

**Logging an exception with its traceback.** Inside an `except` block, `logger.exception("message")` logs at `ERROR` level **and** appends the full traceback. It's how Day 3b's "catch broadly, record, carry on" should look: a loop over independent items, where one bad item mustn't stop the rest. Here three prediction records arrive as JSON text: one good, one cut off, one without a `label`:

```python
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", force=True)
logger = logging.getLogger("summarize")

texts = {
    "a.json": '{"label": "cat", "score": 0.91}',
    "b.json": '{"label": "dog", "score": 0.8',
    "c.json": '{"score": 0.5}',
}
labels, failed = [], 0
for name, text in texts.items():
    try:
        record = json.loads(text)
        labels.append(record["label"])
    except json.JSONDecodeError as e:          # expected: one line is enough
        logger.warning("%s is not valid JSON: %s", name, e)
        failed += 1
    except Exception:                          # unexpected: keep the traceback
        logger.exception("unexpected problem with %s", name)
        failed += 1
    else:
        logger.info("%s: %s", name, record["label"])
print(labels, failed)
# INFO summarize: a.json: cat
# WARNING summarize: b.json is not valid JSON: Expecting ',' delimiter: line 1 column 30 (char 29)
# ERROR summarize: unexpected problem with c.json
# Traceback (most recent call last):
#   File "<cell>", line 13, in <module>
# KeyError: 'label'
# ['cat'] 2
```

The two handlers make a deliberate split. A `JSONDecodeError` is an *expected* problem with the data: one `warning` line with the message says everything. Anything else is *unexpected*, so the broad `except Exception` keeps the loop going but logs the full traceback with `logger.exception`, which here shows the `KeyError` and the line it came from (`<cell>` is the name the ▶ Run button gives each block). Without that handler, the record with no `label` would have stopped the whole loop.

The same pattern as a script on your computer, reading real files. With these three files in a folder:

```text
$ cat a.json b.json c.json
{"label": "cat", "score": 0.91}
{"label": "dog", "score": 0.8
{"score": 0.5}
```

```python
# Run this on your computer (or in Colab), not in the browser.
# summarize.py
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    for path in sorted(Path(".").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            label = record["label"]
        except Exception:
            logger.exception("skipping %s", path)
            continue
        logger.info("%s: %s", path, label)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
```

```text
$ python summarize.py
2026-10-26 19:08:37,656 INFO __main__: a.json: cat
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
2026-10-26 19:08:37,658 ERROR __main__: skipping c.json
Traceback (most recent call last):
  File "/home/you/my_project/summarize.py", line 12, in main
    label = record["label"]
            ~~~~~~^^^^^^^^^
KeyError: 'label'
```

The script uses one broad handler, which is simpler and acceptable here because every failure is logged with its traceback and the loop carries on. The `JSONDecodeError` points at line 2, column 1: the file's text ended (after its final newline) where JSON still expected a `,` or a `}`.

(The library frames, inside `json`, are the kind Day 3b's "Reading a traceback" tells you to skip.) `%(asctime)s` adds the date and time. `continue` (Day 2) skips to the next file. `basicConfig` goes under `if __name__ == "__main__":` (Day 3a): configuring logging is the job of the program that runs, not of modules that get imported. A library that calls `basicConfig` takes that decision away from everyone who imports it.

!!! warning "Common mistake: \"my log messages don't appear\""
    Two causes, both silent:

    - **No configuration.** Without `basicConfig`, the default level is `WARNING`, so `logger.info(...)` shows nothing at all.
    - **A second `basicConfig` is ignored.** `basicConfig` does nothing if the root logger already has a handler. In Jupyter, where a library may already have set one up, or when you re-run a cell with a different level, your new settings have no effect. `force=True` removes the existing handlers first.

    ```python
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s", force=True)
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")   # ignored: already configured
    logging.getLogger("demo").info("you won't see this")
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s", force=True)
    logging.getLogger("demo").info("now you will")
    # INFO: now you will
    ```

    On this page, every block that logs calls `basicConfig(..., force=True)` itself, because the ▶ Run button gives each block its own fresh output, and a handler set up in an earlier block would write to the old one.

**When to use which.** `print` for a notebook you're exploring and for a script's real output (the thing the user asked for). `logging` for progress, warnings and errors in anything that runs unattended: training scripts, data pipelines, servers. `logger.exception` in every broad `except`.

## Testing basics: `assert` and pytest

**What it is.** **pytest** is the standard testing tool for Python. A test is a function whose name starts with `test_`, in a file named `test_*.py` (or `*_test.py`). It checks results with a plain `assert`. You run `pytest` in a terminal; it finds the test files, runs every test function, and reports which failed and why.

**Why it exists.** Python has no compiler to catch type errors, a wrong argument order, or a typo in a branch that rarely runs. Tests are how you find those before your two-hour training run does. In ML, test the small, deterministic pieces: metrics, data cleaning, shape handling, config loading. A wrong metric that runs without error is the worst kind of bug, because every number after it looks plausible.

**Kotlin comparison.**

```kotlin
class MetricsTest {
    @Test fun accuracyTwoOfThree() {
        assertEquals(0.667, accuracy(listOf(1, 0, 0), listOf(1, 0, 1)), 1e-3)
    }
    @Test fun rejectsLengthMismatch() {
        assertThrows<IllegalArgumentException> { accuracy(listOf(1, 0), listOf(1, 0, 1)) }
    }
}
```

pytest needs no class, no annotation and no `assertEquals`: a function and `assert`. `assertEquals` with a delta becomes `pytest.approx`; `assertThrows` becomes `with pytest.raises(...)`.

The code under test, `metrics.py`:

```python
# Run this on your computer (or in Colab), not in the browser.
# metrics.py
def accuracy(preds, labels):
    if len(preds) != len(labels):
        raise ValueError(f"length mismatch: {len(preds)} preds, {len(labels)} labels")
    if not labels:
        raise ValueError("no labels")
    correct = sum(p == y for p, y in zip(preds, labels))
    return correct / len(labels)


def mean(values):
    return sum(values) / len(values)
```

The tests, in `test_metrics.py` next to it:

```python
# Run this on your computer (or in Colab), not in the browser.
# test_metrics.py
import pytest

from metrics import accuracy, mean


def test_accuracy_all_correct():
    assert accuracy([1, 0, 1], [1, 0, 1]) == 1.0


def test_accuracy_two_of_three():
    assert accuracy([1, 0, 0], [1, 0, 1]) == pytest.approx(0.667, abs=1e-3)


def test_mean_of_floats():
    assert mean([0.1, 0.2]) == pytest.approx(0.15)


def test_accuracy_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        accuracy([1, 0], [1, 0, 1])
```

Install and run it inside your virtual environment (Day 1). `-q` ("quiet") prints one character per test, `.` for a pass:

```text
$ pip install pytest
$ pytest -q
....                                                                     [100%]
4 passed in 0.01s
```

A few options you'll use daily. `-v` ("verbose") prints one line per test; `-k accuracy` runs only the tests whose names contain `accuracy`; `-x` stops at the first failure; and a **node id**, `file::test_name`, runs one test:

```text
$ pytest -v
...
test_metrics.py::test_accuracy_all_correct PASSED                        [ 25%]
test_metrics.py::test_accuracy_two_of_three PASSED                       [ 50%]
test_metrics.py::test_mean_of_floats PASSED                              [ 75%]
test_metrics.py::test_accuracy_rejects_length_mismatch PASSED            [100%]

============================== 4 passed in 0.01s ===============================
$ pytest -q -k accuracy
...                                                                      [100%]
3 passed, 1 deselected in 0.01s
$ pytest -q test_metrics.py::test_mean_of_floats
.                                                                        [100%]
1 passed in 0.01s
```

(The `...` replaces `-v`'s header lines, which list your platform, versions and folder.)

- **`pytest.approx(x)`** makes a value that compares equal to anything within a small tolerance of `x` (by default, a relative difference of one part in a million). Floats need it: `0.1 + 0.2` isn't exactly `0.3` (Day 2). `abs=1e-3` sets an absolute tolerance instead, useful when you compare against a rounded number like `0.667`.
- **`pytest.raises(ValueError, match="...")`** is a context manager: the test passes only if the block raises that exception type and its message contains a match for the text (`match` is a regular expression; plain words match themselves). If nothing is raised, the test fails with `DID NOT RAISE`.

**How it works underneath: assertion rewriting.** A plain `assert a == b` that fails only says `AssertionError`. pytest rewrites the `assert` statements in test files as it imports them, so a failure shows the values involved. Here's a test that compares floats with `==`:

```python
# Run this on your computer (or in Colab), not in the browser.
# test_broken.py
from metrics import mean


def test_mean_exact():
    assert mean([0.1, 0.2]) == 0.15
```

```text
$ pytest -q
F....                                                                    [100%]
=================================== FAILURES ===================================
_______________________________ test_mean_exact ________________________________

    def test_mean_exact():
>       assert mean([0.1, 0.2]) == 0.15
E       assert 0.15000000000000002 == 0.15
E        +  where 0.15000000000000002 = mean([0.1, 0.2])

test_broken.py:5: AssertionError
=========================== short test summary info ============================
FAILED test_broken.py::test_mean_exact - assert 0.15000000000000002 == 0.15
1 failed, 4 passed in 0.02s
```

Read it like a traceback: the `>` marks the failing line, the `E` lines show what each side was (`0.15000000000000002`, computed by `mean([0.1, 0.2])`), and the summary names the file, test and line. The fix is `== pytest.approx(0.15)`, as in `test_mean_of_floats`.

**How it goes wrong.** The usual first-time problems:

- **pytest finds no tests** (`no tests ran`): the file or function name doesn't start with `test_`.
- **`ModuleNotFoundError: No module named 'metrics'`** as soon as you move the tests into their own folder, the usual layout. With `metrics.py` in the project root and `tests/test_metrics.py`, plain `pytest` fails while collecting:

    ```text
    $ pytest -q
    ...
    tests/test_metrics.py:3: in <module>
        from metrics import accuracy, mean
    E   ModuleNotFoundError: No module named 'metrics'
    =========================== short test summary info ============================
    ERROR tests/test_metrics.py
    !!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
    1 error in 0.45s
    $ python -m pytest -q
    ....                                                                     [100%]
    4 passed in 0.01s
    ```

    pytest puts the *test file's* folder (`tests/`) on `sys.path` (Day 3a), not the project root, so `metrics` isn't found. `python -m pytest` also adds the current folder, the project root, which fixes it, the same `-m` rule as Day 3a's packages. The permanent fix is a `pytest.ini` file in the project root containing `[pytest]` and `pythonpath = .` on two lines.
- **A test that can't fail**, such as `assert (result, 0.5)` with brackets (the always-true tuple from Day 3b's "Assertions" section) or a test that never calls the code. Check every new test fails once, by breaking the code on purpose.

**In the browser.** pytest isn't available on this page, but the test functions are ordinary functions, and a test function is a function that raises when something's wrong. You can call them by hand to check the logic. Here is a hand-made version of `approx`, to show there's no magic:

```python
def approx_equal(a, b, rel=1e-6):
    return abs(a - b) <= rel * abs(b)

def mean(values):
    return sum(values) / len(values)

def test_mean_of_floats():
    assert approx_equal(mean([0.1, 0.2]), 0.15)

def test_mean_exact():
    assert mean([0.1, 0.2]) == 0.15

for test in [test_mean_of_floats, test_mean_exact]:
    try:
        test()
        print("PASS", test.__name__)
    except AssertionError:
        print("FAIL", test.__name__)
# PASS test_mean_of_floats
# FAIL test_mean_exact
```

That loop is, in miniature, what pytest does: find the test functions, call each one, and report which raised.

---

## Going further

These are real and worth knowing exist, but aren't needed yet:

- **`contextlib`** beyond `contextmanager` and `suppress`, such as `ExitStack` for a variable number of resources: [contextlib docs](https://docs.python.org/3/library/contextlib.html).
- The **`csv`** module's dialects and options (tabs, other quote characters): [csv docs](https://docs.python.org/3/library/csv.html).
- **TOML** config files with the built-in `tomllib`: [tomllib docs](https://docs.python.org/3/library/tomllib.html). YAML needs the third-party PyYAML package.
- **`pickle`**, Python's own object serialisation (and why you must never unpickle files from untrusted sources): [pickle docs](https://docs.python.org/3/library/pickle.html). NumPy's `np.save` comes in Day 4b.
- **`io.StringIO`**, an in-memory text file, handy in tests: [io docs](https://docs.python.org/3/library/io.html).
- **`shutil`** beyond `copy` and `rmtree`: `move`, `copytree` and archives: [shutil docs](https://docs.python.org/3/library/shutil.html).
- **pydantic**, a third-party library that checks a config's types and values from its type hints, the job `__post_init__` did by hand today: [pydantic docs](https://docs.pydantic.dev/latest/).
- **Logging** handlers, file logging and configuration: [Logging cookbook](https://docs.python.org/3/howto/logging-cookbook.html).
- **pytest** fixtures in general (Exercise 3's solution uses two features, `tmp_path` and `@pytest.mark.parametrize`), `conftest.py`, and coverage: [pytest how-to guides](https://docs.pytest.org/en/stable/how-to/index.html).
- **`unittest`**, the standard library's JUnit-style test framework, which pytest can also run: [unittest docs](https://docs.python.org/3/library/unittest.html).

---

## Exercises

**Warm-ups**

1. **Write and read lines.** In a temporary folder, write the labels `["cat", "dog", "fox"]` to `labels.txt`, one per line. Read the file back into a list of clean strings two ways: with `read_text().splitlines()`, and with a `for` loop over an open file. Then append `"owl"` without rewriting the file, and read it again.
2. **A dataset inventory.** In a temporary folder, create `data/train/` and `data/val/` with `mkdir(parents=True, exist_ok=True)`, and write these empty files: `train/cat_1.jpg`, `train/cat_2.jpg`, `train/dog_1.png`, `train/readme.txt`, `val/cat_3.jpg` and `val/dog_2.jpg`. Using only `pathlib`, print: the sorted names of all image files (`.jpg` or `.png`) in every subfolder; the number of files per suffix (use `Counter`); and the sorted labels found in the images, where the label is the part of the stem before the `_`.

**Core**

3. **A config loader, with tests.** Write `load_settings(path)` that reads a JSON file and returns a dict with these defaults filled in for any missing key: `{"lr": 0.001, "epochs": 10, "batch_size": 32}`. Raise a `SettingsError` (your own class) chained with `from` when the file doesn't exist, isn't valid JSON, or has a `batch_size` that isn't a positive `int`. Accept the path as a `str` or a `Path`. Try it in the browser on a good file, a partial file, a file with `"batch_size": 0`, a broken file and a missing one. Then, on your computer, put it in `settings.py` and write `tests/test_settings.py` with pytest tests for: defaults filled in, `lr` read correctly (with `pytest.approx`), bad batch sizes rejected (with `pytest.raises` and `match`), broken JSON keeping its cause, and a missing file. Run them with `python -m pytest -v`.
4. **CSV to JSON, with logging.** Write the CSV below to a file. Read it with `csv.DictReader`, skip rows whose rating isn't a number, and compute the average rating and the app count per category. Instead of `print`, use a logger: `DEBUG` for each good row, `WARNING` for each skipped row (with the app name and the bad value), and `INFO` for the summary. Save the result to `summary.json` with `indent=2` and `sort_keys=True` and read it back. Run it once at `INFO` level and once at `DEBUG`, changing one setting.

    ```text
    name,category,rating
    Snapseed,PHOTOGRAPHY,4.5
    Lens,PHOTOGRAPHY,4.6
    Keep,PRODUCTIVITY,4.4
    "Notes, Lists & Reminders",PRODUCTIVITY,N/A
    Files,PRODUCTIVITY,4.6
    ```

5. **A context manager two ways.** Write a context manager `working_file(path)` that prints `open <name>` on entry, gives the caller an open text file for writing, and on exit closes the file and prints `closed <name> (ok)` or `closed <name> (failed: <ExceptionType>)`, then lets any exception carry on. Write it once as a class and once with `@contextmanager`, and test both with a body that succeeds and one that raises `ValueError`.

**Predict the output**

6. Write down what each `print` shows **before** running it.

    ```python
    import json, tempfile
    from pathlib import Path

    data = {"shape": (2, 3), 1: "one", "ok": True, "none": None}
    text = json.dumps(data)
    print(text)
    back = json.loads(text)
    print(back["shape"] == [2, 3], back == data)
    print(1 in back, "1" in back)

    path = Path(tempfile.mkdtemp()) / "log.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("a\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("b\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write("c")
    print(path.read_text(encoding="utf-8").splitlines())
    with open(path, encoding="utf-8") as f:
        print(f.readlines(), f.read() == "")
    ```

**Find and fix the bug**

7. This code saves a label map and loads it back for an app. It prints `{}` instead of `cat`. Find the two bugs, explain why the output is `{}` and not an error message, and fix both.

    ```python
    import json, tempfile
    from pathlib import Path

    def load_label(path, index):
        try:
            with open(path, encoding="utf-8") as f:
                id2label = json.load(f)
            return id2label[index]
        except:
            return {}

    path = Path(tempfile.mkdtemp()) / "labels.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({0: "cat", 1: "dog"}, f)
    print(load_label(path, 0))       # expected: cat
    ```

**Challenge**

8. **An atomic save.** A training script that crashes while writing `checkpoint.json` leaves a half-written file, and the next run fails to load it. Write a `@contextmanager` called `atomic_write(path)` that gives the caller a file to write, but writes to a temporary file `<name>.tmp` in the same folder; if the body succeeds, it moves the temporary file over `path` with `Path.replace` (an atomic rename on the same file system); if the body raises, it deletes the temporary file and re-raises, leaving any existing `path` untouched. Show that an existing good file survives a failed save, that no `.tmp` file is left behind, and that a successful save replaces the file.

??? success "Solutions"
    **1. Write and read lines.**

    ```python
    import tempfile
    from pathlib import Path

    folder = Path(tempfile.mkdtemp())
    path = folder / "labels.txt"
    labels = ["cat", "dog", "fox"]
    path.write_text("\n".join(labels) + "\n", encoding="utf-8")    # 12

    print(path.read_text(encoding="utf-8").splitlines())     # ['cat', 'dog', 'fox']

    clean = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            clean.append(line.strip())
    print(clean)                                             # ['cat', 'dog', 'fox']

    with open(path, "a", encoding="utf-8") as f:
        f.write("owl\n")
    print(path.read_text(encoding="utf-8").splitlines())     # ['cat', 'dog', 'fox', 'owl']
    ```

    `"\n".join(labels) + "\n"` puts a newline between items and one at the end, so the file ends with a complete line and the append starts on a fresh one. Without that final `"\n"`, the appended text would join the last line (`foxowl`). `splitlines()` and `strip()` remove the newlines that `for line in f` keeps. Mode `"a"` adds to the end; `"w"` would have replaced the three labels.

    **2. A dataset inventory.**

    ```python
    import tempfile
    from collections import Counter
    from pathlib import Path

    root = Path(tempfile.mkdtemp()) / "data"
    files = ["train/cat_1.jpg", "train/cat_2.jpg", "train/dog_1.png", "train/readme.txt",
             "val/cat_3.jpg", "val/dog_2.jpg"]
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    images = sorted(p for p in root.rglob("*") if p.suffix in {".jpg", ".png"})
    print([p.relative_to(root).as_posix() for p in images])
    print(sorted(Counter(p.suffix for p in root.rglob("*") if p.is_file()).items()))
    print(sorted({p.stem.split("_")[0] for p in images}))
    # ['train/cat_1.jpg', 'train/cat_2.jpg', 'train/dog_1.png', 'val/cat_3.jpg', 'val/dog_2.jpg']
    # [('.jpg', 4), ('.png', 1), ('.txt', 1)]
    # ['cat', 'dog']
    ```

    `path.parent.mkdir(parents=True, exist_ok=True)` creates `data/train/` or `data/val/` the first time and does nothing after that. `rglob("*")` yields every file *and* folder under `root`, so the suffix test (a set, Day 2b) keeps only images, and `is_file()` keeps folders out of the count. `Counter` lists equal counts in the order it first saw them, which depends on the file system, so `sorted(...items())` gives a fixed order of `(suffix, count)` pairs. Sorting `Path` objects sorts them by their path text, so the list is in the same order on every machine. `p.stem.split("_")[0]` takes `"cat"` from `"cat_1"`, and the set comprehension removes duplicates before sorting. This "label from the file name" pattern is how many image datasets are organised.

    **3. A config loader with defaults.**

    ```python
    import json, tempfile
    from pathlib import Path

    class SettingsError(Exception):
        pass

    DEFAULTS = {"lr": 0.001, "epochs": 10, "batch_size": 32}

    def load_settings(path):
        path = Path(path)                  # accept a str or a Path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as e:
            raise SettingsError(f"{path.name}: file not found") from e
        except json.JSONDecodeError as e:
            raise SettingsError(f"{path.name}: invalid JSON: {e}") from e
        settings = DEFAULTS | data
        bs = settings["batch_size"]
        if not isinstance(bs, int) or isinstance(bs, bool) or bs <= 0:
            raise SettingsError(f"{path.name}: batch_size must be a positive int, got {bs!r}")
        return settings

    folder = Path(tempfile.mkdtemp())
    files = {
        "good.json": '{"lr": 0.01, "epochs": 3, "batch_size": 64}',
        "partial.json": '{"epochs": 5}',
        "zero.json": '{"batch_size": 0}',
        "broken.json": '{"lr": 0.01',
    }
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")

    for name in [*files, "missing.json"]:
        try:
            print(name, "->", load_settings(folder / name))
        except SettingsError as e:
            print(name, "-> SettingsError:", e, "| cause:", type(e.__cause__).__name__)
    # good.json -> {'lr': 0.01, 'epochs': 3, 'batch_size': 64}
    # partial.json -> {'lr': 0.001, 'epochs': 5, 'batch_size': 32}
    # zero.json -> SettingsError: zero.json: batch_size must be a positive int, got 0 | cause: NoneType
    # broken.json -> SettingsError: broken.json: invalid JSON: Expecting ',' delimiter: line 1 column 12 (char 11) | cause: JSONDecodeError
    # missing.json -> SettingsError: missing.json: file not found | cause: FileNotFoundError
    ```

    `path = Path(path)` first, so `path.name` works whether the caller passed a string or a `Path`. `DEFAULTS | data` (Day 2b's dict merge) takes every default and overrides it with what the file has, so missing keys get defaults. The `try` covers only the reading and parsing; validation comes after it, so a validation error isn't mistaken for a file error. The bad-`batch_size` error has no underlying exception, so its `__cause__` is `None`, while the other two keep the original error as their cause. `isinstance(bs, bool)` is checked because `True` is an `int` in Python (Day 2), and `"batch_size": true` shouldn't count as a batch size of 1.

    The tests, on your computer, with the function above (and its imports, `SettingsError` and `DEFAULTS`) saved as `settings.py` in the project root:

    ```python
    # Run this on your computer (or in Colab), not in the browser.
    # tests/test_settings.py
    import json

    import pytest

    from settings import SettingsError, load_settings


    def write(path, data):
        path.write_text(json.dumps(data), encoding="utf-8")
        return path


    def test_missing_keys_get_defaults(tmp_path):
        path = write(tmp_path / "s.json", {"epochs": 5})
        assert load_settings(path) == {"lr": 0.001, "epochs": 5, "batch_size": 32}


    def test_lr_is_read_as_a_float(tmp_path):
        path = write(tmp_path / "s.json", {"lr": 0.1})
        assert load_settings(path)["lr"] == pytest.approx(0.1)


    @pytest.mark.parametrize("bad", [0, -4, 2.5, True, "32"])
    def test_bad_batch_size_is_rejected(tmp_path, bad):
        path = write(tmp_path / "s.json", {"batch_size": bad})
        with pytest.raises(SettingsError, match="batch_size must be a positive int"):
            load_settings(path)


    def test_broken_json_keeps_the_cause(tmp_path):
        path = tmp_path / "s.json"
        path.write_text('{"lr": 0.01', encoding="utf-8")
        with pytest.raises(SettingsError) as info:
            load_settings(path)
        assert isinstance(info.value.__cause__, json.JSONDecodeError)


    def test_missing_file(tmp_path):
        with pytest.raises(SettingsError, match="file not found"):
            load_settings(tmp_path / "nope.json")
    ```

    ```text
    $ python -m pytest -v
    ...
    tests/test_settings.py::test_missing_keys_get_defaults PASSED            [ 11%]
    tests/test_settings.py::test_lr_is_read_as_a_float PASSED                [ 22%]
    tests/test_settings.py::test_bad_batch_size_is_rejected[0] PASSED        [ 33%]
    tests/test_settings.py::test_bad_batch_size_is_rejected[-4] PASSED       [ 44%]
    tests/test_settings.py::test_bad_batch_size_is_rejected[2.5] PASSED      [ 55%]
    tests/test_settings.py::test_bad_batch_size_is_rejected[True] PASSED     [ 66%]
    tests/test_settings.py::test_bad_batch_size_is_rejected[32] PASSED       [ 77%]
    tests/test_settings.py::test_broken_json_keeps_the_cause PASSED          [ 88%]
    tests/test_settings.py::test_missing_file PASSED                         [100%]

    ============================== 9 passed in 0.03s ===============================
    ```

    Two pytest features appear here for the first time. A test function with a parameter named `tmp_path` gets a fresh, empty temporary folder (a `Path`) from pytest for each test, so tests never see each other's files. `@pytest.mark.parametrize("bad", [...])` runs the same test once per value, and each run gets its own line and id (`[0]`, `[-4]`, ...), so one failing value is reported on its own. `pytest.raises(...) as info` keeps the caught exception in `info.value`, so the test can check its `__cause__`. `python -m pytest` is used because the tests live in `tests/` (see "How it goes wrong" in the pytest section).

    **4. CSV to JSON, with logging.**

    ```python
    import csv, json, logging, tempfile
    from collections import defaultdict
    from pathlib import Path

    logger = logging.getLogger("summary")

    def summarize_csv(csv_path, out_path):
        ratings = defaultdict(list)
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rating = float(row["rating"])
                except ValueError:
                    logger.warning("skipping %s: rating %r is not a number", row["name"], row["rating"])
                    continue
                ratings[row["category"]].append(rating)
                logger.debug("%s: %s %.1f", row["name"], row["category"], rating)
        summary = {
            cat: {"apps": len(vals), "mean_rating": round(sum(vals) / len(vals), 2)}
            for cat, vals in ratings.items()
        }
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("%d categories written to %s", len(summary), out_path.name)
        return json.loads(out_path.read_text(encoding="utf-8"))

    folder = Path(tempfile.mkdtemp())
    csv_path = folder / "apps.csv"
    csv_path.write_text(
        "name,category,rating\n"
        "Snapseed,PHOTOGRAPHY,4.5\n"
        "Lens,PHOTOGRAPHY,4.6\n"
        "Keep,PRODUCTIVITY,4.4\n"
        '"Notes, Lists & Reminders",PRODUCTIVITY,N/A\n'
        "Files,PRODUCTIVITY,4.6\n",
        encoding="utf-8",
    )                                     # 156: characters written

    fmt = "%(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, force=True)
    print(summarize_csv(csv_path, folder / "summary.json"))
    # WARNING summary: skipping Notes, Lists & Reminders: rating 'N/A' is not a number
    # INFO summary: 2 categories written to summary.json
    # {'PHOTOGRAPHY': {'apps': 2, 'mean_rating': 4.55}, 'PRODUCTIVITY': {'apps': 2, 'mean_rating': 4.5}}

    logging.basicConfig(level=logging.DEBUG, format=fmt, force=True)    # the one setting
    summary = summarize_csv(csv_path, folder / "summary.json")
    # DEBUG summary: Snapseed: PHOTOGRAPHY 4.5
    # DEBUG summary: Lens: PHOTOGRAPHY 4.6
    # DEBUG summary: Keep: PRODUCTIVITY 4.4
    # WARNING summary: skipping Notes, Lists & Reminders: rating 'N/A' is not a number
    # DEBUG summary: Files: PRODUCTIVITY 4.6
    # INFO summary: 2 categories written to summary.json
    ```

    `DictReader` handles the quoted name with a comma in it, so the "Notes" row has the right three fields, and its `"N/A"` rating raises `ValueError` in `float`, which is logged and skipped with `continue`. The `try` holds only the conversion. `defaultdict(list)` (Day 2b) creates each category's list on first use. The function never decides *whether* messages are shown: it only says how important each one is. The caller's `basicConfig` level decides, so switching to `DEBUG` shows every row without touching the function. `force=True` is needed for the second call, because `basicConfig` otherwise ignores a second configuration. `%r` in a logging format string works like `{...!r}` in an f-string: it shows the value with its quotes.

    **5. A context manager two ways.**

    ```python
    import tempfile
    from contextlib import contextmanager
    from pathlib import Path

    class WorkingFile:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            print("open", self.path.name)
            self.f = open(self.path, "w", encoding="utf-8")
            return self.f

        def __exit__(self, exc_type, exc, tb):
            self.f.close()
            status = "ok" if exc_type is None else f"failed: {exc_type.__name__}"
            print(f"closed {self.path.name} ({status})")
            return False                      # let the exception carry on

    @contextmanager
    def working_file(path):
        print("open", path.name)
        f = open(path, "w", encoding="utf-8")
        try:
            yield f
        except BaseException as e:
            f.close()
            print(f"closed {path.name} (failed: {type(e).__name__})")
            raise
        else:
            f.close()
            print(f"closed {path.name} (ok)")

    folder = Path(tempfile.mkdtemp())
    for manager in (WorkingFile, working_file):
        with manager(folder / "a.txt") as f:
            f.write("fine\n")
        try:
            with manager(folder / "b.txt") as f:
                f.write("partial\n")
                raise ValueError("bad batch")
        except ValueError as e:
            print("caller caught:", e)
    # open a.txt
    # closed a.txt (ok)
    # open b.txt
    # closed b.txt (failed: ValueError)
    # caller caught: bad batch
    # open a.txt
    # closed a.txt (ok)
    # open b.txt
    # closed b.txt (failed: ValueError)
    # caller caught: bad batch
    ```

    The class receives the exception's type in `__exit__` and returns `False` so it isn't swallowed. In the generator version, an exception in the `with` body appears at the `yield`, so a `try` around the `yield` sees it; `except BaseException` plus a bare `raise` closes the file, reports and re-raises (catching `BaseException` is right here, because nothing is swallowed: even Ctrl+C should close the file and then carry on). `else` runs only when the body succeeded. Both versions print the same lines, because they are the same protocol.

    **6. Predict the output.**

    ```python
    import json, tempfile
    from pathlib import Path

    data = {"shape": (2, 3), 1: "one", "ok": True, "none": None}
    text = json.dumps(data)
    print(text)
    back = json.loads(text)
    print(back["shape"] == [2, 3], back == data)
    print(1 in back, "1" in back)

    path = Path(tempfile.mkdtemp()) / "log.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("a\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("b\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write("c")
    print(path.read_text(encoding="utf-8").splitlines())
    with open(path, encoding="utf-8") as f:
        print(f.readlines(), f.read() == "")
    # {"shape": [2, 3], "1": "one", "ok": true, "none": null}
    # True False
    # False True
    # ['b', 'c']
    # ['b\n', 'c'] True
    ```

    `dumps` writes the tuple as an array, the int key as the string `"1"`, `True` as `true` and `None` as `null`. Loading gives a list and a string key back, so `back["shape"] == [2, 3]` is `True`, but `back == data` is `False`: `(2, 3) != [2, 3]`, and the key `1` became `"1"`. For the file, the second `"w"` empties it before writing `b`, and `"a"` adds `c` with no newline. `readlines()` keeps each line's `\n`, and the last line has none. After `readlines()` the file position is at the end, so `f.read()` returns `''`.

    **7. Find and fix the bug.**

    ```python
    import json, tempfile
    from pathlib import Path

    def load_label(path, index):
        try:
            with open(path, encoding="utf-8") as f:
                id2label = json.load(f)
            return id2label[index]
        except:
            return {}

    path = Path(tempfile.mkdtemp()) / "labels.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({0: "cat", 1: "dog"}, f)
    print(load_label(path, 0))            # {}
    print(path.read_text(encoding="utf-8"))   # {"0": "cat", "1": "dog"}

    def load_label(path, index):
        with open(path, encoding="utf-8") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        try:
            return id2label[index]
        except KeyError:
            raise KeyError(f"no label for index {index}; known: {sorted(id2label)}") from None

    print(load_label(path, 0))            # cat
    try:
        load_label(path, 7)
    except KeyError as e:
        print("KeyError:", e)
    # KeyError: 'no label for index 7; known: [0, 1]'
    ```

    Bug 1: JSON keys are always strings, so the file holds `"0"` and `"1"`, and `id2label[0]` raises `KeyError: 0`. Bug 2: the bare `except:` catches that `KeyError` (and would catch anything else, including a missing file or a typo) and returns `{}`, which is why you see a strange empty dict instead of an error. The fix converts the keys back to `int` after loading, and catches only `KeyError`, re-raising it with a message that helps; a missing or broken file now raises its own clear error. `from None` hides the original `KeyError: 7`, which says nothing the new message doesn't.

    **8. An atomic save.**

    ```python
    import json, os, tempfile
    from contextlib import contextmanager
    from pathlib import Path

    @contextmanager
    def atomic_write(path, encoding="utf-8"):
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        f = open(tmp, "w", encoding=encoding)
        try:
            yield f
        except BaseException:
            f.close()
            tmp.unlink()                  # throw the partial file away
            raise
        else:
            f.flush()                     # hand Python's buffer to the OS
            os.fsync(f.fileno())          # ask the OS to write it to the disk
            f.close()
            tmp.replace(path)             # then swap it in, in one step

    folder = Path(tempfile.mkdtemp())
    ckpt = folder / "checkpoint.json"
    ckpt.write_text(json.dumps({"epoch": 1, "loss": 0.9}), encoding="utf-8")   # 25

    try:
        with atomic_write(ckpt) as f:
            f.write('{"epoch": 2, "lo')       # half a record...
            raise RuntimeError("crash while saving")
    except RuntimeError as e:
        print("save failed:", e)
    print(json.loads(ckpt.read_text(encoding="utf-8")))
    print(sorted(p.name for p in folder.iterdir()))

    with atomic_write(ckpt) as f:
        json.dump({"epoch": 2, "loss": 0.7}, f)
    print(json.loads(ckpt.read_text(encoding="utf-8")))
    print(sorted(p.name for p in folder.iterdir()))
    # save failed: crash while saving
    # {'epoch': 1, 'loss': 0.9}
    # ['checkpoint.json']
    # {'epoch': 2, 'loss': 0.7}
    # ['checkpoint.json']
    ```

    The real file is never opened for writing, so a crash can only damage the `.tmp` file, which the `except` branch deletes before re-raising. On success, `flush` hands the buffered text to the operating system, `os.fsync` asks the operating system to write it to the physical disk (`f.fileno()` is the number the operating system uses for the open file), and only then does `replace` rename the file over the old one. Without the `fsync`, a power cut just after the rename can leave an empty or partial file on some file systems, because the rename reached the disk before the data did. On Linux and macOS, a rename within one file system is atomic (Windows' `os.replace` comes close): another program sees either the old complete file or the new complete file, never half of one. The temporary file goes in the *same folder* because a rename across file systems isn't atomic (and `replace` may fail). Careful checkpointing code does this for exactly this reason.

## Resources

- [Python tutorial §7.2: Reading and writing files](https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files): `open`, modes, file methods, and saving structured data with `json`.
- [pathlib](https://docs.python.org/3/library/pathlib.html): every `Path` method, with a table mapping `os.path` functions to their `pathlib` equivalents.
- [json](https://docs.python.org/3/library/json.html): `dump`, `load`, the conversion table, and options such as `default`, `indent` and `sort_keys`.
- [csv](https://docs.python.org/3/library/csv.html): readers, writers, `DictReader`, `DictWriter`, and why `newline=""` matters.
- [contextlib](https://docs.python.org/3/library/contextlib.html): `contextmanager`, `suppress` and the other context-manager tools.
- [Logging HOWTO](https://docs.python.org/3/howto/logging.html): when to use which level, `basicConfig`, and logging from several modules.
- [pytest: Get started](https://docs.pytest.org/en/stable/getting-started.html): installing pytest, a first test, `pytest.raises`, and how tests are discovered.

## Checklist

- [ ] I can explain `with` through `__enter__` and `__exit__`, and write a context manager as a class and with `@contextmanager`.
- [ ] I open files with the right mode and `encoding="utf-8"`, and I recognise what a wrong mode or encoding does.
- [ ] I can read a text file whole, as a list of lines or line by line, and I know lines keep their `\n`; I can read and write bytes.
- [ ] I can build and inspect paths with `pathlib.Path`, create folders with `mkdir(parents=True, exist_ok=True)`, list files with a sorted `glob`, find files next to my script with `__file__`, and copy and delete with `shutil`.
- [ ] I can read and write CSV with `csv.reader`, `DictReader` and `DictWriter`, and I convert its strings to numbers.
- [ ] I can save and load JSON, I know tuples become lists and keys become strings, and I can fix "not JSON serializable" and `JSONDecodeError`.
- [ ] I can load a JSON config into a dataclass with one clear error type, and read and write JSON Lines.
- [ ] I use `logging` with the right level instead of `print` for scripts that run unattended, can log to a file, and use `logger.exception` in a broad `except`.
- [ ] I can write and run a pytest test with `assert`, `pytest.approx` and `pytest.raises`, and read a failure report.

**Next:** [Day 3d: Iterators and generators](day-03d-iterators-generators.md)
