# Day 2a: Strings, Text and Formatting

**Phase 1 · Week 1 · ~130 minutes**

This is the second of three parts of Day 2. [Day 2](day-02-python-types.md) covered names, numbers, booleans and control flow; [Day 2b](day-02b-collections.md) covers lists, tuples, dicts and sets. Do them on consecutive days.

## Objectives

- [ ] Write string literals with escapes, raw strings and triple quotes, and say why `"C:\new"` is a bug.
- [ ] Explain the difference between `str` and `repr`, and use `print(x)`, `repr(x)` and `{x!r}` deliberately.
- [ ] Index and slice a string, including from the end and with a step, and explain the "fences between characters" model.
- [ ] Use `+`, `*`, `in`, `not in`, `len` and `<` on strings, and say why `"10" < "9"`.
- [ ] Clean, test, search, replace, split and join text with the common `str` methods, including `splitlines` and `partition`.
- [ ] Build a string from many pieces with `"".join(...)` instead of repeated `+=`, and say why.
- [ ] Format values with the f-string format spec: `:.2f`, `:,`, `:.1%`, `:>8`, `:e`, `:03d`, `{x=}` and `!r`.
- [ ] Explain Unicode code points, UTF-8 bytes, `encode` and `decode`, and recognise mojibake, `UnicodeDecodeError` and `UnicodeEncodeError`.
- [ ] Parse a real log file into a formatted report.

**Prerequisites.**

- [Day 1](day-01-setup.md): running Python in the REPL, a script or a notebook, and the ▶ Run button on this site.
- [Day 2](day-02-python-types.md): names and objects, immutability, truthiness, `int()` / `float()` conversion and their `ValueError`, `if`, `for`, `while`, `break`, `continue`, the four basic f-string specs (`{x}`, `:.2f`, `:>8`, `:.1%`), which this lesson completes, and the bits of lists the examples use: a list literal and `.append`, `enumerate`, `zip(..., strict=True)` and unpacking a pair into two names (`a, b = ...`).

---

## The big idea

Almost all the data you will meet arrives as text. A CSV file is text. JSON from an API is text. A `labels.txt` next to a model is text. A log file is text. Column names, file paths, class names and the prompts you send to a language model are text. Before any of it becomes a number or a tensor, someone has to cut, clean and check the text, and that someone is you.

Python's `str` is built for that job, and it works differently from Kotlin's `String` in three ways worth knowing up front:

1. **There is no `Char` type.** A one-character string is an ordinary string of length 1, so the same operations work on one character and on a whole document.
2. **A string is a sequence of Unicode code points, not UTF-16 units.** `len("👍")` is 1 in Python and 2 in Kotlin.
3. **Formatting lives inside the string.** Kotlin templates insert a value (`"$x"`) and leave number formatting to `String.format`. Python's f-strings do both at once, with a small formatting language after a colon: `f"{acc:.1%}"`.

A string is also **immutable**, exactly as in Kotlin: every method returns a *new* string. That one fact explains the most common beginner bug in this lesson, and it is the reason the `join` idiom exists.

!!! info "How this page shows errors"
    Examples that fail on purpose are wrapped in `try` / `except`, so that one failing line doesn't stop the whole ▶ Run block:

    ```python
    try:
        "abc"[5]
    except IndexError as err:
        print("IndexError:", err)
    # IndexError: string index out of range
    ```

    Read it as: "run the indented line; if it raises an `IndexError`, put the error in `err` and run the `except` part instead of stopping". The printed text is the last line of the traceback you'd see without the wrapper. [Day 3b](day-03b-errors-files-json.md) teaches `try` / `except` properly.

## Strings

### Literals, quotes and escapes

**What it is.** A `str` is an immutable sequence of Unicode characters. **Immutable** means it can never be changed after it's created; every operation that looks like a change returns a new string. Single and double quotes mean the same thing. Inside a normal string, a backslash starts an **escape sequence**: `\n` is a newline, `\t` a tab, `\\` a backslash, `\'` and `\"` quotes. Two more name a character by its number: `\xNN` with two hex digits (`"\x41"` is `'A'`), and `\uNNNN` with four (`"\u00e9"` is `'é'`). You'll see `\x` again in bytes, later on this page.

**Kotlin vs Python.** Kotlin uses `"..."` for strings and `'c'` for a `Char`. Python has no character type: `'c'` is a one-character string, and `"abc"[0]` is the string `'a'`.

```python
'it\'s' == "it's"        # True   same string, two spellings
print("col1\tcol2\nrow1")
# col1	col2
# row1
len("\n")                # 1      \n is ONE character
"\u00e9" == "é"          # True   the same character, spelled by its number
"\x41"                   # 'A'
```

**Raw strings** start with `r` and keep backslashes as plain characters. Use them for Windows paths and regular expressions:

```python
print(r"C:\new\table")   # C:\new\table
len(r"\n")               # 2      backslash and n
```

**Multi-line strings** use triple quotes and keep the line breaks:

```python
help_text = """Usage:
  train --epochs 10"""
help_text                # 'Usage:\n  train --epochs 10'
"conv" "net"             # 'convnet'   adjacent literals are joined into one
```

Joining adjacent literals sounds pointless until you need a long string: inside parentheses, you can write it over several lines with no `+`:

```python
message = ("training finished: "
           "see the log for details")
message                  # 'training finished: see the log for details'
```

Two details of triple-quoted strings catch people. A line break right after the opening `"""` becomes part of the string, and inside an indented block, the indentation of each line is kept too. `textwrap.dedent` (Going further) strips the common indentation.

Notice that the echo of `help_text` shows `\n` rather than a line break, and shows the quotes. That's the **repr**, the subject of the next section.

**Kotlin difference.** Kotlin's `"""..."""` strings are *raw*: no escapes are processed. Python's triple-quoted strings are *not* raw; `\n` inside them is still a newline. For a raw multi-line string in Python, combine both: `r"""..."""`.

!!! warning "Common mistake: backslashes in Windows paths"
    In a normal string, `\n` and `\t` inside a path silently become a newline and a tab:

    ```python
    print("C:\new\table")
    # C:
    # ew	able
    ```

    No error, only a wrong path and, later, `FileNotFoundError`. A backslash before a letter that *isn't* an escape, such as `\d`, still works today but warns. Running a script containing `p = "C:\data\new"` prints this before anything else (Python 3.14; 3.12 and 3.13 say `SyntaxWarning: invalid escape sequence '\d'`):

    ```text
    SyntaxWarning: "\d" is an invalid escape sequence. Such sequences will not work in the future. Did you mean "\\d"? A raw string is also an option.
    ```

    **Fix:** use a raw string `r"C:\new\table"`, or forward slashes `"C:/new/table"`, which Windows accepts too. [Day 3c](day-03c-files-json.md) introduces `pathlib`, which avoids the problem.

    Raw strings have one limit: they can't end in a single backslash, because `\"` still stops the quote from closing the string. `p = r"C:\path\"` fails before the script runs:

    ```text
    SyntaxError: unterminated string literal (detected at line 1); perhaps you escaped the end quote?
    ```

    Write `r"C:\path" + "\\"`, or leave the trailing backslash off.

### `repr` and `str`: two ways to show a value

**What it is.** Every Python object can be turned into text in two ways. `str(x)` gives the **plain** form, meant for a person reading output. `repr(x)` gives the **developer** form, meant for you while debugging: for a string, it adds quotes and shows escape sequences, so that invisible characters become visible.

**Why it exists.** It answers a question you've been looking at since Day 2: why does the ▶ Run echo print `'34'` with quotes, when `print` doesn't? Because an echoed value is shown with `repr`, and `print` uses `str`. That difference is what lets you tell the string `"7"` from the number `7` while debugging, and it's why a stray newline or a trailing space in a label becomes visible when you print the repr.

**Kotlin vs Python.** Kotlin has one conversion, `toString()`, which is the plain form, like `str`. The closest thing to `repr` is what the Android Studio debugger shows in its Variables pane: strings in quotes, with `\n` escaped. Python gives you that debugger view as a function you can call anywhere.

**Example.** The same three values, both ways:

```python
for value in ["7", 7, "dog\n"]:
    print("str:", str(value), "| repr:", repr(value))
# str: 7 | repr: '7'
# str: 7 | repr: 7
# str: dog
#  | repr: 'dog\n'
```

Read the third pair carefully: the `str` form printed a real line break in the middle of the line, while the repr showed `'dog\n'`, which is what you need to see when a label won't match. In an f-string, `!r` asks for the repr of that one value:

```python
label = "dog\n"
print(f"label is {label!r}")      # label is 'dog\n'
print(f"label is {label}")        # label is dog
#                                 # (the newline ends the line early)
```

**Underneath.** `str(x)` calls the type's `__str__` method and `repr(x)` calls `__repr__`. A type that defines only `__repr__` uses it for both, which is why many objects print the same either way. When you put objects in a list and print the list, the list's `str` uses the **repr** of each element, which is why list output shows quotes:

```python
labels = ["cat", "dog"]
print(labels)                     # ['cat', 'dog']
print(labels[0])                  # cat
```

Day 3 shows how to write `__repr__` and `__str__` for your own classes.

!!! warning "Common mistake: a number that is really a string"
    `print` hides the difference, so a value read from a file can look fine and still break the next line:

    ```python
    rating = "3.8"               # read from a CSV: text, not a number
    print(rating)                # 3.8        looks like a number
    print(repr(rating))          # '3.8'      the quotes give it away
    try:
        rating + 0.2
    except TypeError as err:
        print("TypeError:", err)
    # TypeError: can only concatenate str (not "float") to str
    ```

    **How to recognise it:** the error says `str`, or a sum comes out as glued-together text (`"3" + "8"` is `'38'`). **Fix:** print the repr, or `type(x)`, when a value behaves oddly, and convert with `float(...)` as soon as the text enters your program.

### Indexing, slicing and immutability

**What it is.** A string is a **sequence**: its characters have positions, starting at 0. `s[i]` returns the character at position `i` (as a one-character string). Negative positions count from the end: `s[-1]` is the last character. `s[start:stop]` is a **slice**: a new string from position `start` up to, but **not including**, `stop`. Either end can be left out.

**Why it matters.** File names, labels, log lines and tokens are all strings you'll cut up. The same slicing syntax works on lists, tuples, NumPy arrays and tensors, so it's worth learning exactly.

**Kotlin vs Python.** `s[0]` works in both. `s.substring(2, 6)` in Kotlin is `s[2:6]` in Python, with the same "stop is excluded" rule as Kotlin's `until`. Kotlin has no negative indices; `s[-1]` replaces `s.last()` and `s[s.length - 1]`.

```python
s = "MobileNet"
s[0]          # 'M'
s[-1]         # 't'
s[0:6]        # 'Mobile'
s[:6]         # 'Mobile'   start defaults to 0
s[6:]         # 'Net'      stop defaults to the end
s[-3:]        # 'Net'      the last three characters
s[::2]        # 'MblNt'    a third number is the step: every 2nd character
s[::-1]       # 'teNeliboM' step -1 walks backwards: reversed
s[-1:-4:-1]   # 'teN'      backwards from the last character, stopping before fence -4
len(s)        # 9
```

With a negative step, the defaults flip: an empty start means "the last character" and an empty stop means "past the first one", which is why `s[::-1]` reverses the whole string. Day 2b has the full slicing rules.

**Underneath.** Picture the positions as fences *between* characters. `s[0:6]` takes everything between fence 0 and fence 6. That's why `s[:6] + s[6:]` always gives back the whole string, and why `stop - start` is the length of the slice:

<figure class="diagram" markdown>
![Animation: the string MobileNet in nine boxes, with positive indices 0 to 9 written on the fences between characters above, and negative indices -9 to -1 on the same fences below. A highlight bracket moves through three slices: s[0:6] covers Mobile, s[6:] covers Net, and s[-3:] covers the same Net from the negative side](../../assets/img/phase-1/day02a-string-slice.svg)
<figcaption>Animated: slice positions are fences between characters, so <code>s[0:6]</code> is six characters and <code>s[6:]</code> picks up exactly where it stopped.</figcaption>
</figure>

Negative indices sit on the same fences, counted from the right-hand end: fence `-1` is one step from the end, so `s[-3:]` starts three characters from the end. There's no fence called `-0`, which is why the stop of `s[-3:]` is left empty. Writing `0` there is a classic bug, because `0` is the *left* end:

```python
s[-3:0]              # ''     from fence -3 to fence 0, going right: nothing
s[-3:]               # 'Net'  leave stop empty to reach the end
```

Slicing in full (slice assignment on lists, and the traps) is on Day 2b. Two rules to know today: indexing outside the string is an error, but slicing outside it is forgiving:

```python
"abc"[1:10]          # 'bc'   slice past the end: clipped, no error
try:
    "abc"[5]
except IndexError as err:
    print("IndexError:", err)
# IndexError: string index out of range
```

**Immutability.** You can't change a character in place, in Python or Kotlin. Every "change" builds a new string:

```python
try:
    s[0] = "m"
except TypeError as err:
    print("TypeError:", err)
# TypeError: 'str' object does not support item assignment
"m" + s[1:]          # 'mobileNet'   build a new string instead
```

### Operators on strings: `+`, `*`, `in`, `len` and comparison

**What they are.** `+` **concatenates** (joins end to end) and `*` **repeats**. `in` and `not in` test whether one string appears inside another. `len` counts characters. `<`, `>`, `<=` and `>=` compare two strings alphabetically.

**Why they matter.** Building a file name, drawing a separator line in a log, checking whether a column name contains "id", and sorting labels are all everyday jobs.

**Kotlin vs Python.** Each one has a Kotlin twin, usually as a method rather than an operator:

| Kotlin | Python | Note |
|---|---|---|
| `"conv" + "2d"` | `"conv" + "2d"` | the same |
| `"-".repeat(10)` | `"-" * 10` | an operator, not a method |
| `"MobileNet".contains("Net")` | `"Net" in "MobileNet"` | the **needle comes first** in Python |
| `!s.contains("x")` | `"x" not in s` | `not in` is one operator |
| `s.length` | `len(s)` | a function, not a property |
| `"a" < "b"` (via `compareTo`) | `"a" < "b"` | Kotlin compares UTF-16 units, Python code points: the same order except for characters beyond the BMP, such as emoji |

```python
"conv" + "2d"            # 'conv2d'
"-" * 10                 # '----------'   repetition
"Net" in "MobileNet"     # True           substring test
"net" in "MobileNet"     # False          case-sensitive
"x" not in "abc"         # True           not in: one operator, reads naturally
"" in "abc"              # True           the empty string is in every string
len("héllo")             # 5
```

**Comparison** is lexicographic, character by character, by Unicode code point (a number for each character; `ord("A")` is 65, `ord("a")` is 97). So upper-case letters sort before lower-case ones, and digits are compared as characters, not numbers:

```python
"apple" < "banana"       # True
"Zebra" < "apple"        # True    'Z' (90) < 'a' (97)
"10" < "9"               # True    '1' < '9', the rest is never looked at
ord("A"), ord("a")       # (65, 97)
chr(65)                  # 'A'     chr is the inverse of ord
```

!!! warning "Common mistake: sorting numbers as strings"
    Checkpoint files named `epoch_9.pt` and `epoch_10.pt` sort with `epoch_10` first, because `"1" < "9"`. You'd load the wrong "latest" checkpoint:

    ```python
    "epoch_10.pt" < "epoch_9.pt"      # True   wrong order for numbers
    int("10") < int("9")              # False  compare numbers as numbers
    "epoch_" + "9".zfill(3)           # 'epoch_009'   or zero-pad names when you create them
    ```

    `zfill(n)` pads a string with leading zeros to length `n`. Zero-padded names sort correctly as text, which is why datasets use `img_00042.png`.

### String methods

**What they are.** A **method** is a function attached to an object, called with a dot: `s.strip()`. String methods never change the string (it's immutable); they **return a new string** or another value. Here are the ones you'll use weekly, grouped by job.

**Cleaning whitespace and prefixes.** `strip()` removes whitespace (spaces, tabs, newlines) from both ends; `lstrip()` and `rstrip()` from one end. `removeprefix(p)` / `removesuffix(p)` remove an exact prefix or suffix if present.

```python
line = "  cat\n"
line.strip()                         # 'cat'
line.rstrip()                        # '  cat'
"model.tflite".removesuffix(".tflite")   # 'model'
"model.onnx".removesuffix(".tflite")     # 'model.onnx'   not present: unchanged
```

**Case.** `lower()`, `upper()`, `title()`, `capitalize()`; `casefold()` is a stronger `lower()` for case-insensitive comparison of any language.

```python
"MobileNet".lower()          # 'mobilenet'
"hello world".title()        # 'Hello World'
"hello world".capitalize()   # 'Hello world'
"Straße".casefold()          # 'strasse'
```

**Testing.** `startswith` / `endswith` (both accept a tuple of options), and the `is...` family: `isdigit()` is `True` when every character is a digit, `isalpha()` when every character is a letter, `isalnum()` when every character is a letter or digit, `isspace()` when every character is whitespace, and `isupper()` / `islower()` test the case. All of them return `False` for the empty string.

```python
"model.tflite".endswith((".tflite", ".lite"))   # True
"IMG_0042.jpg".startswith("IMG_")               # True
"2024".isdigit()                                # True
"-5".isdigit()                                  # False  the minus sign is not a digit
"3.5".isdigit()                                 # False
"".isdigit()                                    # False  empty: no digits at all
"abc".isalpha()                                 # True
"abc123".isalnum()                              # True
"  \t\n".isspace()                              # True
"ABC".isupper(), "abc".islower()                # (True, True)
```

`isdigit` has one more trap: it accepts characters that are digits in Unicode but that `int()` refuses, such as the superscript `²`. `isdecimal()` is the strict test for "safe to pass to `int()`":

```python
"²".isdigit()                                   # True
"²".isdecimal()                                 # False
try:
    int("²")
except ValueError as err:
    print("ValueError:", err)
# ValueError: invalid literal for int() with base 10: '²'
```

**Searching and counting.** `find(sub)` returns the position of the first match or `-1`; `index(sub)` does the same but raises `ValueError` when there's no match; `count(sub)` counts matches that don't overlap: after a match, counting restarts *after* it, so the four `a`s in `"aaaa"` contain two non-overlapping `"aa"`, not three.

```python
"MobileNet".find("Net")      # 6
"MobileNet".find("x")        # -1
"banana".count("a")          # 3
"aaaa".count("aa")           # 2      positions 0-1 and 2-3; the 1-2 overlap isn't counted
try:
    "MobileNet".index("x")
except ValueError as err:
    print("ValueError:", err)
# ValueError: substring not found
```

!!! warning "Common mistake: using `find`'s result without checking for -1"
    `-1` is a valid index (the last character) and it's truthy, while a match at position 0 is falsy. Both mistakes run without an error:

    ```python
    "MobileNet"["MobileNet".find("x"):]   # 't'    not found, so this slices from -1
    bool("MobileNet".find("M"))           # False  found at position 0, but 0 is falsy
    ```

    **Fix:** use `"x" in s` for "is it there?", and `index` when a missing substring is an error you want to see. Use `find` only when you compare its result with `-1` explicitly.

**Replacing.** `replace(old, new)` replaces every match; an optional third argument limits how many.

```python
"a-b-c".replace("-", "_")        # 'a_b_c'
"banana".replace("a", "o", 1)    # 'bonana'
```

**Splitting and joining.** `split()` with no argument splits on any run of whitespace and drops empty pieces. `split(sep)` splits on exactly `sep` and keeps empty pieces. Both return a list of strings. `sep.join(pieces)` is the reverse: it glues a sequence of strings together with `sep` between them. `pieces` can be any sequence of strings, including a string itself, which is a sequence of one-character strings.

```python
" a b  c ".split()               # ['a', 'b', 'c']
"a,b,,c".split(",")              # ['a', 'b', '', 'c']   the empty field is kept
"lr=0.001".split("=")            # ['lr', '0.001']
"a=b=c".split("=", 1)            # ['a', 'b=c']   split at most once
", ".join(["cat", "dog", "bird"])   # 'cat, dog, bird'
"".join(["M", "o", "b"])         # 'Mob'
```

**Splitting lines.** Text files use different line endings: `\n` on Linux and macOS, `\r\n` on Windows. `splitlines()` handles both, and drops the final empty piece that `split("\n")` leaves when the text ends with a line break:

```python
text = "cat\r\ndog\nbird\n"
text.splitlines()                # ['cat', 'dog', 'bird']
text.split("\n")                 # ['cat\r', 'dog', 'bird', '']   \r kept, empty tail
"a\n\nb".splitlines()            # ['a', '', 'b']   a blank line stays, as an empty string
```

Use `splitlines()` for anything that came from a file.

**Splitting once, keeping the separator: `partition`.** `partition(sep)` finds the *first* `sep` and returns exactly three strings: what's before it, the separator itself, and what's after. `rpartition` does the same from the right. Unlike `split`, it always returns three pieces, even when the separator is missing, so the unpacking never fails:

```python
"lr=3e-4".partition("=")         # ('lr', '=', '3e-4')
"novalue".partition("=")         # ('novalue', '', '')   no separator: the rest is empty
"a=b=c".partition("=")           # ('a', '=', 'b=c')     splits at the FIRST =
"a=b=c".rpartition("=")          # ('a=b', '=', 'c')     and rpartition at the last
key, sep, value = "epochs=20".partition("=")
key, value                       # ('epochs', '20')
```

That last line is the `key=value` idiom you'll use for config lines and log fields. `rpartition(".")` is the same trick for splitting a file name from its extension. `rsplit(sep, 1)` is the list-returning cousin: `"a/b/c.tar.gz".rsplit(".", 1)` is `['a/b/c.tar', 'gz']`.

**Kotlin vs Python.** Most names match Kotlin's: `trim()` is `strip()`, `uppercase()` is `upper()`, `indexOf()` is `find()`, `split(",")` is `split(",")`. The big one that's flipped is joining: Kotlin's `list.joinToString(", ")` is Python's `", ".join(list)`. The separator is the object you call the method on.

**Underneath.** Each method call allocates a new string and returns it, so you can chain them, reading left to right:

```python
raw = "  Cat_Food \n"
raw.strip().lower().replace("_", " ")    # 'cat food'
```

Step by step: `strip()` gives `'Cat_Food'`, `lower()` gives `'cat_food'`, `replace` gives `'cat food'`. `raw` itself is unchanged.

**Building a string from many pieces.** Because strings are immutable, `result += piece` means, in principle, "build a new string holding everything so far plus the piece, then rebind the name". In a loop over \(n\) pieces, that copying grows like \(n^2\). CPython often avoids it by growing the string in place when no other name refers to it, so a quick test may look fast. But that's an implementation detail that [PEP 8](https://peps.python.org/pep-0008/#programming-recommendations) tells you not to rely on: a second reference to the string, or another Python implementation such as PyPy, brings the copying back. `join` is guaranteed to take time proportional to the total length, and it says what you mean. The idiom is to collect the pieces in a list and join once at the end:

```python
rows = []                                     # collect the pieces
for name, acc in [("cat", 0.9), ("dog", 0.07)]:
    rows.append(f"{name}={acc:.2f}")          # append adds to the list, in place
"; ".join(rows)                               # 'cat=0.90; dog=0.07'   one new string
```

For a handful of pieces, `+=` is fine and clearer; for a loop over a file, use `join`. It's the same advice as Kotlin's `StringBuilder` or `joinToString` instead of `+=` in a loop.

!!! warning "Common mistake: calling a string method and throwing the result away"
    Because strings are immutable, `label.strip()` on its own line does nothing useful. The cleaned string is returned and lost:

    ```python
    label = "  dog\n"
    label.strip()                    # 'dog'  computed, shown, and thrown away
    label == "dog"                   # False  label is unchanged
    label = label.strip()            # rebind the name to the new string
    label == "dog"                   # True
    ```

    The ▶ Run echo shows `'dog'` for the second line, which makes the bug harder to spot in a notebook: the value appears, but nothing stored it.

!!! warning "Common mistake: `strip` takes a set of characters, not a suffix"
    `strip(".png")` removes any of the characters `.`, `p`, `n`, `g` from both ends, as many as it finds:

    ```python
    "map.png".strip(".png")          # 'ma'    the trailing p of "map" went too
    "map.png".removesuffix(".png")   # 'map'   the fix: removes the exact suffix
    ```

    And `join` only accepts strings, because it doesn't guess how you want a number shown:

    ```python
    try:
        ",".join([1, 2])
    except TypeError as err:
        print("TypeError:", err)
    # TypeError: sequence item 0: expected str instance, int found

    parts = []
    for n in [1, 2]:
        parts.append(str(n))      # convert each one first
    ",".join(parts)               # '1,2'
    ```

    The same thing fits on one line as `",".join(str(n) for n in [1, 2])`, a **generator expression** (Day 2b previews it and [Day 3d](day-03d-iterators-generators.md) covers it). The loop version above is what it does.

**A realistic example.** A `labels.txt` file shipped next to a classifier has one label per line, with stray spaces, a blank line and Windows line endings (`\r\n`). Clean it into a list of labels:

```python
raw_file = " cat\r\ndog \r\n\r\n  bird\r\n"
labels = []                          # an empty list to fill
for line in raw_file.splitlines():   # handles \r\n, and drops the empty tail
    label = line.strip()
    if label:                        # skip the blank line (truthiness!)
        labels.append(label)
labels                               # ['cat', 'dog', 'bird']
```

`splitlines()` removes the `\r\n` endings, `strip()` the stray spaces, and the `if` the blank line. Without the `strip()`, the labels would be `' cat'` and `'dog '`, and a lookup for `'cat'` would quietly fail.

### The format spec mini-language

**What it is.** Day 2 introduced four f-string specs. Here is the whole language. An **f-string** is a string literal with an `f` prefix. Any expression inside `{...}` is evaluated and inserted. After the expression you can add `:` and a **format spec**, a small language that controls how the value is shown: decimals, width, alignment, separators, percentages.

**Why it exists.** Every training log, metrics table and debug print uses it. It's the Python version of Kotlin's string templates plus `String.format`.

**Kotlin vs Python.**

```kotlin
val acc = 0.7523
println("accuracy: $acc")                          // accuracy: 0.7523
println("accuracy: ${"%.1f".format(acc * 100)}%")  // accuracy: 75.2%
```

```python
acc = 0.7523
print(f"accuracy: {acc}")             # accuracy: 0.7523
print(f"accuracy: {acc:.1%}")         # accuracy: 75.2%
```

**The format spec, piece by piece.** The general shape is `{value:[fill][align][sign][0][width][,][.precision][type]}`. Every part is optional. Read each piece:

- **type**, the last letter, says how to show the number: `f` fixed-point decimals, `e` scientific notation, `%` multiply by 100 and add a percent sign, `d` integer, `x` hexadecimal, `b` binary.
- **.precision** after a dot: how many digits after the decimal point (for `f`, `e` and `%`).
- **,** or **_**: group thousands with commas or underscores.
- **width**: the minimum number of characters; shorter values are padded.
- **align**: `<` left, `>` right, `^` centre. Numbers default to right, strings to left.
- **fill**: the padding character, written before the align sign (default: space). A fill only works together with an align sign: `*>8` is valid, `*8` is not.
- **0**: a `0` right before the width pads numbers with zeros after the sign, as in `03d`.
- **sign**: `+` shows a plus sign on positive numbers too.

<figure class="diagram" markdown>
![Diagram of the format spec *>+12,.2f from f"{value:*>+12,.2f}", typeset as one string with each part bracketed underneath. Seven labelled boxes in one row, in spec order, each joined to its bracket by its own straight line: fill (*), align (>), sign (+), width (12), grouping (,), precision (.2) and type (f), each with a one-line description. At the bottom, the result for 1234.567 is shown as ***+1,234.57 with the three padding stars coloured](../../assets/img/phase-1/day02a-format-spec.svg)
<figcaption>Every part of a format spec is optional, and they always come in this order.</figcaption>
</figure>

```python
acc, n, lr = 0.7523, 1234567, 0.0003
f"{acc:.2f}"         # '0.75'          2 decimals
f"{acc:.1%}"         # '75.2%'         times 100, 1 decimal, percent sign
f"{n:,}"             # '1,234,567'     thousands separator
f"{n:_}"             # '1_234_567'
f"{n:,.2f}"          # '1,234,567.00'  separator and decimals together
f"{lr:e}"            # '3.000000e-04'  scientific, 6 decimals by default
f"{lr:.1e}"          # '3.0e-04'
f"{'cat':>8}"        # '     cat'      right-aligned in 8 characters
f"{'cat':<8}|"       # 'cat     |'     left-aligned (the | shows the padding)
f"{'cat':^8}|"       # '  cat   |'     centred
f"{'cat':*^9}"       # '***cat***'     fill with *
f"{7:03d}"           # '007'           zero-pad an integer to width 3
f"{3.14159:8.3f}"    # '   3.142'      width 8, 3 decimals
f"{3:+d}"            # '+3'
f"{255:x}"           # 'ff'
f"{1234.567:*>+12,.2f}"   # '***+1,234.57'   every part at once, as in the diagram
```

That last spec is the diagram, read left to right: fill with `*`, align right (`>`), always show the sign (`+`), width 12, group thousands (`,`), two decimals (`.2`), fixed-point (`f`). You'd never write all of them at once in real code; the point is that they stack in a fixed order.

Two more f-string features you'll use while debugging. `{x=}` prints the expression text and its value. `!r` asks for the repr, which shows quotes and hidden characters:

```python
f"{acc=}"            # 'acc=0.7523'
f"{acc = :.3f}"      # 'acc = 0.752'   the spaces around = are copied into the output
label = "dog\n"
print(f"[{label!r}]")   # ['dog\n']   the repr makes the stray newline visible
width = 10
f"{acc:>{width}.3f}" # '     0.752'   the spec itself can contain {expressions}
f"{{not a field}}"   # '{not a field}'   double braces for a literal brace
```

When a value takes both, `!r` comes first and the `:` spec second, so `{x!r:>8}` means "the repr, right-aligned in 8 characters":

```python
f"[{'cat'!r:>8}]"    # "[   'cat']"   the quotes are part of the 8 characters
```

**Underneath.** `f"{value:spec}"` calls the built-in `format(value, spec)`, which asks the value's type to format itself. That's why the spec letters depend on the type: `f` means something to a float, nothing to a string.

```python
format(0.7523, ".1%")    # '75.2%'
```

**Older styles.** You'll see two older formatting styles in existing code. They produce the same text:

```python
"{} scored {:.1%}".format("model", acc)    # 'model scored 75.2%'
"%s scored %.2f" % ("model", acc)          # 'model scored 0.75'
```

Write f-strings in new code.

!!! warning "Common mistake: a spec the value's type doesn't understand"
    Values read from a file are strings, and numeric specs refuse them:

    ```python
    acc_text = "0.7523"                  # read from a CSV: still text
    try:
        f"{acc_text:.2f}"
    except ValueError as err:
        print("ValueError:", err)
    # ValueError: Unknown format code 'f' for object of type 'str'
    f"{float(acc_text):.2f}"             # '0.75'   convert first
    ```

    Two more `ValueError`s you'll meet. A fill character without an align sign isn't a valid spec, and `d` (integer) refuses a float, even a whole one such as the result of a mean:

    ```python
    for spec_value, spec in [(5, "*8"), (3.0, "d")]:
        try:
            format(spec_value, spec)
        except ValueError as err:
            print("ValueError:", err)
    # ValueError: Invalid format specifier '*8' for object of type 'int'
    # ValueError: Unknown format code 'd' for object of type 'float'
    ```

    The fixes: add the align (`*>8`), and use `.0f` for a float you want shown without decimals, or convert with `int(...)` if it really is a count.

    One more mistake gives no error at all. `%` multiplies by 100, so a value that's already a percentage comes out 100 times too big: `f"{75.23:.1%}"` is `'7523.0%'`. Use `:.1f` and write the `%` yourself for values that are already percentages.

!!! android "Android bridge"
    Kotlin's `"%.2f".format(x)` and `String.format` use the device's default locale, so on a German phone `"%.2f".format(0.75)` gives `"0,75"`. That's right for UI text and wrong for files, logs and anything another program parses, which is why Android code passes `Locale.ROOT` or `Locale.US`. Python's f-strings with `.2f` and `,` always use a dot and commas, whatever the machine's locale. (The rarely used `n` type is the locale-aware one.)

### Unicode, bytes and encodings

**What it is.** A Python `str` is a sequence of Unicode **code points**: one number per character, from 0 to 1,114,111. **Bytes** (`bytes` type, written `b"..."`) are raw 8-bit values, which is what files and networks actually carry. An **encoding** such as UTF-8 is the rule that turns code points into bytes. `s.encode("utf-8")` goes from `str` to `bytes`; `b.decode("utf-8")` goes back.

**Why it matters.** Text from files, the web and tokenizers arrives as bytes and must be decoded. Text models count characters and tokens, so you need to know what one "character" is.

**Kotlin vs Python.** A Kotlin `String` is a sequence of UTF-16 code units: each one is 2 bytes, which covers the first 65,536 code points (the **Basic Multilingual Plane**, or BMP, which holds almost every character in everyday use). A character beyond it, such as most emoji, needs two UTF-16 units, so Kotlin counts it as 2. Python counts code points, so it counts 1:

```kotlin
"👍".length            // 2
```

```python
len("👍")                  # 1
len("é")                   # 1
"é".encode("utf-8")        # b'\xc3\xa9'   two bytes in UTF-8
len("é".encode("utf-8"))   # 2
b"\xc3\xa9".decode("utf-8")   # 'é'
ord("é")                   # 233
```

In the repr of a `bytes` value, each byte that isn't a printable ASCII character is shown as `\x` and two hex digits, so `b'\xc3\xa9'` is two bytes, 0xC3 and 0xA9.

`bytes` looks like `str` but behaves differently in two places. Indexing a `bytes` value gives an `int` (like a Kotlin `ByteArray`, but unsigned, 0–255), not a one-character `bytes`. And `str` and `bytes` never mix:

```python
b"abc"[0]                  # 97    an int: the byte value of 'a'
try:
    "a" + b"b"
except TypeError as err:
    print("TypeError:", err)
# TypeError: can only concatenate str (not "bytes") to str
```

**A code point is not always what a user sees as one character.** Python's `len` counts code points, which is closer to a person's idea of "characters" than Kotlin's UTF-16 count, but still not the same. A **grapheme cluster** (what a user sees as one character) can be several code points: an emoji with a skin-tone modifier, a flag (two "regional indicator" letters), or a letter plus a separate combining accent:

```python
len("👍🏽")                  # 2   thumbs up + skin-tone modifier
len("🇩🇪")                  # 2   regional indicators D and E
len("e\u0301")              # 2   e + combining acute accent: displays as é
```

The last one also means two strings that look identical, `"é"` and `"e\u0301"`, are not equal. Unicode normalisation (Going further) fixes that.

**Underneath: how UTF-8 encodes a code point.** UTF-8 uses one to four bytes per code point, and the first byte says how many follow. A code point below 128 is one byte, identical to the old ASCII encoding, which is why English text is the same size in UTF-8 as it was before Unicode. Above that, the lead byte starts with as many 1 bits as the total number of bytes, and every following ("continuation") byte starts with `10`:

```python
for ch in ["A", "é", "€", "👍"]:
    data = ch.encode("utf-8")
    bits = " ".join(f"{b:08b}" for b in data)   # :08b shows one byte as 8 binary digits
    print(f"{ch}  code point {ord(ch):>6}  {len(data)} byte(s)  {bits}")
# A  code point     65  1 byte(s)  01000001
# é  code point    233  2 byte(s)  11000011 10101001
# €  code point   8364  3 byte(s)  11100010 10000010 10101100
# 👍  code point 128077  4 byte(s)  11110000 10011111 10010001 10001101
```

<figure class="diagram" markdown>
![Animation: four rows, one each for the characters A, e-acute, the euro sign and the thumbs-up emoji. Each row shows the character, its code point, and its UTF-8 bytes as boxes of eight binary digits, with the leading marker bits highlighted: 0 for a single byte, 110 and 10 for two bytes, 1110 and two continuation bytes for three, 11110 and three continuation bytes for four. A moving highlight walks down the rows, and a note at the end of each row gives its byte count: 1, 2, 3 and 4](../../assets/img/phase-1/day02a-utf8-bytes.svg)
<figcaption>Animated: the lead byte's run of 1 bits counts the bytes in the character; every continuation byte starts with 10.</figcaption>
</figure>

That structure is what makes a decoder able to tell where each character starts, and it's why `len(text)` and `len(text.encode("utf-8"))` differ for non-English text. Tokenizers for language models work on these bytes, which is why a model's "token count" for an emoji or a Chinese character is larger than for a Latin letter.

**How it goes wrong.** Decoding bytes with the wrong encoding fails, or produces garbage, depending on the bytes. The examples use **Latin-1** (ISO-8859-1), an older encoding with one byte per character, so it can only hold code points 0–255:

```python
latin1_bytes = "é".encode("latin-1")    # one byte, 0xe9
latin1_bytes                            # b'\xe9'
try:
    latin1_bytes.decode("utf-8")
except UnicodeDecodeError as err:
    print("UnicodeDecodeError:", err)
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 0: unexpected end of data
latin1_bytes.decode("latin-1")          # 'é'   decode with the encoding it was written in
```

Read the error message with the byte table above in mind: `0xe9` is `11101001` in binary, a lead byte whose three leading 1 bits promise a three-byte character, but the data ends after one byte. That's exactly what `unexpected end of data` means. Two other messages tell you something similar. A byte that can't start any character gives `invalid start byte`, and a lead byte followed by a byte that isn't a continuation byte (`10xxxxxx`) gives `invalid continuation byte`. Encoding can fail too, when the target encoding has no byte for a character, as with ASCII (code points 0–127 only):

```python
for data in [b"\x80", b"ok \xe9 here"]:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as err:
        print("UnicodeDecodeError:", err)
try:
    "é".encode("ascii")
except UnicodeEncodeError as err:
    print("UnicodeEncodeError:", err)
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 3: invalid continuation byte
# UnicodeEncodeError: 'ascii' codec can't encode character '\xe9' in position 0: ordinal not in range(128)
```

`0x80` is `10000000`, a continuation byte with no lead byte before it. In the second, `0xe9` promises two continuation bytes, but the next byte is a space.

**Mojibake: the failure with no error.** The opposite mistake is worse, because nothing complains. UTF-8 bytes decoded as Latin-1 always "work", because every byte value is a valid Latin-1 character, so each byte of a two-byte character becomes its own garbled letter. The result has a name: **mojibake**.

```python
"Hello, café".encode("utf-8").decode("latin-1")   # 'Hello, cafÃ©'
```

If you've seen `Ã©`, `â€™` or `Â ` in a web page or a CSV, that's UTF-8 read as Latin-1 or as Windows-1252 (`cp1252`, Microsoft's variant of Latin-1, which adds characters such as `€` and `™`). The data isn't lost: re-encode with the encoding it was wrongly decoded as, and decode as UTF-8:

```python
"cafÃ©".encode("latin-1").decode("utf-8")         # 'café'
"â€™".encode("cp1252").decode("utf-8")            # '’'   a curly apostrophe
```

The second one needs `cp1252`, because `€` and `™` aren't in Latin-1: `"â€™".encode("latin-1")` raises `UnicodeEncodeError`.

**Fix:** always say `encoding="utf-8"` when you open text files ([Day 3c](day-03c-files-json.md)), and decode with the encoding the data was written in. When you can't be sure, `b.decode("utf-8", errors="replace")` puts the **replacement character** U+FFFD (shown as a black diamond with a question mark, �) in place of each bad byte instead of raising, which keeps a data-loading script running:

```python
b"ok \xe9 here".decode("utf-8", errors="replace")   # 'ok � here'
```

Seeing � in your data means the bytes were already damaged or misread; it is a report of a problem, not a fix for one.


!!! android "Android bridge"
    The `labels.txt` you load next to a `.tflite` model is exactly this problem. The file has one class name per line, in the order of the model's output scores, and you read it with `BufferedReader(InputStreamReader(assets.open("labels.txt"), Charsets.UTF_8)).readLines()`, then `trim()` each line. In Python it's `text.splitlines()` and `line.strip()`, and the thing that bites is the encoding. On Android the default charset is always UTF-8, so `InputStreamReader` without a charset happens to work there. Desktop Java before JDK 18, and Python up to 3.14, use the machine's locale instead: `cp1252` on most Windows laptops. Python 3.15 switches its default to UTF-8 ([PEP 686](https://peps.python.org/pep-0686/)). Say `encoding="utf-8"` anyway, so the code doesn't depend on where it runs.

## Putting it together: parsing a training log

Real ML work is full of small text-processing jobs like this one. A training script wrote a log, with a comment line, a blank line and a diverged run at the end. Turn it into a formatted report and stop at the first `nan`. It uses `math.isnan` from Day 2, so the block imports `math` first:

```python
import math

log = """# run 7, lr=3e-4
epoch=1 loss=0.9123 acc=0.6121
epoch=2 loss=0.6044 acc=0.7710
epoch=3 loss=0.4521 acc=0.8432

epoch=4 loss=nan acc=0.1000
epoch=5 loss=nan acc=0.1000"""

print(f"{'epoch':>5} {'loss':>8} {'acc':>7}")
for line in log.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue                                  # skip blank lines and comments
    epoch_part, loss_part, acc_part = line.split()   # three fields separated by spaces
    epoch = int(epoch_part.removeprefix("epoch="))
    loss = float(loss_part.partition("=")[2])      # ('loss', '=', '0.9123') -> item 2
    acc = float(acc_part.partition("=")[2])
    if math.isnan(loss):
        print(f"diverged at epoch {epoch}; stopping")
        break
    print(f"{epoch:>5} {loss:>8.4f} {acc:>7.1%}")
# epoch     loss     acc
#     1   0.9123   61.2%
#     2   0.6044   77.1%
#     3   0.4521   84.3%
# diverged at epoch 4; stopping
```

Line by line: `splitlines()` breaks the text into lines. `strip()` removes stray spaces. Blank lines are falsy, so `not line` catches them; `startswith("#")` catches the comment; `continue` skips both. `line.split()` splits on whitespace into three fields, and `a, b, c = ...` unpacks them into three names (it fails if there aren't exactly three). Each field is cleaned and converted: `removeprefix` for the epoch, and `partition("=")[2]` (the piece after the first `=`) for the others. `float("nan")` parses the text `nan` into a real `nan`, so `math.isnan` catches the divergence and `break` ends the loop. The header f-string formats the three column titles as strings (`'epoch':>5`), and the body one formats numbers (`loss:>8.4f`, `acc:>7.1%`) to the same widths, which is what lines the columns up.

Try changing the log: add a line, remove the `nan` rows, or break a field, and watch which part complains.

## Going further

Real and useful, but not needed in the next month:

- **Regular expressions** for matching patterns beyond `find` and `split`: [`re` module](https://docs.python.org/3/library/re.html).
- **The full format spec grammar**, including `g`, `n`, `!a` and writing `__format__` for your own class: [Format Specification Mini-Language](https://docs.python.org/3/library/string.html#formatspec).
- **`str.translate` and `maketrans`** for fast character-by-character replacement: [`str.translate`](https://docs.python.org/3/library/stdtypes.html#str.translate).
- **`textwrap`** for wrapping and indenting paragraphs, and `textwrap.dedent` for removing the indentation of a triple-quoted string inside a function: [`textwrap`](https://docs.python.org/3/library/textwrap.html).
- **Unicode normalisation**, for when `"é"` is one code point in one file and two in another: [`unicodedata.normalize`](https://docs.python.org/3/library/unicodedata.html).
- **`string.Template`** for user-supplied templates, where f-strings would be unsafe: [`string.Template`](https://docs.python.org/3/library/string.html#template-strings).
- **`bytes` and `bytearray` in full**, including `hex()` and `memoryview`: [Binary sequence types](https://docs.python.org/3/library/stdtypes.html#binaryseq).
- **Names, values and mutability**, the model behind "every method returns a new string": [Ned Batchelder, Facts and myths about Python names and values](https://nedbatchelder.com/text/names.html).
- **String interning**, how Python reuses some identical strings: [`sys.intern`](https://docs.python.org/3/library/sys.html#sys.intern).
- **Grapheme clusters**: counting what users see as characters needs a third-party library, such as the [`regex`](https://pypi.org/project/regex/) module's `\X` pattern; the standard library doesn't do it.

---

## Exercises

Try each one before opening the solutions. Every solution block re-creates its own inputs, so you can run them in any order.

**1. Warm-up: clean the labels.** Given `raw = ["  Cat\n", "DOG ", " bird", "", "  "]`, loop over it and print each non-empty label lower-cased and numbered from 1, right-aligned in a column of width 6, like this:

```text
1:    cat
2:    dog
3:   bird
```

**2. Warm-up: look inside some text.** For each string in `names = ["cafe", "café", "naïve", "👍 ok", "C:\temp\new", r"C:\temp\new"]`, print its repr, its length in characters and its length in UTF-8 bytes. Explain why the two paths differ, and why `café` has 4 characters but 5 bytes. Then repair the mojibake `"cafÃ©"`, and explain why `b"\xe9".decode("utf-8")` fails.

**3. Core: parse a config line.** The string `"lr=3e-4; epochs=20; name=mobilenet_v3; augment=yes; dropout=0.25"` is a list of `key=value` pairs separated by `;`. Loop over the pairs, split each one with `partition`, and convert each value to its proper type: `int` if every character is a digit, `bool` if it's `yes` or `no`, `float` if `float()` can parse it, otherwise leave it as a `str`. Print each key, the converted value's repr and its type name (`type(x).__name__` gives the name as a string).

**4. Core: a metrics table.** Given `names = ["MobileNetV3", "EfficientNet-B0", "ResNet50"]`, `params = [5_400_000, 5_300_000, 25_600_000]` and `top1 = [0.752, 0.773, 0.761]`, print a table with left-aligned names in a 16-character column, parameter counts with thousands separators right-aligned in 12 characters, and accuracies as percentages with one decimal. Then build, with `join`, a single summary line reading `MobileNetV3 (75.2%); EfficientNet-B0 (77.3%); ResNet50 (76.1%)`.

**5. Core: split a file name.** For each name in `["IMG_0042.jpg", "img_7.JPEG", "notes.txt", "photo", "IMG_12.png"]`, print the part before the last dot and the extension in lower case, using `rpartition`. Say what happens for `"photo"`, which has no dot, and print a `(no extension)` note for it.

**6. Predict the output.** Write down what each line prints, then run it:

```python
print("a,b,,c".split(","))
print(" a b ".split())
print("map.png".strip(".png"))
print("-".join("abc"))
print("abc"[::-1], "abc"[1:10], len("👍"))
print("10" < "9", "Zebra" < "apple")
print(f"{0.5:.0%}|{'hi':*^6}|{42:>5}")
print(len("C:\new"), len(r"C:\new"))
```

**7. Find and fix the bug.** This should clean a list of labels: strip the whitespace, lower-case them and drop the empty ones. It runs without an error and returns the wrong thing. Find the bug and say why immutability causes it:

```python
# Exercise 7: this version has a bug. Fix it in the solution.
labels = [" Cat\n", "DOG", "", "bird "]
cleaned = []
for label in labels:
    label.strip().lower()
    if label:
        cleaned.append(label)
print(cleaned)
```

**8. Challenge: run-length encoding.** Write a loop that compresses `"aaabccddddd"` into `"a3b1c2d5"`: each character followed by how many times it repeats in a row. Then write a second loop that decodes it back to the original, and check `decoded == original`. Assume counts can have more than one digit (`"x12"` means twelve `x`s), and characters are never digits. Build both results with `join`, not `+=`.

??? success "Solutions"
    **1. Clean the labels.**

    ```python
    raw = ["  Cat\n", "DOG ", " bird", "", "  "]
    number = 0
    for item in raw:
        label = item.strip().lower()
        if not label:
            continue
        number += 1
        print(f"{number}: {label:>6}")
    # 1:    cat
    # 2:    dog
    # 3:   bird
    ```

    `strip()` turns `"  "` into `""`, which is falsy, so whitespace-only entries are skipped along with the empty one. The counter is only increased for kept labels; using `enumerate` over `raw` would number the skipped entries too. `>6` right-aligns each label in 6 characters.

    **2. Look inside some text.**

    ```python
    names = ["cafe", "café", "naïve", "👍 ok", "C:\temp\new", r"C:\temp\new"]
    for name in names:
        print(f"{name!r:<18} chars={len(name):<3} utf8_bytes={len(name.encode('utf-8'))}")

    print("cafÃ©".encode("latin-1").decode("utf-8"))
    try:
        b"\xe9".decode("utf-8")
    except UnicodeDecodeError as err:
        print("UnicodeDecodeError:", err)
    # 'cafe'             chars=4   utf8_bytes=4
    # 'café'             chars=4   utf8_bytes=5
    # 'naïve'            chars=5   utf8_bytes=6
    # '👍 ok'             chars=4   utf8_bytes=7
    # 'C:\temp\new'      chars=9   utf8_bytes=9
    # 'C:\\temp\\new'    chars=11  utf8_bytes=11
    # café
    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 0: unexpected end of data
    ```

    The repr is what makes the paths readable. In the normal string, `\t` and `\n` are escapes, so it holds a tab and a newline: 9 characters, and the repr shows them as `\t` and `\n`. In the raw string the backslashes are real characters, 11 in total, and the repr doubles each one (`\\`) so you can tell them from escapes. Every ASCII character is one UTF-8 byte, `é` and `ï` are two, and the emoji is four, so `'👍 ok'` is 4 characters but 7 bytes. (Its row is also shifted one column: the emoji takes two columns on screen while counting as one character, a reminder that `len` doesn't measure display width.) `"cafÃ©"` is UTF-8 that was decoded as Latin-1, so encoding it back to Latin-1 recovers the original bytes, and decoding those as UTF-8 gives `café`. `0xe9` is `11101001`, a lead byte that promises a three-byte character, but the data ends after it, hence `unexpected end of data`.

    **3. Parse a config line.**

    ```python
    config = "lr=3e-4; epochs=20; name=mobilenet_v3; augment=yes; dropout=0.25"
    for pair in config.split(";"):
        key, sep, text = pair.strip().partition("=")
        if text.isdigit():
            value = int(text)
        elif text in ("yes", "no"):
            value = text == "yes"
        else:
            try:
                value = float(text)
            except ValueError:
                value = text
        print(f"{key:<8} {value!r:<16} {type(value).__name__}")
    # lr       0.0003           float
    # epochs   20               int
    # name     'mobilenet_v3'   str
    # augment  True             bool
    # dropout  0.25             float
    ```

    The order of tests matters: `"20"` would also parse as a float, so the `isdigit` check comes first to keep it an `int`. `text == "yes"` is itself a `bool`, so it's assigned directly. `partition` splits at the first `=` only and always returns three pieces, so a key with no value wouldn't crash the unpacking (it would give an empty `text`; `"".isdigit()` is `False` and `float("")` raises `ValueError`, so it ends up in the `str` branch). `!r` shows strings with quotes, which makes the `str` case visible in the table.

    **4. A metrics table.**

    ```python
    names = ["MobileNetV3", "EfficientNet-B0", "ResNet50"]
    params = [5_400_000, 5_300_000, 25_600_000]
    top1 = [0.752, 0.773, 0.761]

    print(f"{'model':<16}{'params':>12}{'top-1':>8}")
    for name, p, acc in zip(names, params, top1, strict=True):
        print(f"{name:<16}{p:>12,}{acc:>8.1%}")

    summary = []
    for name, acc in zip(names, top1, strict=True):
        summary.append(f"{name} ({acc:.1%})")
    print("; ".join(summary))
    # model                 params   top-1
    # MobileNetV3        5,400,000   75.2%
    # EfficientNet-B0    5,300,000   77.3%
    # ResNet50          25,600,000   76.1%
    # MobileNetV3 (75.2%); EfficientNet-B0 (77.3%); ResNet50 (76.1%)
    ```

    The spec `>12,` combines alignment, width and the thousands separator; `>8.1%` combines width, precision and percent. The header row formats plain strings to the same widths, which is what makes the columns line up. `strict=True` (Day 2) makes mismatched lists an error instead of silently truncating. The summary collects finished pieces in a list and joins them once, so the separator appears between items and never at the end.

    **5. Split a file name.**

    ```python
    for filename in ["IMG_0042.jpg", "img_7.JPEG", "notes.txt", "photo", "IMG_12.png"]:
        stem, dot, ext = filename.rpartition(".")
        if not dot:
            print(f"{filename:<14} (no extension)")
        else:
            print(f"{filename:<14} stem={stem!r:<12} ext={ext.lower()!r}")
    # IMG_0042.jpg   stem='IMG_0042'   ext='jpg'
    # img_7.JPEG     stem='img_7'      ext='jpeg'
    # notes.txt      stem='notes'      ext='txt'
    # photo          (no extension)
    # IMG_12.png     stem='IMG_12'     ext='png'
    ```

    `rpartition(".")` splits at the **last** dot, which is what you want for a name like `archive.tar.gz`. When there is no dot at all, `rpartition` puts everything in the *third* piece and leaves the first two empty, so `stem` is `''`, `dot` is `''` and `ext` is `'photo'`. Testing `if not dot:` catches that case; using the third piece blindly would report the whole name as its own extension. `.lower()` on the extension makes `.JPEG` and `.jpeg` compare equal. ([Day 3c](day-03c-files-json.md)'s `pathlib` has `.stem` and `.suffix` for this job.)

    **6. Predict the output.**

    ```python
    print("a,b,,c".split(","))
    print(" a b ".split())
    print("map.png".strip(".png"))
    print("-".join("abc"))
    print("abc"[::-1], "abc"[1:10], len("👍"))
    print("10" < "9", "Zebra" < "apple")
    print(f"{0.5:.0%}|{'hi':*^6}|{42:>5}")
    print(len("C:\new"), len(r"C:\new"))
    # ['a', 'b', '', 'c']
    # ['a', 'b']
    # ma
    # a-b-c
    # cba bc 1
    # True True
    # 50%|**hi**|   42
    # 5 6
    ```

    `split(",")` keeps the empty field between the two commas, while `split()` with no argument drops empty pieces. `strip(".png")` removes any of the characters `.`, `p`, `n` and `g` from both ends, which eats the `p` of "map". `join` accepts any iterable of strings, and a string is an iterable of one-character strings, so `"-".join("abc")` interleaves hyphens. `[::-1]` reverses, a slice past the end is clipped, and Python counts the emoji as one code point. `"10" < "9"` compares `'1'` with `'9'`, and `'Z'` (90) is below `'a'` (97). In the last line, `.0%` gives `50%` with no decimals, `*^6` centres `hi` in 6 characters padded with stars, and `>5` right-aligns `42`. Finally, `\n` in the normal string is one newline character, so `"C:\new"` has 5 characters, while the raw string keeps the backslash and the `n`: 6.

    **7. Find and fix the bug.**

    ```python
    labels = [" Cat\n", "DOG", "", "bird "]
    cleaned = []
    for label in labels:
        label = label.strip().lower()     # fix: store the result
        if label:
            cleaned.append(label)
    print(cleaned)
    # ['cat', 'dog', 'bird']
    ```

    The original prints `[' Cat\n', 'DOG', 'bird ']`: the strings are uncleaned, though the empty one was dropped. `label.strip().lower()` computes a cleaned string and returns it, but nothing is assigned, so the new string is discarded immediately. Strings are immutable, so there is no in-place version of `strip` that could have worked: a method can only hand you a new string. The fix is to rebind the name. The empty entry was dropped either way, because `""` is falsy, which is what makes the bug easy to miss: the output *looks* like it did something.

    **8. Run-length encoding.**

    ```python
    original = "aaabccddddd"

    pieces = []
    run_char = original[0]
    run_len = 0
    for ch in original:
        if ch == run_char:
            run_len += 1
        else:
            pieces.append(f"{run_char}{run_len}")
            run_char, run_len = ch, 1
    pieces.append(f"{run_char}{run_len}")      # flush the last run
    encoded = "".join(pieces)
    print(encoded)

    for code, expected in [(encoded, original), ("x12y1", "x" * 12 + "y")]:
        out = []
        char = ""
        digits = ""
        for ch in code + "#":                  # a non-digit sentinel flushes the last run
            if ch.isdigit():
                digits += ch
            else:
                if char:
                    out.append(char * int(digits))
                char, digits = ch, ""
        decoded = "".join(out)
        print(code, "->", decoded, decoded == expected)
    # a3b1c2d5
    # a3b1c2d5 -> aaabccddddd True
    # x12y1 -> xxxxxxxxxxxxy True
    ```

    The encoder keeps a current run (its character and length) and appends a finished piece each time the character changes, plus once after the loop, because the last run never sees a change. The decoder collects digit characters until it meets a letter, then expands the previous letter with `char * int(digits)`; multi-digit counts work because the digits accumulate as text before a single `int()` conversion. Appending `"#"` is a **sentinel**: a fake final character that triggers the last flush, so the flush logic isn't repeated after the loop. Both results are built with `join` over a list, which is the idiom for a loop of unknown length; `digits += ch` is fine because it only ever holds a couple of characters.

## Resources

- [Python tutorial, §3.1.2 Text](https://docs.python.org/3/tutorial/introduction.html#text): strings, quoting and escapes in the official tutorial.
- [Python tutorial, §7.1 Fancier Output Formatting](https://docs.python.org/3/tutorial/inputoutput.html): f-strings, `str.format` and the older `%` style.
- [`str` methods, in Built-in Types](https://docs.python.org/3/library/stdtypes.html#string-methods): the reference list of every method on this page, and the ones that aren't.
- [Format Specification Mini-Language](https://docs.python.org/3/library/string.html#formatspec): the full grammar of what goes after the colon.
- [Unicode HOWTO](https://docs.python.org/3/howto/unicode.html): code points, encodings, and reading and writing Unicode files.
- [Ned Batchelder: Pragmatic Unicode, or, How Do I Stop the Pain?](https://nedbatchelder.com/text/unipain.html): the clearest talk and write-up on bytes versus text.
- [Python Tutor](https://pythontutor.com/): paste code and watch names, strings and lists step by step.

## Checklist

- [ ] I can write escapes, raw strings and triple-quoted strings, and I know why `"C:\new"` is a bug.
- [ ] I can explain `str` vs `repr`, and I print the repr when a value behaves oddly.
- [ ] I can index and slice a string, from either end and with a step, and explain the fences model.
- [ ] I can use `+`, `*`, `in`, `not in`, `len` and `<`, and I know why `"10" < "9"`.
- [ ] I can clean, test, search, replace, split and join text, including `splitlines` and `partition`.
- [ ] I build strings from many pieces with `join`, not `+=` in a loop.
- [ ] I can format values with `:.2f`, `:,`, `:.1%`, `:>8`, `:e`, `:03d`, `{x=}` and `!r`.
- [ ] I can explain code points, UTF-8 bytes, `encode` and `decode`, and recognise mojibake.
- [ ] I parsed a log file into a formatted table.

**Next:** [Day 2b: Collections](day-02b-collections.md)
