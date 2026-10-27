# Day 1: Your AI workstation and first Python program

**Phase 1 · Week 1 · ~150 minutes**

## Objectives

By the end of this lesson you can:

- [ ] Explain why ML is built in Python even though Python itself is slow, and say which parts of an ML program run as Python and which run as native code.
- [ ] Describe how CPython runs a file (source → bytecode → eval loop), look at bytecode with `dis`, and explain why Python reports type errors only when a line runs.
- [ ] Install Python 3.12 or newer, and find out exactly which interpreter a command runs, using `which` / `where` and `sys.executable`.
- [ ] Create, activate, inspect, deactivate and delete a virtual environment, and explain what activation changes underneath.
- [ ] Install, inspect, pin, upgrade and remove packages with `python -m pip`, record them in `requirements.txt`, and rebuild an environment from that file.
- [ ] Write and run your first program, and fix the two errors every beginner meets first: `IndentationError` and "can't open file".
- [ ] Run Python four ways: a script, `python -m`, the REPL and a Jupyter notebook.
- [ ] Use Jupyter safely: kernel, execution counts, Restart and Run All, `%timeit`, `%pip`, a kernel registered for your own environment, and `%autoreload` for files you edit after importing them.
- [ ] Stop a runaway program with <kbd>Ctrl</kbd>+<kbd>C</kbd>, and a runaway cell with Interrupt Kernel, and know when to restart instead.
- [ ] Recognise when a file of yours is hiding a library (`sys.path`), and fix it.
- [ ] Say what this site's ▶ Run button can and can't do.
- [ ] Point VS Code or PyCharm at your environment and stop a program at a breakpoint.
- [ ] Read a traceback from the bottom up and fix the bug it points to.
- [ ] Open a Colab notebook with a GPU, upload a file, and know the free tier's limits.
- [ ] Start a course repo with git, a `.gitignore` and a committed `requirements.txt`.
- [ ] Diagnose the common setup errors from their messages.

**Prerequisites.** None. This is the first lesson. You need a laptop (macOS, Windows or Linux), a terminal, and a free Google account for the Colab part. Kotlin experience is assumed; Python experience isn't.

---

## The big idea

You're about to spend months in a new workshop. Before building anything, you set up the bench: the language, a clean place to install libraries for each project, and the tools to run, inspect and debug code. Every hour spent on this today saves you a day of "it works on my machine" later.

**Why Python?** Not because it's fast. A plain Python loop over numbers is roughly 10 to 100 times slower than the same loop in Kotlin. Python won ML because it is a good *steering wheel*: short, readable code that tells fast libraries written in C, C++ and CUDA (NVIDIA's language for programming GPUs) what to do. When you write `a @ b` to multiply two big matrices with NumPy, Python makes **one** call; optimised native code then does billions of multiply-adds. The steering wheel is slow to turn by hand, but it's attached to a very fast engine.

That's why every major ML library (NumPy, pandas, PyTorch, JAX, scikit-learn, Hugging Face Transformers), most research code, and most tutorials target Python first. The language is the glue; the speed lives underneath.

The single rule that follows from this, and that runs through the whole course: **don't loop over numbers in Python; ask a library to process the whole array at once.** Day 4 measures exactly why. Today you'll see a first glimpse in Jupyter.

!!! android "Android bridge"
    It's the same split as Kotlin and the Android NDK. Your app logic is Kotlin, but a camera filter, an audio codec or a game engine runs in C++ behind JNI (the Java Native Interface, which lets JVM code call compiled C/C++ functions). In ML, Python is the app logic and NumPy or PyTorch is the native layer. You will use Kotlin again later in the course, to *ship* trained models inside Android apps, but you'll *build and train* them in Python.

## How Python runs your code

**What it is.** When you type `python hello.py`, the program that runs is an **interpreter**: a program that reads your source code and executes it directly, without a separate build step that produces an app. The standard one is **CPython**, written in C, and it's what you download from python.org. CPython works in two stages, both at run time:

1. **Compile** the source file into **bytecode**: a compact list of simple instructions such as "load variable `a`", "add the top two values", "return". Bytecode is not machine code; your CPU can't run it.
2. **Execute** that bytecode in the **eval loop** (short for evaluation loop): a big C loop inside CPython that takes one instruction, does it, then takes the next.

**Why it works this way.** No build step means you edit a file and run it within a second. There is no Gradle sync, no APK, no install. That fast edit-run loop is what makes Python pleasant for exploring data. The price is that much of the checking Kotlin's compiler does happens, in Python, only while the program runs.

**Kotlin comparison.** You know this pipeline:

```kotlin
// Main.kt
fun main() {
    val years: Int = 8
    println("Hello! " + years)
}
```

`kotlinc` checks every type and compiles `Main.kt` to JVM bytecode; D8 converts that to `classes.dex`; the APK is installed; ART on the phone compiles dex to machine code ahead of time (AOT) or just in time (JIT). All the type checking happened on your laptop, before anyone ran anything.

<figure class="diagram" markdown>
![Animation in two lanes. Top lane, Kotlin on Android: Main.kt, then kotlinc plus D8, then classes.dex, all inside a box labelled on your laptop at build time, then an install arrow to ART on the phone. Type errors stop the build at the kotlinc step. Bottom lane, Python: hello.py, then compile to bytecode in memory, then the eval loop with a repeat arrow, then NumPy or PyTorch in compiled C, C++ and CUDA, all inside one box labelled every time you run python hello.py. The highlight walks along each lane in turn, and in the Python lane it bounces between the eval loop and NumPy](../../assets/img/phase-1/day01-runtime-pipeline.svg)
<figcaption>Animated: Kotlin checks and compiles before the app runs. Python compiles and runs in one go, every launch, and hands array maths to native code.</figcaption>
</figure>

What's the same: both compile to bytecode first, and both end up running native code for the heavy work. What's different: Python does it all at run time, on the machine where the program runs, and Python's compile step checks only **syntax** (is the code grammatically valid), not **types** (does `a + b` make sense for these values).

**A minimal example.** The standard-library module `dis` (disassembler) prints a function's bytecode. Run this block with the ▶ Run button under it (more on that button later):

```python
import dis

def add(a, b):
    return a + b

dis.dis(add)
add(2, 3)
add("Kot", "lin")
```

In the browser, which runs Python 3.14, the output is:

```text
  3           RESUME                   0

  4           LOAD_FAST_BORROW_LOAD_FAST_BORROW 1 (a, b)
              BINARY_OP                0 (+)
              RETURN_VALUE
5
'Kotlin'
```

Read it as: line 3 (`def add`) starts the function (`RESUME` is housekeeping); line 4 loads `a` and `b`, applies `+`, and returns the result. On Python 3.13 the load instruction is called `LOAD_FAST_LOAD_FAST`. Instruction names change between versions; the shape of the idea doesn't.

**How it works underneath.** Notice there's one `BINARY_OP (+)` instruction, and it handled both `2 + 3` and `"Kot" + "lin"`. The bytecode doesn't know the types. Each time the eval loop reaches that instruction, it looks at the two actual objects, asks "what does `+` mean for these?", and dispatches to integer addition or string joining. That per-instruction type check is the reason a Python loop over a million numbers is slow (a million checks), and the reason NumPy is fast (one check, then a tight C loop over raw numbers).

CPython also caches bytecode so it doesn't recompile unchanged files. When you **import** a file (load it as a module from another file; [Day 3a](day-03a-classes-modules.md) covers imports), CPython saves its bytecode in a `__pycache__` folder. A script you run directly (`python hello.py`) is compiled in memory and never written to disk; only imported files get a cache.

In the commands below, `printf '...'` prints text (`\n` means "new line"), and `> helpers.py` **redirects** that output into a file called `helpers.py` instead of the screen, so the first line creates a two-line Python file. You could equally type the file in an editor.

```bash
# Run this on your computer (or in Colab), not in the browser.
printf 'def double(x):\n    return 2 * x\n' > helpers.py
python3 -c "import helpers; print(helpers.double(21))"
ls __pycache__
```

```text
42
helpers.cpython-313.pyc
```

`.pyc` means "Python compiled". The `cpython-313` part says which interpreter and version made it, so two Python versions never trample each other's cache. You never edit or commit these files; they're regenerated whenever the source changes. (The `python3 -c "..."` form runs a one-line program given as text; you'll meet it again below.)

**How it goes wrong.** Because types are checked only when an instruction runs, a bug can sit in a branch for weeks and stay silent until that branch executes. Here it is in the REPL (Python's interactive prompt, covered below):

```pycon
>>> def risky(flag):
...     if flag:
...         return len(42)
...     return "fine"
...
>>> risky(False)
'fine'
>>> risky(True)
Traceback (most recent call last):
  File "<python-input-2>", line 1, in <module>
    risky(True)
    ~~~~~^^^^^^
  File "<python-input-0>", line 3, in risky
    return len(42)
TypeError: object of type 'int' has no len()
```

Kotlin would never have compiled `42.length`. Python compiled `len(42)` happily, ran `risky(False)` happily, and failed only when line 3 ran. Fixing this one bug is easy once you've seen it; fixing the *habit* means to run every path of your code (tests, Day 3c) and to use an editor that reads type hints and underlines suspicious calls before you run anything ([Day 3](day-03-functions-classes.md) covers type hints).

**Syntax** errors are different: the compile step catches them before *any* line runs. This file has a `print` on line 1 and a missing colon on line 3:

```bash
# Run this on your computer (or in Colab), not in the browser.
cat late.py
python3 late.py
```

```text
print("start")

def ratio(a, b)
    return a / b

print("end")
  File "/home/you/ai-course/late.py", line 3
    def ratio(a, b)
                   ^
SyntaxError: expected ':'
```

Nothing printed, not even `start`. The whole file is compiled first, and compiling failed. If your script prints nothing at all and shows a `SyntaxError`, the problem is grammar, not logic. (Paths in outputs on this page are shortened to `/home/you/...`; yours will show your own folders.)

!!! note "CPython is not the only Python"
    "Python" is a language; CPython is one implementation of it. Others exist: PyPy (with a JIT compiler), and **Pyodide**, which is CPython compiled to **WebAssembly** (a portable low-level instruction format that every modern browser can run at close to native speed) so it can run inside a browser tab. Pyodide is what this site's ▶ Run button uses. Everything in this course targets CPython.

## Installing Python

**What it is.** Installing Python puts the CPython interpreter, its standard library (the modules that ship with Python, like `dis`, `math` and `json`) and pip (the package installer) on your machine, and makes a command available in your terminal to start it.

**Why it matters which one.** Most machines end up with several Pythons: one the operating system uses for its own tools, one you install, maybe one from Homebrew or Anaconda. Every Python has its own set of installed packages. Half of all setup problems are "I installed the package into one Python and ran another". So today you'll learn both how to install it and how to check which one is running.

**Kotlin comparison.** You've lived this with JDKs. A typical Android machine has a system JDK, the JetBrains Runtime bundled inside Android Studio, maybe another from SDKMAN, and Gradle picks one through `JAVA_HOME` or the "Gradle JDK" setting. A build that works in Android Studio and fails on the command line is usually two different JDKs. "Which Python?" is exactly the same problem, with one twist: in Python the installed *libraries* belong to the interpreter too, so picking the wrong interpreter also means picking the wrong libraries.

**Which version.** Use **3.12 or newer**; this course was tested on 3.13 on a laptop and 3.14 in the browser. Don't rush to a version released in the last month or two: big ML libraries publish ready-made builds for a new Python some weeks after it comes out, and until then installs can fail.

=== "macOS"

    Download the macOS installer from [python.org/downloads](https://www.python.org/downloads/) and run it. Then, in Terminal:

    ```bash
    python3 --version
    which -a python3
    ```

    ```text
    Python 3.13.9
    /Library/Frameworks/Python.framework/Versions/3.13/bin/python3
    /usr/bin/python3
    ```

    (An example from a typical Mac; your version numbers will differ.) The python.org installer puts its `python3` first on `PATH`. The `/usr/bin/python3` below it is Apple's stub from the Xcode Command Line Tools; leave it alone.

    Or use Homebrew. `brew install python` installs Homebrew's default Python and links the plain `python3` command to it, under `/opt/homebrew/bin` on Apple Silicon Macs (`/usr/local/bin` on Intel Macs). A *versioned* formula such as `brew install python@3.13` installs only `python3.13`, so you'd then type `python3.13 --version` and later `python3.13 -m venv .venv`.

    macOS has no `python` command by default, only `python3`. That's normal.

    **The certificate error.** After a python.org install, the first download from Python (a `pip install`, or code that fetches a URL) can fail with:

    ```text
    ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
    ```

    (Typical wording; pip wraps it in its own retry messages.) The python.org build doesn't use macOS's certificate store; it needs its own list of trusted certificate authorities. Fix: open **Applications → Python 3.13** in Finder and double-click **Install Certificates.command**, once per installed version.

=== "Windows"

    From Python 3.14 on, python.org's recommended route on Windows is the **Python install manager** (also in the Microsoft Store as "Python Install Manager"). After installing it, you install interpreters with the `py` command:

    ```powershell
    py install 3.13
    py list
    py -V:3.13 --version
    ```

    The older route still works: download the classic Windows installer from [python.org/downloads](https://www.python.org/downloads/), and on its first screen tick **"Add python.exe to PATH"**. With the classic installer, `py --list` shows what's installed and `py -3.13` picks a version:

    ```text
    PS> py --list
     -V:3.13 *        Python 3.13 (64-bit)
     -V:3.12          Python 3.12 (64-bit)
    PS> py -3.13 --version
    Python 3.13.9
    ```

    (An example from a typical Windows machine; the `*` marks the default.) `py` is the **Python launcher for Windows**: it finds every Python you installed, whichever route you used. Check the [Windows page of the Python docs](https://docs.python.org/3/using/windows.html) for the current options; this area changed in 2025.

=== "Linux (Ubuntu/Debian)"

    ```bash
    sudo apt update
    sudo apt install python3 python3-venv python3-pip
    python3 --version
    ```

    ```text
    Python 3.13.16
    ```

    (My machine; Ubuntu 24.04 ships 3.12.) Use your distribution's packages for the interpreter, never `sudo pip` for libraries (you'll see why below).

    `python3-venv` is a separate package on Debian and Ubuntu. Without it, the first `python3 -m venv .venv` stops with a message like this (typical wording; I couldn't reproduce it here because my machine has the package):

    ```text
    The virtual environment was not created successfully because ensurepip is not
    available.  On Debian/Ubuntu systems, you need to install the python3-venv
    package using the following command.

        apt install python3.12-venv
    ```

    Fix: run the `apt install` it names (with `sudo`), delete the half-made `.venv` folder, and create the environment again.

Any 3.12 or newer is fine.

### `python`, `python3` and `py`

**What and why.** These are three command names for "start an interpreter". Which ones exist depends on the OS and how Python was installed, so instructions that say `python` fail on some machines and work on others:

- `python3` exists on macOS and Linux and always means Python 3.
- `python` may not exist at all (macOS, stock Ubuntu), or on very old systems it may be Python 2. On Windows the classic installer creates it when you tick "Add python.exe to PATH". On Ubuntu, the package `python-is-python3` adds it.
- `py` exists only on Windows.
- Inside an **activated virtual environment** (next section), `python` always exists on every OS and means that environment's interpreter. From then on this course writes `python`.

**Example.** On my Linux machine:

```bash
# Run this on your computer (or in Colab), not in the browser.
python3 --version
```

```text
Python 3.13.16
```

**How it goes wrong.** Typing a name that doesn't exist. On macOS (zsh) and Ubuntu (bash with the `command-not-found` helper) the messages look like this (typical examples, not from my machine):

```text
zsh: command not found: python
```

```text
Command 'python' not found, did you mean:
  command 'python3' from deb python3
  command 'python' from deb python-is-python3
```

Fix: type `python3`, or activate your venv, after which `python` exists.

On Windows the trap is different. A fresh Windows has a `python.exe` **App execution alias**: a stand-in that opens the Microsoft Store instead of running Python. Typing `python` in a terminal then prints (typical wording):

```text
Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut from Settings > Manage App Execution Aliases.
```

Fix: install Python as above, then either use `py`, or turn off the `python.exe` and `python3.exe` entries under **Settings → Apps → Advanced app settings → App execution aliases** so the real Python on `PATH` is found.

### PATH: how the terminal finds `python`

**What it is.** Your **shell** (the program reading what you type in the terminal: zsh on macOS, bash on Linux, PowerShell on Windows) keeps an **environment variable** called `PATH`. An environment variable is a named piece of text every program inherits from the shell that started it. `PATH` holds a list of folders. When you type a command name such as `python`, the shell looks in each folder, in order, and runs the **first** match.

**Kotlin and Android comparison.** You've done this when `adb` "wasn't found" until you added `platform-tools` to `PATH`. Same mechanism. What's new is that with Python, *which* folder wins decides which interpreter, and therefore which packages, you get.

**Example.** Ask the shell where commands come from. `which -a` lists every match in PATH order (macOS and Linux); on Windows use `where python` in cmd, or `Get-Command python -All` in PowerShell.

```bash
# Run this on your computer (or in Colab), not in the browser.
echo $PATH                 # PowerShell: $env:Path -split ';'
which -a python3           # PowerShell: Get-Command python -All
```

```text
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
/usr/bin/python3
/bin/python3
```

(My `PATH` was longer; I trimmed it to the standard folders.) The first line of `which -a` is the one that runs. Here the two matches are the same file: on current Linux distributions `/bin` is a symlink (a pointer) to `/usr/bin`, so the same program shows up under both names. Two matches in *different* real folders would mean two installations.

**Underneath: ask Python itself.** The most reliable check is to ask the running interpreter. The `sys` module (short for system) describes the interpreter that's executing your code:

```python
import sys

print(sys.version)
print(sys.executable)
print(sys.prefix)
print(sys.base_prefix)
print(sys.prefix != sys.base_prefix)
```

`sys.version` is the version text, `sys.executable` is the path of the interpreter program, `sys.prefix` is the folder its packages come from, and `sys.base_prefix` is the folder of the original installation. The two prefixes differ only inside a virtual environment, so the last line prints `True` there (`!=` is "not equal", as in Kotlin). On my laptop, inside the course environment you're about to build:

```text
3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]
/home/you/ai-course/.venv/bin/python
/home/you/ai-course/.venv
/usr
True
```

In the browser the version starts with `3.14`, the paths point inside the browser's sandbox rather than at your disk, and the last line is `False`.

**Where `import` looks: `sys.path`.** One more attribute matters from day one. `sys.path` is the list of folders Python searches, in order, when you write `import something`. Its **first** entry is the folder of the script you ran; the standard library comes next, and site-packages last. A script that prints the first and last entries, run from my venv:

```bash
# Run this on your computer (or in Colab), not in the browser.
printf 'import sys\nprint(sys.path[0])\nprint(sys.path[-1])\n' > showpath.py
python showpath.py
```

```text
/home/you/experiments
/home/you/ai-course/.venv/lib/python3.13/site-packages
```

(`[0]` is the first item of a list and `[-1]` the last; Day 2 explains indexing.) Because your own folder is searched first, a file of yours can hide a library with the same name.

!!! warning "Common mistake: naming your file after a library"
    You save your first NumPy experiment as `numpy.py`, or a dice script next to a file called `random.py`. `import numpy` then finds *your* file first and imports it instead of the real library. Python 3.13 even says so:

    ```text
    Traceback (most recent call last):
      File "/home/you/experiments/numpy.py", line 2, in <module>
        import numpy as np
      File "/home/you/experiments/numpy.py", line 4, in <module>
        print(np.array([1, 2, 3]))
              ^^^^^^^^
    AttributeError: module 'numpy' has no attribute 'array' (consider renaming '/home/you/experiments/numpy.py' if it has the same name as a library you intended to import)
    ```

    Notice the traceback's two frames are both *your* file: the script imported itself. For a standard-library name the hint is even clearer: `AttributeError: module 'random' has no attribute 'randint' (consider renaming '/home/you/experiments/random.py' since it has the same name as the standard library module named 'random' and prevents importing that standard library module)`. Fix: rename the file (`np_test.py`), and delete the `__pycache__` folder next to it if one was created. Never name files `numpy.py`, `pandas.py`, `torch.py`, `random.py`, `math.py` or `json.py`.

**How it goes wrong.** You install a package with one Python and run another. On my machine, the system Python has NumPy and the environment's Python doesn't:

```bash
# Run this on your computer (or in Colab), not in the browser.
python3 -c "import numpy; print(numpy.__version__)"
.venv/bin/python -c "import numpy"
```

```text
2.5.3
Traceback (most recent call last):
  File "<string>", line 1, in <module>
    import numpy
ModuleNotFoundError: No module named 'numpy'
```

Same machine, same moment, two answers. Whenever you see `ModuleNotFoundError` for something you're sure you installed, print `sys.executable` from the failing program and compare it with the Python you ran `pip` with.

## Virtual environments

**What it is.** A **virtual environment** (venv) is a folder, by convention named `.venv` inside your project, that holds a private place to install packages plus a small `python` launcher that uses it. Packages installed while it's active go into that folder and nowhere else.

**Why it exists.** Out of the box, each Python installation has *one* shared folder of installed packages, called **site-packages**. If project A needs `numpy 1.26` and project B needs `numpy 2.5`, a single shared folder can't satisfy both. A venv gives every project its own site-packages. It also makes a project reproducible: delete the folder, recreate it from a list of packages, and you get the same setup back. Every project in this course, and every ML repo you'll clone, assumes you work inside one.

**Kotlin comparison.** In Gradle, each module declares its own dependencies, and Gradle builds a separate classpath per module from a shared download cache. You never "install" a library into the JDK. Python has no build step, so the "classpath" is a real folder on disk that the interpreter reads at run time, and a venv is how you give each project its own. What's the same: per-project dependency sets. What's different: a venv is something you create and activate yourself, and Python doesn't stop you from forgetting.

**Example: create it.**

```bash
# Run this on your computer (or in Colab), not in the browser.
mkdir ai-course
cd ai-course
python3 -m venv .venv        # Windows: py -m venv .venv
```

`python3 -m venv .venv` means "run the standard-library module `venv`, and make an environment in the folder `.venv`". The leading dot hides the folder in normal listings on macOS and Linux.

**Underneath: what's in `.venv`.** Look inside (output from Linux; macOS is the same, Windows has `Scripts\` instead of `bin/` and `Lib\site-packages` instead of `lib/python3.13/site-packages`):

```bash
# Run this on your computer (or in Colab), not in the browser.
ls .venv .venv/bin                     # PowerShell: dir .venv, .venv\Scripts
cat .venv/pyvenv.cfg                   # PowerShell: type .venv\pyvenv.cfg
ls .venv/lib/python3.13/site-packages  # PowerShell: dir .venv\Lib\site-packages
```

(`ls` lists a folder and `cat` prints a file.)

```text
.venv:
bin  include  lib  lib64  pyvenv.cfg

.venv/bin:
Activate.ps1  activate  activate.csh  activate.fish  pip  pip3  pip3.13  python  python3  python3.13

home = /usr/bin
include-system-site-packages = false
version = 3.13.16
executable = /usr/bin/python3.13
command = /usr/bin/python3 -m venv /home/you/ai-course/.venv
pip
pip-26.2.1.dist-info
```

Piece by piece:

- `bin/python` is not a copy of Python. On macOS and Linux it's a **symlink** (a file that points at another file) to the real interpreter, here `/usr/bin/python3`.
- `pyvenv.cfg` is the key. When Python starts, it looks for this file next to (or one folder above) the program that was launched. If it finds one, it says "I'm inside a venv": it sets `sys.prefix` to `.venv` and looks for packages in `.venv/lib/python3.13/site-packages` instead of the system folder. `include-system-site-packages = false` means "and ignore the system's packages entirely".
- `site-packages` starts with only pip in it. Everything you install lands here.
- `include/` is where packages that compile C code could put C header files; it stays empty for everything in this course. `lib64` is a symlink to `lib` that Linux creates on 64-bit systems, so both names reach the same folder.
- `activate` (bash/zsh), `activate.fish`, `activate.csh` and `Activate.ps1` (PowerShell) are activation scripts for different shells.
- Since Python 3.13, `venv` also writes a `.gitignore` containing `*` inside `.venv`, so git ignores the folder even if you forget to.

**Example: activate, check, deactivate.**

```bash
# Run this on your computer (or in Colab), not in the browser.
source .venv/bin/activate     # Windows PowerShell: .venv\Scripts\Activate.ps1
                              # Windows cmd:        .venv\Scripts\activate.bat
which python
echo $VIRTUAL_ENV
deactivate
which python3
```

```text
/home/you/ai-course/.venv/bin/python
/home/you/ai-course/.venv
/usr/bin/python3
```

After `deactivate` I ask for `python3`, because a stock Ubuntu or macOS has no plain `python` and `which python` would print nothing.

While active, your prompt starts with `(.venv)`. `deactivate` undoes it.

<figure class="diagram" markdown>
![Animation of the shell running python hello.py. Left panel, before activation: PATH lists /usr/local/bin, /usr/bin and /bin; the search highlight finds no python in /usr/local/bin, then finds python in /usr/bin, which runs /usr/bin/python and imports from the system site-packages shared by every project. Right panel, after source .venv/bin/activate: PATH starts with ~/ai-course/.venv/bin, where python is found immediately; it reads .venv/pyvenv.cfg and imports from .venv/lib/python3.13/site-packages, this project's packages only](../../assets/img/phase-1/day01-venv-path.svg)
<figcaption>Animated: activation puts the venv's bin folder at the front of PATH, so the shell finds the venv's python first.</figcaption>
</figure>

**Underneath: what activation changes.** Activation is much less magic than it looks. Compare `PATH` before and after on my machine (trimmed):

```text
before: /usr/local/bin:/usr/bin:/bin
after:  /home/you/ai-course/.venv/bin:/usr/local/bin:/usr/bin:/bin
```

That's all: it puts `.venv/bin` at the front of `PATH`, sets `VIRTUAL_ENV`, changes the prompt, and defines a `deactivate` command. Nothing is copied or installed. Which means you can skip activation entirely and call the venv's interpreter by its path; it finds `pyvenv.cfg` on its own:

```bash
# Run this on your computer (or in Colab), not in the browser.
.venv/bin/python whereami.py     # Windows: .venv\Scripts\python whereami.py
```

That's handy in scripts, cron jobs and CI, where nobody types `activate`.

**Deleting it.** A venv is disposable. Delete the folder and it's gone; nothing is registered anywhere else.

```bash
# Run this on your computer (or in Colab), not in the browser.
deactivate                # if it's active
rm -rf .venv              # Windows PowerShell: Remove-Item -Recurse -Force .venv
```

**How it goes wrong.** Two classic mistakes.

1. *Forgetting to activate*, then installing. Your packages go into a system Python, or the system refuses (see "never `sudo pip`" below). Check the prompt for `(.venv)`, or run `which python` (Windows: `Get-Command python`).
2. *Moving or renaming the project folder.* A venv stores absolute paths. Here I created a venv in `old/`, renamed the folder to `new/`, and tried to use it:

```bash
# Run this on your computer (or in Colab), not in the browser.
python3 -m venv old/.venv
mv old new
./new/.venv/bin/pip --version
head -1 new/.venv/bin/pip
source new/.venv/bin/activate
which python3
```

```text
bash: ./new/.venv/bin/pip: cannot execute: required file not found
#!/home/you/old/.venv/bin/python3
/usr/bin/python3
```

`pip` fails because its first line, the **shebang** (`#!` followed by the program that should run this file), still names `old/.venv/bin/python3`, which no longer exists; `head -1` prints a file's first line. Worse, activation "works" but puts the *old* path on `PATH`, so `python3` silently falls through to the system's `/usr/bin/python3`. On macOS, zsh reports the first error as `zsh: /Users/you/new/.venv/bin/pip: bad interpreter: /Users/you/old/.venv/bin/python3: no such file or directory` (typical example, not from my machine). The fix is never to repair a venv: delete it and recreate it from `requirements.txt` (next section).

!!! warning "Common mistake: PowerShell won't run Activate.ps1"
    On Windows, the first `.venv\Scripts\Activate.ps1` often fails with
    `File C:\...\.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`
    PowerShell blocks scripts by default. Allow scripts you created yourself, for your user only, once:
    `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
    then activate again. (Or use cmd and `.venv\Scripts\activate.bat`.)

## Installing packages with pip

**What it is.** **pip** is Python's package installer. It downloads packages from **PyPI** (the Python Package Index, pypi.org) and unpacks them into the site-packages of the Python that runs it. Most packages arrive as **wheels**: zip files with a `.whl` extension holding ready-to-use code, including compiled C extensions built for your OS, CPU and Python version. If no matching wheel exists, pip falls back to a **source distribution** and tries to compile it on your machine, which usually fails without a C compiler.

A wheel's file name says exactly what it was built for. The NumPy wheel pip picked for my machine:

```text
numpy-2.5.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

Read it in dashes: package `numpy`, version `2.5.4`, then `cp313` twice (built for **C**Python 3.13, the same tag as in `cpython-313.pyc`), then the platform: `manylinux_2_27`/`2_28` means "most Linux distributions from about 2018 on", and `x86_64` is the CPU type. A Mac with Apple Silicon gets a `macosx_..._arm64` wheel instead, Windows a `win_amd64` one. If no wheel has your Python tag, you're in source-build territory.

**Why you'll use it daily.** Every library in this course (NumPy, pandas, matplotlib, PyTorch) comes from PyPI through pip or a pip-compatible tool.

**Kotlin comparison.**

```kotlin
// build.gradle.kts
dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
}
```

```bash
python -m pip install "numpy==2.5.4"
```

PyPI plays the role of Maven Central, and a pinned version plays the role of the version in the coordinate. What's different: Gradle resolves dependencies on every build and keeps them in a cache, while pip resolves **once**, at install time, and writes files into your environment. Nothing re-checks them later unless you ask (`pip check`, below). And there is no `build.gradle.kts` that pip reads automatically; `requirements.txt` is a file you write and pass to pip yourself.

!!! tip "Always `python -m pip`, not bare `pip`"
    `python -m pip install x` runs pip *with the exact interpreter named `python`*, so the package lands where that `python` will look. A bare `pip` command is found on `PATH` separately and can belong to a different Python. Inside an activated venv both are the same, but `python -m pip` is never wrong.

**Example: the everyday commands.** With the venv active (all output real, from my machine; I've hidden pip's download progress lines):

```bash
# Run this on your computer (or in Colab), not in the browser.
python -m pip install numpy pandas matplotlib
```

```text
Successfully installed contourpy-1.4.0 cycler-0.12.1 fonttools-4.67.0 kiwisolver-1.5.1 matplotlib-3.11.2 numpy-2.5.4 packaging-26.3 pandas-3.1.0 pillow-13.0.0 pyparsing-3.3.3 python-dateutil-3.0.0 six-1.17.0 tzdata-2026.5
```

You asked for three packages and got thirteen: the others are their **dependencies** (packages they need), resolved and installed automatically, like Gradle's transitive dependencies.

`pip list` shows what's installed; `pip show` shows one package's details:

```bash
# Run this on your computer (or in Colab), not in the browser.
python -m pip list
python -m pip show pandas
```

```text
Package         Version
--------------- -------
contourpy       1.4.0
cycler          0.12.1
fonttools       4.67.0
kiwisolver      1.5.1
matplotlib      3.11.2
numpy           2.5.4
packaging       26.3
pandas          3.1.0
pillow          13.0.0
pip             26.2.1
pyparsing       3.3.3
python-dateutil 3.0.0
six             1.17.0
tzdata          2026.5
Name: pandas
Version: 3.1.0
Summary: Powerful data structures for data analysis, time series, and statistics
Location: /home/you/ai-course/.venv/lib/python3.13/site-packages
Requires: numpy, python-dateutil
Required-by:
```

(I cut `pip show`'s author, licence and home-page lines.) `Location` confirms it went into the venv. `Requires` lists its dependencies; `Required-by` lists what depends on it.

**Version specifiers** say which versions you accept:

| Specifier | Meaning |
|---|---|
| `numpy` | the newest version that fits everything else |
| `numpy==2.5.4` | exactly 2.5.4 |
| `numpy>=2.3` | 2.3 or newer |
| `numpy>=2.3,<3` | at least 2.3, but not 3.x |
| `numpy~=2.5.0` | "compatible release": 2.5.0 or newer, but below 2.6 |

Quote them in the shell (`"numpy>=2.3,<3"`), because `>` and `<` mean redirection to bash and PowerShell. Here's a range in action, in a fresh venv:

```bash
# Run this on your computer (or in Colab), not in the browser.
python -m pip install "numpy>=2.3,<2.5"
```

```text
Successfully installed numpy-2.4.6
```

2.5.4 exists, but the range excludes it, so pip picked the newest version below 2.5. When your pins contradict what a package needs, pip refuses rather than installing something broken:

```bash
# Run this on your computer (or in Colab), not in the browser.
python -m pip install "pandas==3.1.0" "numpy<2.0"
```

```text
The conflict is caused by:
    The user requested numpy<2.0
    pandas 3.1.0 depends on numpy>=2.0.2; python_version < "3.14"
...
ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts
```

(I cut pip's build attempts and its "To fix this you could try to" advice.) Read "The conflict is caused by": two requirements that can't both hold. Fix: drop or loosen one pin.

The rest of the daily set:

```bash
# Run this on your computer (or in Colab), not in the browser.
python -m pip install --upgrade pandas      # or -U: move to the newest allowed version
python -m pip uninstall -y pandas           # -y: don't ask "Proceed (Y/n)?"
python -m pip check                         # do the installed versions agree with each other?
```

```text
Found existing installation: pandas 3.1.0
Uninstalling pandas-3.1.0:
  Successfully uninstalled pandas-3.1.0
No broken requirements found.
```

(The upgrade printed only `Requirement already satisfied` lines, because pandas was already the newest version.)

`pip check` earns its keep when something *is* broken. Installing with `--no-deps` (skip dependencies) is one way to get there; it's also what happens after a half-finished install or a careless uninstall:

```bash
# Run this on your computer (or in Colab), not in the browser.
python -m pip install --no-deps "pandas==3.1.0"
python -m pip check
```

```text
Successfully installed pandas-3.1.0
pandas 3.1.0 requires python-dateutil, which is not installed.
```

`import pandas` would now crash. Fix: `python -m pip install pandas` again without `--no-deps`, or rebuild the venv.

**Underneath: what an install does.** pip (1) reads the requirement, (2) asks PyPI which versions exist, (3) picks versions that satisfy your package *and* all its dependencies, (4) downloads the wheel for your OS, CPU and Python, and (5) unzips it into site-packages, adding a `*.dist-info` folder that records the version and every file it wrote. After installing `numpy==2.3.4` into an empty venv, the folder holds:

```text
numpy
numpy-2.3.4.dist-info
numpy.libs
pip
pip-26.2.1.dist-info
```

`import numpy` later finds the `numpy` folder there. `numpy.libs` holds compiled shared libraries bundled inside the wheel, such as OpenBLAS (the fast linear-algebra library NumPy calls for matrix maths), so you don't need to install them separately. `pip list` and `pip show` read the `.dist-info` folders. `pip uninstall` deletes the files that `.dist-info` lists, and nothing else. That last point matters: uninstalling pandas does **not** remove NumPy or `python-dateutil`, which pandas pulled in. They stay, because pip doesn't know whether you also want them. To get back to a clean state, recreate the venv.

### Recording and rebuilding: `requirements.txt`

`pip freeze` prints every installed package with its exact version, in the format pip reads back:

```bash
# Run this on your computer (or in Colab), not in the browser.
python -m pip freeze > requirements.txt
cat requirements.txt
```

```text
contourpy==1.4.0
cycler==0.12.1
fonttools==4.67.0
kiwisolver==1.5.1
matplotlib==3.11.2
numpy==2.5.4
packaging==26.3
pandas==3.1.0
pillow==13.0.0
pyparsing==3.3.3
python-dateutil==3.0.0
six==1.17.0
tzdata==2026.5
```

`>` redirects the output into a file. Commit this file; never commit `.venv`. Anyone (including you, next month, on a new laptop) rebuilds the identical environment with:

```bash
# Run this on your computer (or in Colab), not in the browser.
rm -rf .venv
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # Windows: .venv\Scripts\python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

(On Windows, delete the folder with `Remove-Item -Recurse -Force .venv` and create it with `py -m venv .venv`.)

```text
Successfully installed contourpy-1.4.0 cycler-0.12.1 fonttools-4.67.0 kiwisolver-1.5.1 matplotlib-3.11.2 numpy-2.5.4 packaging-26.3 pandas-3.1.0 pillow-13.0.0 pyparsing-3.3.3 python-dateutil-3.0.0 six-1.17.0 tzdata-2026.5
No broken requirements found.
```

A frozen file pins *everything*, including dependencies you never asked for. Some projects prefer a short hand-written file with only the top-level packages (`numpy>=2.3`, `pandas>=3.0`), which is easier to read but less exactly reproducible. For this course, freeze.

**How it goes wrong.**

*Never `sudo pip`.* On modern Linux distributions and Homebrew Python, installing into the system Python is blocked, because the OS's own tools depend on its packages. This is the real message (from Ubuntu's Python 3.12):

```text
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try apt install
    python3-xyz, where xyz is the package you are trying to
    install.

    If you wish to install a non-Debian-packaged Python package,
    create a virtual environment using python3 -m venv path/to/venv.
    Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make
    sure you have python3-full installed.
    ...
note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
```

The message is right: don't pass `--break-system-packages`, and don't reach for `sudo`. Activate your venv and install there. `sudo pip` on older systems overwrote OS files as root, which is how people broke their package manager.

*An old pin with no wheel for your Python.* Old tutorials pin old versions. `numpy==1.26.4` has no wheel for Python 3.13, so pip would try to compile it from source and print a long compiler failure. Telling pip "wheels only, never build" (`--only-binary=:all:`) makes the real cause readable:

```bash
# Run this on your computer (or in Colab), not in the browser.
python -m pip install "numpy==1.26.4" --only-binary=:all:
```

```text
ERROR: Could not find a version that satisfies the requirement numpy==1.26.4 (from versions: 2.1.0, 2.1.1, 2.1.2, 2.1.3, 2.2.0, 2.2.1, 2.2.2, 2.2.3, 2.2.4, 2.2.5, 2.2.6, 2.3.0, 2.3.1, 2.3.2, 2.3.3, 2.3.4, 2.3.5, 2.4.0rc1, 2.4.1, 2.4.2, 2.4.3, 2.4.4, 2.4.5, 2.4.6, 2.5.0rc1, 2.5.0, 2.5.1, 2.5.2, 2.5.3, 2.5.4)
ERROR: No matching distribution found for numpy==1.26.4
```

The list shows only versions with wheels for *this* Python. Fix: loosen the pin to a version in the list, or use the Python version the old project expects. A typo or a version that doesn't exist at all (`numpy==99.0`) ends the same way: `ERROR: No matching distribution found for numpy==99.0`.

??? tip "Optional: `uv`, a faster drop-in"
    [uv](https://docs.astral.sh/uv/) is a newer tool that does the work of `venv` and `pip` (and more), written in Rust and much faster. Many modern ML repos use it. The same environment, with uv:

    ```bash
    # Run this on your computer (or in Colab), not in the browser.
    uv venv
    uv pip install numpy pandas matplotlib
    ```

    ```text
    Using CPython 3.14.6
    Creating virtual environment at: .venv
    Activate with: source .venv/bin/activate
    Resolved 13 packages in 714ms
    ...
    Installed 13 packages in 30ms
    ```

    On my machine pip took 23 seconds for the same install and uv under 4 (network-dependent; yours will differ). The result is an ordinary `.venv` that `python` and `activate` treat exactly as before. Either tool is fine for this course; the lessons write `python -m pip`.

## Your first program

**What it is.** A **script** is a `.py` file you run with `python file.py`. Python executes it top to bottom, one statement at a time.

**Why.** Scripts are how you'll run training jobs, data-cleaning steps and anything you want to repeat exactly.

**Kotlin comparison.**

```kotlin
fun main() {
    val name = "Android engineer"
    val years = 8
    println("Hello, $name! $years years of Kotlin, day 1 of AI.")
    for (day in 1..3) {
        println("Day $day: learning")
    }
    println("Done.")
}
```

Create `hello.py` in your `ai-course` folder with this content (you can also press ▶ Run to try it here first):

```python
# hello.py: your first Python program
name = "Android engineer"
years = 8

print(f"Hello, {name}! {years} years of Kotlin, day 1 of AI.")

for day in range(1, 4):
    print(f"Day {day}: learning")

print("Done.")
```

Run it from the terminal, in the folder that contains it:

```bash
python hello.py
```

```text
Hello, Android engineer! 8 years of Kotlin, day 1 of AI.
Day 1: learning
Day 2: learning
Day 3: learning
Done.
```

Line by line, against the Kotlin:

- `#` starts a comment, like `//`.
- No `fun main()`. A script's top level *is* the program; Python runs it from line 1.
- No `val` or `var`. `name = "Android engineer"` creates the name on first assignment. (Day 2 explains what a Python variable really is.)
- `print(...)` is `println(...)`.
- `f"...{name}..."` is an **f-string** (formatted string): the `f` before the quote turns `{expression}` into its value, like Kotlin's `"...$name..."` or `"${expr}"`.
- `for day in range(1, 4):` loops over 1, 2, 3. `range(1, 4)` stops *before* 4, like Kotlin's `1 until 4`, not `1..4`. Day 2 covers loops fully.
- **No braces.** The colon `:` opens a block, and the block is every following line indented more than the `for`. `print("Done.")` is back at the left edge, so it runs once, after the loop. The convention is four spaces per level.

**Underneath.** Indentation isn't style in Python; it's grammar. The compile step turns indentation changes into the same "begin block / end block" markers that braces are in Kotlin. That's why an indentation mistake is a `SyntaxError` family error (reported before anything runs), not a runtime surprise.

**How it goes wrong.** Two errors you'll certainly meet this week.

*Indentation.* Forget to indent the loop body:

```text
  File "/home/you/ai-course/bad_indent.py", line 2
    print(f"Day {day}")
    ^^^^^
IndentationError: expected an indented block after 'for' statement on line 1
```

Indent a line that shouldn't be (here, line 3 indented further than line 2 for no reason):

```text
  File "/home/you/ai-course/mixed.py", line 3
    print(f"Day {day}")
IndentationError: unexpected indent
```

Fix: make every line of a block start at the same column, and configure your editor to insert four spaces when you press Tab. Mixing tabs and spaces causes the same errors in invisible ways.

*Wrong folder or wrong name.* Python looks for the file relative to the folder the terminal is in:

```text
python: can't open file '/home/you/ai-course/helo.py': [Errno 2] No such file or directory
```

`[Errno 2]` is the operating system's "no such file" code. Check the spelling, and run `ls` (`dir` on Windows) to see what's actually in the current folder, or `cd` into the right one.

## Four ways to run Python

You've run a script. There are three more ways, and you'll use all of them.

### `python -m module`: run a module by name

**What it is.** `python -m name` finds a module (a Python file or package) called `name` wherever this interpreter looks for imports, and runs it as the main program. You've already used it three times: `python -m venv`, `python -m pip`, `python -m ipykernel` (coming up).

**Why.** It guarantees that the tool runs with *this* interpreter, and it lets you run tools that ship as modules without needing a separate command on `PATH`. A useful example from the standard library, `timeit`, which times a small piece of code:

```bash
# Run this on your computer (or in Colab), not in the browser.
python -m timeit "sum(range(1000))"
```

```text
20000 loops, best of 5: 9.22 usec per loop
```

(A timing: it depends on the machine; yours will differ.) How `-m` decides what "running as the main program" means is tied to a special variable called `__name__`, which [Day 3a](day-03a-classes-modules.md) explains.

**How it goes wrong.** `-m` takes a *module name*, not a file name. Adding `.py` gives a confusing result: Python imports `hello` (which runs your whole script!) and then complains that `hello` has no sub-module called `py`:

```text
Hello, Android engineer! 8 years of Kotlin, day 1 of AI.
Day 1: learning
Day 2: learning
Day 3: learning
Done.
/home/you/ai-course/.venv/bin/python: Error while finding module specification for 'hello.py' (ModuleNotFoundError: __path__ attribute not found on 'hello' while trying to find 'hello.py'). Try using 'hello' instead of 'hello.py' as the module name.
```

Use `python hello.py` for a file, `python -m hello` for a module name.

**`python -c`** runs a program given as a string, handy for one-line checks:

```bash
python -c "print(2 ** 100)"
```

```text
1267650600228229401496703205376
```

(`**` is "to the power of". Python integers have no size limit, unlike Kotlin's `Int` and `Long`; Day 2 explains.)

### The REPL

**What it is.** Type `python` with no arguments and you get the **REPL** (Read–Eval–Print Loop): Python reads one statement, runs it, prints the result, and waits for the next. `>>>` is the prompt; `...` means "continue the statement you started".

**Why.** Quick experiments: what does this function return, what type is this, does this method exist. It's the Kotlin REPL or a scratch file, but instant.

```pycon
>>> 2 ** 10
1024
>>> import math
>>> math.sqrt(2)
1.4142135623730951
>>> type(3.0)
<class 'float'>
>>> 21 * 2
42
>>> _ + 1
43
>>> help(math.sqrt)
Help on built-in function sqrt in module math:

sqrt(x, /)
    Return the square root of x.

>>> exit()
```

**Underneath.** The REPL echoes the value of any bare expression, which a script does not: in a script, `2 ** 10` on its own line computes 1024 and throws it away silently. `_` holds the last echoed value. `type(x)` tells you what kind of object `x` is. `help(thing)` shows its documentation (press <kbd>q</kbd> to leave the help pager if it opens one); `dir(thing)` lists the names it contains. Leave with `exit()` or <kbd>Ctrl</kbd>+<kbd>D</kbd> (<kbd>Ctrl</kbd>+<kbd>Z</kbd> then <kbd>Enter</kbd> on Windows).

**How it goes wrong.** Errors in the REPL print a traceback and return you to `>>>`, with everything defined so far still there. In a fresh session:

```pycon
>>> "7" + 1
Traceback (most recent call last):
  File "<python-input-0>", line 1, in <module>
    "7" + 1
    ~~~~^~~
TypeError: can only concatenate str (not "int") to str
```

`<python-input-0>` means "the first thing you typed in this session" (inputs are counted from 0, so the fifth would be `<python-input-4>`). The other trap: anything you type in the REPL is gone when you exit. Copy what works into a file.

**Pasting code.** Since Python 3.13 the REPL accepts a pasted block of several lines (a whole function, say) as one input, keeps its indentation, and shows colour and multi-line editing; press <kbd>F3</kbd> for an explicit paste mode. Older REPLs need a blank line after an indented block before they run it.

**`python -i`: a script, then the REPL.** `python -i script.py` runs the script and then, instead of exiting, drops you into the REPL with all the script's variables still alive. It's the quickest way to poke at a script's results, or at the state it was in when it crashed. With a two-line `stats_i.py` that computes `scores` and `average`:

```text
$ python -i stats_i.py
>>> average
85.0
>>> scores
[72, 88, 95]
>>> exit()
```

### Jupyter notebooks

**What it is.** A **Jupyter notebook** (`.ipynb` file) is a document made of **cells**. A code cell holds Python you can run on its own; a Markdown cell holds notes. Output, including tables and charts, appears under each cell and is saved in the file. The Python that runs the cells is a separate process called the **kernel**. The browser page is only a front end that sends your code to the kernel and shows what comes back.

**Why.** ML work is exploratory: load data, look at it, plot something, decide the next step. A notebook lets you run one step at a time, keep the data in memory between steps, and see results inline, without re-running a whole script that takes a minute to load a dataset. Most ML tutorials, Kaggle and Colab are notebooks.

**Kotlin comparison.** The nearest things are Android Studio's Kotlin scratch files or the Kotlin Notebook plugin in IntelliJ. What's different is the separate kernel: a notebook is a long-running Python process plus an editor that talks to it.

**Setting it up.** Install JupyterLab (the notebook app) into your venv, and register the venv as a kernel so notebooks use *its* packages:

```bash
# Run this on your computer (or in Colab), not in the browser.
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name ai-course --display-name "Python (ai-course)"
jupyter lab
```

```text
Installed kernelspec ai-course in /home/you/.local/share/jupyter/kernels/ai-course
```

**ipykernel** is the package that lets a Python act as a Jupyter kernel. In the second command, `--user` registers the kernel for your user account only (no admin rights needed), `--name ai-course` is the internal folder name Jupyter stores it under, and `--display-name` is the label you'll see in menus.

`jupyter lab` opens a browser tab. Click **Python (ai-course)** under *Notebook*, type into a cell and press <kbd>Shift</kbd>+<kbd>Enter</kbd> to run it and move to the next cell. Each cell shows its last expression's value, like the REPL.

**Two modes and the keys worth learning.** A cell is in **edit mode** when the cursor is inside it (typing changes the code) and **command mode** when it's selected but not being edited (keys act on cells). <kbd>Esc</kbd> switches to command mode, <kbd>Enter</kbd> back to edit mode. In command mode: <kbd>A</kbd> inserts a cell above, <kbd>B</kbd> below, <kbd>D</kbd> <kbd>D</kbd> (twice) deletes the cell, <kbd>M</kbd> turns it into a Markdown (notes) cell, <kbd>Y</kbd> back into a code cell. <kbd>Shift</kbd>+<kbd>Enter</kbd> runs a cell in either mode. The same keys work in VS Code's notebook view.

**Underneath: what a kernel is.** The registration wrote a small file, `kernel.json`, that tells Jupyter how to start the kernel. Mine says:

```text
{
 "argv": [
  "/home/you/ai-course/.venv/bin/python",
  "-Xfrozen_modules=off",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python (ai-course)",
 ...
}
```

So "the kernel" is your venv's `python`, started with `-m ipykernel_launcher`. (`-Xfrozen_modules=off` is an interpreter option that keeps the notebook debugger able to step into standard-library code, and `{connection_file}` is a placeholder Jupyter replaces with the path of a small file holding the network ports the browser page and the kernel talk over.) It stays running while the notebook is open, and every cell you run executes in that one process, sharing all its variables. Each run of a cell bumps a counter shown as `[1]:`, `[2]:` and so on next to the cell: the **execution count**, the order in which cells actually ran.

<figure class="diagram" markdown>
![Animation: a notebook with three cells, A: lr = 0.1, B: lr = lr * 10, C: print(lr), and a kernel box showing the value of lr. You run A and the counter shows [1], lr is 0.1. You run B, counter [2], lr is 1.0. You run B again, its counter becomes [3], lr is 10.0. You run C, counter [4], and the output under C is 10.0. A note explains that reading top to bottom suggests 1.0, but the kernel holds 10.0, and that the gap in the counters is the clue](../../assets/img/phase-1/day01-notebook-state.svg)
<figcaption>Animated: the kernel remembers what ran, in the order it ran, not what the page shows.</figcaption>
</figure>

**How it goes wrong: hidden state.** Cells can run in any order, any number of times, so the kernel's memory can drift away from what the page shows. Try it with this page's ▶ Run buttons, which share variables in the same way (blocks on a page are like cells in one notebook). Run the first block, then run the second block **twice**:

```python
lr = 0.1          # cell A: a learning rate
```

```python
lr = lr * 10      # cell B: "scale it up"
lr
```

The first run of cell B prints `1.0`. The second prints `10.0`, though nothing on the page changed. In a real notebook this is how a "working" analysis turns out to depend on a cell you ran twice an hour ago, or on a cell you since deleted.

The fix is a habit: before trusting a result or sharing a notebook, use **Kernel → Restart Kernel and Run All Cells**. That throws away the kernel's memory and runs every cell once, top to bottom. If the notebook still gives the same answers, it's honest. (On this site, the Reset button next to ▶ Run does the same: it forgets every variable on the page.)

**Magics and shell commands.** Lines starting with `%` are **magics**: commands for the notebook itself, not Python. Lines starting with `!` run a shell command. The ones you'll use:

- `%timeit expr` runs `expr` many times and reports how long one run takes.
- `%who` lists the variables the kernel holds. In a fresh kernel after running `import time` and `x = 5`, it printed `time	 x`: modules count as variables too.
- `%pip install x` installs into the *kernel's* Python.
- `!ls` (or `!dir`) runs a shell command and shows its output.

All of these come from **IPython** (Interactive Python): an enhanced interactive Python shell, and the engine inside Jupyter's Python kernel. Magics are an IPython feature, not part of Python itself, which is why `%timeit` is a syntax error in a `.py` file. Typing `ipython` in a terminal gives you the same shell without the notebook.

Here's a first look at the steering-wheel idea. This cell sums a million numbers twice: once with Python's built-in `sum` looping in the eval loop, once with NumPy's C loop. `np.arange(1_000_001)` makes an array of 0 to 1,000,000 (underscores in numbers are only for readability), and `.sum()` adds it up:

```python
import numpy as np

sum(range(1_000_001))
np.arange(1_000_001).sum()
%timeit sum(range(1_000_001))
%timeit np.arange(1_000_001).sum()
```

The first two lines show the same total: `500000500000` from Python, and `np.int64(500000500000)` from NumPy (a NumPy 64-bit integer; Day 4 explains these types). In IPython on my laptop the timings were:

```text
12.9 ms ± 1.44 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
795 μs ± 59.8 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

That's about 16 times faster for NumPy, and NumPy also had to *create* the array. Timings depend on the machine; yours will differ. The ▶ Run button supports `%timeit` too and prints its own shorter format, marked `[browser timing]`. Day 4 takes this measurement apart properly.

!!! warning "Common mistake: `!pip install` in a notebook"
    `!pip install x` runs whichever `pip` the shell finds on `PATH`, which may belong to a different Python than the kernel. The notebook then says `ModuleNotFoundError: No module named 'x'` right after a "successful" install. Use `%pip install x`, which always targets the kernel, and restart the kernel afterwards. If an import fails in a notebook but works in your terminal, run `import sys; sys.executable` in a cell (the `;` puts two statements on one line, here an import and then an expression to display): the notebook is using a different interpreter, usually because the wrong kernel is selected (top-right corner of JupyterLab).

**Notebooks vs scripts.** A notebook file is JSON (text with nested `{...}` structures) containing code, outputs and images. That makes it great for exploring and explaining, and awkward for version control (diffs are noisy) and for code you import from elsewhere. The working pattern in ML: explore in a notebook; once a piece of code works, move it into a `.py` file and import it from the notebook. Training runs that take hours belong in scripts. That pattern has one trap, next.

#### Editing a file the notebook has imported

**What it is.** When a notebook imports your own module (say `prep.py`, with a `scale` function), the kernel loads it once and keeps that copy in memory. Editing and saving `prep.py` afterwards does **not** change what the notebook runs, and running `import prep` again does nothing, because Python imports each module only once per process ([Day 3a](day-03a-classes-modules.md) shows the cache, `sys.modules`).

**Why it matters.** The "explore in a notebook, move code to a `.py` file" workflow means you'll edit imported files constantly. Without knowing this, you fix a bug, re-run the cell, see the old result, and start doubting your fix.

**Kotlin comparison.** It's like changing a library module's source while the app keeps running on the device: nothing changes until you rebuild and relaunch. Android Studio's Apply Changes is the closest thing to the fix below, swapping code into a running process.

**Example: the stale result.** In a real kernel, with `prep.py` containing `return x * 10`:

```text
In [1]: import prep
        prep.scale(3)
Out[1]: 30

        # now edit prep.py in your editor to return x * 100, and save

In [3]: import prep
        prep.scale(3)
Out[3]: 30
```

Still 30. (Cell 2 was the one that rewrote the file.)

**Underneath.** `import prep` first checks the kernel's table of already-loaded modules. `prep` is there, so Python hands back the old module object without reading the file. The new code exists only on disk. A fresh process has an empty table, so a **Kernel → Restart** followed by the import gives `300`. Restarting costs you every variable, though, which is why IPython has an extension that re-reads changed files for you:

```text
In [1]: %load_ext autoreload
        %autoreload 2

In [2]: import prep
        prep.scale(3)
Out[2]: 30

        # edit prep.py to return x * 100, and save

In [4]: prep.scale(3)
Out[4]: 300
```

`%load_ext autoreload` loads the extension; `%autoreload 2` tells it to check, before *every* cell, whether any imported module's file changed, and reload it if so.

**How it goes wrong.** Turning autoreload on *after* the import and the edit didn't help in my run: the extension treats what's loaded when it starts as up to date, so `prep.scale(3)` still gave 30. Put the two autoreload lines in the **first** cell of the notebook. And autoreload can't fix everything (objects created before the edit keep their old behaviour), so when results look impossible, restart the kernel and Run All.

## Stopping a running program

**What it is.** Sooner or later you'll start a loop that never ends, or a step that takes far longer than you meant. In a terminal, <kbd>Ctrl</kbd>+<kbd>C</kbd> sends the program an *interrupt*; Python turns it into an error called `KeyboardInterrupt`, raised at whatever line was running, and the program stops with a traceback. In Jupyter, the equivalent is **Kernel → Interrupt Kernel** (the ■ button in the toolbar, or <kbd>I</kbd> <kbd>I</kbd> in command mode).

**Why.** [Day 2](day-02-python-types.md) teaches `while` loops this week, and a `while True:` without a way out is the classic first hang. In ML, interrupting a slow training run to change a setting is routine.

**Kotlin comparison.** <kbd>Ctrl</kbd>+<kbd>C</kbd> on a JVM program in a terminal kills it, and Android Studio's red Stop button kills the app process. Python's version is gentler: it raises an exception inside the program, so the program could react to it (Day 3b shows how), and in Jupyter the *kernel survives* with all its variables.

**Example.** A script that never stops:

```text
# loop.py: never stops on its own
count = 0
while True:
    count = count + 1
```

Run it with `python loop.py`, wait a second, press <kbd>Ctrl</kbd>+<kbd>C</kbd>:

```text
^CTraceback (most recent call last):
  File "/home/you/ai-course/loop.py", line 3, in <module>
    while True:
          ^^^^
KeyboardInterrupt
```

(`^C` is the terminal echoing your keypress.) The traceback says which line was running when you pressed it, which is handy for finding *where* a program is stuck. (The line can be 3 or 4 depending on the instant you press.)

**Underneath: interrupt versus restart in Jupyter.** The same loop in a notebook cell, then **Interrupt Kernel**, in a real kernel:

```text
---------------------------------------------------------------------------
KeyboardInterrupt                         Traceback (most recent call last)
Cell In[2], line 1
----> 1 while True:
      2     count = count + 1

KeyboardInterrupt:
```

Running `count > 0` in the next cell gives `True`: the kernel is still alive and `count` kept the value it had reached. **Interrupt** stops the current cell and keeps memory; **Restart** kills the kernel process and starts a fresh one with nothing defined. Interrupt first; restart when interrupt doesn't respond or the state is a mess.

**How it goes wrong.** Interrupt works by raising an exception in Python code, so it waits until the program is back in the eval loop. If the program is deep inside a long native call (a huge NumPy operation, a model download), <kbd>Ctrl</kbd>+<kbd>C</kbd> or Interrupt can seem to do nothing for a while. Wait a few seconds; if it still runs, close the terminal or use **Kernel → Restart**. In the browser, this site's ▶ Run runs on the page itself, so a block that never ends freezes the tab; reload the page to stop it. (That's one more reason the page never contains endless loops.)

## The ▶ Run button on this site

**What it is.** Every Python block on this site has a ▶ Run button. It runs the block in your browser using **Pyodide** (CPython 3.14 compiled to WebAssembly). Nothing runs on a server, and nothing touches your disk. The first click downloads Python (about 15 MB); later clicks are instant.

**How it behaves.**

- Blocks on one page share variables, top to bottom, like notebook cells. If a block says `NameError`, run the blocks above it first. **Reset** forgets everything.
- Like Jupyter and the REPL, it shows the value of every bare expression line, which is why code on this site often ends with a bare name to show it.
- numpy, pandas and matplotlib are available, and are loaded the first time a block imports them. Charts appear under the block.
- `%timeit` works; other `%` and `!` lines are skipped with a note.

**What it can't do.** There's no `pip install` of arbitrary packages, no PyTorch, no GPU, no access to files on your computer, no `input()` and no threads. Timings reflect your browser, not a real install. Ask it what it's running on:

```python
import sys

sys.platform
```

In the browser this shows `'emscripten'` (the WebAssembly toolchain Pyodide is built with); in a terminal on my Linux laptop, `'linux'` (macOS says `'darwin'`, Windows `'win32'`). Blocks that need a real computer start with the comment `# Run this on your computer (or in Colab), not in the browser.` On this page that includes every block that's a shell command or a file you're meant to save and run.

## Editors and the debugger

**What it is.** An editor that knows your interpreter gives you completion, go-to-definition, inline errors and a debugger. The two common choices are **VS Code** with Microsoft's *Python* and *Jupyter* extensions, and **PyCharm** (by JetBrains, the company behind IntelliJ, on which Android Studio is built).

**Why the interpreter setting matters.** The editor's completions and error checking come from the packages installed in the interpreter you select. Select the wrong one and the editor underlines `import numpy` in red even though your program runs fine, or the reverse.

- **VS Code:** open the `ai-course` folder, press <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>Shift</kbd>+<kbd>P</kbd>, choose **Python: Select Interpreter**, and pick the one under `.venv`. VS Code usually offers it automatically. New terminals inside VS Code then activate the venv for you.
- **PyCharm:** **Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter**, then choose the existing environment at `.venv`.

**Using the debugger.** Click in the gutter left of a line number to set a breakpoint (a red dot). In VS Code, open the Run and Debug panel and choose **Python Debugger: Debug Python File**; in PyCharm, right-click the file and choose **Debug**. The controls are the ones you know from Android Studio:

- **Continue** (<kbd>F5</kbd> in VS Code, <kbd>F9</kbd> in PyCharm): run to the next breakpoint.
- **Step over** (<kbd>F10</kbd> / <kbd>F8</kbd>): run the current line, including any function it calls, and stop on the next line.
- **Step into** (<kbd>F11</kbd> / <kbd>F7</kbd>): if the line calls one of your functions, stop on that function's first line.
- **Step out** (<kbd>Shift</kbd>+<kbd>F11</kbd> / <kbd>Shift</kbd>+<kbd>F8</kbd>): finish the current function and stop back in its caller.
- **Variables** pane: every local and global name with its current value; **Watch** pane: expressions you type, re-evaluated at each stop.

A trace of what you'll see, with a breakpoint on line 5 (`total = total + s`) of the `debug_me.py` file shown below (without its `breakpoint()` line): the first stop shows `scores = [72, 88, 95]`, `total = 0`, `s = 72` in the Variables pane. Continue, and it stops on the same line again with `total = 72`, `s = 88`; once more, `total = 160`, `s = 95`. Add `total / len(scores)` to Watch and it shows `0.0`, then `24.0`, then `53.333333333333336`, recomputed at each stop. (Those are the values the code produces; the exact pane layout depends on your editor version.)

**Notebooks inside the editor.** You don't have to use JupyterLab in a browser. Open a `.ipynb` file in VS Code (with the *Jupyter* extension) or PyCharm and it shows as cells. Pick the kernel with **Select Kernel** (top right in VS Code; choose **Python Environments → .venv**) or the interpreter selector in PyCharm; the same "is this the right Python?" rule applies.

!!! android "Android bridge"
    PyCharm's debugger, keymap and refactorings are the same family as Android Studio's. Go to declaration (<kbd>Cmd</kbd>+<kbd>B</kbd> / <kbd>Ctrl</kbd>+<kbd>B</kbd>) works too, and it can jump into NumPy's Python source the way you'd jump into AndroidX. One difference: in a Kotlin app, the debugger attaches to a process on a device; here it starts your script locally under a debugging wrapper.

**Without an editor: `breakpoint()`.** Python has a built-in debugger, `pdb`, and the built-in function `breakpoint()` pauses your program there and opens it in the terminal:

```text
# debug_me.py
def average(scores):
    total = 0
    for s in scores:
        total = total + s
    breakpoint()          # execution pauses here
    return total / len(scores)

print(average([72, 88, 95]))
```

A real session (`p` prints a value, `l` lists the code around the current line, `n` runs the next line, `c` continues to the end):

```text
> /home/you/ai-course/debug_me.py(6)average()
-> breakpoint()          # execution pauses here
(Pdb) p total
255
(Pdb) p scores
[72, 88, 95]
(Pdb) l
  1  	# debug_me.py
  2  	def average(scores):
  3  	    total = 0
  4  	    for s in scores:
  5  	        total = total + s
  6  ->	    breakpoint()          # execution pauses here
  7  	    return total / len(scores)
  8  	
  9  	print(average([72, 88, 95]))
[EOF]
(Pdb) n
> /home/you/ai-course/debug_me.py(7)average()
-> return total / len(scores)
(Pdb) c
85.0
```

Remove `breakpoint()` lines before committing; they'd pause the program for everyone. The ▶ Run button can't run pdb (it would open a browser prompt for every command), so this file is shown as text: save it and run it in your terminal with `python debug_me.py`.

## Reading your first traceback

**What it is.** When an error isn't handled, Python stops and prints a **traceback**: the chain of function calls that were active when the error happened, from the outermost call down to the line that failed, followed by the error's **type** and **message**.

**Why.** It's the single most useful debugging tool you have, and most of the time it tells you the exact line and the reason. Day 3b covers exceptions in depth; today you learn to read one.

**Kotlin comparison.** A JVM stack trace lists the failing frame **first** and callers below it (`at ...` lines). Python prints the opposite order: outermost call first, failing line **last**, and the error type and message at the very bottom. So read a Python traceback **from the bottom up**.

**Example.** Save this as `first_bug.py` and run it:

```python
# Run this on your computer (or in Colab), not in the browser.
def greet(name, years):
    return "Hello, " + name + "! Years of Kotlin: " + years

def main():
    message = greet("Android engineer", 8)
    print(message)

main()
```

```text
Traceback (most recent call last):
  File "/home/you/ai-course/first_bug.py", line 9, in <module>
    main()
    ~~~~^^
  File "/home/you/ai-course/first_bug.py", line 6, in main
    message = greet("Android engineer", 8)
  File "/home/you/ai-course/first_bug.py", line 3, in greet
    return "Hello, " + name + "! Years of Kotlin: " + years
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~
TypeError: can only concatenate str (not "int") to str
```

**Underneath: reading it bottom-up.**

1. **Last line: what went wrong.** `TypeError` is the kind of error: an operation got a value of the wrong type. The message says `+` was asked to join a `str` (string) and an `int`. Python won't convert the number for you, unlike Kotlin's `"..." + 8`, which calls `toString()` implicitly.
2. **The frame above it: where.** File `first_bug.py`, line 3, inside function `greet`, followed by the line itself. The `^` under the second-to-last `+` and the `~` under its two operands point at the exact operation: everything to the left (a string) plus `years`.
3. **Frames further up: how you got there.** Line 6 in `main` called `greet(..., 8)`, and line 9 at the top level (`<module>` means "not inside any function") called `main()`. Read these to find where the bad value came from: the `8` is passed on line 6.

**Fix.** Turn the number into text. Either convert it, `"..." + str(years)`, or use an f-string, which converts for you:

```python
def greet(name, years):
    return f"Hello, {name}! Years of Kotlin: {years}"

greet("Android engineer", 8)
```

```text
'Hello, Android engineer! Years of Kotlin: 8'
```

**How it goes wrong (reading it).** The common misreading is to start at the top, see `main()` on line 9, and look for the bug there. The top frame is only the entry point. The useful information is at the bottom: the error type, the message, and the last frame that's in *your* code. When a traceback runs through library code (NumPy, pandas), the last frames are inside the library; scroll up to the last frame whose file is yours, because that's usually where the bad value came from.

## Colab: a free GPU in the cloud

**What it is.** [Google Colab](https://colab.research.google.com/) is a hosted Jupyter notebook. Your notebook runs on a virtual machine in Google's cloud, which can include a GPU, with NumPy, pandas, matplotlib and PyTorch preinstalled.

**Why.** Training neural networks, from later in the course, is far faster on a GPU, and you don't need to buy one. Colab is also the quickest way to run code from a tutorial without touching your laptop's setup.

**Using a GPU.**

1. Open Colab and create a new notebook (it's saved to your Google Drive).
2. **Runtime → Change runtime type**, choose a GPU such as **T4 GPU**, then **Save**.
3. Run:

```python
# Run this on your computer (or in Colab), not in the browser.
!nvidia-smi
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

`!nvidia-smi` is a shell command that lists NVIDIA GPUs. On a T4 runtime the output looks like this (an example from a typical Colab session, shortened; your driver and library versions will differ):

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
...
2.8.0+cu126
True
```

The table names the GPU (`Tesla T4`). `torch.__version__` ends in `+cu126`, meaning "built for CUDA 12.6". `True` means PyTorch can use the GPU. On a CPU-only runtime, `!nvidia-smi` instead fails with a message that the command isn't found or can't talk to a driver, and the last line prints `False`: go back to **Runtime → Change runtime type**.

**Uploading files.** A Colab machine starts empty apart from its software. Two ways to get your data in:

- Open the **Files** panel (the folder icon on the left) and drag a file in, or run:

```python
# Run this on your computer (or in Colab), not in the browser.
from google.colab import files
uploaded = files.upload()      # opens a file picker; files land in /content
```

- For anything you want to keep, mount Google Drive: `from google.colab import drive` then `drive.mount("/content/drive")`, and read and write under `/content/drive/MyDrive/`.

**Limits of the free tier (and how it goes wrong).** The machine is temporary. When the runtime disconnects or is recycled, everything in `/content` that isn't on Drive is gone, including uploaded files, installed packages and trained models. Free sessions disconnect after a period of inactivity and have a maximum lifetime (Google's FAQ says up to 12 hours), and GPUs are not guaranteed: at busy times or after heavy use Colab may tell you it cannot connect to a GPU backend because of usage limits, and you have to wait or use CPU. Google adjusts these limits without notice, so check the FAQ in Resources. Save checkpoints to Drive, and keep `%pip install` lines at the top of the notebook so a fresh runtime can be rebuilt with Run All.

[Kaggle Notebooks](https://www.kaggle.com/code) are an alternative with a weekly free GPU quota.

## A course repo with git

**What and why.** Put your course work in a git repository from day one: your notes, scripts and notebooks get history, and a private GitHub repo is your backup. The rule for Python projects is the same as for Android's `build/` folder: commit the *recipe* (`requirements.txt`), never the *products* (`.venv/`, `__pycache__/`).

```bash
# Run this on your computer (or in Colab), not in the browser.
cd ai-course
git init -b main
printf '.venv/\n__pycache__/\n*.pyc\n.ipynb_checkpoints/\n.DS_Store\n' > .gitignore
git status --short
python -m pip freeze > requirements.txt
git add .gitignore requirements.txt hello.py
git commit -m "Day 1: environment and first program"
git status
```

On Windows, create `.gitignore` in your editor with those five lines, one per line (`printf` isn't available in PowerShell). The output, from a demo folder on my machine:

```text
Initialized empty Git repository in /home/you/ai-course/.git/
?? .gitignore
?? hello.py
?? requirements.txt
[main (root-commit) 455f540] Day 1: environment and first program
 3 files changed, 28 insertions(+)
 create mode 100644 .gitignore
 create mode 100644 hello.py
 create mode 100644 requirements.txt
On branch main
nothing to commit, working tree clean
```

`-b main` names the first branch `main`. `git status --short` lists untracked files with `??`: notice `.venv/` is *not* among them, so the `.gitignore` works. What each `.gitignore` line excludes: the venv; the bytecode cache folder and any loose `.pyc` files; the hidden autosave copies Jupyter writes next to notebooks; and macOS's folder-metadata files. If `.venv/` ever appears in `git status`, the `.gitignore` is missing or misspelled.

**Backing it up to GitHub.** Create an empty **private** repository on github.com (no README, so there's nothing to merge), then connect and push:

```bash
# Run this on your computer (or in Colab), not in the browser.
git remote add origin https://github.com/<your-user>/ai-course.git
git push -u origin main
```

`remote add origin` gives the GitHub URL the short name `origin`; `push -u` uploads `main` and remembers the link, so later a plain `git push` is enough. GitHub asks you to sign in the first time (a browser login, or a personal access token instead of a password).

On a new machine: `git clone`, then `python3 -m venv .venv`, activate, and `python -m pip install -r requirements.txt`. Whenever you install something new, re-run `pip freeze > requirements.txt` and commit the change, the way you'd commit an edited `build.gradle.kts`.

!!! warning "Common mistake: committing `.venv`"
    A venv holds hundreds of megabytes of platform-specific files and absolute paths to *your* machine. Committed, it bloats the repo and doesn't work on anyone else's computer (or on yours after you move the folder). If you already committed it: `git rm -r --cached .venv`, add `.venv/` to `.gitignore`, and commit.

## Troubleshooting: setup errors and fixes

Every row below was taught above; this is the summary to come back to.

| You see | What it means | Fix |
|---|---|---|
| `zsh: command not found: python` (macOS) or `Command 'python' not found` (Ubuntu) | No program called `python` on `PATH`. | Use `python3`, or activate your venv, where `python` always exists. |
| `Python was not found; run without arguments to install from the Microsoft Store...` (Windows) | Windows' Store shortcut is answering instead of a real Python. | Install Python (install manager, or the classic installer with "Add python.exe to PATH" ticked); turn off the `python.exe` entries under Settings → Apps → Advanced app settings → App execution aliases; or use `py`. |
| `ModuleNotFoundError: No module named 'numpy'` | The Python that ran your code doesn't have it, often because it's a different Python from the one you installed into. | Print `sys.executable`; activate the right venv; install with `python -m pip install numpy`. In a notebook, select the right kernel and use `%pip`. |
| `AttributeError: module 'numpy' has no attribute 'array' (consider renaming '.../numpy.py' ...)` | Your own file is named like the library and is imported instead of it. | Rename the file, delete the `__pycache__` next to it. |
| `SSL: CERTIFICATE_VERIFY_FAILED` on macOS after a python.org install | That Python has no list of trusted certificates yet. | Run **Install Certificates.command** from the Python folder in Applications. |
| `ResolutionImpossible` from pip | Two pins can't both be satisfied. | Read "The conflict is caused by", loosen one pin. |
| `pip check` reports `x requires y, which is not installed` | The environment is inconsistent. | `python -m pip install x` again, or rebuild the venv. |
| A notebook keeps using old code after you edit an imported `.py` | The kernel cached the module at first import. | Restart the kernel, or start the notebook with `%load_ext autoreload` and `%autoreload 2`. |
| A program or cell never finishes | An endless loop or a very long step. | <kbd>Ctrl</kbd>+<kbd>C</kbd> in a terminal; **Kernel → Interrupt Kernel** in Jupyter; restart if it doesn't respond. |
| `error: externally-managed-environment` | You tried to install into the system Python. | Activate a venv and install there. Not `sudo`, not `--break-system-packages`. |
| `The virtual environment was not created successfully because ensurepip is not available` (Debian/Ubuntu) | The `venv` support package is missing. | `sudo apt install python3-venv` (or the versioned name it suggests), delete the half-made `.venv`, recreate. |
| `...Activate.ps1 cannot be loaded because running scripts is disabled on this system` | PowerShell's execution policy blocks scripts. | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, or use cmd and `activate.bat`. |
| `cannot execute: required file not found` (bash) / `bad interpreter: ...: no such file or directory` (zsh) running `.venv/bin/pip` | The project folder was moved or renamed after the venv was made. | Delete `.venv`, recreate it, `pip install -r requirements.txt`. |
| `ERROR: No matching distribution found for x==y` | That version doesn't exist, or has no wheel for your Python/OS. | Check the spelling; pick a version from pip's list; or use the Python version the project expects. |
| `can't open file '...': [Errno 2] No such file or directory` | Wrong folder, or a typo in the file name. | `ls` / `dir`, then `cd` to the right folder. |
| `IndentationError: expected an indented block` / `unexpected indent` | Block structure is wrong. | Four spaces per level, consistently; editor set to insert spaces. |
| `SyntaxError` and nothing printed | The file failed to compile. | Fix the line shown (often a missing `:` or bracket); nothing ran. |
| Notebook result changes after Restart and Run All | Hidden state from out-of-order runs. | Trust the Run All result; reorder cells so top-to-bottom is the real order. |

## Going further

- **pyproject.toml and packaging**: the modern project file that names dependencies and builds installable packages. [Python Packaging User Guide](https://packaging.python.org/en/latest/)
- **Lock files and project managers** (uv's `uv lock`, Poetry, pip-tools): exact, cross-platform reproducibility. [uv projects guide](https://docs.astral.sh/uv/guides/projects/)
- **conda / mamba**: environments that also manage non-Python libraries such as CUDA. [conda documentation](https://docs.conda.io/en/latest/)
- **Managing several Python versions** with pyenv or `uv python install`. [pyenv](https://github.com/pyenv/pyenv)
- **Linters and formatters** such as Ruff, which catch many bugs before you run. [Ruff](https://docs.astral.sh/ruff/)
- **CPython internals**: the eval loop, the GIL (global interpreter lock), the experimental JIT and free-threaded builds. [Python Developer's Guide](https://devguide.python.org/)
- **Docker** for shipping a whole environment, OS included. [Docker documentation](https://docs.docker.com/)

---

## Exercises

1. **Warm-up: where am I?** Create the `ai-course` folder and its venv. Write `whereami.py` that prints the Python version, the interpreter's path, and whether it's running inside a venv. Run it with the venv active, then deactivated, and compare.
2. **Warm-up: about me.** Write `about_me.py` that prints your name, then the numbers 1 to 10 and whether each is even or odd. (Hint: `n % 2` is the remainder after dividing by 2, as in Kotlin; `a if condition else b` is Python's version of Kotlin's `if (condition) a else b` expression.)
3. **Core: the requirements round trip.** In your venv, install `numpy`, `pandas`, `matplotlib`, `jupyterlab` and `ipykernel`. Freeze them to `requirements.txt`. Delete `.venv` completely, recreate it from the file, and prove with `pip check` and one import line that it's back.
4. **Core: hidden state on purpose.** In a Jupyter notebook, make three cells: `count = 0`, `count = count + 1`, `print(count)`. Run them so that the last cell prints `3`. Then use Restart Kernel and Run All Cells and note what it prints and why. Finally, time `sum(range(1_000_001))` against `np.arange(1_000_001).sum()` with `%timeit` and write down both numbers.
5. **Core: the right kernel.** Register your venv as a Jupyter kernel named `ai-course`, open a notebook with it, and prove in a cell that the notebook runs your venv's Python. Then install a small package that isn't installed yet (for example `tqdm`) from inside the notebook, the safe way.
6. **Predict the output.** Answer first, then check. (a) A notebook has cells A `count = 0`, B `count = count + 1`, C `print(count)`. You run them in the order A, B, B, C, B, C. What do the two `print`s show? (b) A file `late.py` contains `print("start")` on line 1 and `def ratio(a, b)` (no colon) on line 3. What does `python late.py` print before the error?
7. **Find and fix the bug.** This script crashes. Read the traceback from the bottom up, explain the cause, and fix it so it prints the average with one decimal place.

    ```text
    Traceback (most recent call last):
      File "/home/you/ai-course/stats.py", line 4, in <module>
        print("Average score: " + average)
              ~~~~~~~~~~~~~~~~~~^~~~~~~~~
    TypeError: can only concatenate str (not "float") to str
    ```

    ```text
    # stats.py
    scores = [72, 88, 94]
    average = sum(scores) / len(scores)
    print("Average score: " + average)
    ```

8. **Challenge: an environment report.** Write `env_report.py` that prints the Python version (only the number, like `3.13.16`), the interpreter path, whether it's in a venv, and for each of `numpy`, `pandas`, `matplotlib` and `torch` either its installed version or `not installed`, without crashing when one is missing. Use `importlib.util.find_spec(name)`, which returns `None` when a package can't be found, and `importlib.metadata.version(name)`, which returns an installed package's version string. Two pieces of syntax you haven't met yet: an `if`/`else` statement is written `if condition:` then an indented block, then `else:` and another indented block (the same colon-and-indent rule as `for`); and the test for "is this `None`?" is written `x is None` (Day 2 explains why `is`, not `==`). Note also that `importlib.util` and `importlib.metadata` are separate sub-modules: `import importlib` alone doesn't load them, so import each one by its full name. Run it in your venv and in the ▶ Run sandbox, and explain the difference.

??? success "Solutions"

    **1. Where am I?**

    ```python
    # whereami.py
    import sys

    print(sys.version)
    print(sys.executable)
    print(sys.prefix != sys.base_prefix)
    ```

    With the venv active, on my laptop:

    ```text
    3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]
    /home/you/ai-course/.venv/bin/python
    True
    ```

    Deactivated (`python3 whereami.py`):

    ```text
    3.13.16 (main, Oct  1 2026, 15:40:24) [GCC 13.3.0]
    /usr/bin/python3
    False
    ```

    Same version both times, because the venv's `python` is a link to the same installation. What differs is `sys.executable` and the prefix: inside the venv, Python found `.venv/pyvenv.cfg` and switched its `sys.prefix` (and so its package folder) to `.venv`. That one comparison is the most reliable "am I in a venv?" check.

    **2. About me.**

    ```python
    # about_me.py
    print("My name is Alex")
    for n in range(1, 11):
        kind = "even" if n % 2 == 0 else "odd"
        print(n, kind)
    ```

    ```text
    My name is Alex
    1 odd
    2 even
    3 odd
    4 even
    5 odd
    6 even
    7 odd
    8 even
    9 odd
    10 even
    ```

    `range(1, 11)` stops before 11, so it yields 1 to 10, like `1 until 11`. `n % 2 == 0` is true for even numbers; the conditional expression picks `"even"` or `"odd"`. `print(n, kind)` with two arguments puts a space between them, so no f-string is needed here.

    **3. The requirements round trip.**

    ```bash
    # Run this on your computer (or in Colab), not in the browser.
    source .venv/bin/activate
    python -m pip install numpy pandas matplotlib jupyterlab ipykernel
    python -m pip freeze > requirements.txt
    deactivate
    rm -rf .venv
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m pip check
    python -c "import numpy, pandas, matplotlib; print(numpy.__version__, pandas.__version__, matplotlib.__version__)"
    ```

    The last two lines, on my machine (install lines omitted):

    ```text
    No broken requirements found.
    2.5.4 3.1.0 3.11.2
    ```

    Deleting `.venv` loses nothing that matters, because `requirements.txt` records every package at its exact version, and pip installs the same wheels again. That's why the venv is disposable and the text file is what you commit. (Your version numbers will be whatever was newest when you froze.)

    **4. Hidden state on purpose.** Run A once, then B three times, then C: the counters read `[1]` for A, `[4]` for B and `[5]` for C, and C prints `3`. Restart Kernel and Run All Cells prints `1`, because it runs A, B and C once each, top to bottom, with an empty memory. On my laptop, `%timeit` gave:

    ```text
    22.3 ms ± 1.05 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    788 μs ± 31.5 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    ```

    (Machine-dependent; on an earlier run the first number was 12.9 ms.) The lesson of the first part: the page showed nothing unusual, only the execution counts gave it away. The lesson of the second: the same sum is more than ten times faster when NumPy's C loop does it instead of the eval loop.

    **5. The right kernel.**

    ```bash
    # Run this on your computer (or in Colab), not in the browser.
    source .venv/bin/activate
    python -m pip install ipykernel jupyterlab
    python -m ipykernel install --user --name ai-course --display-name "Python (ai-course)"
    jupyter lab
    ```

    In a notebook using the **Python (ai-course)** kernel:

    ```python
    # Run this on your computer (or in Colab), not in the browser.
    import sys
    print(sys.executable)
    %pip install tqdm
    ```

    The output ends with (the `%pip` download lines omitted):

    ```text
    /home/you/ai-course/.venv/bin/python
    Successfully installed tqdm-4.70.1
    Note: you may need to restart the kernel to use updated packages.
    ```

    The first line, `/home/you/ai-course/.venv/bin/python` (with your own folder), proves the kernel is the venv's interpreter: `ipykernel install` wrote that path into the kernel's `kernel.json`. `%pip` installs into exactly that interpreter, whereas `!pip` would use whatever `pip` the shell finds. Restart the kernel after installing so the new package is importable cleanly.

    **6. Predict the output.**

    (a) `2`, then `3`. Replaying the run order as one script shows it:

    ```python
    count = 0              # A
    count = count + 1      # B
    count = count + 1      # B again
    print(count)           # C
    count = count + 1      # B
    print(count)           # C
    ```

    ```text
    2
    3
    ```

    The kernel executes whatever you run, in the order you run it, and keeps the result. The page never shows "B ran three times"; only the execution counts do.

    (b) Nothing. The output is only the error:

    ```text
      File "/home/you/ai-course/late.py", line 3
        def ratio(a, b)
                       ^
    SyntaxError: expected ':'
    ```

    CPython compiles the whole file to bytecode before running line 1, and compilation fails at line 3, so `print("start")` never executes. A `TypeError`, by contrast, happens while running, so earlier lines would have printed.

    **7. Find and fix the bug.** Bottom line first: `TypeError`, `+` can't join a `str` and a `float`. The frame above points at line 4 (line 1 is the `# stats.py` comment), and the `^` sits under the `+` between `"Average score: "` and `average`. `sum(scores) / len(scores)` is a `float` (in Python, `/` always gives a float, here `84.66666666666667`), and Python never converts numbers to text implicitly. Fix with an f-string:

    ```python
    # stats.py, fixed
    scores = [72, 88, 94]
    average = sum(scores) / len(scores)
    print(f"Average score: {average}")
    print(f"Average score: {average:.1f}")
    ```

    ```text
    Average score: 84.66666666666667
    Average score: 84.7
    ```

    Inside the braces, `:.1f` is a **format spec**: after the colon, `.1` means one digit after the decimal point and `f` means "fixed-point number", so the value is rounded to `84.7` for display. (Day 2 covers format specs in full.) `"Average score: " + str(average)` also works, but can't control the decimals.

    **8. Challenge: an environment report.**

    ```python
    # env_report.py
    import sys
    import importlib.util
    import importlib.metadata

    print("Python     :", sys.version.split()[0])
    print("Interpreter:", sys.executable)
    print("In a venv  :", sys.prefix != sys.base_prefix)

    for name in ["numpy", "pandas", "matplotlib", "torch"]:
        if importlib.util.find_spec(name) is None:
            print(f"{name:<11}: not installed")
        else:
            print(f"{name:<11}: {importlib.metadata.version(name)}")
    ```

    In my venv:

    ```text
    Python     : 3.13.16
    Interpreter: /home/you/ai-course/.venv/bin/python
    In a venv  : True
    numpy      : 2.5.4
    pandas     : 3.1.0
    matplotlib : 3.11.2
    torch      : not installed
    ```

    New pieces: `sys.version.split()` breaks the version text at spaces into a list, and `[0]` takes its first item, `"3.13.16"`. `["numpy", ...]` is a list and `for name in` walks it. `{name:<11}` is a format spec meaning "left-align in a field 11 characters wide", which lines the colons up. `find_spec` looks for a package *without importing it*, so a missing package gives `None` instead of a `ModuleNotFoundError` crash, and only then does the script ask `importlib.metadata` for the version it recorded in `.dist-info`. In the ▶ Run sandbox, the version starts with `3.14`, `In a venv` is `False`, `torch` is `not installed`, and packages that no block on the page has imported yet can show as `not installed` too, because Pyodide downloads numpy, pandas and matplotlib only when a block imports them. That difference is itself the point: the same script tells you exactly which Python you're in and what it can see.

## Resources

- [The Python tutorial: Using the Python interpreter](https://docs.python.org/3/tutorial/interpreter.html): the official guide to the REPL and running scripts; sections 3 and 4 pair well with Day 2.
- [venv: creation of virtual environments](https://docs.python.org/3/library/venv.html): what a venv contains and how activation works, from the standard library docs.
- [Python Packaging User Guide: Installing packages](https://packaging.python.org/en/latest/tutorials/installing-packages/): pip, PyPI, requirements files and version specifiers.
- [VS Code: Python tutorial](https://code.visualstudio.com/docs/python/python-tutorial): interpreter selection, running and debugging.
- [JupyterLab documentation](https://jupyterlab.readthedocs.io/en/latest/): the notebook interface, kernels and shortcuts.
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html): the free tier's limits, straight from Google.
- [uv documentation](https://docs.astral.sh/uv/): if you choose the faster installer.

## Checklist

- [ ] I can explain why ML uses Python and which work happens in native code.
- [ ] I can describe source → bytecode → eval loop, and I've looked at bytecode with `dis`.
- [ ] Python 3.12+ is installed, and I can say which interpreter `python` runs (`which` / `where`, `sys.executable`).
- [ ] I've created, activated, inspected, deactivated and deleted a venv, and I know activation only changes `PATH`.
- [ ] I can install, show, list, pin, upgrade and uninstall packages with `python -m pip`, and rebuild a venv from `requirements.txt`.
- [ ] `hello.py` runs, and I can fix an `IndentationError` and a "can't open file" error.
- [ ] I've run Python as a script, with `-m`, in the REPL and in Jupyter.
- [ ] I've seen hidden notebook state, fixed it with Restart and Run All, and used `%timeit`, `%pip` and `%autoreload` with my own kernel.
- [ ] I can stop a runaway script (<kbd>Ctrl</kbd>+<kbd>C</kbd>) and a runaway cell (Interrupt Kernel).
- [ ] I know why a file named `numpy.py` breaks `import numpy`.
- [ ] I know what the ▶ Run button can't do.
- [ ] My editor uses `.venv`, and I've stopped a program at a breakpoint.
- [ ] I can read a traceback bottom-up and fix the bug it points to.
- [ ] I've run a GPU check in Colab and know its limits.
- [ ] My course repo has a `.gitignore` and a committed `requirements.txt`, and no `.venv`.
- [ ] I can match a setup error message to its fix.

**Next:** [Day 2: Python's core types, through Kotlin eyes](day-02-python-types.md)
