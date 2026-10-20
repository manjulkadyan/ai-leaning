# How Lessons Work

Every lesson follows the same shape so you always know where you are.

| Section | What it gives you |
|---|---|
| **Objectives** | What you'll be able to do by the end, as checkable statements |
| **Prerequisites** | Lessons you need first, linked |
| **The big idea** | The concept in plain language with an everyday analogy |
| **How it works** | The internals, with diagrams and, where motion helps, animations |
| **The math** | Notation introduced one symbol at a time, with a worked numeric example |
| **Code** | Runnable Python you type yourself, with expected output |
| **Android bridge** | How the idea maps to something you already know from Android and Kotlin |
| **Common mistakes** | What trips people up, and how to notice it |
| **Exercises** | Practice, with hidden solutions |
| **Resources** | Free courses, docs, videos and repos with direct links |
| **Checklist** | Tick these before moving on |

## Running the code

Every Python block has a **▶ Run** button under it. It runs the code in your browser with [Pyodide](https://pyodide.org/), a full Python with NumPy, pandas and Matplotlib compiled to WebAssembly, so nothing needs installing.

- The first run on a page downloads Python (about 15 MB plus the packages it imports), which takes a few seconds. Later runs are instant.
- Blocks on one page share their variables, like notebook cells, so run them top to bottom. **Reset** forgets everything.
- The value of each bare expression is printed, the way the `# comment` after it shows. Plots appear as images.
- A few blocks can't run in a browser: PyTorch, shell commands such as `!nvidia-smi`, and anything that needs your files or GPU. They say so when you run them. Timings from `%timeit` are slower than on your computer.

Each lesson also has a **Download this lesson's code as a Jupyter notebook** link under its title. Open the file in Jupyter, VS Code or [Google Colab](https://colab.research.google.com/) (File → Upload notebook) to run everything on a real machine.

## Callout boxes

!!! android "Android bridge"
    Connects the new idea to Android or Kotlin knowledge you already have.

!!! warning "Common mistake"
    Something that catches almost everyone the first time.

!!! example "Try it"
    A small experiment to run right now.

??? success "Solution"
    Solutions are collapsed so you try first. Click to open.

## Notation

Math appears like this: inline \( y = wx + b \), or on its own line:

\[
\hat{y} = \sum_{i=1}^{n} w_i x_i + b
\]

Every symbol is defined in words the first time it appears.
