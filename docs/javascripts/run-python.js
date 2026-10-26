// "Run" buttons for Python code blocks. Runs the code in the browser with
// Pyodide (CPython compiled to WebAssembly), so lessons work without installing
// anything. Blocks on a page share one namespace, like cells in a notebook.
(function () {
  const PYODIDE_VERSION = "314.0.7";
  const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

  // Python side: runs a cell statement by statement and echoes the value of
  // every bare expression, so `a.shape  # (3, 4)` prints like the lesson shows.
  const HELPER = String.raw`
import ast, base64, io, os, sys, traceback
os.environ.setdefault("MPLBACKEND", "Agg")

def _patch_pandas():
    pd = sys.modules.get("pandas")
    if pd is None or getattr(pd, "_course_patched", False):
        return
    from pyodide.http import open_url
    orig = pd.read_csv
    def read_csv(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith(("http://", "https://")):
            path = open_url(path)
        return orig(path, *args, **kwargs)
    pd.read_csv = read_csv
    pd._course_patched = True

_saved = []

def _patch_matplotlib():
    # Lessons often save a figure and close it; keep a copy so it still shows.
    fig_mod = sys.modules.get("matplotlib.figure")
    if fig_mod is None or getattr(fig_mod.Figure.savefig, "_course", False):
        return
    orig = fig_mod.Figure.savefig
    def savefig(self, fname, *args, **kwargs):
        result = orig(self, fname, *args, **kwargs)
        if isinstance(fname, (str, os.PathLike)):
            buf = io.BytesIO()
            orig(self, buf, format="png", dpi=100, bbox_inches="tight")
            _saved.append((id(self), base64.b64encode(buf.getvalue()).decode()))
        return result
    savefig._course = True
    fig_mod.Figure.savefig = savefig

def _timeit(expr, ns):
    import timeit
    t = timeit.Timer(expr, globals=ns)
    n, _ = t.autorange()
    best = min(t.repeat(repeat=3, number=n)) / n
    for unit, scale in (("s", 1), ("ms", 1e3), ("µs", 1e6), ("ns", 1e9)):
        if best * scale >= 1:
            break
    print(f"{best * scale:.3g} {unit} per loop (best of 3, {n} loops each)  [browser timing]")

def _prepare(src):
    lines = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("%timeit "):
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}_course_timeit({s[len('%timeit '):]!r}, globals())")
        elif s.startswith(("%", "!")):
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}print({('skipped notebook-only line: ' + s)!r})")
        else:
            lines.append(line)
    return "\n".join(lines)

def _figures():
    latest = dict(_saved)  # a figure saved twice (png and svg) shows once
    shown = set(latest)
    images = list(latest.values())
    _saved.clear()
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is None:
        return images
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if id(fig) not in shown:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            images.append(base64.b64encode(buf.getvalue()).decode())
    plt.close("all")
    return images

def _is_plot_object(value):
    # ax.set_title(...) and friends return artists; echoing them is noise.
    items = value if isinstance(value, (list, tuple)) and value else [value]
    return all(type(v).__module__.startswith("matplotlib") for v in items)

def course_run(src, ns):
    ns.setdefault("_course_timeit", _timeit)
    out = io.StringIO()
    old = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = out
    ok = True
    try:
        tree = ast.parse(_prepare(src), "<cell>")
        for node in tree.body:
            _patch_pandas()
            _patch_matplotlib()
            if isinstance(node, ast.Expr):
                value = eval(compile(ast.Expression(node.value), "<cell>", "eval"), ns)
                if value is not None and not _is_plot_object(value):
                    print(repr(value))
            else:
                exec(compile(ast.Module([node], []), "<cell>", "exec"), ns)
            _patch_pandas()
            _patch_matplotlib()
    except BaseException as e:
        ok = False
        print("".join(traceback.format_exception_only(type(e), e)).rstrip())
        if isinstance(e, ModuleNotFoundError):
            print("Hint: this package isn't available in browser Python. Run this block on your computer or in Colab (Day 1).")
        if isinstance(e, NameError):
            print("Hint: this block may use a variable from an earlier block. Run the blocks above it first.")
    finally:
        sys.stdout, sys.stderr = old
    return ok, out.getvalue(), _figures()
`;

  let pyodideReady = null;
  let pageNamespace = null;

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error("Couldn't download Python for the browser. Check your connection."));
      document.head.appendChild(s);
    });
  }

  function getPyodide() {
    if (!pyodideReady) {
      pyodideReady = (async () => {
        if (typeof loadPyodide === "undefined") await loadScript(INDEX_URL + "pyodide.js");
        const py = await loadPyodide({ indexURL: INDEX_URL });
        py.runPython(HELPER);
        return py;
      })();
      pyodideReady.catch(() => (pyodideReady = null));
    }
    return pyodideReady;
  }

  function freshNamespace(py) {
    if (pageNamespace) pageNamespace.destroy();
    pageNamespace = py.globals.get("dict")();
    pageNamespace.set("__name__", "__main__");
    return pageNamespace;
  }

  function codeOf(block) {
    const code = block.querySelector("pre > code") || block.querySelector("pre");
    // Material wraps lines in spans; textContent gives the plain source.
    return code ? code.textContent.replace(/\n$/, "") : "";
  }

  function render(output, ok, text, images) {
    output.hidden = false;
    output.classList.toggle("run-error", !ok);
    output.replaceChildren();
    const pre = document.createElement("pre");
    pre.textContent = text || (images.length ? "" : "(no output)");
    if (pre.textContent) output.appendChild(pre);
    for (const b64 of images) {
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + b64;
      img.alt = "Figure produced by the code above";
      output.appendChild(img);
    }
  }

  async function run(block, button, output) {
    button.disabled = true;
    const label = button.textContent;
    output.hidden = false;
    output.classList.remove("run-error");
    output.replaceChildren(Object.assign(document.createElement("pre"), {
      textContent: pyodideReady ? "Running…" : "Loading Python in your browser (first run only, ~15 MB)…",
    }));
    try {
      const py = await getPyodide();
      const src = codeOf(block);
      button.textContent = "Loading packages…";
      try {
        await py.loadPackagesFromImports(src);
      } catch (e) {
        // e.g. torch: not available in the browser; the run below shows the ImportError.
      }
      button.textContent = "Running…";
      const ns = pageNamespace || freshNamespace(py);
      const result = py.globals.get("course_run")(src, ns);
      const [ok, text, images] = result.toJs();
      result.destroy();
      render(output, ok, text, images);
    } catch (e) {
      render(output, false, String(e && e.message ? e.message : e), []);
    } finally {
      button.textContent = label;
      button.disabled = false;
    }
  }

  function addButtons() {
    pageNamespace = null; // new page, new namespace (instant navigation keeps the JS alive)
    document.querySelectorAll(".md-typeset .language-python.highlight:not([data-run])").forEach((block) => {
      // Blocks labelled "Run this on your computer" need a real machine
      // (files, torch, pip...), so they get no Run button.
      if (/^\s*# Run this on your computer/.test(block.querySelector("code")?.textContent || "")) {
        block.dataset.run = "local";
        return;
      }
      block.dataset.run = "1";
      const bar = document.createElement("div");
      bar.className = "run-bar";
      const runBtn = document.createElement("button");
      runBtn.type = "button";
      runBtn.className = "run-btn";
      runBtn.textContent = "▶ Run";
      runBtn.title = "Run this code in your browser. Blocks on this page share variables, so run them top to bottom.";
      const resetBtn = document.createElement("button");
      resetBtn.type = "button";
      resetBtn.className = "run-reset";
      resetBtn.textContent = "Reset";
      resetBtn.title = "Forget all variables from earlier blocks on this page";
      resetBtn.hidden = true;
      const output = document.createElement("div");
      output.className = "run-output";
      output.hidden = true;
      runBtn.addEventListener("click", () => {
        resetBtn.hidden = false;
        run(block, runBtn, output);
      });
      resetBtn.addEventListener("click", async () => {
        if (pyodideReady) freshNamespace(await getPyodide());
        document.querySelectorAll(".run-output").forEach((o) => {
          o.hidden = true;
          o.replaceChildren();
        });
      });
      bar.append(runBtn, resetBtn);
      block.after(bar, output);
    });
  }

  if (typeof document$ !== "undefined") document$.subscribe(addButtons);
  else document.addEventListener("DOMContentLoaded", addButtons);
})();
