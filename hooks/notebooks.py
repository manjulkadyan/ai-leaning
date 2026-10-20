"""Build a Jupyter notebook for every lesson that has Python code.

Each notebook holds the lesson's Python blocks in page order, with the nearest
heading above each one, and a download link is added under the lesson title.
"""
import json
import os
import re

FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,})python[^\n]*\n(?P<body>.*?)^(?P=indent)(?P=fence)[ \t]*$", re.M | re.S)
HEADING = re.compile(r"^(#{2,3}) (.+)$", re.M)

_notebooks = {}


def _cells(markdown, title, url):
    cells = [_md(f"# {title}\n\nCode from the lesson at {url}. Run the cells top to bottom; "
                 "the explanations are on the page.")]
    headings = [(m.start(), m.group(0)) for m in HEADING.finditer(markdown)]
    last_heading = None
    for m in FENCE.finditer(markdown):
        above = [h for pos, h in headings if pos < m.start()]
        if above and above[-1] != last_heading:
            last_heading = above[-1]
            cells.append(_md(last_heading))
        indent = len(m.group("indent"))
        src = "\n".join(line[indent:] for line in m.group("body").rstrip("\n").splitlines())
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": src.splitlines(keepends=True)})
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i}"
    return cells


def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def on_page_markdown(markdown, page, config, files):
    if not page.file.src_uri.startswith("lessons/") or not FENCE.search(markdown):
        return markdown
    name = os.path.splitext(os.path.basename(page.file.src_uri))[0] + ".ipynb"
    url = config["site_url"].rstrip("/") + "/" + page.url if config.get("site_url") else page.url
    _notebooks[page.file.src_uri] = (name, {
        "cells": _cells(markdown, page.title, url),
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    })
    return markdown


def on_page_content(html, page, config, files):
    entry = _notebooks.get(page.file.src_uri)
    if not entry:
        return html
    link = (f'<p class="notebook-link"><a href="{entry[0]}" download>'
            "Download this lesson's code as a Jupyter notebook</a> "
            "(open it in Jupyter, VS Code or Colab), or press <strong>▶ Run</strong> "
            "under any code block to run it here.</p>")
    return re.sub(r"(</h1>)", r"\1" + link.replace("\\", r"\\"), html, count=1)


def on_post_page(output, page, config):
    entry = _notebooks.get(page.file.src_uri)
    if entry:
        out_dir = os.path.dirname(os.path.join(config["site_dir"], page.file.dest_uri))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, entry[0]), "w", encoding="utf-8") as f:
            json.dump(entry[1], f, indent=1, ensure_ascii=False)
    return output
