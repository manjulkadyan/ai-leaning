// Interactive matrix-multiplication stepper: edit A (2x3) and B (3x2), then
// press Step (or hover a cell of C) to see which row of A and column of B
// produce each cell, with the full sum written out.
(function () {
  const A0 = [[1, 2, 3], [4, 5, 6]];
  const B0 = [[7, 8], [9, 10], [11, 12]];
  const ORDER = [[0, 0], [0, 1], [1, 0], [1, 1]];

  // Show integers as integers, otherwise up to 2 decimals.
  const fmt = (n) => {
    const r = Math.round(n * 100) / 100;
    return Number.isInteger(r) ? String(r) : r.toFixed(2).replace(/0+$/, "");
  };
  // Wrap negatives in brackets inside a product, e.g. 2×(-3).
  const term = (n) => (n < 0 ? `(${fmt(n)})` : fmt(n));

  function mount(root) {
    const A = A0.map((r) => r.slice());
    const B = B0.map((r) => r.slice());
    let step = -1; // index into ORDER of the last stepped cell
    const revealed = new Set(); // cells of C shown so far, as "i,j"

    const make = (tag, cls, parent) => {
      const e = document.createElement(tag);
      if (cls) e.className = cls;
      if (parent) parent.appendChild(e);
      return e;
    };

    const board = make("div", "mm-board", root);

    function matrix(label, rows, cols, cls) {
      const wrap = make("div", "mm-matrix " + cls, board);
      make("div", "mm-label", wrap).textContent = `${label}  (${rows}×${cols})`;
      const grid = make("div", "mm-grid", wrap);
      grid.style.gridTemplateColumns = `repeat(${cols}, auto)`;
      const cells = [];
      for (let i = 0; i < rows; i++) {
        cells.push([]);
        for (let j = 0; j < cols; j++) cells[i].push(grid);
      }
      return { wrap, grid, cells };
    }

    function inputs(M, label, cls) {
      const m = matrix(label, M.length, M[0].length, cls);
      for (let i = 0; i < M.length; i++) {
        for (let j = 0; j < M[0].length; j++) {
          const inp = make("input", "mm-cell mm-input", m.grid);
          inp.type = "number";
          inp.step = "any";
          inp.value = M[i][j];
          inp.setAttribute("aria-label", `${label}[${i},${j}]`);
          inp.addEventListener("input", () => {
            const v = parseFloat(inp.value);
            M[i][j] = Number.isFinite(v) ? v : 0;
            render();
          });
          m.cells[i][j] = inp;
        }
      }
      return m;
    }

    const mA = inputs(A, "A", "mm-a");
    make("div", "mm-op", board).textContent = "@";
    const mB = inputs(B, "B", "mm-b");
    make("div", "mm-op", board).textContent = "=";
    const mC = matrix("C", 2, 2, "mm-c");
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        const c = make("div", "mm-cell mm-out", mC.grid);
        c.tabIndex = 0;
        c.setAttribute("aria-label", `C[${i},${j}]`);
        c.addEventListener("mouseenter", () => render([i, j]));
        c.addEventListener("focus", () => render([i, j]));
        c.addEventListener("mouseleave", () => render());
        c.addEventListener("blur", () => render());
        mC.cells[i][j] = c;
      }
    }

    const controls = make("div", "mm-controls", root);
    const stepBtn = make("button", "md-button md-button--primary mm-btn", controls);
    stepBtn.type = "button";
    stepBtn.textContent = "Step";
    const resetBtn = make("button", "md-button mm-btn", controls);
    resetBtn.type = "button";
    resetBtn.textContent = "Reset";
    const readout = make("div", "mm-readout", root);
    readout.setAttribute("aria-live", "polite");

    const product = () =>
      [0, 1].map((i) => [0, 1].map((j) => A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]));

    // focus: [i, j] of a hovered cell, or undefined to show the stepped cell.
    function render(focus) {
      const C = product();
      const cur = focus || (step >= 0 ? ORDER[step] : null);

      for (let i = 0; i < 2; i++) {
        for (let k = 0; k < 3; k++) mA.cells[i][k].classList.toggle("mm-hl", !!cur && cur[0] === i);
      }
      for (let k = 0; k < 3; k++) {
        for (let j = 0; j < 2; j++) mB.cells[k][j].classList.toggle("mm-hl", !!cur && cur[1] === j);
      }
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          const c = mC.cells[i][j];
          const isCur = !!cur && cur[0] === i && cur[1] === j;
          const show = revealed.has(`${i},${j}`) || isCur;
          c.textContent = show ? fmt(C[i][j]) : "?";
          c.classList.toggle("mm-hl", isCur);
          c.classList.toggle("mm-pending", !show);
        }
      }

      if (!cur) {
        readout.innerHTML = "Press <b>Step</b>, or hover a cell of <b>C</b>, to see where it comes from.";
        return;
      }
      const [i, j] = cur;
      const terms = [0, 1, 2]
        .map((k) => `<span class="mm-ta">${term(A[i][k])}</span>×<span class="mm-tb">${term(B[k][j])}</span>`)
        .join(" + ");
      readout.innerHTML =
        `<div><b class="mm-tc">C[${i},${j}]</b> = row ${i} of A · column ${j} of B</div>` +
        `<div><b class="mm-tc">C[${i},${j}]</b> = ${terms} = <b class="mm-tc">${fmt(C[i][j])}</b></div>`;
    }

    stepBtn.addEventListener("click", () => {
      step = (step + 1) % ORDER.length;
      revealed.add(ORDER[step].join(","));
      render();
    });
    resetBtn.addEventListener("click", () => {
      step = -1;
      revealed.clear();
      A0.forEach((r, i) => r.forEach((v, j) => { A[i][j] = v; mA.cells[i][j].value = v; }));
      B0.forEach((r, i) => r.forEach((v, j) => { B[i][j] = v; mB.cells[i][j].value = v; }));
      render();
    });

    render();
  }

  function init() {
    document.querySelectorAll(".matmul-stepper:not([data-mounted])").forEach((root) => {
      root.dataset.mounted = "1";
      mount(root);
    });
  }

  if (typeof document$ !== "undefined") document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
