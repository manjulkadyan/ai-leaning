// Interactive tangent explorer: pick a function, drag the point along the curve,
// and compare the tangent line (the true derivative) with a secant of width h.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const sig = (x) => 1 / (1 + Math.exp(-x));
  const FUNCS = {
    square: {
      label: "x²", f: (x) => x * x, df: (x) => 2 * x, dfText: "2x",
      x: [-3, 3], y: [-1.5, 9], start: 1,
    },
    cubic: {
      label: "x³ − x", f: (x) => x * x * x - x, df: (x) => 3 * x * x - 1, dfText: "3x² − 1",
      x: [-1.8, 1.8], y: [-4.5, 4.5], start: 1,
    },
    sine: {
      label: "sin x", f: Math.sin, df: Math.cos, dfText: "cos x",
      x: [-6.5, 6.5], y: [-1.6, 1.6], start: 1,
    },
    sigmoid: {
      label: "sigmoid σ(x)", f: sig, df: (x) => sig(x) * (1 - sig(x)), dfText: "σ(x)(1 − σ(x))",
      x: [-6, 6], y: [-0.25, 1.25], start: 0,
    },
  };

  function mount(root) {
    const W = 520, H = 340, PAD = 28;
    const state = { key: "square", a: 1, logh: Math.log10(0.5) };

    // Function picker
    const bar = document.createElement("div");
    bar.className = "te-bar";
    const buttons = {};
    for (const key in FUNCS) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "md-button te-btn";
      b.textContent = FUNCS[key].label;
      b.addEventListener("click", () => {
        state.key = key;
        state.a = FUNCS[key].start;
        render();
      });
      buttons[key] = b;
      bar.appendChild(b);
    }

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "te-svg");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Plot of the chosen function with a draggable point, its tangent line and a secant line");
    svg.style.touchAction = "none";
    const el = (tag, attrs, parent = svg) => {
      const e = document.createElementNS(NS, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      parent.appendChild(e);
      return e;
    };
    const clipId = "te-clip-" + Math.random().toString(36).slice(2, 8);
    const defs = el("defs", {});
    const cp = el("clipPath", { id: clipId }, defs);
    el("rect", { x: PAD, y: 6, width: W - PAD - 6, height: H - 6 - PAD }, cp);
    const gridG = el("g", {});
    const plotG = el("g", { "clip-path": `url(#${clipId})` });
    const curve = el("path", { class: "te-curve" }, plotG);
    const secant = el("line", { class: "te-secant" }, plotG);
    const tangent = el("line", { class: "te-tangent" }, plotG);
    const qDot = el("circle", { r: 5, class: "te-q" }, plotG);
    const handle = el("circle", { r: 9, class: "te-handle", tabindex: 0, role: "slider", "aria-label": "x position of the point" });

    // h slider
    const ctrl = document.createElement("label");
    ctrl.className = "te-ctrl";
    ctrl.innerHTML = `<span>secant width <b>h</b></span>`;
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "-3";
    slider.max = String(Math.log10(2));
    slider.step = "0.01";
    slider.value = String(state.logh);
    slider.addEventListener("input", () => {
      state.logh = parseFloat(slider.value);
      render();
    });
    const hOut = document.createElement("code");
    hOut.className = "te-h";
    ctrl.appendChild(slider);
    ctrl.appendChild(hOut);

    const readout = document.createElement("div");
    readout.className = "te-readout";
    readout.setAttribute("aria-live", "polite");

    root.appendChild(bar);
    root.appendChild(svg);
    root.appendChild(ctrl);
    root.appendChild(readout);

    let sx, sy;
    function scales(F) {
      const [x0, x1] = F.x, [y0, y1] = F.y;
      sx = (x) => PAD + ((x - x0) / (x1 - x0)) * (W - PAD - 6);
      sy = (y) => 6 + ((y1 - y) / (y1 - y0)) * (H - 6 - PAD);
    }
    function niceStep(span) {
      const raw = span / 8, p = Math.pow(10, Math.floor(Math.log10(raw)));
      for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
      return 10 * p;
    }
    function drawGrid(F) {
      while (gridG.firstChild) gridG.removeChild(gridG.firstChild);
      const [x0, x1] = F.x, [y0, y1] = F.y;
      const fmt = (v) => String(Math.round(v * 100) / 100);
      const xs = niceStep(x1 - x0), ys = niceStep(y1 - y0);
      for (let v = Math.ceil(x0 / xs) * xs; v <= x1 + 1e-9; v += xs) {
        el("line", { x1: sx(v), y1: 6, x2: sx(v), y2: H - PAD, class: Math.abs(v) < 1e-9 ? "te-axis" : "te-grid" }, gridG);
        if (Math.abs(v) > 1e-9) el("text", { x: sx(v), y: H - PAD + 14, class: "te-tick", "text-anchor": "middle" }, gridG).textContent = fmt(v);
      }
      for (let v = Math.ceil(y0 / ys) * ys; v <= y1 + 1e-9; v += ys) {
        el("line", { x1: PAD, y1: sy(v), x2: W - 6, y2: sy(v), class: Math.abs(v) < 1e-9 ? "te-axis" : "te-grid" }, gridG);
        el("text", { x: PAD - 4, y: sy(v) + 4, class: "te-tick", "text-anchor": "end" }, gridG).textContent = fmt(v);
      }
    }
    const lineThrough = (node, x, y, m, F) => {
      const [x0, x1] = F.x;
      node.setAttribute("x1", sx(x0));
      node.setAttribute("y1", sy(y + m * (x0 - x)));
      node.setAttribute("x2", sx(x1));
      node.setAttribute("y2", sy(y + m * (x1 - x)));
    };
    const fmt = (v, d = 4) => (Object.is(Math.round(v * 10 ** d) / 10 ** d, -0) ? 0 : v).toFixed(d);
    const sci = (v) => (v === 0 ? "0" : Math.abs(v) >= 1e-3 ? v.toFixed(5) : v.toExponential(2));

    let lastKey = null;
    function render() {
      const F = FUNCS[state.key];
      scales(F);
      if (lastKey !== state.key) {
        drawGrid(F);
        let d = "";
        const n = 300;
        for (let i = 0; i <= n; i++) {
          const x = F.x[0] + ((F.x[1] - F.x[0]) * i) / n;
          d += (i ? " L" : "M") + sx(x).toFixed(2) + "," + sy(F.f(x)).toFixed(2);
        }
        curve.setAttribute("d", d);
        for (const k in buttons) buttons[k].classList.toggle("md-button--primary", k === state.key);
        lastKey = state.key;
      }
      const a = state.a, h = Math.pow(10, state.logh);
      const fa = F.f(a), slope = F.df(a);
      const fwd = (F.f(a + h) - fa) / h;
      const ctr = (F.f(a + h) - F.f(a - h)) / (2 * h);
      lineThrough(tangent, a, fa, slope, F);
      lineThrough(secant, a, fa, fwd, F);
      qDot.setAttribute("cx", sx(a + h));
      qDot.setAttribute("cy", sy(F.f(a + h)));
      handle.setAttribute("cx", sx(a));
      handle.setAttribute("cy", sy(fa));
      handle.setAttribute("aria-valuenow", a.toFixed(2));
      hOut.textContent = "h = " + (h >= 0.01 ? h.toFixed(3) : h.toExponential(1));
      readout.innerHTML =
        `<div>f(x) = <b>${F.label.replace("sigmoid ", "")}</b> &nbsp; x = ${fmt(a, 2)} &nbsp; f(x) = ${fmt(fa)}</div>` +
        `<div><span class="te-t">tangent slope</span> f′(x) = ${F.dfText} = <b>${fmt(slope)}</b></div>` +
        `<div><span class="te-s">secant slope</span> (f(x+h) − f(x)) / h = <b>${fmt(fwd)}</b> &nbsp; error ${sci(fwd - slope)}</div>` +
        `<div>central (f(x+h) − f(x−h)) / 2h = <b>${fmt(ctr)}</b> &nbsp; error ${sci(ctr - slope)}</div>`;
    }

    // Dragging: the point follows the pointer's x position along the curve.
    const pt = svg.createSVGPoint();
    function xFromEvent(evt) {
      pt.x = evt.clientX;
      pt.y = evt.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      const F = FUNCS[state.key], [x0, x1] = F.x;
      const x = x0 + ((p.x - PAD) / (W - PAD - 6)) * (x1 - x0);
      return Math.round(Math.max(x0, Math.min(x1, x)) * 100) / 100;
    }
    let dragging = false;
    svg.addEventListener("pointerdown", (e) => {
      dragging = true;
      svg.setPointerCapture(e.pointerId);
      state.a = xFromEvent(e);
      render();
    });
    svg.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      state.a = xFromEvent(e);
      render();
    });
    const stop = () => (dragging = false);
    svg.addEventListener("pointerup", stop);
    svg.addEventListener("pointercancel", stop);
    handle.addEventListener("keydown", (e) => {
      const F = FUNCS[state.key], step = (F.x[1] - F.x[0]) / 100;
      if (e.key === "ArrowRight" || e.key === "ArrowUp") state.a = Math.min(F.x[1], state.a + step);
      else if (e.key === "ArrowLeft" || e.key === "ArrowDown") state.a = Math.max(F.x[0], state.a - step);
      else return;
      e.preventDefault();
      render();
    });
    render();
  }

  function init() {
    document.querySelectorAll(".tangent-explorer:not([data-mounted])").forEach((root) => {
      root.dataset.mounted = "1";
      mount(root);
    });
  }

  if (typeof document$ !== "undefined") document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
