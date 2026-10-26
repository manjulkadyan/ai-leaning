// Gradient-descent playground: pick a 2-D loss, click to choose a starting point,
// set the learning rate and watch w <- w - lr * grad L(w) walk across the contour map.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const LOSSES = {
    round: {
      label: "round bowl",
      formula: "L = x² + y²",
      f: (x, y) => x * x + y * y,
      g: (x, y) => [2 * x, 2 * y],
      x: [-3, 3], y: [-2, 2], start: [-2.5, 1.5], lr: 0.1,
      note: "curvature 2 in every direction, so it diverges for lr > 2 / 2 = 1",
    },
    stretched: {
      label: "stretched bowl",
      formula: "L = x² + 10y²",
      f: (x, y) => x * x + 10 * y * y,
      g: (x, y) => [2 * x, 20 * y],
      x: [-3, 3], y: [-2, 2], start: [-2.5, 1.5], lr: 0.09,
      note: "curvatures 2 (along x) and 20 (along y): diverges for lr > 2 / 20 = 0.1",
    },
    twomin: {
      label: "two minima",
      formula: "L = (x² − 1)² + y² + 0.3x",
      f: (x, y) => (x * x - 1) ** 2 + y * y + 0.3 * x,
      g: (x, y) => [4 * x * (x * x - 1) + 0.3, 2 * y],
      x: [-2, 2], y: [-1.333, 1.333], start: [0.2, 1.0], lr: 0.05,
      note: "a deeper valley on the left, a shallower one on the right: the start decides which you reach",
    },
  };
  const MAX_STEPS = 500;

  function mount(root) {
    const W = 540, H = 360;
    const state = { key: "round", path: [], lr: 0.1, timer: null, status: "" };

    const bar = document.createElement("div");
    bar.className = "gdp-bar";
    const buttons = {};
    for (const key in LOSSES) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "md-button gdp-btn";
      b.textContent = LOSSES[key].label;
      b.addEventListener("click", () => choose(key));
      buttons[key] = b;
      bar.appendChild(b);
    }

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "gdp-svg");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Contour map of the chosen loss with the gradient-descent path. Click or tap to choose a starting point (this needs a mouse, touch or pen; with a keyboard, use the default start and the buttons).");
    svg.style.touchAction = "manipulation";   // taps choose a start; the page still scrolls
    const el = (tag, attrs, parent = svg) => {
      const e = document.createElementNS(NS, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      parent.appendChild(e);
      return e;
    };
    const clipId = "gdp-clip-" + Math.random().toString(36).slice(2, 8);
    const cp = el("clipPath", { id: clipId }, el("defs", {}));
    el("rect", { x: 0, y: 0, width: W, height: H }, cp);
    el("rect", { x: 0, y: 0, width: W, height: H, class: "gdp-bg" });
    const contourG = el("g", {});
    const pathG = el("g", { "clip-path": `url(#${clipId})` });
    const trail = el("polyline", { class: "gdp-trail" }, pathG);
    const dotsG = el("g", {}, pathG);
    const minG = el("g", {}, pathG);
    const current = el("circle", { r: 7, class: "gdp-current" }, pathG);

    // Learning-rate slider over a list of round values (so 0.5, 0.105, 1.1 are exactly reachable),
    // plus a number box for any other value.
    const LRS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.09, 0.095, 0.1, 0.105, 0.11,
      0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.5];
    const ctrl = document.createElement("div");
    ctrl.className = "gdp-ctrl";
    const lab = document.createElement("label");
    lab.innerHTML = "<span>learning rate</span>";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(LRS.length - 1);
    slider.step = "1";
    slider.setAttribute("aria-label", "learning rate");
    const lrOut = document.createElement("input");
    lrOut.type = "number";
    lrOut.min = "0.0001";
    lrOut.max = "5";
    lrOut.step = "any";
    lrOut.className = "gdp-lr";
    lrOut.setAttribute("aria-label", "learning rate value");
    const lrLab = document.createElement("span");
    lrLab.textContent = "lr =";
    const lrGroup = document.createElement("span");   // keeps "lr =" and its box on one line
    lrGroup.className = "gdp-lrgroup";
    lrGroup.appendChild(lrLab);
    lrGroup.appendChild(lrOut);
    lab.appendChild(slider);
    lab.appendChild(lrGroup);
    ctrl.appendChild(lab);
    const mkBtn = (text, fn) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "md-button gdp-btn";
      b.textContent = text;
      b.addEventListener("click", fn);
      ctrl.appendChild(b);
      return b;
    };
    mkBtn("Step", () => { stop(); step(); render(); });
    const runBtn = mkBtn("Run", () => (state.timer ? stop() : run()));
    mkBtn("Reset", () => { stop(); reset(state.path[0]); });

    const readout = document.createElement("div");
    readout.className = "gdp-readout";
    readout.setAttribute("aria-live", "polite");

    root.appendChild(bar);
    root.appendChild(svg);
    root.appendChild(ctrl);
    root.appendChild(readout);

    let sx, sy, ix, iy;
    function scales(F) {
      const [x0, x1] = F.x, [y0, y1] = F.y;
      sx = (x) => ((x - x0) / (x1 - x0)) * W;
      sy = (y) => ((y1 - y) / (y1 - y0)) * H;
      ix = (px) => x0 + (px / W) * (x1 - x0);
      iy = (py) => y1 - (py / H) * (y1 - y0);
    }

    // Contours by marching squares on a grid: each cell contributes 0-2 segments per level.
    function drawContours(F) {
      while (contourG.firstChild) contourG.removeChild(contourG.firstChild);
      const nx = 120, ny = 80;
      const vals = [];
      let lo = Infinity, hi = -Infinity;
      for (let j = 0; j <= ny; j++) {
        const row = [];
        for (let i = 0; i <= nx; i++) {
          const v = F.f(ix((i / nx) * W), iy((j / ny) * H));
          row.push(v);
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
        vals.push(row);
      }
      const nLev = 14;
      for (let k = 1; k <= nLev; k++) {
        const level = lo + (hi - lo) * Math.pow(k / (nLev + 1), 2);
        let d = "";
        for (let j = 0; j < ny; j++) {
          for (let i = 0; i < nx; i++) {
            const c = [vals[j][i], vals[j][i + 1], vals[j + 1][i + 1], vals[j + 1][i]];
            const p = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
            const pts = [];
            for (let e = 0; e < 4; e++) {
              const a = c[e], b = c[(e + 1) % 4];
              if ((a < level) !== (b < level)) {
                const t = (level - a) / (b - a);
                const pa = p[e], pb = p[(e + 1) % 4];
                pts.push([((pa[0] + t * (pb[0] - pa[0])) / nx) * W, ((pa[1] + t * (pb[1] - pa[1])) / ny) * H]);
              }
            }
            for (let q = 0; q + 1 < pts.length; q += 2) {
              d += `M${pts[q][0].toFixed(1)},${pts[q][1].toFixed(1)}L${pts[q + 1][0].toFixed(1)},${pts[q + 1][1].toFixed(1)}`;
            }
          }
        }
        el("path", { d, class: "gdp-contour", "stroke-opacity": (0.85 - (0.5 * k) / nLev).toFixed(2) }, contourG);
      }
      // axes through 0
      el("line", { x1: sx(0), y1: 0, x2: sx(0), y2: H, class: "gdp-axis" }, contourG);
      el("line", { x1: 0, y1: sy(0), x2: W, y2: sy(0), class: "gdp-axis" }, contourG);
      el("text", { x: W - 6, y: sy(0) - 5, class: "gdp-tick", "text-anchor": "end" }, contourG).textContent = "x";
      el("text", { x: sx(0) + 6, y: 14, class: "gdp-tick" }, contourG).textContent = "y";
    }

    function drawMinima(F) {
      while (minG.firstChild) minG.removeChild(minG.firstChild);
      const mins = state.key === "twomin" ? [[-1.0356, 0], [0.9601, 0]] : [[0, 0]];
      for (const [mx, my] of mins) el("circle", { cx: sx(mx), cy: sy(my), r: 4.5, class: "gdp-min" }, minG);
    }

    const cur = () => state.path[state.path.length - 1];
    function step() {
      if (state.status === "diverged" || state.status === "converged" || state.status === "max") return false;
      const F = LOSSES[state.key];
      const [x, y] = cur();
      const [gx, gy] = F.g(x, y);
      const nx = x - state.lr * gx, ny = y - state.lr * gy;
      state.path.push([nx, ny]);
      const [hx, hy] = F.g(nx, ny);
      const gn = Math.hypot(hx, hy);
      if (!isFinite(nx) || !isFinite(ny) || Math.abs(F.f(nx, ny)) > 1e6) state.status = "diverged";
      else if (gn < 1e-4) state.status = "converged";
      else if (state.path.length - 1 >= MAX_STEPS) state.status = "max";
      else state.status = "";
      return state.status === "";
    }
    function run() {
      runBtn.textContent = "Pause";
      runBtn.classList.add("md-button--primary");
      state.timer = setInterval(() => {
        const more = step();
        render();
        if (!more) stop();
      }, 90);
    }
    function stop() {
      if (state.timer) clearInterval(state.timer);
      state.timer = null;
      runBtn.textContent = "Run";
      runBtn.classList.remove("md-button--primary");
    }
    function reset(start) {
      state.path = [start.slice()];
      state.status = "";
      render();
    }
    function choose(key) {
      stop();
      state.key = key;
      const F = LOSSES[key];
      scales(F);
      drawContours(F);
      drawMinima(F);
      for (const k in buttons) buttons[k].classList.toggle("md-button--primary", k === key);
      setLr(F.lr);
      reset(F.start);
    }
    function setLr(lr) {
      state.lr = lr;
      // put the slider on the nearest listed value (on a log scale)
      let best = 0;
      LRS.forEach((v, i) => { if (Math.abs(Math.log(v / lr)) < Math.abs(Math.log(LRS[best] / lr))) best = i; });
      slider.value = String(best);
      lrOut.value = String(lr);
    }
    const lrChanged = () => {
      if (state.path.length > 1) { stop(); reset(state.path[0]); }
      else render();
    };
    slider.addEventListener("input", () => {
      setLr(LRS[parseInt(slider.value, 10)]);
      lrChanged();
    });
    lrOut.addEventListener("change", () => {
      const v = parseFloat(lrOut.value);
      if (!(v > 0) || !isFinite(v)) { lrOut.value = String(state.lr); return; }
      setLr(v);
      lrChanged();
    });

    const clampPx = (v, lim) => Math.max(-lim, Math.min(2 * lim, v));
    const fmt = (v, d = 3) => {
      if (!isFinite(v)) return String(v);
      if (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0)) return v.toExponential(2);
      const r = v.toFixed(d);
      return /^-0\.0*$/.test(r) ? r.slice(1) : r;
    };
    function render() {
      const F = LOSSES[state.key];
      const pts = state.path.map(([x, y]) => [clampPx(sx(x), W), clampPx(sy(y), H)]);
      trail.setAttribute("points", pts.map((p) => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" "));
      while (dotsG.firstChild) dotsG.removeChild(dotsG.firstChild);
      pts.forEach((p, i) => el("circle", { cx: p[0], cy: p[1], r: i === 0 ? 5 : 2.6, class: i === 0 ? "gdp-start" : "gdp-dot" }, dotsG));
      const last = pts[pts.length - 1];
      current.setAttribute("cx", last[0]);
      current.setAttribute("cy", last[1]);
      const [x, y] = cur();
      const L = F.f(x, y);
      const [gx, gy] = F.g(x, y);
      const status = {
        "": "",
        converged: ' &nbsp; <b class="gdp-ok">converged</b> (gradient length below 0.0001)',
        diverged: ' &nbsp; <b class="gdp-bad">diverged</b>: lower the learning rate and press Reset',
        max: ` &nbsp; stopped after ${MAX_STEPS} steps`,
      }[state.status];
      readout.innerHTML =
        `<div><b>${F.formula}</b> &nbsp; step ${state.path.length - 1}${status}</div>` +
        `<div>w = (${fmt(x)}, ${fmt(y)}) &nbsp; loss = <b>${fmt(L, 4)}</b> &nbsp; |gradient| = ${fmt(Math.hypot(gx, gy), 4)}</div>` +
        `<div class="gdp-note">${F.note}. Click the map to choose a new start.</div>`;
    }

    svg.addEventListener("click", (e) => {   // click, not pointerdown, so a scroll gesture never moves the start
      const pt = svg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      stop();
      reset([Math.round(ix(p.x) * 100) / 100, Math.round(iy(p.y) * 100) / 100]);
    });

    choose("round");
  }

  function init() {
    document.querySelectorAll(".gd-playground:not([data-mounted])").forEach((root) => {
      root.dataset.mounted = "1";
      mount(root);
    });
  }

  if (typeof document$ !== "undefined") document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
