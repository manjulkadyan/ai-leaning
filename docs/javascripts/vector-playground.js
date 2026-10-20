// Interactive 2-D vector playground: drag the tips of a and b to see
// length, dot product, angle and cosine similarity update live.
(function () {
  function mount(root) {
    const NS = "http://www.w3.org/2000/svg";
    const W = 360, H = 360, S = 40, OX = W / 2, OY = H / 2; // 1 unit = 40px
    const state = { a: [3, 1], b: [1, 2.5] };

    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("class", "vp-svg");
    svg.style.touchAction = "none";

    const el = (tag, attrs, parent = svg) => {
      const e = document.createElementNS(NS, tag);
      for (const k in attrs) e.setAttribute(k, attrs[k]);
      parent.appendChild(e);
      return e;
    };

    for (let i = -4; i <= 4; i++) {
      el("line", { x1: OX + i * S, y1: 0, x2: OX + i * S, y2: H, class: i === 0 ? "vp-axis" : "vp-grid" });
      el("line", { x1: 0, y1: OY + i * S, x2: W, y2: OY + i * S, class: i === 0 ? "vp-axis" : "vp-grid" });
    }
    const defs = el("defs", {});
    for (const [id, cls] of [["vpa", "vp-a"], ["vpb", "vp-b"]]) {
      const m = el("marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", class: cls + "-fill" }, m);
    }
    const arc = el("path", { class: "vp-arc" });
    const lines = {
      a: el("line", { x1: OX, y1: OY, class: "vp-a", "marker-end": "url(#vpa)" }),
      b: el("line", { x1: OX, y1: OY, class: "vp-b", "marker-end": "url(#vpb)" }),
    };
    const handles = {
      a: el("circle", { r: 12, class: "vp-handle vp-a-fill" }),
      b: el("circle", { r: 12, class: "vp-handle vp-b-fill" }),
    };
    const labels = { a: el("text", { class: "vp-label vp-a-fill" }), b: el("text", { class: "vp-label vp-b-fill" }) };
    labels.a.textContent = "a";
    labels.b.textContent = "b";

    const readout = document.createElement("div");
    readout.className = "vp-readout";
    root.appendChild(svg);
    root.appendChild(readout);

    const toPx = ([x, y]) => [OX + x * S, OY - y * S];
    const f = (n) => (Math.round(n * 100) / 100).toFixed(2);

    function render() {
      for (const k of ["a", "b"]) {
        const [px, py] = toPx(state[k]);
        lines[k].setAttribute("x2", px);
        lines[k].setAttribute("y2", py);
        handles[k].setAttribute("cx", px);
        handles[k].setAttribute("cy", py);
        labels[k].setAttribute("x", px + 10);
        labels[k].setAttribute("y", py - 10);
      }
      const [ax, ay] = state.a, [bx, by] = state.b;
      const la = Math.hypot(ax, ay), lb = Math.hypot(bx, by);
      const dot = ax * bx + ay * by;
      const cos = la && lb ? dot / (la * lb) : 0;
      const angle = (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;

      const r = 28, t1 = Math.atan2(ay, ax), t2 = Math.atan2(by, bx);
      let d = t2 - t1;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      const p1 = [OX + r * Math.cos(t1), OY - r * Math.sin(t1)];
      const p2 = [OX + r * Math.cos(t1 + d), OY - r * Math.sin(t1 + d)];
      arc.setAttribute("d", `M${p1[0]},${p1[1]} A${r},${r} 0 0 ${d > 0 ? 0 : 1} ${p2[0]},${p2[1]}`);

      readout.innerHTML =
        `<div><b class="vp-a-text">a</b> = [${f(ax)}, ${f(ay)}] &nbsp; ‖a‖ = ${f(la)}</div>` +
        `<div><b class="vp-b-text">b</b> = [${f(bx)}, ${f(by)}] &nbsp; ‖b‖ = ${f(lb)}</div>` +
        `<div>a · b = ${f(ax)}×${f(bx)} + ${f(ay)}×${f(by)} = <b>${f(dot)}</b></div>` +
        `<div>angle = <b>${f(angle)}°</b> &nbsp; cosine similarity = <b>${f(cos)}</b></div>`;
    }

    let dragging = null;
    const pt = svg.createSVGPoint();
    function toUnits(evt) {
      pt.x = evt.clientX;
      pt.y = evt.clientY;
      const p = pt.matrixTransform(svg.getScreenCTM().inverse());
      const snap = (v) => Math.round(v * 4) / 4;
      return [snap(Math.max(-4, Math.min(4, (p.x - OX) / S))), snap(Math.max(-4, Math.min(4, (OY - p.y) / S)))];
    }
    for (const k of ["a", "b"]) {
      handles[k].addEventListener("pointerdown", (e) => {
        dragging = k;
        svg.setPointerCapture(e.pointerId);
      });
    }
    svg.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      state[dragging] = toUnits(e);
      render();
    });
    svg.addEventListener("pointerup", () => (dragging = null));
    render();
  }

  function init() {
    document.querySelectorAll(".vector-playground:not([data-mounted])").forEach((root) => {
      root.dataset.mounted = "1";
      mount(root);
    });
  }

  if (typeof document$ !== "undefined") document$.subscribe(init);
  else document.addEventListener("DOMContentLoaded", init);
})();
