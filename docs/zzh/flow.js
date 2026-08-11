// torchtitan learning docs - lightweight SVG flowchart engine (offline, no deps)
// Spec format (see 02_training_flow.html):
//   Flow.reg(name, {
//     width, nodeWidth,
//     blocks: [
//       {id, text, shape:'rect'|'diamond', cls:'accent'|'green'|'purple'|'amber'},
//       { branch:true, tracks: [ [node,...], [node,...] ] },   // parallel paths, fan-out from prev, fan-in to next
//     ],
//     extraEdges: [ {from, to, kind:'loop', label} ]   // side loop-back arrow
//   })
(function () {
  const SVGNS = "http://www.w3.org/2000/svg";
  const ROW_H = 94;
  const PAD = 22;

  function nodeH(n) {
    const lines = (n.text || "").split("\n").length;
    if (n.shape === "diamond") return Math.max(78, lines * 15 + 46);
    return Math.max(48, lines * 19 + 26);
  }
  function mk(tag, attrs) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }
  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function textDiv(text, diamond) {
    const d = document.createElement("div");
    d.className = "flow-text" + (diamond ? " flow-text-diamond" : "");
    d.innerHTML = text
      .split("\n")
      .map((l) => "<div>" + esc(l) + "</div>")
      .join("");
    return d;
  }

  window.__FLOWS__ = window.__FLOWS__ || {};

  function render(name, el) {
    const spec = window.__FLOWS__[name];
    if (!spec) {
      el.textContent = "[missing flow: " + name + "]";
      return;
    }
    const W = spec.width || 860;
    const NW = spec.nodeWidth || 226;
    const nodes = {};
    const edges = [];
    let cursor = 0,
      prev = null,
      pendingFanIn = null;

    spec.blocks.forEach((b) => {
      if (b.branch) {
        const tracks = b.tracks;
        const span = Math.max.apply(null, tracks.map((t) => t.length));
        const laneW = (W - 2 * PAD) / tracks.length;
        const top = cursor * ROW_H;
        tracks.forEach((track, ti) => {
          const cx = PAD + ti * laneW + laneW / 2;
          track.forEach((n, ri) => {
            const cy = top + ri * ROW_H + ROW_H / 2;
            const h = nodeH(n);
            nodes[n.id] = {
              cx,
              cy,
              x: cx - NW / 2,
              y: cy - h / 2,
              w: NW,
              h,
              shape: n.shape || "rect",
              cls: n.cls || "",
              text: n.text,
            };
          });
        });
        if (prev)
          tracks.forEach((t) => edges.push({ from: prev, to: t[0].id, kind: "fanout" }));
        b._lasts = tracks.map((t) => t[t.length - 1].id);
        pendingFanIn = b;
        cursor += span;
        prev = null;
      } else {
        if (pendingFanIn) {
          pendingFanIn._lasts.forEach((id) =>
            edges.push({ from: id, to: b.id, kind: "fanin" })
          );
          pendingFanIn = null;
        }
        const cy = cursor * ROW_H + ROW_H / 2;
        const h = nodeH(b);
        nodes[b.id] = {
          cx: W / 2,
          cy,
          x: W / 2 - NW / 2,
          y: cy - h / 2,
          w: NW,
          h,
          shape: b.shape || "rect",
          cls: b.cls || "",
          text: b.text,
        };
        if (prev) edges.push({ from: prev, to: b.id, kind: "seq" });
        cursor++;
        prev = b.id;
      }
    });
    (spec.extraEdges || []).forEach((e) => edges.push(e));

    const H = cursor * ROW_H + PAD;
    const svg = mk("svg", {
      viewBox: "0 0 " + W + " " + H,
      class: "flow-svg",
      preserveAspectRatio: "xMidYMin meet",
    });

    const defs = mk("defs", {});
    [
      ["arw", "#64748b"],
      ["arwB", "#2563eb"],
    ].forEach(([id, c]) => {
      const m = mk("marker", {
        id,
        viewBox: "0 0 10 10",
        refX: 9,
        refY: 5,
        markerWidth: 7,
        markerHeight: 7,
        orient: "auto-start-reverse",
      });
      m.appendChild(mk("path", { d: "M0,0 L10,5 L0,10 z", fill: c }));
      defs.appendChild(m);
    });
    svg.appendChild(defs);

    edges.forEach((e) => {
      const s = nodes[e.from],
        t = nodes[e.to];
      if (!s || !t) return;
      let d,
        color = "#94a3b8",
        mk2 = "arw";
      if (e.kind === "loop") {
        const leftX = Math.min(s.x, t.x) - 30;
        d =
          "M " +
          s.x +
          " " +
          s.cy +
          " L " +
          leftX +
          " " +
          s.cy +
          " L " +
          leftX +
          " " +
          t.cy +
          " L " +
          t.x +
          " " +
          t.cy;
        color = "#475569";
        mk2 = "arw";
      } else if (Math.abs(s.cx - t.cx) < 1) {
        d = "M " + s.cx + " " + (s.y + s.h / 2) + " L " + t.cx + " " + (t.y - t.h / 2);
      } else {
        const midY = (s.y + s.h / 2 + (t.y - t.h / 2)) / 2;
        d =
          "M " +
          s.cx +
          " " +
          (s.y + s.h / 2) +
          " L " +
          s.cx +
          " " +
          midY +
          " L " +
          t.cx +
          " " +
          midY +
          " L " +
          t.cx +
          " " +
          (t.y - t.h / 2);
        color = "#2563eb";
        mk2 = "arwB";
      }
      svg.appendChild(
        mk("path", {
          d,
          fill: "none",
          stroke: color,
          "stroke-width": e.kind === "seq" ? 1.6 : 2,
          "marker-end": "url(#" + mk2 + ")",
        })
      );
      if (e.label) {
        const tx = mk("text", {
          x: Math.min(s.x, t.x) - 34,
          y: (s.cy + t.cy) / 2 + 4,
          class: "flow-loop-label",
        });
        tx.textContent = e.label;
        svg.appendChild(tx);
      }
    });

    for (const id in nodes) {
      const n = nodes[id];
      const g = mk("g", {});
      if (n.shape === "diamond") {
        const pts =
          n.cx +
          "," +
          n.y +
          " " +
          (n.x + n.w) +
          "," +
          n.cy +
          " " +
          n.cx +
          "," +
          (n.y + n.h) +
          " " +
          n.x +
          "," +
          n.cy;
        g.appendChild(mk("polygon", { points: pts, class: "node node-" + n.cls }));
        const fo = mk("foreignObject", {
          x: n.x + n.w * 0.14,
          y: n.y + 6,
          width: n.w * 0.72,
          height: n.h - 12,
        });
        fo.appendChild(textDiv(n.text, true));
        g.appendChild(fo);
      } else {
        g.appendChild(
          mk("rect", {
            x: n.x,
            y: n.y,
            width: n.w,
            height: n.h,
            rx: 10,
            class: "node node-" + n.cls,
          })
        );
        const fo = mk("foreignObject", {
          x: n.x + 8,
          y: n.y + 4,
          width: n.w - 16,
          height: n.h - 8,
        });
        fo.appendChild(textDiv(n.text, false));
        g.appendChild(fo);
      }
      svg.appendChild(g);
    }
    el.appendChild(svg);
  }

  window.Flow = {
    reg: function (name, spec) {
      window.__FLOWS__[name] = spec;
    },
    render: render,
  };

  // Auto-render any container with data-flow once DOM is ready.
  function boot() {
    document.querySelectorAll("[data-flow]").forEach((el) => {
      const name = el.getAttribute("data-flow");
      if (!el.childNodes.length) render(name, el);
    });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
