// Model picker over the current set: filter chips (method, code, tags, anchors, data, step), presets, checkboxes.
// Chips and presets select everything they match; checkboxes fine-tune the selection.
import { h, methodColor } from "../lib/dom.js";

const FACETS = [
  { id: "method", label: "Method", of: (m) => [m.method] },
  { id: "code", label: "Code", of: (m) => [m.code] },
  { id: "tags", label: "Tags", of: (m) => (m.tags.length ? m.tags : ["none"]) },
  { id: "anchors", label: "Anchors", of: (m) => [m.anchors] },
  { id: "data", label: "SFT data", of: (m) => [m.data] },
  { id: "step", label: "Step", of: (m) => [m.step] },
];
const filterState = new Map(); // "<experiment>:<pool>" -> { facet: Set(values) }, kept while the page is open

function facetLabel(exp, facet, v) {
  if (facet === "method") return exp.methodsById[v]?.label || v;
  if (facet === "anchors") return v === true ? "with" : v === false ? "without" : "unknown";
  return v == null ? "unknown" : String(v);
}

function matches(m, filters, query) {
  for (const f of FACETS) {
    const want = filters[f.id];
    if (want?.size && !f.of(m).some((v) => want.has(v))) return false;
  }
  if (query) {
    const hay = [m.label, m.abbrev, m.method, m.code, ...m.tags, m.sft?.run, m.pretrain?.run].join(" ").toLowerCase();
    return query.toLowerCase().split(/\s+/).every((q) => hay.includes(q));
  }
  return true;
}

/** Returns the toolbar button; opening it shows the popover. `onChange(ids[])` gets the new selection. */
export function modelPicker(exp, pool, selected, onChange) {
  const total = pool.length;
  const btn = h("button", { class: "btn", type: "button", "aria-haspopup": "dialog", "aria-expanded": "false" },
    h("span", {}, "Models"), h("span", { class: "count" }, `${selected.size} / ${total}`),
    h("svg", { viewBox: "0 0 16 16", width: "12", height: "12", html: '<path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>' }));
  btn.addEventListener("click", () => open(exp, pool, btn, new Set(selected), onChange));
  return btn;
}

function open(exp, pool, anchor, sel, onChange) {
  document.querySelector(".popover")?.remove();
  const stateKey = `${exp.id}:${pool.map((m) => m.id).join(",")}`;
  if (!filterState.has(stateKey)) filterState.set(stateKey, {});
  const filters = filterState.get(stateKey);
  let query = "";

  const pop = h("div", { class: "popover", role: "dialog", "aria-label": "Choose models" });
  const r = anchor.getBoundingClientRect();
  pop.style.top = `${r.bottom + window.scrollY + 6}px`;
  pop.style.left = `${Math.max(16, Math.min(r.left + window.scrollX, window.innerWidth - 16 - Math.min(620, window.innerWidth - 32)))}px`;

  const commit = () => onChange(pool.map((m) => m.id).filter((id) => sel.has(id)));
  const visible = () => pool.filter((m) => matches(m, filters, query));
  const selectVisible = () => { sel = new Set(visible().map((m) => m.id)); };

  const search = h("input", { class: "search", type: "search", placeholder: "Search name, code, tag, run…",
    oninput: (e) => { query = e.target.value; renderList(); } });
  const filterGrid = h("div", { class: "filters" });
  const list = h("div", { class: "list" });
  const footer = h("footer", {});
  pop.append(h("header", {}, search, filterGrid), list, footer);

  function renderFilters() {
    filterGrid.replaceChildren();
    if (exp.manifest.presets?.length) {
      filterGrid.append(h("span", { class: "label" }, "Presets"), h("div", { class: "chips" },
        exp.manifest.presets.map((p) => h("button", { class: "chip", type: "button", onclick: () => {
          for (const k of Object.keys(filters)) delete filters[k];
          for (const [k, vs] of Object.entries(p.filter || {})) filters[k] = new Set(vs);
          selectVisible(); renderAll();
        } }, p.name)),
        h("button", { class: "chip", type: "button", onclick: () => {
          for (const k of Object.keys(filters)) delete filters[k];
          selectVisible(); renderAll();
        } }, "Reset")));
    }
    for (const f of FACETS) {
      const counts = new Map();
      for (const m of pool) for (const v of f.of(m)) counts.set(v, (counts.get(v) || 0) + 1);
      if (counts.size < 2 && !filters[f.id]?.size) continue;
      let values = [...counts.keys()];
      if (f.id === "method") values = exp.manifest.methods.map((x) => x.id).filter((v) => counts.has(v));
      else if (f.id === "tags") values.sort((a, b) => (a === "none") - (b === "none") || counts.get(b) - counts.get(a) || String(a).localeCompare(b));
      else values.sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true }));
      filterGrid.append(h("span", { class: "label" }, f.label), h("div", { class: "chips" }, values.map((v) => {
        const on = !!filters[f.id]?.has(v);
        return h("button", { class: "chip", type: "button", "aria-pressed": String(on), onclick: () => {
          filters[f.id] ||= new Set();
          on ? filters[f.id].delete(v) : filters[f.id].add(v);
          selectVisible(); renderAll();
        } },
        f.id === "method" && h("span", { class: "dot", style: { background: methodColor(exp.methodsById[v]) } }),
        facetLabel(exp, f.id, v), h("span", { class: "n" }, counts.get(v)));
      })));
    }
  }

  function renderList() {
    list.replaceChildren();
    const vis = visible();
    for (const meth of exp.manifest.methods) {
      const ms = vis.filter((m) => m.method === meth.id);
      if (!ms.length) continue;
      list.append(h("div", { class: "grp" }, meth.label));
      for (const m of ms) {
        list.append(h("label", { class: "row" },
          h("input", { type: "checkbox", checked: sel.has(m.id), onchange: (e) => {
            e.target.checked ? sel.add(m.id) : sel.delete(m.id); renderFooter(); commit();
          } }),
          h("span", { class: "dot", style: { background: methodColor(meth) } }),
          h("span", {}, m.label),
          h("span", { class: "sub" }, [m.code && `code ${m.code}`, `step ${m.step}`].filter(Boolean).join(" · "))));
      }
    }
    if (!vis.length) list.append(h("p", { class: "section-note", style: { padding: "16px 8px" } }, "No models match."));
    renderFooter();
  }

  function renderFooter() {
    footer.replaceChildren(
      h("span", { class: "grow" }, `${sel.size} selected · ${visible().length} shown`),
      h("button", { class: "btn small ghost", type: "button", onclick: () => { selectVisible(); renderList(); commit(); } }, "Select shown"),
      h("button", { class: "btn small ghost", type: "button", onclick: () => { sel = new Set(pool.map((m) => m.id)); renderList(); commit(); } }, "All"),
      h("button", { class: "btn small ghost", type: "button", onclick: () => { sel = new Set(); renderList(); commit(); } }, "None"),
      h("button", { class: "btn small", type: "button", onclick: close }, "Done"));
  }

  function renderAll() { renderFilters(); renderList(); commit(); }

  function close() {
    pop.remove();
    anchor.setAttribute("aria-expanded", "false");
    document.removeEventListener("mousedown", outside, true);
    document.removeEventListener("keydown", esc);
  }
  const outside = (e) => { if (!pop.contains(e.target) && !anchor.contains(e.target)) close(); };
  const esc = (e) => { if (e.key === "Escape") { close(); anchor.focus(); } };

  renderFilters(); renderList();
  document.body.append(pop);
  anchor.setAttribute("aria-expanded", "true");
  document.addEventListener("mousedown", outside, true);
  document.addEventListener("keydown", esc);
  search.focus();
}
