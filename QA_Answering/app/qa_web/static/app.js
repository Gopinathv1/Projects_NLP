/* Ranked extractive QA — front end.
   Talks to POST /api/answer and renders the ranked list. Clicking a candidate
   shows the passage it came from with the span highlighted. */

const SAMPLES = JSON.parse(document.getElementById("samples-data").textContent);

const $ = (id) => document.getElementById(id);
const passagesEl = $("passages");
const resultsEl = $("results");
const statusEl = $("status");
const metaEl = $("meta");
const evidenceEl = $("evidence");

let lastPassages = [];

/* ---------------- passage inputs ---------------- */
function addPassage(text = "") {
  const wrap = document.createElement("div");
  wrap.className = "passage";

  const ta = document.createElement("textarea");
  ta.placeholder = "Paste a document, article or paragraph…";
  ta.value = text;

  const drop = document.createElement("button");
  drop.type = "button";
  drop.className = "drop";
  drop.title = "Remove this passage";
  drop.textContent = "×";
  drop.onclick = () => {
    wrap.remove();
    if (!passagesEl.children.length) addPassage();
    renumber();
  };

  const tag = document.createElement("span");
  tag.className = "tag";

  wrap.append(ta, drop, tag);
  passagesEl.append(wrap);
  renumber();
}

function renumber() {
  [...passagesEl.querySelectorAll(".passage")].forEach((p, i) => {
    p.querySelector(".tag").textContent = "passage " + i;
  });
}

function readPassages() {
  return [...passagesEl.querySelectorAll("textarea")]
    .map((t) => t.value.trim())
    .filter(Boolean);
}

/* ---------------- rendering ---------------- */
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function showEvidence(cand) {
  if (cand.is_no_answer || cand.passage_index === null || cand.passage_index === undefined) {
    evidenceEl.hidden = true;
    return;
  }
  const ctx = lastPassages[cand.passage_index] || "";
  const s = cand.start_char, e = cand.end_char;
  const body =
    s === null || e === null
      ? escapeHtml(ctx)
      : escapeHtml(ctx.slice(0, s)) + "<mark>" + escapeHtml(ctx.slice(s, e)) + "</mark>" +
        escapeHtml(ctx.slice(e));
  $("evidence-src").textContent = `passage ${cand.passage_index} · chars ${s}–${e}`;
  $("evidence-body").innerHTML = body;
  evidenceEl.hidden = false;
}

function render(data) {
  resultsEl.innerHTML = "";
  evidenceEl.hidden = true;

  const answers = data.answers || [];
  if (!answers.length) {
    resultsEl.innerHTML = '<p class="empty">No candidate answer was found in these passages.</p>';
    metaEl.textContent = "";
    return;
  }

  const top = answers[0].confidence || 1;

  answers.forEach((cand, i) => {
    const li = document.createElement("li");
    li.className = "answer" + (i === 0 ? " top" : "") + (cand.is_no_answer ? " none" : "");

    const pct = (100 * (cand.confidence / top)).toFixed(0);
    const prov = cand.is_no_answer
      ? "the model scores 'no answer' above every span"
      : `passage ${cand.passage_index}` +
        (cand.support > 1 ? ` · <span class="support">found in ${cand.support} passages</span>` : "");

    li.innerHTML = `
      <div class="answer-head">
        <span class="rank">${cand.rank}.</span>
        <span class="answer-text">${escapeHtml(cand.text)}</span>
        <span class="score">${cand.confidence.toFixed(3)}</span>
      </div>
      <div class="bar"><span style="width:${pct}%"></span></div>
      <div class="prov">${prov}</div>`;

    li.onclick = () => {
      [...resultsEl.children].forEach((c) => c.classList.remove("active"));
      li.classList.add("active");
      showEvidence(cand);
    };
    resultsEl.append(li);
  });

  resultsEl.firstChild.classList.add("active");
  showEvidence(answers[0]);

  const bits = [
    `engine: ${data.engine}`,
    `passages: ${data.n_passages}`,
    data.n_windows != null ? `windows scanned: ${data.n_windows}` : null,
    `top_k: ${data.top_k}`,
    data.latency_ms != null ? `latency: ${data.latency_ms} ms` : null,
  ].filter(Boolean);
  metaEl.textContent = bits.join("   ·   ");
}

/* ---------------- request ---------------- */
async function run() {
  const question = $("question").value.trim();
  const contexts = readPassages();

  if (!question || !contexts.length) {
    statusEl.textContent = "Enter a question and at least one passage.";
    statusEl.className = "status error";
    return;
  }

  lastPassages = contexts;
  $("run").disabled = true;
  statusEl.className = "status";
  statusEl.textContent = "Scoring candidate spans…";

  try {
    const res = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        contexts,
        top_k: parseInt($("topk").value, 10),
        engine: $("engine").value,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `request failed (${res.status})`);
    render(data);
    statusEl.textContent = "";
  } catch (err) {
    statusEl.className = "status error";
    statusEl.textContent = err.message;
    resultsEl.innerHTML = "";
    metaEl.textContent = "";
    evidenceEl.hidden = true;
  } finally {
    $("run").disabled = false;
  }
}

/* ---------------- wiring ---------------- */
$("add-passage").onclick = () => addPassage();
$("run").onclick = run;
$("topk").oninput = (e) => ($("topk-value").textContent = e.target.value);
$("question").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });

$("sample").onchange = (e) => {
  const s = SAMPLES[e.target.value];
  if (!s) return;
  $("question").value = s.question;
  passagesEl.innerHTML = "";
  s.contexts.forEach((c) => addPassage(c));
  e.target.value = "";
};

addPassage();
