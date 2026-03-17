const DATA_URL = "greenwashing_claim_history.json";
// Use localhost for local dev; in production, call the deployed backend on Vercel.
const BACKEND_BASE_URL =
  window.BACKEND_BASE_URL ||
  (location.hostname === "localhost"
    ? "http://localhost:8001"
    : "https://claims-backend-sigma.vercel.app/confidence");

let flattenedSnippets = null;
let dataLoadError = null;
const confidenceCache = new Map();

async function loadClaimsData() {
  try {
    const res = await fetch(DATA_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();

    const claims = json.claims || {};
    const snippets = [];

    for (const [claimId, claimObj] of Object.entries(claims)) {
      const superText = claimObj.current_text || "";
      const history = Array.isArray(claimObj.history) ? claimObj.history : [];
      const superclaimId = claimId.startsWith("SC_")
        ? claimId
        : `SC_${claimId.replace(/^(NC_|SC_)/, "")}`;

      history.forEach((entry, idx) => {
        if (!entry || !entry.source_snippet) return;
        const rawSubId = entry.source_article_id ?? idx;
        const rawSubIdText = String(rawSubId);
        const subclaimId = rawSubIdText.startsWith("NC_")
          ? rawSubIdText
          : `NC_${rawSubIdText.replace(/^(NC_|SC_)/, "")}`;

        snippets.push({
          sentenceSnippet: entry.source_snippet,
          snippetLower: entry.source_snippet.toLowerCase(),
          superclaimText: superText,
          superclaimId,
          subclaimId,
        });
      });
    }

    flattenedSnippets = snippets;
  } catch (err) {
    console.error("Failed to load claim history:", err);
    dataLoadError = err;
  }
}

function splitIntoSentences(text) {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (!cleaned) return [];

  const regex = /[^.!?]+[.!?]+/g;
  const matches = cleaned.match(regex) || [];

  const remainder = cleaned.slice(matches.join("").length).trim();
  if (remainder) matches.push(remainder);

  return matches.map((s) => s.trim()).filter(Boolean);
}

function computeMatchConfidence(sentenceLower, snippetLower) {
  // NOTE: This keeps the *mapping selection* behavior as-is.
  // The displayed confidence score is computed live by the backend LLM call.
  if (!sentenceLower || !snippetLower) return 0;
  if (sentenceLower === snippetLower) return 1;
  if (snippetLower.includes(sentenceLower) || sentenceLower.includes(snippetLower)) return 1;

  const minLen = Math.min(sentenceLower.length, snippetLower.length);
  if (minLen < 12) return 0;

  const intersection = longestCommonSubstr(sentenceLower, snippetLower);
  return intersection.length / minLen;
}

function formatConfidence(score) {
  if (!Number.isFinite(score)) return "N/A";
  return score.toFixed(2);
}

async function fetchLlMConfidence({ subclaimId, sentenceSnippet, superclaimId, superclaimText }) {
  const key = `${subclaimId}|${superclaimId}|${sentenceSnippet}|${superclaimText}`;
  if (confidenceCache.has(key)) return confidenceCache.get(key);

  const p = (async () => {
    const res = await fetch(`${BACKEND_BASE_URL}/confidence`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        subclaim_id: subclaimId,
        superclaim_id: superclaimId,
        subclaim_text: sentenceSnippet,
        superclaim_text: superclaimText,
      }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Backend error ${res.status}: ${text}`);
    }

    const json = await res.json();
    const confidence = Number(json.confidence);
    return {
      verdict: json.verdict,
      confidence: Number.isFinite(confidence) ? confidence : 0,
      reason: typeof json.reason === "string" ? json.reason : "",
    };
  })();

  confidenceCache.set(key, p);
  return p;
}

function findBestMatchesForSentence(sentence) {
  if (!flattenedSnippets || flattenedSnippets.length === 0) return [];

  const lowerSentence = sentence.toLowerCase();

  const candidates = flattenedSnippets.filter((s) => {
    const confidence = computeMatchConfidence(lowerSentence, s.snippetLower);
    return confidence >= 0.6;
  }).map((s) => ({
    ...s,
    mappingConfidence: computeMatchConfidence(lowerSentence, s.snippetLower),
  }));

  const dedupByKey = new Map();

  for (const c of candidates) {
    const key = `${c.superclaimId}|${c.subclaimId}`;
    const existing = dedupByKey.get(key);
    if (!existing || c.mappingConfidence > existing.mappingConfidence) {
      dedupByKey.set(key, c);
    }
  }

  const dedup = Array.from(dedupByKey.values()).sort(
    (a, b) => b.mappingConfidence - a.mappingConfidence
  );

  return dedup.slice(0, 4);
}

function longestCommonSubstr(a, b) {
  const dp = Array(a.length + 1)
    .fill(null)
    .map(() => Array(b.length + 1).fill(0));
  let longest = 0;
  let endIdx = 0;

  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1;
        if (dp[i][j] > longest) {
          longest = dp[i][j];
          endIdx = i;
        }
      }
    }
  }

  return a.slice(endIdx - longest, endIdx);
}

function renderResults(sentencesWithMatches) {
  const container = document.getElementById("results-container");
  container.innerHTML = "";

  if (!sentencesWithMatches.length) {
    const p = document.createElement("p");
    p.className = "placeholder";
    p.textContent = "No sentences found. Paste article text and click “Analyze sentences”.";
    container.appendChild(p);
    return;
  }

  const table = document.createElement("table");
  table.className = "results-table";

  const thead = document.createElement("thead");
  thead.innerHTML = `
    <tr>
      <th>Sentence</th>
      <th>Subclaim(s)</th>
      <th>Superclaim(s)</th>
    </tr>
  `;

  const tbody = document.createElement("tbody");

  sentencesWithMatches.forEach(({ sentence, matches }) => {
    if (!matches.length) {
      const tr = document.createElement("tr");

      const tdSentence = document.createElement("td");
      tdSentence.className = "sentence-cell";
      tdSentence.textContent = sentence;

      const tdSub = document.createElement("td");
      const tdSuper = document.createElement("td");

      tdSub.innerHTML = `<div class="no-match"><strong>No subclaim mapping found</strong></div>`;
      tdSuper.innerHTML = `<div class="no-match"><strong>No superclaim mapping found</strong></div>`;

      tr.appendChild(tdSentence);
      tr.appendChild(tdSub);
      tr.appendChild(tdSuper);
      tbody.appendChild(tr);
      return;
    }

    const rowSpan = matches.length;

    matches.forEach((m, idx) => {
      const tr = document.createElement("tr");

      if (idx === 0) {
        const tdSentence = document.createElement("td");
        tdSentence.className = "sentence-cell";
        tdSentence.textContent = sentence;
        tdSentence.rowSpan = rowSpan;
        tr.appendChild(tdSentence);
      }

      const tdSub = document.createElement("td");
      const tdSuper = document.createElement("td");

      tdSub.innerHTML = `
        <div class="claim-label">Subclaim <span class="claim-id">(${m.subclaimId})</span></div>
        <div class="claim-text">${m.sentenceSnippet}</div>
      `;

      tdSuper.innerHTML = `
        <div class="claim-label">Superclaim <span class="claim-id">(${m.superclaimId})</span></div>
        <div class="claim-text">${m.superclaimText}</div>
        <div class="claim-meta">
          Confidence score:
          <span
            class="confidence-score"
            data-subclaim-id="${m.subclaimId}"
            data-superclaim-id="${m.superclaimId}"
            data-subclaim-text="${escapeHtmlAttr(m.sentenceSnippet)}"
            data-superclaim-text="${escapeHtmlAttr(m.superclaimText)}"
          >
            …
          </span>
        </div>
      `;

      tr.appendChild(tdSub);
      tr.appendChild(tdSuper);
      tbody.appendChild(tr);
    });
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  container.appendChild(table);

  // Populate confidence scores live via backend LLM calls.
  hydrateConfidenceScores(container).catch((e) => {
    console.error("Failed to hydrate confidence scores:", e);
  });
}

function escapeHtmlAttr(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function hydrateConfidenceScores(container) {
  const nodes = Array.from(container.querySelectorAll(".confidence-score"));
  await Promise.all(nodes.map(async (node) => {
    const subclaimId = node.dataset.subclaimId;
    const superclaimId = node.dataset.superclaimId;
    const subclaimText = node.dataset.subclaimText;
    const superclaimText = node.dataset.superclaimText;

    try {
      const result = await fetchLlMConfidence({
        subclaimId,
        sentenceSnippet: subclaimText,
        superclaimId,
        superclaimText,
      });
      node.textContent = formatConfidence(result.confidence);
      const verdict = result.verdict ? String(result.verdict) : "unknown";
      const reason = result.reason ? String(result.reason) : "No reason provided";
      node.title = `LLM-scored confidence (gpt-5-mini)\nverdict: ${verdict}\nreason: ${reason}`;
    } catch (e) {
      node.textContent = formatConfidence(0);
      node.title = String(e && e.message ? e.message : e);
    }
  }));
}

async function handleAnalyzeClick() {
  const btn = document.getElementById("analyze-btn");
  const statusEl = document.getElementById("status");
  const text = document.getElementById("article-input").value;

  statusEl.textContent = "";

  if (!text.trim()) {
    statusEl.textContent = "Please paste an article first.";
    statusEl.classList.remove("error-text");
    return;
  }

  btn.disabled = true;
  statusEl.textContent = "Loading claim history and analyzing…";
  statusEl.classList.remove("error-text");

  if (!flattenedSnippets && !dataLoadError) {
    await loadClaimsData();
  }

  if (dataLoadError) {
    statusEl.textContent = "Unable to load claim history JSON. Check that it is served from the same folder.";
    statusEl.classList.add("error-text");
    btn.disabled = false;
    return;
  }

  const sentences = splitIntoSentences(text);
  const withMatches = sentences.map((s) => ({
    sentence: s,
    matches: findBestMatchesForSentence(s),
  }));

  renderResults(withMatches);
  statusEl.textContent = `Analyzed ${sentences.length} sentence${sentences.length === 1 ? "" : "s"}.`;
  btn.disabled = false;
}

window.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("analyze-btn");
  btn.addEventListener("click", handleAnalyzeClick);
});

