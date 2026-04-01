const DATA_URL = "greenwashing_claim_history.json";
const VALIDATED_MAPPINGS_URL = "8B model/validated_mappings.jsonl";
const FLAGGED_MAPPINGS_URL = "8B model/flagged_mappings.jsonl";
const COLLAPSE_MAP_URL = "subclaim_bertopic_collapse.json";

let flattenedSnippets = null;
/** @type {string | null} */
let artifactBundleVersion = null;
/** @type {Map<string, { topicId: number, collapseFlag: boolean, collapseWith: string[], topicLabel?: string, hierarchyConfidence?: number }> | null} */
let collapseBySubclaim = null;
let dataLoadError = null;

async function loadClaimsData() {
  try {
    // Load subclaim definitions (ids + current_text + history).
    const res = await fetch(DATA_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();

    // BERTopic collapse flags (offline artifact; optional).
    collapseBySubclaim = new Map();
    try {
      const collapseRes = await fetch(COLLAPSE_MAP_URL);
      if (collapseRes.ok) {
        const collapseJson = await collapseRes.json();
        if (typeof collapseJson.claims_bundle_version === "string") {
          artifactBundleVersion = collapseJson.claims_bundle_version;
        }
        const sub = collapseJson.subclaims || {};
        for (const [sid, row] of Object.entries(sub)) {
          if (!row || typeof row !== "object") continue;
          const collapseWith = Array.isArray(row.collapse_with)
            ? row.collapse_with.map(String)
            : [];
          const hc = row.hierarchy_confidence;
          const entry = {
            topicId: Number(row.topic_id),
            collapseFlag: Boolean(row.collapse_flag),
            collapseWith: collapseWith,
            topicLabel:
              typeof row.topic_label === "string" ? row.topic_label : undefined,
          };
          if (typeof hc === "number" && Number.isFinite(hc)) {
            entry.hierarchyConfidence = hc;
          }
          collapseBySubclaim.set(sid, entry);
        }
      }
    } catch {
      // Missing or invalid collapse file: UI continues without flags.
    }

    const claims = json.claims || {};

    // Load validated and flagged subclaim→superclaim mappings (ids + superclaim text).
    const validatedRes = await fetch(VALIDATED_MAPPINGS_URL);
    if (!validatedRes.ok) throw new Error(`HTTP ${validatedRes.status}`);
    const validatedText = await validatedRes.text();

    const flaggedRes = await fetch(FLAGGED_MAPPINGS_URL);
    if (!flaggedRes.ok) throw new Error(`HTTP ${flaggedRes.status}`);
    const flaggedText = await flaggedRes.text();

    const superclaimBySubclaim = new Map();

    function ingestMappings(rawText, { preferIfMissing = false } = {}) {
      rawText
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)
      .forEach((line) => {
        try {
          const rec = JSON.parse(line);
          if (!rec.subclaim_id || !rec.superclaim_id) return;
          // Prefer mappings that have an explicit superclaim_text.
          const existing = superclaimBySubclaim.get(rec.subclaim_id);
          if (existing && existing.superclaimText && !preferIfMissing) return;
          superclaimBySubclaim.set(rec.subclaim_id, {
            superclaimId: rec.superclaim_id,
            superclaimText: rec.superclaim_text || "",
          });
        } catch {
          // Ignore malformed lines.
        }
      });
    }

    // Validated mappings win; flagged fill in any gaps.
    ingestMappings(validatedText);
    ingestMappings(flaggedText, { preferIfMissing: true });

    const snippets = [];

    // Each entry in `claims` represents a *subclaim* (e.g., "NC_46").
    // Its `history` contains all source snippets that have been mapped to that subclaim.
    for (const [claimId, claimObj] of Object.entries(claims)) {
      const subclaimId = claimId.startsWith("NC_")
        ? claimId
        : `NC_${claimId.replace(/^(NC_|SC_)/, "")}`;
      const subclaimText = claimObj.current_text || "";

      const mapping = superclaimBySubclaim.get(subclaimId);
      if (!mapping) continue; // If we don't know its superclaim, skip for this UI.

      const { superclaimId, superclaimText } = mapping;
      const history = Array.isArray(claimObj.history) ? claimObj.history : [];

      history.forEach((entry) => {
        if (!entry || !entry.source_snippet) return;

        snippets.push({
          sentenceSnippet: entry.source_snippet,
          snippetLower: entry.source_snippet.toLowerCase(),
          superclaimText,
          superclaimId,
          subclaimId,
          subclaimText,
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

  // Simple abbreviation handling so things like "U.S." are not split as sentences.
  const abbreviations = new Set([
    "U.S.",
    "U.K.",
    "U.N.",
    "Mr.",
    "Ms.",
    "Mrs.",
    "Dr.",
    "Prof.",
    "Inc.",
    "Ltd.",
    "Co.",
    "Jr.",
    "Sr.",
    "e.g.",
    "i.e.",
  ]);

  const sentences = [];
  let startIdx = 0;

  for (let i = 0; i < cleaned.length; i++) {
    const ch = cleaned[i];
    if (ch === "." || ch === "!" || ch === "?") {
      // Look backwards to see if this punctuation ends a known abbreviation.
      let wordStart = i;
      while (wordStart > 0 && cleaned[wordStart - 1] !== " ") {
        wordStart--;
      }
      const candidateWord = cleaned.slice(wordStart, i + 1);
      const isAbbrev = abbreviations.has(candidateWord);

      // Also require that this is likely an end-of-sentence: next char is space or end of string.
      const nextChar = cleaned[i + 1] || "";
      const likelyBoundary = !nextChar || nextChar === " ";

      if (!isAbbrev && likelyBoundary) {
        const sentence = cleaned.slice(startIdx, i + 1).trim();
        if (sentence) sentences.push(sentence);
        startIdx = i + 1;
      }
    }
  }

  const remainder = cleaned.slice(startIdx).trim();
  if (remainder) sentences.push(remainder);

  return sentences;
}

function computeMatchConfidence(sentenceLower, snippetLower) {
  // NOTE: Keeps *mapping selection* behavior (local overlap / LCS).
  if (!sentenceLower || !snippetLower) return 0;
  if (sentenceLower === snippetLower) return 1;
  if (snippetLower.includes(sentenceLower) || sentenceLower.includes(snippetLower)) return 1;

  const minLen = Math.min(sentenceLower.length, snippetLower.length);
  if (minLen < 12) return 0;

  const intersection = longestCommonSubstr(sentenceLower, snippetLower);
  return intersection.length / minLen;
}

function formatHierarchyLine(row) {
  if (
    !row ||
    typeof row.hierarchyConfidence !== "number" ||
    !Number.isFinite(row.hierarchyConfidence)
  ) {
    return { html: "", titlePart: "" };
  }
  const v = row.hierarchyConfidence;
  const html = `<div class="hierarchy-confidence">Offline semantic confidence (subclaim↔superclaim): <strong>${v.toFixed(2)}</strong></div>`;
  return {
    html,
    titlePart: `Offline embedding similarity: ${v.toFixed(2)}`,
  };
}

function formatCollapseMeta(subclaimId) {
  if (!collapseBySubclaim || !collapseBySubclaim.size) {
    return {
      html: `<span class="collapse-meta">No artifact loaded (run <code>python scripts/build_subclaim_collapse_bertopic.py</code>).</span>`,
      title: "Generate subclaim_bertopic_collapse.json",
    };
  }
  const row = collapseBySubclaim.get(subclaimId);
  if (!row) {
    return {
      html: `<span class="collapse-meta">No BERTopic row for this subclaim.</span>`,
      title: "",
    };
  }
  const hier = formatHierarchyLine(row);
  const peers = row.collapseWith || [];
  const label = row.topicLabel ? escapeHtml(String(row.topicLabel)) : "";
  if (!row.collapseFlag || peers.length === 0) {
    const tid = Number.isFinite(row.topicId) ? row.topicId : "";
    const labelBit = label
      ? `<div class="collapse-topic-label">${label}</div>`
      : "";
    const titleBits = [hier.titlePart, label ? `BERTopic: ${row.topicLabel}` : "", "Singleton or outlier — not flagged for merge"].filter(Boolean);
    return {
      html: `
        <div class="collapse-meta">
          ${hier.html}
          <span class="collapse-badge collapse-badge-quiet">Unique topic${tid !== "" ? ` (${tid})` : ""}</span>
          ${labelBit}
        </div>
      `,
      title: titleBits.join(" · "),
    };
  }
  const peerList = peers
    .slice(0, 8)
    .map((id) => `<span class="claim-id">${escapeHtml(id)}</span>`)
    .join(", ");
  const more =
    peers.length > 8 ? ` <span class="collapse-more">+${peers.length - 8} more</span>` : "";
  const labelBlock = label
    ? `<div class="collapse-topic-label">${label}</div>`
    : "";
  const titleBits = [
    hier.titlePart,
    `Same BERTopic cluster as: ${peers.join(", ")}`,
  ].filter(Boolean);
  return {
    html: `
      <div class="collapse-meta">
        ${hier.html}
        <span class="collapse-badge">May collapse with</span>
        ${labelBlock}
        <div class="collapse-peers">${peerList}${more}</div>
      </div>
    `,
    title: titleBits.join(" · "),
  };
}

function findBestMatchesForSentence(sentence) {
  if (!flattenedSnippets || flattenedSnippets.length === 0) return [];

  const lowerSentence = sentence.toLowerCase();

  const candidates = flattenedSnippets
    .filter((s) => {
      const confidence = computeMatchConfidence(lowerSentence, s.snippetLower);
      return confidence >= 0.6;
    })
    .map((s) => ({
      ...s,
      mappingConfidence: computeMatchConfidence(
        lowerSentence,
        s.snippetLower
      ),
    }));

  // For each sentence, ensure at most one match per subclaim.
  const bestBySubclaim = new Map();

  for (const c of candidates) {
    const key = c.subclaimId;
    const existing = bestBySubclaim.get(key);
    if (!existing || c.mappingConfidence > existing.mappingConfidence) {
      bestBySubclaim.set(key, c);
    }
  }

  const dedup = Array.from(bestBySubclaim.values()).sort(
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
        <div class="claim-text">${m.subclaimText}</div>
      `;

      const collapse = formatCollapseMeta(m.subclaimId);
      tdSuper.innerHTML = `
        <div class="claim-label">Superclaim <span class="claim-id">(${m.superclaimId})</span></div>
        <div class="claim-text">${m.superclaimText}</div>
        <div class="claim-meta collapse-block" title="${escapeHtmlAttr(collapse.title)}">
          ${collapse.html}
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
}

function escapeHtmlAttr(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
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
  const bundleBit =
    artifactBundleVersion != null
      ? ` · artifact ${artifactBundleVersion}`
      : "";
  statusEl.textContent = `Analyzed ${sentences.length} sentence${sentences.length === 1 ? "" : "s"}${bundleBit}.`;
  btn.disabled = false;
}

window.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("analyze-btn");
  btn.addEventListener("click", handleAnalyzeClick);
});

