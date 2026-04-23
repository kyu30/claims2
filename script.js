const CLAIM_HISTORY_URL = "greenwashing_claim_history.json";
const SUPERCLAIMS_URL = "greenwashing_superclaims.json";
const CODEBOOK_URL = "greenwashing_codebook.json";
const CLAIM_SUPERCLAIM_MAP_URL = "claim_superclaim_map.json";
const COLLAPSE_MAP_URL = "subclaim_bertopic_collapse.json";

let flattenedSnippets = null;
/** @type {string | null} */
let artifactBundleVersion = null;
/** @type {Map<string, { topicId: number, collapseFlag: boolean, collapseWith: string[], topicLabel?: string, hierarchyConfidence?: number }> | null} */
let collapseBySubclaim = null;
/** @type {Map<string, { superclaimId: string, superclaimText: string }> | null} subclaim → mapped superclaim (from loaded JSON) */
let superclaimMappingBySubclaim = null;
let dataLoadError = null;

function normalizeSubclaimId(raw) {
  const s = String(raw).trim();
  if (!s) return "";
  if (s.startsWith("NC_")) return s;
  return `NC_${s.replace(/^(NC_|SC_)/, "")}`;
}

function normalizeSuperclaimId(raw) {
  const s = String(raw).trim();
  if (!s) return "";
  if (s.startsWith("SC_")) return s;
  return `SC_${s.replace(/^(SC_|NC_)/, "")}`;
}

/**
 * @param {unknown} json
 * @param {"subclaim" | "superclaim"} kind
 * @returns {Map<string, string>}
 */
function loadIdTextMap(json, kind) {
  if (!json || typeof json !== "object" || Array.isArray(json)) {
    const label =
      kind === "subclaim" ? "greenwashing_codebook.json" : "greenwashing_superclaims.json";
    throw new Error(`${label} must be a JSON object of {id: text}`);
  }
  const out = new Map();
  for (const [k, v] of Object.entries(json)) {
    const text = String(v == null ? "" : v).trim();
    const id =
      kind === "subclaim"
        ? normalizeSubclaimId(k)
        : normalizeSuperclaimId(k);
    if (!id) continue;
    out.set(id, text);
  }
  return out;
}

/**
 * @param {unknown} obj
 * @returns {{ subclaimId: string, superclaimId: string, mapSuperText: string }[]}
 */
function parseClaimSuperclaimMap(obj) {
  const rows = [];
  if (obj == null) return rows;

  if (typeof obj === "object" && !Array.isArray(obj)) {
    const keys = Object.keys(obj);
    const first = keys[0];
    const sample = first != null ? obj[first] : undefined;
    const isCombined =
      sample != null &&
      typeof sample === "object" &&
      !Array.isArray(sample) &&
      (Object.prototype.hasOwnProperty.call(sample, "superclaim_id") ||
        Object.prototype.hasOwnProperty.call(sample, "superclaimId") ||
        Object.prototype.hasOwnProperty.call(sample, "sc_id"));

    if (isCombined) {
      for (const [subId, record] of Object.entries(obj)) {
        if (!record || typeof record !== "object" || Array.isArray(record)) continue;
        const sc =
          record.superclaim_id ??
          record.superclaimId ??
          record.sc_id ??
          record.SC;
        if (sc == null) continue;
        const mapSuperText = String(
          record.superclaim_text ??
            record.superclaimText ??
            record.superclaim ??
            ""
        ).trim();
        rows.push({
          subclaimId: normalizeSubclaimId(subId),
          superclaimId: normalizeSuperclaimId(sc),
          mapSuperText,
        });
      }
      return rows;
    }

    for (const [nc, sc] of Object.entries(obj)) {
      rows.push({
        subclaimId: normalizeSubclaimId(nc),
        superclaimId: normalizeSuperclaimId(sc),
        mapSuperText: "",
      });
    }
    return rows;
  }

  if (Array.isArray(obj)) {
    for (const item of obj) {
      if (Array.isArray(item) && item.length >= 2) {
        rows.push({
          subclaimId: normalizeSubclaimId(item[0]),
          superclaimId: normalizeSuperclaimId(item[1]),
          mapSuperText: "",
        });
        continue;
      }
      if (item && typeof item === "object") {
        const nc =
          item.subclaim_id ??
          item.nc_id ??
          item.subclaim ??
          item.NC;
        const sc =
          item.superclaim_id ??
          item.sc_id ??
          item.superclaim ??
          item.SC;
        if (nc == null || sc == null) continue;
        const mapSuperText = String(
          item.superclaim_text ?? item.superclaimText ?? ""
        ).trim();
        rows.push({
          subclaimId: normalizeSubclaimId(nc),
          superclaimId: normalizeSuperclaimId(sc),
          mapSuperText,
        });
      }
    }
  }

  return rows;
}

async function loadClaimsData() {
  try {
    const [historyRes, superRes, codebookRes, mapRes] = await Promise.all([
      fetch(CLAIM_HISTORY_URL),
      fetch(SUPERCLAIMS_URL),
      fetch(CODEBOOK_URL),
      fetch(CLAIM_SUPERCLAIM_MAP_URL),
    ]);

    if (!historyRes.ok) throw new Error(`claim history HTTP ${historyRes.status}`);
    if (!superRes.ok) throw new Error(`superclaims HTTP ${superRes.status}`);
    if (!codebookRes.ok) throw new Error(`codebook HTTP ${codebookRes.status}`);
    if (!mapRes.ok) throw new Error(`claim_superclaim_map HTTP ${mapRes.status}`);

    const [historyJson, superJson, codebookJson, mapJson] = await Promise.all([
      historyRes.json(),
      superRes.json(),
      codebookRes.json(),
      mapRes.json(),
    ]);

    // BERTopic collapse (offline artifact; optional).
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
            collapseWith,
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
      // Missing or invalid collapse file: UI continues without collapse hints.
    }

    const superclaimsById = loadIdTextMap(superJson, "superclaim");
    const codebookById = loadIdTextMap(codebookJson, "subclaim");
    const mapRows = parseClaimSuperclaimMap(mapJson);

    const superclaimBySubclaim = new Map();
    for (const { subclaimId, superclaimId, mapSuperText } of mapRows) {
      if (!subclaimId || !superclaimId) continue;
      superclaimBySubclaim.set(subclaimId, {
        superclaimId,
        superclaimText:
          superclaimsById.get(superclaimId) || mapSuperText || "",
      });
    }

    superclaimMappingBySubclaim = superclaimBySubclaim;

    const claims = historyJson.claims || {};
    const snippets = [];

    for (const [claimId, claimObj] of Object.entries(claims)) {
      if (!claimObj || typeof claimObj !== "object") continue;
      const subclaimId = claimId.startsWith("NC_")
        ? claimId
        : `NC_${claimId.replace(/^(NC_|SC_)/, "")}`;
      const subclaimText =
        codebookById.get(subclaimId) ||
        claimObj.current_text ||
        "";

      const mapping = superclaimBySubclaim.get(subclaimId);
      if (!mapping) continue;

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
    console.error("Failed to load claim data:", err);
    dataLoadError = err;
  }
}

/** Paragraphs = blocks separated by line breaks (after normalizing newlines). */
function splitIntoParagraphs(text) {
  const normalized = text.replace(/\r\n/g, "\n").trim();
  if (!normalized) return [];
  return normalized
    // Treat each line as its own paragraph; ignore empty lines.
    .split(/\n+/)
    .map((p) => p.replace(/\s+/g, " ").trim())
    .filter(Boolean);
}

function computeMatchConfidence(paragraphLower, snippetLower) {
  // NOTE: Keeps *mapping selection* behavior (local overlap / LCS).
  if (!paragraphLower || !snippetLower) return 0;
  if (paragraphLower === snippetLower) return 1;
  if (snippetLower.includes(paragraphLower) || paragraphLower.includes(snippetLower)) return 1;

  const minLen = Math.min(paragraphLower.length, snippetLower.length);
  if (minLen < 12) return 0;

  const intersection = longestCommonSubstr(paragraphLower, snippetLower);
  return intersection.length / minLen;
}

/**
 * Unique superclaims mapped from peer subclaims in the same BERTopic cluster.
 * @param {string[]} peerSubclaimIds
 * @param {string} [currentSuperclaimId] matched superclaim for this row (for labels)
 */
function superclaimTargetsFromPeers(peerSubclaimIds, currentSuperclaimId) {
  const map = superclaimMappingBySubclaim;
  if (!map || !peerSubclaimIds.length) return [];

  const bySc = new Map();
  for (const raw of peerSubclaimIds) {
    const sid = normalizeSubclaimId(raw);
    if (!sid) continue;
    const m = map.get(sid);
    if (!m || !m.superclaimId) continue;
    if (!bySc.has(m.superclaimId)) {
      bySc.set(m.superclaimId, {
        superclaimId: m.superclaimId,
        superclaimText: m.superclaimText || "",
      });
    }
  }

  const list = Array.from(bySc.values()).map((entry) => ({
    ...entry,
    isCurrentMapping:
      Boolean(currentSuperclaimId) && entry.superclaimId === currentSuperclaimId,
  }));

  list.sort((a, b) => {
    if (a.isCurrentMapping !== b.isCurrentMapping) return a.isCurrentMapping ? 1 : -1;
    return a.superclaimId.localeCompare(b.superclaimId);
  });

  return list;
}

function formatSuperclaimTargetsHtml(targets, { mappedPeerCount, artifactPeerCount }) {
  if (!targets.length) {
    return `<div class="collapse-peer-superclaims collapse-peer-superclaims--empty">No superclaim targets could be resolved from cluster peers that still appear in your map.</div>`;
  }
  const maxShow = 8;
  const slice = targets.slice(0, maxShow);
  const more =
    targets.length > maxShow
      ? `<div class="collapse-more">+${targets.length - maxShow} more superclaim(s)</div>`
      : "";
  const staleNote =
    artifactPeerCount > mappedPeerCount
      ? `<p class="collapse-peer-superclaims-stale">${artifactPeerCount - mappedPeerCount} offline cluster peer(s) are not in your current <code>claim_superclaim_map</code> and were ignored.</p>`
      : "";
  const items = slice
    .map((t) => {
      const idHtml = `<span class="claim-id">${escapeHtml(t.superclaimId)}</span>`;
      const badge = t.isCurrentMapping
        ? ` <span class="collapse-mapping-badge">same as match</span>`
        : "";
      const textShort = t.superclaimText
        ? escapeHtml(
            t.superclaimText.length > 160
              ? `${t.superclaimText.slice(0, 157)}…`
              : t.superclaimText
          )
        : "";
      return `<li class="collapse-sc-target-item">
        <div class="collapse-sc-target-line">${idHtml}${badge}</div>
        ${textShort ? `<div class="collapse-sc-target-text">${textShort}</div>` : ""}
      </li>`;
    })
    .join("");
  return `
    <div class="collapse-peer-superclaims">
      <div class="collapse-peer-superclaims-title">Potential superclaims to collapse toward</div>
      <p class="collapse-peer-superclaims-hint">Using <strong>${mappedPeerCount}</strong> cluster peer subclaim${mappedPeerCount === 1 ? "" : "s"} that still exist in your map${artifactPeerCount !== mappedPeerCount ? ` (of ${artifactPeerCount} in the offline artifact)` : ""}. Targets below are their mapped superclaims.</p>
      ${staleNote}
      <ul class="collapse-sc-target-list">${items}</ul>
      ${more}
    </div>
  `;
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
  const html = `<div class="hierarchy-confidence">Offline cosine similarity (subclaim↔superclaim): <strong>${v.toFixed(2)}</strong></div>`;
  return {
    html,
    titlePart: `Embedding cosine similarity: ${v.toFixed(2)}`,
  };
}

/**
 * @param {string} subclaimId
 * @param {string} [currentSuperclaimId] superclaim id for the matched row (for target labeling)
 */
function formatCollapseMeta(subclaimId, currentSuperclaimId) {
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
  // UI display rule: only show collapse *targets* when the offline hierarchy cosine
  // similarity is above a minimum threshold. (We still show the score itself.)
  const MIN_COLLAPSE_SIMILARITY = 0.6;
  const hier = formatHierarchyLine(row);
  const peersArtifact = (row.collapseWith || []).map(String);
  const peersMapped = superclaimMappingBySubclaim
    ? peersArtifact.filter((p) =>
        superclaimMappingBySubclaim.has(normalizeSubclaimId(p))
      )
    : [...peersArtifact];

  const label = row.topicLabel ? escapeHtml(String(row.topicLabel)) : "";
  if (!row.collapseFlag || peersArtifact.length === 0) {
    const tid = Number.isFinite(row.topicId) ? row.topicId : "";
    const labelBit = label
      ? `<div class="collapse-topic-label">${label}</div>`
      : "";
    const titleBits = [
      hier.titlePart,
      label ? `Topic label: ${row.topicLabel}` : "",
      "Singleton or outlier — not flagged for merge",
    ].filter(Boolean);
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

  if (peersMapped.length === 0) {
    const labelBlock = label
      ? `<div class="collapse-topic-label">${label}</div>`
      : "";
    const titleBits = [
      hier.titlePart,
      label ? `Topic label: ${row.topicLabel}` : "",
      "Cluster peers missing from current map",
    ].filter(Boolean);
    return {
      html: `
        <div class="collapse-meta">
          ${hier.html}
          <span class="collapse-badge">Topic cluster</span>
          ${labelBlock}
          <div class="collapse-stale-cluster-msg">
            The offline artifact lists ${peersArtifact.length} cluster peer subclaim${peersArtifact.length === 1 ? "" : "s"}, but none appear in your current <code>claim_superclaim_map</code>. Regenerate <code>subclaim_bertopic_collapse.json</code> or refresh the map so collapse targets stay in sync.
          </div>
        </div>
      `,
      title: titleBits.join(" · "),
    };
  }

  if (
    typeof row.hierarchyConfidence === "number" &&
    Number.isFinite(row.hierarchyConfidence) &&
    row.hierarchyConfidence < MIN_COLLAPSE_SIMILARITY
  ) {
    const labelBlock = label
      ? `<div class="collapse-topic-label">${label}</div>`
      : "";
    const titleBits = [
      hier.titlePart,
      label ? `Topic label: ${row.topicLabel}` : "",
      `Collapse suggestions hidden (similarity < ${MIN_COLLAPSE_SIMILARITY.toFixed(2)})`,
    ].filter(Boolean);
    return {
      html: `
        <div class="collapse-meta">
          ${hier.html}
          <span class="collapse-badge">Topic cluster</span>
          ${labelBlock}
          <div class="collapse-stale-cluster-msg">
            Cluster peers exist, but collapse targets are hidden because the offline cosine similarity is below <strong>${MIN_COLLAPSE_SIMILARITY.toFixed(
              2
            )}</strong>.
          </div>
        </div>
      `,
      title: titleBits.join(" · "),
    };
  }

  const scTargets = superclaimTargetsFromPeers(peersMapped, currentSuperclaimId);
  const targetsHtml = formatSuperclaimTargetsHtml(scTargets, {
    mappedPeerCount: peersMapped.length,
    artifactPeerCount: peersArtifact.length,
  });
  const labelBlock = label
    ? `<div class="collapse-topic-label">${label}</div>`
    : "";
  const titleBits = [
    hier.titlePart,
    label ? `Topic label: ${row.topicLabel}` : "",
    scTargets.length
      ? `Superclaims: ${scTargets.map((t) => t.superclaimId).join(", ")}`
      : "",
  ].filter(Boolean);
  return {
    html: `
      <div class="collapse-meta">
        ${hier.html}
        <span class="collapse-badge">Topic cluster</span>
        ${labelBlock}
        ${targetsHtml}
      </div>
    `,
    title: titleBits.join(" · "),
  };
}

function findBestMatchesForParagraph(paragraph) {
  if (!flattenedSnippets || flattenedSnippets.length === 0) return [];

  const lowerParagraph = paragraph.toLowerCase();

  const candidates = flattenedSnippets
    .filter((s) => {
      const confidence = computeMatchConfidence(lowerParagraph, s.snippetLower);
      return confidence >= 0.6;
    })
    .map((s) => ({
      ...s,
      mappingConfidence: computeMatchConfidence(
        lowerParagraph,
        s.snippetLower
      ),
    }));

  // For each paragraph, ensure at most one match per subclaim.
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

function renderResults(paragraphsWithMatches) {
  const container = document.getElementById("results-container");
  container.innerHTML = "";

  if (!paragraphsWithMatches.length) {
    const p = document.createElement("p");
    p.className = "placeholder";
    p.textContent = "No paragraphs found. Paste article text (paragraphs separated by blank lines) and click “Analyze paragraphs”.";
    container.appendChild(p);
    return;
  }

  const ledger = document.createElement("div");
  ledger.className = "results-ledger";
  const total = paragraphsWithMatches.length;

  paragraphsWithMatches.forEach(({ paragraph, matches, proposals: paragraphProposals = [] }, idx) => {
    const card = document.createElement("article");
    card.className = "paragraph-result-card";
    card.setAttribute("aria-labelledby", `paragraph-result-title-${idx}`);

    const header = document.createElement("header");
    header.className = "paragraph-result-header";
    const badge = document.createElement("span");
    badge.className = "paragraph-result-badge";
    badge.textContent = String(idx + 1);
    const headerText = document.createElement("div");
    headerText.className = "paragraph-result-header-text";
    const titleEl = document.createElement("div");
    titleEl.className = "paragraph-result-title";
    titleEl.id = `paragraph-result-title-${idx}`;
    titleEl.textContent = `Paragraph ${idx + 1} of ${total}`;
    const subEl = document.createElement("div");
    subEl.className = "paragraph-result-sub";
    subEl.textContent = "Maps below apply only to this paragraph.";
    headerText.appendChild(titleEl);
    headerText.appendChild(subEl);
    header.appendChild(badge);
    header.appendChild(headerText);
    card.appendChild(header);

    const table = document.createElement("table");
    table.className = "results-table results-table--in-card";

    const thead = document.createElement("thead");
    thead.innerHTML = `
      <tr>
        <th scope="col">Paragraph text</th>
        <th scope="col">Subclaim(s)</th>
        <th scope="col">Superclaim(s)</th>
      </tr>
    `;

    const tbody = document.createElement("tbody");

    if (!matches.length) {
      const tr = document.createElement("tr");

      const tdSentence = document.createElement("td");
      tdSentence.className = "sentence-cell paragraph-cell";
      tdSentence.textContent = paragraph;

      const tdSub = document.createElement("td");
      const tdSuper = document.createElement("td");

      tdSub.innerHTML = `<div class="no-match"><strong>No subclaim mapping found</strong></div>`;
      tdSuper.innerHTML = `<div class="no-match"><strong>No superclaim mapping found</strong></div>`;

      tr.appendChild(tdSentence);
      tr.appendChild(tdSub);
      tr.appendChild(tdSuper);
      tbody.appendChild(tr);
    } else {
      const rowSpan = matches.length;

      matches.forEach((m, mIdx) => {
        const tr = document.createElement("tr");

        if (mIdx === 0) {
          const tdSentence = document.createElement("td");
          tdSentence.className = "sentence-cell paragraph-cell";
          tdSentence.textContent = paragraph;
          tdSentence.rowSpan = rowSpan;
          tr.appendChild(tdSentence);
        }

        const tdSub = document.createElement("td");
        const tdSuper = document.createElement("td");

        tdSub.innerHTML = `
        <div class="claim-label">Subclaim <span class="claim-id">(${m.subclaimId})</span></div>
        <div class="claim-text">${m.subclaimText}</div>
      `;

        const collapse = formatCollapseMeta(m.subclaimId, m.superclaimId);
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
    }

    table.appendChild(thead);
    table.appendChild(tbody);
    card.appendChild(table);

    ledger.appendChild(card);
  });

  container.appendChild(ledger);
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

const REVIEWER_STORAGE_KEY = "CLAIMS_REVIEWER_NAME";

function getReviewerName() {
  const el = document.getElementById("reviewer-name");
  const fromInput = el && el.value != null ? String(el.value).trim() : "";
  if (fromInput) return fromInput;
  try {
    const fromStorage = localStorage.getItem(REVIEWER_STORAGE_KEY);
    return fromStorage ? String(fromStorage).trim() : "";
  } catch {
    return "";
  }
}

function persistReviewerName(name) {
  const el = document.getElementById("reviewer-name");
  if (el) el.value = name;
  try {
    localStorage.setItem(REVIEWER_STORAGE_KEY, name);
  } catch {
    // ignore
  }
}

function requireReviewerName() {
  const name = getReviewerName();
  if (!name) {
    alert("Please enter your reviewer name before approving, rejecting, or applying proposals.");
    const el = document.getElementById("reviewer-name");
    if (el) el.focus();
    return null;
  }
  persistReviewerName(name);
  return name;
}

function formatProposalMetaHtml(p) {
  const bits = [];
  if (p.reviewedBy) {
    bits.push(
      `<div class="proposal-line"><strong>Reviewed by:</strong> ${escapeHtml(
        p.reviewedBy
      )}</div>`
    );
  }
  if (p.appliedBy) {
    bits.push(
      `<div class="proposal-line"><strong>Applied by:</strong> ${escapeHtml(
        p.appliedBy
      )}</div>`
    );
  }
  return bits.length ? `<div class="proposal-meta">${bits.join("")}</div>` : "";
}

function getApiCandidates() {
  const out = [];
  /** @param {string|null|undefined} v */
  const pushUnique = (v) => {
    if (v === "") {
      if (!out.includes("")) out.push("");
      return;
    }
    if (v == null) return;
    const s = String(v).trim().replace(/\/+$/, "");
    if (!s) return;
    if (!out.includes(s)) out.push(s);
  };

  const host =
    typeof window !== "undefined" && window.location && window.location.hostname
      ? window.location.hostname
      : "";
  const isDevHost = host === "localhost" || host === "127.0.0.1";
  const isLocalApiUrl = (base) =>
    /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/?$/i.test(String(base || "").trim());

  const meta = document.querySelector('meta[name="claims-api-base"]');
  const fromMeta = meta && meta.getAttribute("content");

  // 1) Same-origin `/api/*` (Vercel rewrites, or a dev proxy) — always try first in the browser.
  if (typeof window !== "undefined") {
    pushUnique("");
  }

  // 2) Local uvicorn before any hardcoded remote meta/storage so localhost dev is not blocked by a dead URL.
  if (isDevHost) {
    pushUnique("http://localhost:8001");
  }

  try {
    const fromStorage = localStorage.getItem("CLAIMS_API_BASE");
    if (fromStorage != null && String(fromStorage).trim() !== "") {
      if (!isLocalApiUrl(fromStorage) || isDevHost) {
        pushUnique(fromStorage);
      }
    }
  } catch {
    // ignore
  }

  if (fromMeta != null && String(fromMeta).trim() !== "") {
    if (!isLocalApiUrl(fromMeta) || isDevHost) {
      pushUnique(fromMeta);
    }
  }

  return out;
}

function resolveApiUrl(base, path) {
  if (!path) return base || "";
  if (/^https?:\/\//i.test(path)) return path;
  const b = base || "";
  if (!b) return path;
  return `${b}${path.startsWith("/") ? "" : "/"}${path}`;
}

/** Same-origin calls must send cookies so Vercel Deployment Protection can authorize `/api/*`. Cross-origin keeps `omit` so `Access-Control-Allow-Origin: *` stays valid. */
function apiFetchCredentials(target) {
  try {
    if (typeof window === "undefined" || !window.location) return "omit";
    const resolved = new URL(target, window.location.href);
    if (resolved.origin === window.location.origin) return "same-origin";
  } catch {
    // ignore
  }
  return "omit";
}

/** Prefer a short hint when the server returned HTML (e.g. Vercel auth) instead of JSON. */
function shortenApiFailureMessage(res, text) {
  const raw = text == null ? "" : String(text);
  const ct = (res && res.headers && res.headers.get("content-type")) || "";
  if (
    raw.includes("Authentication Required") ||
    (ct.includes("text/html") && raw.includes("vercel") && raw.length > 400)
  ) {
    return (
      "Vercel Deployment Protection blocked this API request (HTML login page instead of JSON). " +
      "Options: turn off protection for this environment, use a production deployment without protection, " +
      "or stay signed in—same-origin requests now send cookies so protected previews can work after you open the site once."
    );
  }
  if (ct.includes("text/html") && raw.length > 400) {
    return `Unexpected HTML from API (HTTP ${res.status}). Check the URL and deployment protection settings.`;
  }
  return raw.length > 900 ? `${raw.slice(0, 900)}…` : raw;
}

async function postJson(url, body) {
  const bases = getApiCandidates();
  let lastErr = null;
  for (const base of bases) {
    try {
      const target = resolveApiUrl(base, url);
      const creds = apiFetchCredentials(target);
      const res = await fetch(target, {
        method: "POST",
        credentials: creds,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const text = await res.text();
      let data = null;
      try {
        data = text ? JSON.parse(text) : null;
      } catch {
        // ignore
      }
      if (!res.ok) {
        const msg =
          (data && (data.detail || data.message)) ||
          shortenApiFailureMessage(res, text) ||
          `HTTP ${res.status}`;
        // IMPORTANT: never fall back to a different backend on a valid HTTP error response.
        // Doing so can mix proposal IDs between deployments (list from one backend, apply on another).
        throw new Error(msg);
      }
      return data;
    } catch (e) {
      const target = resolveApiUrl(base, url);
      const msg = e && e.message ? String(e.message) : String(e);
      lastErr = new Error(
        msg === "Failed to fetch"
          ? `Failed to fetch (${target}). Confirm the API is deployed (open /api/health in a tab), CORS allows this origin, and meta claims-api-base / localStorage CLAIMS_API_BASE are correct or cleared.`
          : `${msg} (${target})`
      );
      // If we got a real application error (not a network failure), stop here.
      // "Proposal not found" is a good example: falling back will only make it worse.
      if (msg !== "Failed to fetch") break;
    }
  }
  throw lastErr || new Error("Request failed.");
}

async function getJson(url) {
  const bases = getApiCandidates();
  let lastErr = null;
  for (const base of bases) {
    try {
      const target = resolveApiUrl(base, url);
      const creds = apiFetchCredentials(target);
      const res = await fetch(target, { method: "GET", credentials: creds });
      const text = await res.text();
      let errData = null;
      if (!res.ok) {
        try {
          errData = text ? JSON.parse(text) : null;
        } catch {
          // ignore
        }
        const msg =
          (errData && (errData.detail || errData.message)) ||
          shortenApiFailureMessage(res, text) ||
          `HTTP ${res.status}`;
        throw new Error(msg);
      }
      if (text != null && String(text).trim()) {
        try {
          return JSON.parse(text);
        } catch (parseErr) {
          throw new Error(
            `Invalid JSON (HTTP ${res.status}): ${parseErr && parseErr.message ? parseErr.message : String(parseErr)}`
          );
        }
      }
      return null;
    } catch (e) {
      const target = resolveApiUrl(base, url);
      const msg = e && e.message ? String(e.message) : String(e);
      lastErr = new Error(
        msg === "Failed to fetch"
          ? `Failed to fetch (${target}). Confirm the API is deployed (open /api/health in a tab), CORS allows this origin, and meta claims-api-base / localStorage CLAIMS_API_BASE are correct or cleared.`
          : `${msg} (${target})`
      );
      if (msg !== "Failed to fetch") break;
    }
  }
  throw lastErr || new Error("Request failed.");
}

function formatNewSuperclaimSectionsHtml(p) {
  const article = escapeHtml(p.paragraph || "");
  const payload = p.payload || {};
  const scText = escapeHtml(String(payload.superclaimText || "").trim());
  return `
    <div class="proposal-section">
      <div class="proposal-section-heading">Article</div>
      <div class="proposal-section-body">${article}</div>
    </div>
    <div class="proposal-section">
      <div class="proposal-section-heading">Suggested original superclaim</div>
      <div class="proposal-section-body proposal-section-body--superclaim">${scText}</div>
    </div>
  `;
}

function formatProposalTitle(p) {
  const type = p.type || "";
  if (type === "new_subclaim") return "New subclaim";
  if (type === "new_superclaim") return "Suggested original superclaim";
  if (type === "link_subclaim_to_superclaim") return "Remap subclaim → superclaim";
  if (type === "merge_subclaims") return "Merge subclaims";
  if (type === "merge_superclaims") return "Merge superclaims";
  return type || "Proposal";
}

function formatProposalBodyHtml(p) {
  const payload = p.payload || {};
  if (p.type === "merge_subclaims") {
    const ids = Array.isArray(payload.mergeSubclaimIds)
      ? payload.mergeSubclaimIds.map((x) => String(x)).join(", ")
      : "";
    const canonText = String(payload.canonicalSubclaimText || "").trim();
    const removeText = String(payload.removeSubclaimText || "").trim();
    const mergeTexts =
      canonText || removeText
        ? `<div class="proposal-merge-claims">
            <div class="proposal-merge-claim"><span class="proposal-merge-label">Keep (canonical)</span><div class="proposal-merge-body">${escapeHtml(
              canonText || "(no text in payload)"
            )}</div><code class="proposal-merge-id">${escapeHtml(
              payload.canonicalSubclaimId || ""
            )}</code></div>
            <div class="proposal-merge-claim"><span class="proposal-merge-label">Merge away</span><div class="proposal-merge-body">${escapeHtml(
              removeText || "(no text in payload)"
            )}</div><code class="proposal-merge-id">${escapeHtml(
              payload.removeSubclaimId || ""
            )}</code></div>
          </div>`
        : "";
    const idLines =
      mergeTexts === ""
        ? `<div class="proposal-line"><strong>Canonical:</strong> <code>${escapeHtml(
            payload.canonicalSubclaimId || ""
          )}</code></div>
          <div class="proposal-line"><strong>Remove:</strong> <code>${escapeHtml(
            payload.removeSubclaimId || ""
          )}</code></div>`
        : "";
    const cos =
      typeof payload.pairCosine === "number"
        ? `<div class="proposal-line"><strong>Pair cosine (TF‑IDF):</strong> ${payload.pairCosine.toFixed(
            3
          )}</div>`
        : "";
    return `
      ${mergeTexts}
      ${idLines}
      <div class="proposal-line"><strong>Group:</strong> <code>${escapeHtml(ids)}</code></div>
      <div class="proposal-line"><strong>Shared superclaim:</strong> <code>${escapeHtml(
        payload.sharedSuperclaimId || ""
      )}</code></div>
      ${cos}
      ${p.rationale ? `<div class="proposal-line proposal-reason">${escapeHtml(p.rationale)}</div>` : ""}
    `;
  }
  if (p.type === "merge_superclaims") {
    const ids = Array.isArray(payload.mergeSuperclaimIds)
      ? payload.mergeSuperclaimIds.map((x) => String(x)).join(", ")
      : "";
    const canonText = String(payload.canonicalSuperclaimText || "").trim();
    const removeText = String(payload.removeSuperclaimText || "").trim();
    const mergeTexts =
      canonText || removeText
        ? `<div class="proposal-merge-claims">
            <div class="proposal-merge-claim"><span class="proposal-merge-label">Keep (canonical)</span><div class="proposal-merge-body">${escapeHtml(
              canonText || "(no text in payload)"
            )}</div><code class="proposal-merge-id">${escapeHtml(
              payload.canonicalSuperclaimId || ""
            )}</code></div>
            <div class="proposal-merge-claim"><span class="proposal-merge-label">Merge away</span><div class="proposal-merge-body">${escapeHtml(
              removeText || "(no text in payload)"
            )}</div><code class="proposal-merge-id">${escapeHtml(
              payload.removeSuperclaimId || ""
            )}</code></div>
          </div>`
        : "";
    const idLines =
      mergeTexts === ""
        ? `<div class="proposal-line"><strong>Canonical:</strong> <code>${escapeHtml(
            payload.canonicalSuperclaimId || ""
          )}</code></div>
          <div class="proposal-line"><strong>Remove:</strong> <code>${escapeHtml(
            payload.removeSuperclaimId || ""
          )}</code></div>`
        : "";
    const cos =
      typeof payload.pairCosine === "number"
        ? `<div class="proposal-line"><strong>Pair cosine (TF‑IDF):</strong> ${payload.pairCosine.toFixed(
            3
          )}</div>`
        : "";
    return `
      ${mergeTexts}
      ${idLines}
      <div class="proposal-line"><strong>Group:</strong> <code>${escapeHtml(ids)}</code></div>
      ${cos}
      ${p.rationale ? `<div class="proposal-line proposal-reason">${escapeHtml(p.rationale)}</div>` : ""}
    `;
  }
  if (p.type === "new_subclaim") {
    const sc = payload.suggestedSuperclaimId
      ? `<div class="proposal-line"><strong>Suggested superclaim:</strong> <code>${escapeHtml(
          payload.suggestedSuperclaimId
        )}</code> ${payload.suggestedSuperclaimText ? `— ${escapeHtml(payload.suggestedSuperclaimText)}` : ""}</div>`
      : "";
    const conf =
      typeof payload.confidence === "number"
        ? `<div class="proposal-line"><strong>LLM confidence:</strong> ${(payload.confidence * 100).toFixed(0)}%</div>`
        : "";
    return `
      ${sc}
      ${conf}
      ${p.rationale ? `<div class="proposal-line proposal-reason">${escapeHtml(p.rationale)}</div>` : ""}
    `;
  }
  if (p.type === "new_superclaim") {
    const conf =
      typeof payload.confidence === "number"
        ? `<div class="proposal-line"><strong>LLM confidence (best candidate):</strong> ${(payload.confidence * 100).toFixed(0)}%</div>`
        : "";
    const near =
      payload.fromLowConfidenceMapping && payload.nearbySuperclaimId
        ? `<div class="proposal-line"><strong>Closest taxonomy superclaim:</strong> <code>${escapeHtml(
            String(payload.nearbySuperclaimId)
          )}</code>${payload.nearbySuperclaimText ? ` — ${escapeHtml(String(payload.nearbySuperclaimText))}` : ""}</div>`
        : "";
    const reason = p.rationale
      ? `<div class="proposal-line proposal-reason"><strong>Reasoning:</strong> ${escapeHtml(p.rationale)}</div>`
      : "";
    return `
      ${conf}
      ${near}
      ${reason}
    `;
  }
  return `
    <div class="proposal-line"><strong>Payload:</strong> <code>${escapeHtml(
      JSON.stringify(payload)
    )}</code></div>
    ${p.rationale ? `<div class="proposal-line proposal-reason">${escapeHtml(p.rationale)}</div>` : ""}
  `;
}

async function refreshPendingProposals() {
  const container = document.getElementById("proposals-container");
  if (!container) return;

  container.innerHTML = `<p class="placeholder">Loading pending proposals…</p>`;
  let proposals = [];
  try {
    const raw = await getJson("/api/proposals?status=pending");
    if (!Array.isArray(raw)) {
      throw new Error(
        raw == null
          ? "Empty response from /api/proposals (expected a JSON array)."
          : `Invalid response from /api/proposals: expected an array, got ${typeof raw}.`
      );
    }
    proposals = raw;
  } catch (e) {
    container.innerHTML = `<p class="placeholder error-text">Unable to load proposals: ${escapeHtml(
      e.message || String(e)
    )}</p>`;
    return;
  }

  if (proposals.length === 0) {
    container.innerHTML = `<p class="placeholder">No pending proposals yet.</p>
      <p class="placeholder proposal-empty-hint">This list only shows proposals with status <strong>pending</strong> (approved/rejected/applied disappear here).</p>
      <p class="placeholder proposal-empty-hint">Proposals are written to the database only when <strong>Analyze paragraphs</strong> uses the <strong>backend</strong> (<code>/api/analyze</code>). If the status line says the backend failed and switched to a local heuristic, nothing is saved—open DevTools → Console / Network to see the <code>/api/analyze</code> error (often missing <code>OPENAI_API_KEY</code> or a 4xx/5xx).</p>
      <p class="placeholder proposal-empty-hint">If analyze completed via the backend but this stays empty, open <a href="/api/health" target="_blank" rel="noopener"><code>/api/health</code></a> (<code>postgresOk</code> with <code>DATABASE_URL</code>) and <a href="/api/proposals" target="_blank" rel="noopener"><code>/api/proposals</code></a> to see whether any rows exist at all.</p>`;
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "results-ledger";
  const byId = new Map();

  proposals.forEach((p) => {
    if (p && p.id) byId.set(String(p.id), p);
    const card = document.createElement("article");
    card.className = "paragraph-result-card pending-proposal-card";
    const isNewSuperclaim = p.type === "new_superclaim";
    const topSections = isNewSuperclaim
      ? formatNewSuperclaimSectionsHtml(p)
      : `<div class="proposal-paragraph">${escapeHtml(p.paragraph || "")}</div>`;
    const badgeText = formatProposalTitle(p).slice(0, 1).toUpperCase();

    card.innerHTML = `
      <header class="paragraph-result-header pending-proposal-header">
        <div class="paragraph-result-badge pending-proposal-badge">${escapeHtml(badgeText)}</div>
        <div class="paragraph-result-header-text">
          <div class="paragraph-result-title">${escapeHtml(formatProposalTitle(p))}</div>
          <div class="paragraph-result-sub"><code>${escapeHtml(p.id || "")}</code></div>
        </div>
      </header>
      <div class="pending-proposal-top">${topSections}</div>
      <div class="pending-proposal-body">${formatProposalBodyHtml(p)}${formatProposalMetaHtml(p)}</div>
      <div class="pending-proposal-actions">
        <button class="action-btn" data-action="approve" data-id="${escapeHtmlAttr(p.id || "")}">Approve</button>
        <button class="action-btn action-btn--danger" data-action="reject" data-id="${escapeHtmlAttr(p.id || "")}">Reject</button>
        <button class="action-btn action-btn--accent" data-action="apply" data-id="${escapeHtmlAttr(p.id || "")}">Apply</button>
      </div>
    `;
    wrap.appendChild(card);
  });

  container.innerHTML = "";
  container.appendChild(wrap);

  container.querySelectorAll("button[data-action]").forEach((btn) => {
    btn.addEventListener("click", async (ev) => {
      const el = ev.currentTarget;
      const action = el.getAttribute("data-action");
      const id = el.getAttribute("data-id");
      if (!id || !action) return;

      const card = el.closest("article");
      const cardButtons = card ? Array.from(card.querySelectorAll("button[data-action]")) : [el];
      cardButtons.forEach((b) => (b.disabled = true));
      try {
        const reviewer = requireReviewerName();
        if (!reviewer) {
          cardButtons.forEach((b) => (b.disabled = false));
          return;
        }
        // Apply implies approval: approve first, then apply.
        if (action === "apply") {
          await postJson(`/api/proposals/${encodeURIComponent(id)}/approve`, {
            reviewer_name: reviewer,
          });
        }
        await postJson(`/api/proposals/${encodeURIComponent(id)}/${action}`, {
          reviewer_name: reviewer,
        });
        // Applying should also refresh claims data next run; for now just refresh the list.
        await refreshPendingProposals();
      } catch (e) {
        cardButtons.forEach((b) => (b.disabled = false));
        alert(`Proposal ${action} failed: ${e.message || String(e)}`);
      }
    });
  });
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
  statusEl.textContent = "Analyzing…";
  statusEl.classList.remove("error-text");

  // BERTopic collapse + snippet index live in static JSON; load them even when the backend
  // handles /api/analyze, otherwise collapseBySubclaim stays null and the UI shows "No artifact loaded".
  if (!flattenedSnippets && !dataLoadError) {
    await loadClaimsData();
  }

  // Prefer the backend LLM API (supports proposals + human approval). Fallback to local heuristic if unavailable.
  try {
    const api = await postJson("/api/analyze", { text });
    const rows = Array.isArray(api?.paragraphs) ? api.paragraphs : [];
    const withMatches = rows.map((r) => ({
      paragraph: r.paragraph,
      matches: Array.isArray(r.matches) ? r.matches : [],
      proposals: Array.isArray(r.proposals) ? r.proposals : [],
    }));

    renderResults(withMatches);
    statusEl.textContent = `Analyzed ${withMatches.length} paragraph${withMatches.length === 1 ? "" : "s"} · bundle ${escapeHtml(
      api?.bundleVersion || ""
    )}.`;
    await refreshPendingProposals();
    btn.disabled = false;
    return;
  } catch (e) {
    console.warn("Backend /api/analyze failed; falling back to local matching.", e);
  }

  // Local fallback (does not POST proposals to the server—only backend /api/analyze does)
  statusEl.textContent = "Backend unavailable; running local heuristic…";
  if (!flattenedSnippets && !dataLoadError) {
    await loadClaimsData();
  }
  if (dataLoadError) {
    statusEl.textContent =
      "Unable to load claim JSON files. Serve the folder over HTTP and check that the four data files are present.";
    statusEl.classList.add("error-text");
    btn.disabled = false;
    return;
  }

  const paragraphs = splitIntoParagraphs(text);
  const withMatches = paragraphs.map((p) => ({
    paragraph: p,
    matches: findBestMatchesForParagraph(p),
  }));
  renderResults(withMatches);
  const bundleBit =
    artifactBundleVersion != null ? ` · artifact ${artifactBundleVersion}` : "";
  statusEl.textContent = `Analyzed ${paragraphs.length} paragraph${paragraphs.length === 1 ? "" : "s"}${bundleBit}. Local matching only—proposals are not saved (fix /api/analyze, e.g. OPENAI_API_KEY, to persist to Postgres).`;
  statusEl.classList.add("error-text");
  btn.disabled = false;
  void refreshPendingProposals();
}

window.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("analyze-btn");
  btn.addEventListener("click", handleAnalyzeClick);

  const reviewerEl = document.getElementById("reviewer-name");
  if (reviewerEl) {
    try {
      const saved = localStorage.getItem(REVIEWER_STORAGE_KEY);
      if (saved) reviewerEl.value = saved;
    } catch {
      // ignore
    }
    reviewerEl.addEventListener("change", () => {
      const v = String(reviewerEl.value || "").trim();
      if (v) persistReviewerName(v);
    });
  }

  void loadClaimsData();
  refreshPendingProposals();
});
