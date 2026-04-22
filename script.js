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

    if (Array.isArray(paragraphProposals) && paragraphProposals.length > 0) {
      const propBlock = document.createElement("div");
      propBlock.className = "paragraph-proposals";
      propBlock.innerHTML = `<h3 class="paragraph-proposals-title">Taxonomy proposals (this paragraph)</h3>
        <p class="paragraph-proposals-hint">Shown from the analyze response. The <strong>Pending taxonomy proposals</strong> list below updates when the server has stored them (e.g. Supabase or a writable data directory).</p>`;
      paragraphProposals.forEach((p) => {
        if (!p || !p.type) return;
        const sub = document.createElement("article");
        sub.className = "proposal-card proposal-card--embedded";
        sub.innerHTML = `
          <header class="proposal-header">
            <div class="proposal-title">${escapeHtml(formatProposalTitle(p))}</div>
            <div class="proposal-id"><code>${escapeHtml(p.id || "")}</code></div>
          </header>
          <div class="proposal-body">${formatProposalBodyHtml(p)}${formatProposalMetaHtml(p)}</div>
        `;
        propBlock.appendChild(sub);
      });
      card.appendChild(propBlock);
    }

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

function getSupabaseConfig() {
  const metaUrl = document.querySelector('meta[name="supabase-url"]');
  const metaKey = document.querySelector('meta[name="supabase-anon-key"]');
  const url = (metaUrl && metaUrl.getAttribute("content")) || "";
  const key = (metaKey && metaKey.getAttribute("content")) || "";
  return { url: String(url || "").trim(), key: String(key || "").trim() };
}

function getSupabaseClient() {
  const cfg = getSupabaseConfig();
  if (!cfg.url || !cfg.key) return null;
  if (typeof window === "undefined") return null;
  if (!window.supabase || typeof window.supabase.createClient !== "function") return null;
  try {
    return window.supabase.createClient(cfg.url, cfg.key);
  } catch {
    return null;
  }
}

function nextIdFromRows(rows, prefix) {
  let max = 0;
  for (const r of rows || []) {
    const id = String(r?.id || "");
    if (!id.startsWith(prefix)) continue;
    const n = parseInt(id.slice(prefix.length), 10);
    if (!Number.isNaN(n)) max = Math.max(max, n);
  }
  return `${prefix}${max + 1}`;
}

async function applyProposalToSupabaseTaxonomy(p) {
  const sb = getSupabaseClient();
  if (!sb) throw new Error("Supabase taxonomy config missing (set meta supabase-url and supabase-anon-key).");

  const type = String(p?.type || "");
  const payload = p?.payload || {};

  if (type === "new_superclaim") {
    const text = String(payload.superclaimText || "").trim();
    if (!text) throw new Error("Missing superclaimText");

    const { data: existing, error: selErr } = await sb
      .from("taxonomy_superclaims")
      .select("id")
      .order("id", { ascending: false });
    if (selErr) throw new Error(selErr.message || String(selErr));

    const newId = nextIdFromRows(existing || [], "SC_");
    const { error: insErr } = await sb.from("taxonomy_superclaims").insert({ id: newId, text });
    if (insErr) throw new Error(insErr.message || String(insErr));
    return;
  }

  if (type === "merge_subclaims") {
    const canonical = String(payload.canonicalSubclaimId || "").trim();
    const remove = String(payload.removeSubclaimId || "").trim();
    if (!canonical || !remove || canonical === remove) throw new Error("Missing canonical/remove subclaim ids");

    const { data: rows, error } = await sb
      .from("taxonomy_subclaims")
      .select("id,superclaim_id")
      .in("id", [canonical, remove]);
    if (error) throw new Error(error.message || String(error));
    const canonRow = (rows || []).find((r) => String(r.id) === canonical);
    const remRow = (rows || []).find((r) => String(r.id) === remove);
    if (!canonRow || !remRow) throw new Error("Unknown subclaim id(s) in taxonomy_subclaims");
    if (String(canonRow.superclaim_id) !== String(remRow.superclaim_id)) {
      throw new Error("Refusing merge: subclaims are not mapped to the same superclaim.");
    }

    const { error: delErr } = await sb.from("taxonomy_subclaims").delete().eq("id", remove);
    if (delErr) throw new Error(delErr.message || String(delErr));
    return;
  }

  if (type === "merge_superclaims") {
    const canonical = String(payload.canonicalSuperclaimId || "").trim();
    const remove = String(payload.removeSuperclaimId || "").trim();
    if (!canonical || !remove || canonical === remove) throw new Error("Missing canonical/remove superclaim ids");

    const { data: rows, error } = await sb
      .from("taxonomy_superclaims")
      .select("id")
      .in("id", [canonical, remove]);
    if (error) throw new Error(error.message || String(error));
    if (!Array.isArray(rows) || rows.length !== 2) throw new Error("Unknown superclaim id(s) in taxonomy_superclaims");

    const { error: updErr } = await sb
      .from("taxonomy_subclaims")
      .update({ superclaim_id: canonical })
      .eq("superclaim_id", remove);
    if (updErr) throw new Error(updErr.message || String(updErr));

    const { error: delErr } = await sb.from("taxonomy_superclaims").delete().eq("id", remove);
    if (delErr) throw new Error(delErr.message || String(delErr));
    return;
  }

  throw new Error(`Unsupported proposal type for browser taxonomy apply: ${type}`);
}

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

async function postJson(url, body) {
  const bases = getApiCandidates();
  let lastErr = null;
  for (const base of bases) {
    try {
      const target = resolveApiUrl(base, url);
      const res = await fetch(target, {
        method: "POST",
        credentials: "omit",
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
          text ||
          `HTTP ${res.status}`;
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
      const res = await fetch(target, { method: "GET", credentials: "omit" });
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
          text ||
          `HTTP ${res.status}`;
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
    proposals = await getJson("/api/proposals?status=pending");
  } catch (e) {
    container.innerHTML = `<p class="placeholder error-text">Unable to load proposals: ${escapeHtml(
      e.message || String(e)
    )}</p>`;
    return;
  }

  if (!Array.isArray(proposals) || proposals.length === 0) {
    container.innerHTML = `<p class="placeholder">No pending proposals yet.</p>`;
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "proposal-list";
  const byId = new Map();

  proposals.forEach((p) => {
    if (p && p.id) byId.set(String(p.id), p);
    const card = document.createElement("article");
    card.className = "proposal-card";
    const isNewSuperclaim = p.type === "new_superclaim";
    const topSections = isNewSuperclaim
      ? formatNewSuperclaimSectionsHtml(p)
      : `<div class="proposal-paragraph">${escapeHtml(p.paragraph || "")}</div>`;

    card.innerHTML = `
      <header class="proposal-header">
        <div class="proposal-title">${escapeHtml(formatProposalTitle(p))}</div>
        <div class="proposal-id"><code>${escapeHtml(p.id || "")}</code></div>
      </header>
      ${topSections}
      <div class="proposal-body">${formatProposalBodyHtml(p)}${formatProposalMetaHtml(p)}</div>
      <div class="proposal-actions">
        <button class="action-btn" data-action="approve" data-id="${escapeHtmlAttr(
          p.id || ""
        )}">Approve</button>
        <button class="action-btn action-btn--danger" data-action="reject" data-id="${escapeHtmlAttr(
          p.id || ""
        )}">Reject</button>
        <button class="action-btn action-btn--accent" data-action="apply" data-id="${escapeHtmlAttr(
          p.id || ""
        )}">Apply</button>
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

      el.disabled = true;
      try {
        const reviewer = requireReviewerName();
        if (!reviewer) {
          el.disabled = false;
          return;
        }
        if (action === "apply" && getSupabaseClient()) {
          const p = byId.get(String(id));
          if (!p) throw new Error("Missing proposal payload in UI.");
          await applyProposalToSupabaseTaxonomy(p);
          await postJson(`/api/proposals/${encodeURIComponent(id)}/${action}`, {
            reviewer_name: reviewer,
            skip_taxonomy_update: true,
          });
        } else {
          await postJson(`/api/proposals/${encodeURIComponent(id)}/${action}`, {
            reviewer_name: reviewer,
          });
        }
        // Applying should also refresh claims data next run; for now just refresh the list.
        await refreshPendingProposals();
      } catch (e) {
        el.disabled = false;
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

  // Local fallback
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
  statusEl.textContent = `Analyzed ${paragraphs.length} paragraph${paragraphs.length === 1 ? "" : "s"}${bundleBit}.`;
  btn.disabled = false;
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

  refreshPendingProposals();
});
