/* ── Console helper ── */
function logToTerminal(message, type = "info") {
    const el = document.getElementById("consoleOutput");
    const p = document.createElement("p");

    if (type === "error")   p.style.color = "#ef4444";
    if (type === "warning") p.style.color = "#f59e0b";

    const ts = new Date().toLocaleTimeString("ro-RO");
    p.textContent = `[${ts}] > ${message}`;
    el.appendChild(p);
    el.scrollTop = el.scrollHeight;
}

/* ── Read sensor inputs ── */
function getFeatures() {
    return [
        parseFloat(document.getElementById("temp").value),
        parseFloat(document.getElementById("humidity").value),
        parseFloat(document.getElementById("co2").value),
        parseFloat(document.getElementById("light").value),
        parseFloat(document.getElementById("ph").value),
    ];
}

/* ── Unified prediction call ── */
async function predict(endpoint, resultId) {
    const card = document.getElementById(resultId);
    card.textContent = "⏳ Se analizează…";

    try {
        const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: getFeatures() }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Eroare server");

        if (endpoint === "/predict") {
            const ok   = !data.is_anomaly;
            const icon = ok ? "✅" : "🚨";
            const txt  = ok ? "Normal" : "ANOMALIE";
            card.innerHTML = `<b style="color:${ok ? "var(--green)" : "var(--red)"}">${icon} ${txt}</b> &nbsp;·&nbsp; scor: ${data.anomaly_score}`;
            logToTerminal(`IsoForest → scor ${data.anomaly_score} → ${txt}`, ok ? "info" : "error");
        }
        else if (endpoint === "/predict/yield") {
            card.innerHTML = `🍅 <b>Randament estimat:</b> ${data.predicted_yield_kg_per_m2} kg / m²`;
            logToTerminal(`LinReg → ${data.predicted_yield_kg_per_m2} kg/m²`);
        }
        else if (endpoint === "/predict/cluster") {
            card.innerHTML = `📊 <b>Cluster:</b> Scenariul ${data.cluster_id}`;
            logToTerminal(`KMeans → Cluster ${data.cluster_id}`);
        }
        else if (endpoint === "/predict/crop") {
            card.innerHTML = `🌱 <b>Plantă prezisă:</b> ${data.predicted_crop_type}`;
            logToTerminal(`RandomForest → ${data.predicted_crop_type}`);
        }
    } catch (e) {
        card.innerHTML = `❌ ${e.message}`;
        logToTerminal(`Eroare ${endpoint}: ${e.message}`, "error");
    }
}

/* ── Drift detection ── */
async function detectDrift() {
    logToTerminal("Evidently AI – analiză drift în curs…", "warning");

    try {
        const res = await fetch("/drift/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);

        const pct = (data.drift_share * 100).toFixed(1);
        if (data.drifted) {
            logToTerminal(`🚨 DATA DRIFT CONFIRMAT! ${pct}% din features afectate.`, "error");
        } else {
            logToTerminal(`✅ Stabil – doar ${pct}% deviație.`);
        }
    } catch (e) {
        logToTerminal(`Eroare drift: ${e.message}`, "error");
    }
}

/* ── Retrain ── */
async function retrainModel() {
    logToTerminal("Pipeline CI/CD retrain pornit…", "warning");

    try {
        const res = await fetch("/drift/retrain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);

        if (data.retrained) {
            logToTerminal("🔧 Modelele au fost re-antrenate cu succes!");
        } else {
            logToTerminal("ℹ️ Drift sub threshold – re-antrenare neactivată.");
        }
    } catch (e) {
        logToTerminal(`Eroare retrain: ${e.message}`, "error");
    }
}

/* ── Load metrics on page open ── */
async function loadMetrics() {
    try {
        const res = await fetch("/metrics");
        if (!res.ok) return;

        const data  = await res.json();
        const panel = document.getElementById("metricsPanel");
        const box   = document.getElementById("metricsContent");

        const rf = data.classification;
        const lr = data.regression;

        box.innerHTML = `
            <div class="metric-card" style="background:${cssVar("purple-dim")}; border-color:rgba(139,92,246,0.25);">
                <h3 style="color:#a78bfa;">🌱 Random Forest (Clasificare Plante)</h3>
                <div class="stat">Acuratețe: <b>${rf.accuracy}%</b></div>
                <div class="stat">Precizie:  <b>${rf.precision}%</b></div>
                <div class="stat">Recall:    <b>${rf.recall}%</b></div>
                <div class="highlight" style="color:#c4b5fd;">🏆 F1 Score: ${rf.f1_score}%</div>
            </div>
            <div class="metric-card" style="background:${cssVar("blue-dim")}; border-color:rgba(59,130,246,0.25);">
                <h3 style="color:#60a5fa;">📈 Regresie Liniară (Estimare Recoltă)</h3>
                <div class="stat">RMSE: <b>${lr.rmse} kg/m²</b></div>
                <div class="highlight" style="color:#93c5fd;">R² Score: ${lr.r2}</div>
            </div>
        `;

        panel.style.display = "block";
        logToTerminal("Metricile modelelor încărcate via /metrics.");
    } catch {
        logToTerminal("Metrics indisponibile – antrenează modelele mai întâi.", "warning");
    }
}

function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim();
}

window.addEventListener("DOMContentLoaded", loadMetrics);
