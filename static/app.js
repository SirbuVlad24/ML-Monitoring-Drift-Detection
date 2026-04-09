// Helper function to append to terminal
function logToTerminal(message, type="info") {
    const consoleBody = document.getElementById("consoleOutput");
    const p = document.createElement("p");
    
    // Add color based on log type
    if (type === "error") p.style.color = "#ef4444";
    if (type === "warning") p.style.color = "#f59e0b";
    
    const time = new Date().toLocaleTimeString('ro-RO');
    p.textContent = `[${time}] > ${message}`;
    consoleBody.appendChild(p);
    consoleBody.scrollTop = consoleBody.scrollHeight;
}

// Get feature inputs
function getFeatures() {
    const temp = parseFloat(document.getElementById("temp").value);
    const humidity = parseFloat(document.getElementById("humidity").value);
    const co2 = parseFloat(document.getElementById("co2").value);
    const light = parseFloat(document.getElementById("light").value);
    const ph = parseFloat(document.getElementById("ph").value);
    
    return [temp, humidity, co2, light, ph];
}

// Unified Prediction Function
async function predict(endpoint, resultElementId) {
    const resultCard = document.getElementById(resultElementId);
    resultCard.innerHTML = `<span class="pulse dot" style="margin-right: 8px;"></span> Analizare date în curs...`;
    
    try {
        const features = getFeatures();
        
        const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: features })
        });
        
        const data = await res.json();
        
        if (!res.ok) throw new Error(data.detail || "Eroare la server");
        
        // Format the output based on endpoint
        if (endpoint === "/predict") {
            const isAnomaly = data.is_anomaly;
            const emoji = isAnomaly ? "🚨" : "✅";
            const status = isAnomaly ? "ANOMALIE DETECTATĂ" : "Parametri normali";
            const color = isAnomaly ? "var(--danger)" : "var(--primary)";
            resultCard.innerHTML = `<b style="color: ${color}">${emoji} ${status}</b> (Scor: ${data.anomaly_score})`;
            logToTerminal(`Inference IsoForest: Scor ${data.anomaly_score} -> ${status}`, isAnomaly ? "error" : "info");
        } 
        else if (endpoint === "/predict/yield") {
            const yieldKg = data.predicted_yield_kg_per_m2;
            resultCard.innerHTML = `🍅 <b>Estimare Randament:</b> ${yieldKg} kg / m²`;
            logToTerminal(`Inference LinReg: Estimat ${yieldKg} kg/m²`);
        }
        else if (endpoint === "/predict/cluster") {
            const cluster = data.cluster_id;
            resultCard.innerHTML = `📊 <b>Profil Mediu:</b> Scenariul / Clusterul ${cluster}`;
            logToTerminal(`Inference KMeans: Asignare Cluster ${cluster}`);
        }
    } catch (e) {
        resultCard.innerHTML = `❌ Eroare: ${e.message}`;
        logToTerminal(`Eroare ${endpoint}: ${e.message}`, "error");
    }
}

// MLOps Drift Detection
async function detectDrift() {
    logToTerminal("Se rulează analiza de Drift (Evidently AI)...", "warning");
    
    try {
        const res = await fetch("/drift/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}) // Default paths are fine
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);
        
        const drifted = data.drifted;
        const percentage = (data.drift_share * 100).toFixed(1);
        
        if (drifted) {
             logToTerminal(`🚨 DATA DRIFT CONFIRMAT! ${percentage}% din features sunt afectate.`, "error");
        } else {
             logToTerminal(`✅ Datele sunt stabile. Doar ${percentage}% deviație vizibilă.`);
        }
        
    } catch(e) {
         logToTerminal(`Eroare Drift: ${e.message}`, "error");
    }
}

// MLOps Retrain
async function retrainModel() {
    logToTerminal("Se rulează CI/CD pipeline de re-antrenare. Așteaptă...", "warning");
    
    try {
        const res = await fetch("/drift/retrain", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({})
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);
        
        if (data.retrained) {
            logToTerminal(`🔧 Modelele au fost re-antrenate cu succes pe noile date!`, "info");
        } else {
            logToTerminal(`ℹ️ Nu este nevoie de reantrenare (Drift sub threshold).`, "info");
        }
        
    } catch(e) {
         logToTerminal(`Eroare Retrain: ${e.message}`, "error");
    }
}
