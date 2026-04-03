/* WasteScan Pro — Neural Inference Engine (Adapted Version) */

const LABELS = [
    "Cardboard", "Food Organics", "Glass", "Metal",
    "Miscellaneous Trash", "Paper", "Plastic",
    "Textile Trash", "Vegetation"
];

const CONFIG = {
    modelPath: "model.onnx",
    imageSize: 224,
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225]
};

let session = null;

document.addEventListener('DOMContentLoaded', () => {

    // Target current IDs from index.html
    const dropZone       = document.getElementById('drop-zone');
    const fileInput      = document.getElementById('file-input');
    const resultsPanel   = document.getElementById('result-view');     // Mapping results-panel -> result-view
    const imagePreview   = document.getElementById('result-img');      // Mapping image-preview -> result-img
    const predictionText = document.getElementById('prediction-text');
    const confidenceText = document.getElementById('confidence-text');
    const confidenceFill = document.getElementById('confidence-fill');
    const modelStatus    = document.getElementById('engine-status');    // Mapping model-status -> engine-status
    const progressFill   = document.getElementById('progress-fill');
    const resetBtn       = document.getElementById('reset-btn');
    const placeholder    = document.getElementById('placeholder');      // For toggling view

    // Safety wrapper to avoid "null" property errors
    function safeSetText(el, text) {
        if (el) el.textContent = text;
    }

    async function init() {
        try {
            safeSetText(modelStatus, "LOADING ENGINE...");
            if (progressFill) progressFill.style.width = "30%";

            session = await ort.InferenceSession.create(CONFIG.modelPath);

            // Explicitly set to 100% and solidified state
            if (progressFill) {
                progressFill.style.width = "100%";
                progressFill.style.backgroundColor = "var(--accent, #4ade80)"; 
            }
            
            safeSetText(modelStatus, "🧠 AI ENGINE ONLINE");
            
            // Status dot styling
            const dot = document.getElementById('status-dot');
            if (dot) dot.className = 'status-dot online';

            setTimeout(() => {
                const si = document.querySelector('.status-dot');
                if (si) si.style.opacity = "0.8";
            }, 3000);

            console.log("ONNX Model Loaded. imageSize:", CONFIG.imageSize);
            await warmupSession();

        } catch (e) {
            console.error("Model loading failed:", e);
            safeSetText(modelStatus, "ENGINE ERROR");
            if (modelStatus) modelStatus.style.color = "#ff4d4d";
        }
    }

    async function warmupSession() {
        try {
            const size = CONFIG.imageSize;
            const dummy = new ort.Tensor(
                'float32',
                new Float32Array(1 * 3 * size * size),
                [1, 3, size, size]
            );
            await session.run({ [session.inputNames[0]]: dummy });
            console.log("Warmup complete.");
        } catch (e) {
            console.warn("Warmup failed (non-critical):", e);
        }
    }

    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput && fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "var(--accent, #4ade80)";
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.style.borderColor = "rgba(255,255,255,0.1)";
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = "rgba(255,255,255,0.1)";
            if (e.dataTransfer.files.length > 0) handleImage(e.dataTransfer.files[0]);
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) handleImage(e.target.files[0]);
        });
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            if (resultsPanel) resultsPanel.classList.add('hidden-el');
            if (placeholder) placeholder.classList.remove('hidden-el');
            if (fileInput) fileInput.value = "";
            safeSetText(predictionText, "—");
            if (confidenceFill) confidenceFill.style.width = "0%";
            safeSetText(confidenceText, "Match Strength: 0%");
        });
    }

    async function handleImage(file) {
        if (!session) { alert("Please wait for the engine to initialize."); return; }
        if (!file.type.startsWith('image/')) return;

        const reader = new FileReader();
        reader.onload = async (e) => {
            const dataURL = e.target.result;
            if (imagePreview) imagePreview.src = dataURL;
            
            // Toggle placeholder and result view
            if (placeholder) placeholder.classList.add('hidden-el');
            if (resultsPanel) {
                resultsPanel.classList.remove('hidden-el');
                resultsPanel.style.opacity = "0.7"; // Visual feedback for processing
            }

            const img = new Image();
            img.src = dataURL;
            img.onload = async () => {
                await runInference(img);
                if (resultsPanel) resultsPanel.style.opacity = "1";
            };
        };
        reader.readAsDataURL(file);
    }

    function preprocessImage(img) {
        const size = CONFIG.imageSize;

        // scale so shortest side = 224 (preserve aspect ratio)
        const scale = size / Math.min(img.width, img.height);
        const dw = Math.round(img.width * scale);
        const dh = Math.round(img.height * scale);

        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = dw;
        tempCanvas.height = dh;
        tempCanvas.getContext('2d').drawImage(img, 0, 0, dw, dh);

        const cx = Math.floor((dw - size) / 2);
        const cy = Math.floor((dh - size) / 2);

        const cropCanvas = document.createElement('canvas');
        cropCanvas.width = size;
        cropCanvas.height = size;
        cropCanvas.getContext('2d').drawImage(
            tempCanvas,
            cx, cy, size, size,
            0, 0, size, size
        );

        const { data } = cropCanvas.getContext('2d').getImageData(0, 0, size, size);
        const pixels = size * size;
        const input = new Float32Array(3 * pixels);

        for (let i = 0; i < pixels; i++) {
            input[i] = (data[i * 4] / 255 - CONFIG.mean[0]) / CONFIG.std[0]; // R
            input[i + pixels] = (data[i * 4 + 1] / 255 - CONFIG.mean[1]) / CONFIG.std[1]; // G
            input[i + pixels * 2] = (data[i * 4 + 2] / 255 - CONFIG.mean[2]) / CONFIG.std[2]; // B
        }

        return input;
    }

    async function runInference(img) {
        const t0 = performance.now();

        try {
            const inputData = preprocessImage(img);
            const size = CONFIG.imageSize;
            const tensor = new ort.Tensor('float32', inputData, [1, 3, size, size]);
            const feeds = { [session.inputNames[0]]: tensor };
            const output = await session.run(feeds);
            const logits = Array.from(output[session.outputNames[0]].data);

            const maxLogit = Math.max(...logits);
            const expScores = logits.map(l => Math.exp(l - maxLogit));
            const sumExp = expScores.reduce((a, b) => a + b, 0);
            const probs = expScores.map(s => s / sumExp);

            const results = probs
                .map((prob, i) => ({ label: LABELS[i], confidence: prob }))
                .sort((a, b) => b.confidence - a.confidence);

            const best = results[0];
            const ms = (performance.now() - t0).toFixed(0);

            console.log("Inference:", ms + "ms |", best.label, (best.confidence * 100).toFixed(1) + "%");

            safeSetText(predictionText, best.label);
            
            // Animation for text
            if (predictionText) {
                predictionText.style.animation = 'none';
                void predictionText.offsetWidth;
                predictionText.style.animation = 'fadeInUp 0.6s ease-out';
            }

            // Sync confidence UI
            setTimeout(() => {
                const pct = Math.max(10, best.confidence * 100);
                safeSetText(confidenceText, `Neural Match Strength: ${pct.toFixed(1)}%`);
                if (confidenceFill) {
                    confidenceFill.style.width = `${pct}%`;
                    confidenceFill.style.opacity = "1";
                }
                const latencyEl = document.getElementById('latency-val');
                if (latencyEl) latencyEl.textContent = ms + "ms";
            }, 50);

        } catch (err) {
            console.error("Inference Failed:", err);
            safeSetText(predictionText, "Error — see console");
        }
    }

    // Launch
    init();

}); // end DOMContentLoaded
