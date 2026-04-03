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

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultsPanel = document.getElementById('results-panel');
const imagePreview = document.getElementById('image-preview');
const predictionText = document.getElementById('prediction-text');
const confidenceText = document.getElementById('confidence-text');
const confidenceFill = document.getElementById('confidence-fill');
const modelStatus = document.getElementById('model-status');
const progressFill = document.getElementById('progress-fill');
const resetBtn = document.getElementById('reset-btn');

async function init() {
    try {
        modelStatus.textContent = "Loading model...";
        progressFill.style.width = "30%";

        session = await ort.InferenceSession.create(CONFIG.modelPath);

        // Explicitly set to 100% and solidified state
        progressFill.style.width = "100%";
        progressFill.style.backgroundColor = "var(--accent)"; 
        
        modelStatus.textContent = "🧠 AI Engine Online";
        setTimeout(() => {
            const si = document.querySelector('.status-indicator');
            if (si) si.style.opacity = "0.4";
        }, 3000);

        console.log("ONNX Model Loaded. imageSize:", CONFIG.imageSize);
        console.log("Input names:", session.inputNames);
        console.log("Output names:", session.outputNames);

        await warmupSession();

    } catch (e) {
        console.error("Model loading failed:", e);
        modelStatus.textContent = "Error loading model";
        modelStatus.style.color = "#ff4d4d";
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
        console.log("Warmup complete — input shape [1, 3,", size + ",", size + "]");
    } catch (e) {
        console.warn("Warmup failed (non-critical):", e);
    }
}

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = "var(--accent)";
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = "var(--border)";
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) handleImage(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleImage(e.target.files[0]);
});

resetBtn.addEventListener('click', () => {
    resultsPanel.classList.add('hidden');
    document.getElementById('placeholder-text').classList.remove('hidden');
    fileInput.value = "";
});

async function handleImage(file) {
    if (!session) { alert("Please wait for the model to finish loading."); return; }

    const reader = new FileReader();
    reader.onload = async (e) => {
        imagePreview.src = e.target.result;
        document.getElementById('placeholder-text').classList.add('hidden');
        resultsPanel.classList.remove('hidden');
        resultsPanel.classList.add('loading');

        const img = new Image();
        img.src = e.target.result;
        img.onload = async () => {
            await runInference(img);
            resultsPanel.classList.remove('loading');
        };
    };
    reader.readAsDataURL(file);
}

function preprocessImage(img) {
    const size = CONFIG.imageSize;

    // Step 1: scale so shortest side = 224 (preserve aspect ratio)
    const scale = size / Math.min(img.width, img.height);
    const dw = Math.round(img.width * scale);
    const dh = Math.round(img.height * scale);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = dw;
    tempCanvas.height = dh;
    tempCanvas.getContext('2d').drawImage(img, 0, 0, dw, dh);

    // Step 2: center crop to 224x224
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

    // Step 3: read pixels
    const { data } = cropCanvas.getContext('2d').getImageData(0, 0, size, size);
    const pixels = size * size;
    const input = new Float32Array(3 * pixels);

    // Step 4: HWC (RGBA canvas) -> CHW (ONNX) + ImageNet normalize
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

        // softmax with max subtraction for numerical stability
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
        console.log("Top 3:", results.slice(0, 3).map(r => r.label + ": " + (r.confidence * 100).toFixed(1) + "%"));

        predictionText.textContent = best.label;
        
        // Finalize the bar as a 'Match Strength' score
        setTimeout(() => {
            const pct = Math.max(10, best.confidence * 100); // Minimum 10% for visibility
            confidenceText.textContent = `Neural Match Strength: ${pct.toFixed(1)}%`;
            confidenceFill.style.width = `${pct}%`;
            confidenceFill.style.opacity = "1";
        }, 50);

    } catch (err) {
        console.error("Inference Failed:", err);
        predictionText.textContent = "Error — check console";
        confidenceText.textContent = "";
    }
}

init();