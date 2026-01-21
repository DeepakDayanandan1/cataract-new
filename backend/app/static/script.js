
// --- Fundus Logic ---
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loader = document.getElementById('loader');
const results = document.getElementById('results');

if (dropZone) {
    setupDragDrop(dropZone, fileInput, handleFundusFile);
}

function handleFundusFile(file) {
    processFile(file, '/predict', loader, results, displayFundusResults, resetFundus);
}

function displayFundusResults(data) {
    // Set Images
    document.getElementById('img-original').src = data.visualizations.original;
    document.getElementById('img-green').src = data.visualizations.green;
    document.getElementById('img-denoised').src = data.visualizations.denoised;
    document.getElementById('img-clahe').src = data.visualizations.clahe;

    // Set Text
    const statusParams = document.getElementById('diagnosis-text');
    statusParams.textContent = data.prediction;
    statusParams.className = `status ${data.prediction}`;

    document.getElementById('confidence-score').textContent = data.confidence;
}

function resetFundus() {
    dropZone.classList.remove('hidden');
    results.classList.add('hidden');
    loader.classList.add('hidden');
    fileInput.value = '';
}


// --- Fundus Multi-Class Logic ---
const dropZoneMulti = document.getElementById('drop-zone-multi');
const fileInputMulti = document.getElementById('file-input-multi');
const loaderMulti = document.getElementById('loader-multi');
const resultsMulti = document.getElementById('results-multi');

if (dropZoneMulti) {
    setupDragDrop(dropZoneMulti, fileInputMulti, handleMultiFile);
}

function handleMultiFile(file) {
    processFile(file, '/predict_multiclass', loaderMulti, resultsMulti, displayMultiResults, resetMulti);
}

function displayMultiResults(data) {
    // Set Images
    document.getElementById('multi-img-original').src = data.visualizations.original;
    document.getElementById('multi-img-green').src = data.visualizations.green;
    document.getElementById('multi-img-denoised').src = data.visualizations.denoised;
    document.getElementById('multi-img-clahe').src = data.visualizations.clahe;

    // Set Text
    const statusParams = document.getElementById('multi-diagnosis-text');
    statusParams.textContent = data.prediction;
    statusParams.className = `status ${data.prediction.toLowerCase()}`; // e.g. .mild, .severe

    // document.getElementById('multi-confidence-score').textContent = data.confidence;
    document.getElementById('multi-acc').textContent = data.metrics.accuracy;
    document.getElementById('multi-prec').textContent = data.metrics.precision;
    document.getElementById('multi-rec').textContent = data.metrics.recall;
    document.getElementById('multi-f1').textContent = data.metrics.f1;
}

function resetMulti() {
    dropZoneMulti.classList.remove('hidden');
    resultsMulti.classList.add('hidden');
    loaderMulti.classList.add('hidden');
    fileInputMulti.value = '';
}


// --- Slit-Lamp Logic ---
const dropZoneSlit = document.getElementById('drop-zone-slit');
const fileInputSlit = document.getElementById('file-input-slit');
const loaderSlit = document.getElementById('loader-slit');
const resultsSlit = document.getElementById('results-slit');

if (dropZoneSlit) {
    setupDragDrop(dropZoneSlit, fileInputSlit, handleSlitFile);
}

function handleSlitFile(file) {
    processFile(file, '/predict_slit_lamp', loaderSlit, resultsSlit, displaySlitResults, resetSlit);
}

function displaySlitResults(data) {
    // Set Images
    document.getElementById('slit-img-original').src = data.visualizations.original;
    // document.getElementById('slit-img-green').src = data.visualizations.green; // Removed
    document.getElementById('slit-img-denoised').src = data.visualizations.denoised;
    document.getElementById('slit-img-clahe').src = data.visualizations.clahe;

    // Set Text
    const statusParams = document.getElementById('slit-diagnosis-text');
    statusParams.textContent = data.prediction;
    // Map classes to colors/status if needed, or just status class
    statusParams.className = `status ${data.prediction}`;

    document.getElementById('slit-confidence-score').textContent = data.confidence;
}

function resetSlit() {
    dropZoneSlit.classList.remove('hidden');
    resultsSlit.classList.add('hidden');
    loaderSlit.classList.add('hidden');
    fileInputSlit.value = '';
}


// --- Shared Helper Functions ---

function setupDragDrop(zone, input, handler) {
    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handler(e.dataTransfer.files[0]);
        }
    });

    input.addEventListener('change', () => {
        if (input.files.length) {
            handler(input.files[0]);
        }
    });
}

function processFile(file, endpoint, loaderEl, resultsEl, displayCallback, resetCallback) {
    if (!file.type.startsWith('image/')) {
        alert("Please upload a valid image file.");
        return;
    }

    // Hide Upload, Show Loader
    // We want to hide the specific drop zone, which we can find by traversing up or using globals
    // Better to use the globals defined above logic-blocks or pass them in.
    // However, existing logic hides dropZone. Let's do that.

    // Logic: 
    // 1. Hide the drop zone associated with this loader (passed in? No, we need dropZone element)
    // Actually, resetCallback can handle resetting, but here we need to hide drop zone.
    // Let's pass dropZone as well or deduce it.
    // In `handleFundusFile`, I use `dropZone`.

    if (endpoint === '/predict') {
        dropZone.classList.add('hidden');
    } else if (endpoint === '/predict_multiclass') {
        dropZoneMulti.classList.add('hidden');
    } else {
        dropZoneSlit.classList.add('hidden');
    }

    loaderEl.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    fetch(endpoint, {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                resetCallback();
                return;
            }
            displayCallback(data);
            resultsEl.classList.remove('hidden');
        })
        .catch(err => {
            console.error(err);
            alert("An error occurred during analysis.");
            resetCallback();
        })
        .finally(() => {
            loaderEl.classList.add('hidden');
        });
}
