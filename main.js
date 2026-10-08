/**
 * MediCare AI - Interactive Frontend Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // State management
    let allSymptoms = [];
    const selectedSymptoms = new Set();
    let currentCategoryFilter = 'all';

    // DOM Elements
    const themeToggleBtn = document.getElementById('themeToggle');
    const symptomsGrid = document.getElementById('symptomsGrid');
    const categoryTabs = document.getElementById('categoryTabs');
    const symptomSearch = document.getElementById('symptomSearch');
    const clearSearchBtn = document.getElementById('clearSearchBtn');
    const selectedChipsTray = document.getElementById('selectedChipsTray');
    const selectedCountBadge = document.getElementById('selectedCountBadge');
    
    const severitySlider = document.getElementById('severitySlider');
    const severityVal = document.getElementById('severityVal');
    const predictionForm = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');

    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const resultsContent = document.getElementById('resultsContent');

    const resDiseaseName = document.getElementById('resDiseaseName');
    const resTriageBadge = document.getElementById('resTriageBadge');
    const resConfidenceVal = document.getElementById('resConfidenceVal');
    const gaugeCircle = document.getElementById('gaugeCircle');
    const resDescription = document.getElementById('resDescription');
    const resSpecialist = document.getElementById('resSpecialist');
    const resPrecautions = document.getElementById('resPrecautions');
    const resEmergencyText = document.getElementById('resEmergencyText');
    const resDifferentialList = document.getElementById('resDifferentialList');

    const printReportBtn = document.getElementById('printReportBtn');
    const resetFormBtn = document.getElementById('resetFormBtn');

    // 1. Theme Toggle Handler
    themeToggleBtn.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        themeToggleBtn.querySelector('i').className = newTheme === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun';
    });

    // 2. Severity Slider Feedback
    severitySlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        if (val <= 3) {
            severityVal.textContent = `${val} - Mild`;
            severityVal.className = 'severity-badge low';
        } else if (val <= 7) {
            severityVal.textContent = `${val} - Moderate`;
            severityVal.className = 'severity-badge mod';
        } else {
            severityVal.textContent = `${val} - Severe`;
            severityVal.className = 'severity-badge high';
        }
    });

    // 3. Gender Selector Buttons
    document.querySelectorAll('.gender-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.gender-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            btn.querySelector('input').checked = true;
        });
    });

    // API Base URL resolution for standalone frontend
    const API_BASE_URL = 'http://127.0.0.1:8000';

    const getApiUrl = (endpoint) => `${API_BASE_URL}${endpoint}`;

    // 4. Fetch Symptom Catalog from Backend API
    async function fetchSymptoms() {
        try {
            const res = await fetch(getApiUrl('/api/symptoms'));

            if (!res.ok) {
                throw new Error(`HTTP error ${res.status}`);
            }

            const data = await res.json();

            allSymptoms = data.symptoms || [];
            renderSymptoms();

        } catch (err) {
            console.error('Error fetching symptoms:', err);

            symptomsGrid.innerHTML = `
                <div class="error-msg">
                    <i class="fa-solid fa-triangle-exclamation"></i>
                    <strong>Unable to load symptoms</strong><br>
                    Please make sure the FastAPI backend is running.
                </div>`;
        }
    }

    // 5. Render Symptoms Grid based on Category & Search filter
    function renderSymptoms() {
        const searchQuery = symptomSearch.value.toLowerCase().trim();
        symptomsGrid.innerHTML = '';

        const filtered = allSymptoms.filter(sym => {
            const matchesCategory = currentCategoryFilter === 'all' || sym.category === currentCategoryFilter;
            const matchesSearch = sym.name.toLowerCase().includes(searchQuery) || sym.category.toLowerCase().includes(searchQuery);
            return matchesCategory && matchesSearch;
        });

        if (filtered.length === 0) {
            symptomsGrid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-dim); padding: 1.5rem;">No matching symptoms found. Try another search.</div>`;
            return;
        }

        filtered.forEach(sym => {
            const isChecked = selectedSymptoms.has(sym.id);
            const card = document.createElement('label');
            card.className = `symptom-checkbox-card ${isChecked ? 'selected' : ''}`;
            card.innerHTML = `
                <input type="checkbox" value="${sym.id}" ${isChecked ? 'checked' : ''}>
                <span>${sym.name}</span>
            `;

            card.querySelector('input').addEventListener('change', (e) => {
                if (e.target.checked) {
                    selectedSymptoms.add(sym.id);
                } else {
                    selectedSymptoms.delete(sym.id);
                }
                updateSelectedState();
                renderSymptoms();
            });

            symptomsGrid.appendChild(card);
        });
    }

    // 6. Update Selected Symptoms Chips & Counters
    function updateSelectedState() {
        selectedCountBadge.textContent = `${selectedSymptoms.size} Symptom${selectedSymptoms.size === 1 ? '' : 's'} Selected`;
        selectedChipsTray.innerHTML = '';

        if (selectedSymptoms.size === 0) {
            selectedChipsTray.innerHTML = `<span class="chips-placeholder">No symptoms selected yet. Click options below to add.</span>`;
            return;
        }

        selectedSymptoms.forEach(symId => {
            const symObj = allSymptoms.find(s => s.id === symId);
            const name = symObj ? symObj.name : symId;

            const chip = document.createElement('div');
            chip.className = 'symptom-chip';
            chip.innerHTML = `
                <span>${name}</span>
                <i class="fa-solid fa-xmark" title="Remove"></i>
            `;
            chip.querySelector('i').addEventListener('click', () => {
                selectedSymptoms.delete(symId);
                updateSelectedState();
                renderSymptoms();
            });
            selectedChipsTray.appendChild(chip);
        });
    }

    // 7. Category Filter Tabs Listener
    categoryTabs.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            categoryTabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentCategoryFilter = btn.getAttribute('data-category');
            renderSymptoms();
        });
    });

    // 8. Search Bar Listener
    symptomSearch.addEventListener('input', () => {
        clearSearchBtn.style.display = symptomSearch.value ? 'block' : 'none';
        renderSymptoms();
    });

    clearSearchBtn.addEventListener('click', () => {
        symptomSearch.value = '';
        clearSearchBtn.style.display = 'none';
        renderSymptoms();
    });

    // 9. Form Submission & Predict API Call
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (selectedSymptoms.size === 0) {
            alert('Please select at least 1 symptom to run disease prediction.');
            return;
        }

        const age = parseInt(document.getElementById('ageInput').value) || 28;
        const gender = document.querySelector('input[name="gender"]:checked').value;
        const duration_days = parseInt(document.getElementById('durationInput').value) || 3;
        const severity_level = parseInt(severitySlider.value) || 5;

        const payload = {
            age,
            gender,
            duration_days,
            severity_level,
            symptoms: Array.from(selectedSymptoms)
        };

        // Loading state UI
        predictBtn.disabled = true;
        predictBtn.querySelector('span').style.display = 'none';
        predictBtn.querySelector('.btn-spinner').style.display = 'block';

        try {
            let res;
            try {
                res = await fetch(getApiUrl('/api/predict'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            } catch (e) {
                res = await fetch('http://127.0.0.1:8000/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
            }

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || 'Prediction failed');
            }

            const data = await res.json();
            renderPredictionResults(data);

        } catch (err) {
            alert(`Error: ${err.message}`);
        } finally {
            predictBtn.disabled = false;
            predictBtn.querySelector('span').style.display = 'inline-flex';
            predictBtn.querySelector('.btn-spinner').style.display = 'none';
        }
    });

    // 10. Render Prediction Results in Dashboard
    function renderPredictionResults(data) {
        resultsPlaceholder.style.display = 'none';
        resultsContent.style.display = 'block';

        const primary = data.primary_diagnosis;

        // Disease Name & Triage Badge
        resDiseaseName.textContent = primary.name;
        resTriageBadge.textContent = `${primary.triage_severity} Severity`;
        resTriageBadge.className = `triage-badge ${primary.triage_severity}`;

        // Confidence gauge SVG calculation (Circumference = 2 * PI * r = 2 * 3.14159 * 50 = 314)
        const confidence = primary.confidence;
        resConfidenceVal.textContent = `${confidence}%`;
        const offset = 314 - (314 * (confidence / 100));
        gaugeCircle.style.strokeDashoffset = offset;

        // Description & Specialist
        resDescription.textContent = primary.description;
        resSpecialist.textContent = primary.recommended_specialist;

        // Precautions
        resPrecautions.innerHTML = '';
        (primary.precautions || []).forEach(prec => {
            const precCard = document.createElement('div');
            precCard.className = 'precaution-card';
            precCard.innerHTML = `
                <i class="fa-solid fa-circle-check"></i>
                <span>${prec}</span>
            `;
            resPrecautions.appendChild(precCard);
        });

        // Emergency Text
        resEmergencyText.textContent = primary.emergency_warning || "Persistent severe pain, difficulty breathing, or high fever.";

        // Differential Diagnoses Horizontal Bars
        resDifferentialList.innerHTML = '';
        (data.differential_diagnoses || []).forEach(diff => {
            const item = document.createElement('div');
            item.className = 'diff-item';
            item.innerHTML = `
                <div class="diff-info">
                    <span>${diff.disease}</span>
                    <span>${diff.confidence_percent}%</span>
                </div>
                <div class="diff-bar-bg">
                    <div class="diff-bar-fill" style="width: 0%;"></div>
                </div>
            `;
            resDifferentialList.appendChild(item);

            // Animate bar width
            setTimeout(() => {
                item.querySelector('.diff-bar-fill').style.width = `${diff.confidence_percent}%`;
            }, 100);
        });

        // Smooth scroll to results
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }

    // 11. Print Clinical Report Button
    printReportBtn.addEventListener('click', () => {
        window.print();
    });

    // 12. Reset Form Button
    resetFormBtn.addEventListener('click', () => {
        selectedSymptoms.clear();
        updateSelectedState();
        renderSymptoms();
        resultsContent.style.display = 'none';
        resultsPlaceholder.style.display = 'block';
        predictionForm.reset();
        severitySlider.value = 5;
        severityVal.textContent = '5 - Moderate';
        severityVal.className = 'severity-badge mod';
        document.getElementById('checkerSection').scrollIntoView({ behavior: 'smooth' });
    });

    // Initial Fetch
    fetchSymptoms();
});
