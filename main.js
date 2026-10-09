/**
 * MediCare AI - Interactive Frontend Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // Comprehensive symptom fallback catalog (45 symptoms)
    const DEFAULT_SYMPTOMS = [
        {"id": "fever", "name": "Fever / Elevated Temperature", "category": "General"},
        {"id": "fatigue", "name": "Fatigue / Extreme Tiredness", "category": "General"},
        {"id": "chills", "name": "Chills / Shivering", "category": "General"},
        {"id": "night_sweats", "name": "Night Sweats", "category": "General"},
        {"id": "weight_loss", "name": "Unexplained Weight Loss", "category": "General"},
        {"id": "malaise", "name": "General Malaise / Weakness", "category": "General"},
        {"id": "swollen_lymph_nodes", "name": "Swollen Lymph Nodes", "category": "General"},
        {"id": "cough", "name": "Persistent Cough", "category": "Respiratory"},
        {"id": "shortness_of_breath", "name": "Shortness of Breath / Dyspnea", "category": "Respiratory"},
        {"id": "sore_throat", "name": "Sore / Scratchy Throat", "category": "Respiratory"},
        {"id": "runny_nose", "name": "Runny / Congested Nose", "category": "Respiratory"},
        {"id": "chest_pain", "name": "Chest Pain / Tightness", "category": "Respiratory"},
        {"id": "wheezing", "name": "Wheezing / Whistling Breath", "category": "Respiratory"},
        {"id": "sneezing", "name": "Frequent Sneezing", "category": "Respiratory"},
        {"id": "headache", "name": "Headache", "category": "Neurological"},
        {"id": "dizziness", "name": "Dizziness / Lightheadedness", "category": "Neurological"},
        {"id": "blurred_vision", "name": "Blurred Vision", "category": "Neurological"},
        {"id": "stiff_neck", "name": "Stiff Neck", "category": "Neurological"},
        {"id": "insomnia", "name": "Insomnia / Sleep Trouble", "category": "Neurological"},
        {"id": "confusion", "name": "Mental Confusion", "category": "Neurological"},
        {"id": "anxiety", "name": "Anxiety / Nervousness", "category": "Neurological"},
        {"id": "nausea", "name": "Nausea", "category": "Digestive"},
        {"id": "vomiting", "name": "Vomiting", "category": "Digestive"},
        {"id": "diarrhea", "name": "Loose Motion / Diarrhea", "category": "Digestive"},
        {"id": "abdominal_pain", "name": "Abdominal / Stomach Pain", "category": "Digestive"},
        {"id": "stomach_cramps", "name": "Severe Stomach Cramps", "category": "Digestive"},
        {"id": "indigestion", "name": "Indigestion / Dyspepsia", "category": "Digestive"},
        {"id": "blood_in_stool", "name": "Blood in Stool", "category": "Digestive"},
        {"id": "loss_of_appetite", "name": "Loss of Appetite", "category": "Digestive"},
        {"id": "acid_reflux", "name": "Acid Reflux", "category": "Digestive"},
        {"id": "heartburn", "name": "Heartburn", "category": "Digestive"},
        {"id": "constipation", "name": "Constipation", "category": "Digestive"},
        {"id": "bloating", "name": "Abdominal Bloating / Gas", "category": "Digestive"},
        {"id": "joint_pain", "name": "Joint Pain", "category": "Musculoskeletal"},
        {"id": "muscle_ache", "name": "Muscle Aches / Body Pain", "category": "Musculoskeletal"},
        {"id": "back_pain", "name": "Back Pain", "category": "Musculoskeletal"},
        {"id": "swollen_joints", "name": "Swollen Joints", "category": "Musculoskeletal"},
        {"id": "skin_rash", "name": "Skin Rash / Lesions", "category": "Skin & Allergies"},
        {"id": "itchy_eyes", "name": "Itchy / Red Eyes", "category": "Skin & Allergies"},
        {"id": "jaundice", "name": "Yellow Skin / Eyes (Jaundice)", "category": "Skin & Allergies"},
        {"id": "frequent_urination", "name": "Frequent Urination", "category": "Metabolic & Renal"},
        {"id": "burning_urination", "name": "Pain / Burning Urination", "category": "Metabolic & Renal"},
        {"id": "increased_thirst", "name": "Extreme Thirst", "category": "Metabolic & Renal"},
        {"id": "high_blood_pressure", "name": "Elevated Blood Pressure", "category": "Metabolic & Renal"},
        {"id": "palpitations", "name": "Heart Palpitations", "category": "Metabolic & Renal"}
    ];

    // State management
    let allSymptoms = [...DEFAULT_SYMPTOMS];
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
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', newTheme);
            const icon = themeToggleBtn.querySelector('i');
            if (icon) {
                icon.className = newTheme === 'dark' ? 'fa-solid fa-moon' : 'fa-solid fa-sun';
            }
        });
    }

    // 2. Severity Slider Feedback
    if (severitySlider && severityVal) {
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
    }

    // 3. Gender Selector Buttons
    document.querySelectorAll('.gender-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.gender-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const radio = btn.querySelector('input');
            if (radio) radio.checked = true;
        });
    });

    // Smart API Base URL resolution
    const getApiEndpoints = () => {
        const endpoints = [];
        if (window.MEDICARE_BACKEND_URL) {
            endpoints.push(window.MEDICARE_BACKEND_URL.replace(/\/+$/, ''));
        }
        // If loaded from origin with port 8000
        if (window.location.origin && window.location.port === '8000') {
            endpoints.push('');
        }
        // Local default backend
        endpoints.push('http://127.0.0.1:8000');
        endpoints.push('http://localhost:8000');
        endpoints.push('');
        return [...new Set(endpoints)];
    };

    // 4. Fetch Symptom Catalog from Backend API (with fallback)
    async function fetchSymptoms() {
        const candidateHosts = getApiEndpoints();
        let loaded = false;

        for (const host of candidateHosts) {
            try {
                const url = `${host}/api/symptoms`;
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 2000);
                const res = await fetch(url, { signal: controller.signal });
                clearTimeout(timeoutId);

                if (res.ok) {
                    const data = await res.json();
                    if (data.symptoms && data.symptoms.length > 0) {
                        allSymptoms = data.symptoms;
                        loaded = true;
                        break;
                    }
                }
            } catch (e) {
                // Try next endpoint
            }
        }

        if (!loaded) {
            allSymptoms = [...DEFAULT_SYMPTOMS];
        }

        renderSymptoms();
    }

    // 5. Render Symptoms Grid based on Category & Search filter
    function renderSymptoms() {
        if (!symptomsGrid) return;
        const searchQuery = (symptomSearch ? symptomSearch.value : '').toLowerCase().trim();
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
        if (selectedCountBadge) {
            selectedCountBadge.textContent = `${selectedSymptoms.size} Symptom${selectedSymptoms.size === 1 ? '' : 's'} Selected`;
        }
        if (!selectedChipsTray) return;
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
    if (categoryTabs) {
        categoryTabs.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                categoryTabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentCategoryFilter = btn.getAttribute('data-category');
                renderSymptoms();
            });
        });
    }

    // 8. Search Bar Listener
    if (symptomSearch && clearSearchBtn) {
        symptomSearch.addEventListener('input', () => {
            clearSearchBtn.style.display = symptomSearch.value ? 'block' : 'none';
            renderSymptoms();
        });

        clearSearchBtn.addEventListener('click', () => {
            symptomSearch.value = '';
            clearSearchBtn.style.display = 'none';
            renderSymptoms();
        });
    }

    // 9. Form Submission & Predict API Call
    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            if (selectedSymptoms.size === 0) {
                alert('Please select at least 1 symptom to run disease prediction.');
                return;
            }

            const ageInput = document.getElementById('ageInput');
            const durationInput = document.getElementById('durationInput');
            const genderRadio = document.querySelector('input[name="gender"]:checked');

            const age = ageInput ? parseInt(ageInput.value) || 28 : 28;
            const gender = genderRadio ? genderRadio.value : 'other';
            const duration_days = durationInput ? parseInt(durationInput.value) || 3 : 3;
            const severity_level = severitySlider ? parseInt(severitySlider.value) || 5 : 5;

            const payload = {
                age,
                gender,
                duration_days,
                severity_level,
                symptoms: Array.from(selectedSymptoms)
            };

            // Loading state UI
            if (predictBtn) {
                predictBtn.disabled = true;
                const span = predictBtn.querySelector('span');
                const spinner = predictBtn.querySelector('.btn-spinner');
                if (span) span.style.display = 'none';
                if (spinner) spinner.style.display = 'block';
            }

            let resultData = null;
            let lastError = null;
            const candidateHosts = getApiEndpoints();

            for (const host of candidateHosts) {
                try {
                    const url = `${host}/api/predict`;
                    const res = await fetch(url, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });

                    if (res.ok) {
                        resultData = await res.json();
                        break;
                    } else {
                        const err = await res.json().catch(() => ({}));
                        lastError = err.detail || `Server error (${res.status})`;
                    }
                } catch (err) {
                    lastError = err.message;
                }
            }

            if (predictBtn) {
                predictBtn.disabled = false;
                const span = predictBtn.querySelector('span');
                const spinner = predictBtn.querySelector('.btn-spinner');
                if (span) span.style.display = 'inline-flex';
                if (spinner) spinner.style.display = 'none';
            }

            if (resultData) {
                renderPredictionResults(resultData);
            } else {
                alert(`Could not connect to FastAPI server (http://127.0.0.1:8000). Please start the backend with: python app.py\n\nDetails: ${lastError || 'Connection refused'}`);
            }
        });
    }

    // 10. Render Prediction Results in Dashboard
    function renderPredictionResults(data) {
        if (!resultsPlaceholder || !resultsContent) return;
        resultsPlaceholder.style.display = 'none';
        resultsContent.style.display = 'block';

        const primary = data.primary_diagnosis;

        // Disease Name & Triage Badge
        if (resDiseaseName) resDiseaseName.textContent = primary.name;
        if (resTriageBadge) {
            resTriageBadge.textContent = `${primary.triage_severity} Severity`;
            resTriageBadge.className = `triage-badge ${primary.triage_severity}`;
        }

        // Confidence gauge SVG calculation (Circumference = 2 * PI * r = 2 * 3.14159 * 50 = 314)
        const confidence = primary.confidence;
        if (resConfidenceVal) resConfidenceVal.textContent = `${confidence}%`;
        if (gaugeCircle) {
            const offset = 314 - (314 * (confidence / 100));
            gaugeCircle.style.strokeDashoffset = offset;
        }

        // Description & Specialist
        if (resDescription) resDescription.textContent = primary.description;
        if (resSpecialist) resSpecialist.textContent = primary.recommended_specialist;

        // Precautions
        if (resPrecautions) {
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
        }

        // Emergency Text
        if (resEmergencyText) {
            resEmergencyText.textContent = primary.emergency_warning || "Persistent severe pain, difficulty breathing, or high fever.";
        }

        // Differential Diagnoses Horizontal Bars
        if (resDifferentialList) {
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
                    const fill = item.querySelector('.diff-bar-fill');
                    if (fill) fill.style.width = `${diff.confidence_percent}%`;
                }, 100);
            });
        }

        // Smooth scroll to results
        const resSec = document.getElementById('resultsSection');
        if (resSec) resSec.scrollIntoView({ behavior: 'smooth' });
    }

    // 11. Print Clinical Report Button
    if (printReportBtn) {
        printReportBtn.addEventListener('click', () => {
            window.print();
        });
    }

    // 12. Reset Form Button
    if (resetFormBtn) {
        resetFormBtn.addEventListener('click', () => {
            selectedSymptoms.clear();
            updateSelectedState();
            renderSymptoms();
            if (resultsContent) resultsContent.style.display = 'none';
            if (resultsPlaceholder) resultsPlaceholder.style.display = 'block';
            if (predictionForm) predictionForm.reset();
            if (severitySlider && severityVal) {
                severitySlider.value = 5;
                severityVal.textContent = '5 - Moderate';
                severityVal.className = 'severity-badge mod';
            }
            const checkSec = document.getElementById('checkerSection');
            if (checkSec) checkSec.scrollIntoView({ behavior: 'smooth' });
        });
    }

    // Initial Fetch & Render
    renderSymptoms();
    fetchSymptoms();
});
