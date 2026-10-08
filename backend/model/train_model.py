import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Ensure directory exists
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Define Master Symptom List with Categories & Descriptions
SYMPTOMS_CATALOG = [
    # Respiratory
    {"id": "cough", "name": "Cough", "category": "Respiratory"},
    {"id": "shortness_of_breath", "name": "Shortness of Breath", "category": "Respiratory"},
    {"id": "sore_throat", "name": "Sore Throat", "category": "Respiratory"},
    {"id": "runny_nose", "name": "Runny Nose / Congestion", "category": "Respiratory"},
    {"id": "sneezing", "name": "Sneezing", "category": "Respiratory"},
    {"id": "wheezing", "name": "Wheezing", "category": "Respiratory"},
    {"id": "loss_of_taste_smell", "name": "Loss of Taste / Smell", "category": "Respiratory"},
    {"id": "chest_pain", "name": "Chest Pain", "category": "Respiratory"},

    # General / Systemic
    {"id": "fever", "name": "Fever", "category": "General"},
    {"id": "fatigue", "name": "Fatigue & Weakness", "category": "General"},
    {"id": "chills", "name": "Chills & Shivering", "category": "General"},
    {"id": "sweating", "name": "Excessive Sweating", "category": "General"},
    {"id": "weight_loss", "name": "Unexplained Weight Loss", "category": "General"},
    {"id": "swollen_lymph_nodes", "name": "Swollen Lymph Nodes", "category": "General"},

    # Neurological & Head
    {"id": "headache", "name": "Headache", "category": "Neurological"},
    {"id": "dizziness", "name": "Dizziness / Lightheadedness", "category": "Neurological"},
    {"id": "blurred_vision", "name": "Blurred Vision", "category": "Neurological"},
    {"id": "stiff_neck", "name": "Stiff Neck", "category": "Neurological"},
    {"id": "insomnia", "name": "Insomnia / Sleep Trouble", "category": "Neurological"},
    {"id": "confusion", "name": "Mental Confusion", "category": "Neurological"},
    {"id": "anxiety", "name": "Anxiety / Nervousness", "category": "Neurological"},

    # Digestive & Abdominal (Enhanced for stomach pain & loose motions)
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

    # Musculoskeletal
    {"id": "joint_pain", "name": "Joint Pain", "category": "Musculoskeletal"},
    {"id": "muscle_ache", "name": "Muscle Aches / Body Pain", "category": "Musculoskeletal"},
    {"id": "back_pain", "name": "Back Pain", "category": "Musculoskeletal"},
    {"id": "swollen_joints", "name": "Swollen Joints", "category": "Musculoskeletal"},

    # Skin & Allergies
    {"id": "skin_rash", "name": "Skin Rash / Lesions", "category": "Skin & Allergies"},
    {"id": "itchy_eyes", "name": "Itchy / Red Eyes", "category": "Skin & Allergies"},
    {"id": "jaundice", "name": "Yellow Skin / Eyes (Jaundice)", "category": "Skin & Allergies"},

    # Metabolic & Renal
    {"id": "frequent_urination", "name": "Frequent Urination", "category": "Metabolic & Renal"},
    {"id": "burning_urination", "name": "Pain / Burning Urination", "category": "Metabolic & Renal"},
    {"id": "increased_thirst", "name": "Extreme Thirst", "category": "Metabolic & Renal"},
    {"id": "high_blood_pressure", "name": "Elevated Blood Pressure", "category": "Metabolic & Renal"},
    {"id": "palpitations", "name": "Heart Palpitations", "category": "Metabolic & Renal"}
]

SYMPTOM_IDS = [s["id"] for s in SYMPTOMS_CATALOG]

# 2. Define Disease Definitions with Characteristic Symptoms (Primary & Secondary)
DISEASE_PROFILES = {
    # --- STOMACH PAIN & LOOSE MOTION / GI DISEASES ---
    "Food Poisoning": {
        "primary": ["diarrhea", "vomiting", "nausea", "stomach_cramps", "fever"],
        "secondary": ["fatigue", "chills", "headache", "abdominal_pain"],
        "info": {
            "description": "Acute gastrointestinal illness caused by ingesting food or water contaminated with toxic bacteria or parasites.",
            "severity": "Moderate",
            "specialist": "Gastroenterologist / General Physician",
            "precautions": [
                "Sip Oral Rehydration Salts (ORS) continuously to replace lost electrolytes.",
                "Refrain from solid foods until vomiting subsides, then eat bland foods (rice, toast, bananas).",
                "Avoid caffeine, dairy products, fatty foods, and alcohol.",
                "Seek urgent care if unable to retain fluids for over 24 hours."
            ],
            "emergency_warning": "High fever (>102°F), severe dehydration, bloody stool, or continuous vomiting."
        }
    },
    "Gastroenteritis (Stomach Flu)": {
        "primary": ["diarrhea", "nausea", "vomiting", "abdominal_pain"],
        "secondary": ["fever", "chills", "fatigue", "loss_of_appetite", "stomach_cramps"],
        "info": {
            "description": "Intestinal infection marked by watery diarrhea, abdominal cramps, nausea, vomiting, and sometimes fever.",
            "severity": "Moderate",
            "specialist": "Gastroenterologist / General Physician",
            "precautions": [
                "Sip oral rehydration solution (ORS) or electrolyte fluids continuously.",
                "Follow a bland diet (BRAT: Bananas, Rice, Applesauce, Toast) once vomiting stops.",
                "Avoid dairy products, caffeine, alcohol, and fatty foods.",
                "Practice thorough hand hygiene to prevent spreading infection."
            ],
            "emergency_warning": "Inability to keep liquids down for 24 hours, signs of severe dehydration, or blood in stool."
        }
    },
    "Peptic Ulcer Disease": {
        "primary": ["abdominal_pain", "heartburn", "acid_reflux", "indigestion"],
        "secondary": ["nausea", "bloating", "loss_of_appetite", "stomach_cramps"],
        "info": {
            "description": "Painful sores or erosions on the inner protective lining of the stomach or upper small intestine.",
            "severity": "Moderate",
            "specialist": "Gastroenterologist",
            "precautions": [
                "Avoid pain relievers like NSAIDs (ibuprofen/aspirin) which aggravate stomach lining.",
                "Take prescribed proton pump inhibitors (PPIs) or H2 blockers as directed.",
                "Eliminate spicy, acidic, fried foods, and coffee.",
                "Eat smaller, more frequent meals to neutralize stomach acid."
            ],
            "emergency_warning": "Vomiting blood or dark coffee-ground material, sudden severe sharp stomach pain, or dark tarry stool."
        }
    },
    "Inflammatory Bowel Disease (IBD / Crohn's)": {
        "primary": ["diarrhea", "abdominal_pain", "stomach_cramps", "blood_in_stool", "weight_loss"],
        "secondary": ["fatigue", "fever", "loss_of_appetite"],
        "info": {
            "description": "Chronic autoimmune inflammatory condition affecting the digestive tract, causing recurrent abdominal cramps and bloody diarrhea.",
            "severity": "High",
            "specialist": "Gastroenterologist",
            "precautions": [
                "Follow a low-residue diet during inflammatory flare-ups.",
                "Take prescribed anti-inflammatory or immunomodulator medications consistently.",
                "Stay well hydrated and take recommended vitamin supplements.",
                "Keep a food diary to identify personal dietary trigger items."
            ],
            "emergency_warning": "Severe blood loss in stool, persistent high fever, intense unrelenting abdominal pain, or bowel obstruction."
        }
    },
    "Amoebic Dysentery (Amoebiasis)": {
        "primary": ["diarrhea", "abdominal_pain", "stomach_cramps", "blood_in_stool", "fever"],
        "secondary": ["fatigue", "weight_loss", "nausea", "chills"],
        "info": {
            "description": "Parasitic infection of the intestines caused by Entamoeba histolytica, leading to painful bloody stools and abdominal cramps.",
            "severity": "High",
            "specialist": "Gastroenterologist / Infectious Disease Specialist",
            "precautions": [
                "Undergo stool routine and culture diagnostic evaluation promptly.",
                "Complete full course of anti-parasitic antibiotics (e.g., metronidazole).",
                "Drink only boiled, filtered, or sealed bottled water.",
                "Maintain rigorous personal sanitation and hand washing."
            ],
            "emergency_warning": "Frequent bloody diarrhea, high fever, or signs of liver abscess (right upper abdominal pain)."
        }
    },
    "Lactose Intolerance": {
        "primary": ["diarrhea", "abdominal_pain", "bloating", "stomach_cramps"],
        "secondary": ["nausea", "acid_reflux", "indigestion"],
        "info": {
            "description": "Digestive disorder caused by inability to digest lactose, the sugar found in milk, leading to bloating and loose motions after dairy consumption.",
            "severity": "Low",
            "specialist": "Gastroenterologist / Clinical Dietitian",
            "precautions": [
                "Replace dairy milk with lactose-free, almond, oat, or soy alternatives.",
                "Take over-the-counter lactase enzyme drops or pills before consuming dairy.",
                "Read ingredient labels carefully for hidden whey or milk solids.",
                "Ensure adequate calcium and Vitamin D intake through non-dairy sources."
            ],
            "emergency_warning": "Unintended severe weight loss or symptoms persisting despite zero dairy intake."
        }
    },
    "Gallstones / Acute Cholecystitis": {
        "primary": ["abdominal_pain", "nausea", "vomiting", "fever"],
        "secondary": ["bloating", "jaundice", "indigestion", "stomach_cramps"],
        "info": {
            "description": "Hardened deposits in the gallbladder that obstruct bile ducts, resulting in sudden intense upper right stomach pain.",
            "severity": "High",
            "specialist": "Gastroenterologist / General Surgeon",
            "precautions": [
                "Seek medical evaluation and abdominal ultrasound imaging.",
                "Avoid high-fat, greasy, and fried foods that trigger gallbladder contractions.",
                "Maintain a healthy body weight with gradual weight loss if needed.",
                "Consult a general surgeon if recurrent painful gallbladder attacks occur."
            ],
            "emergency_warning": "Jaundice (yellow eyes/skin), high fever with chills, or severe pain lasting more than 6 hours."
        }
    },
    "Appendicitis": {
        "primary": ["abdominal_pain", "nausea", "vomiting", "loss_of_appetite"],
        "secondary": ["fever", "constipation", "diarrhea", "bloating", "stomach_cramps"],
        "info": {
            "description": "Acute inflammation of the appendix requiring prompt surgical evaluation to prevent rupture.",
            "severity": "Emergency",
            "specialist": "General Surgeon / Emergency Medicine",
            "precautions": [
                "Seek immediate emergency hospital evaluation if sudden sharp pain moves to lower right abdomen.",
                "Do NOT eat, drink, or take laxatives/painkillers before doctor evaluation.",
                "Avoid applying heat packs to the abdomen.",
                "Prepare for diagnostic ultrasound or CT scan."
            ],
            "emergency_warning": "Sudden severe pain starting near navel and shifting to right lower abdomen with fever/vomiting."
        }
    },
    "Irritable Bowel Syndrome (IBS)": {
        "primary": ["abdominal_pain", "bloating", "diarrhea", "constipation", "stomach_cramps"],
        "secondary": ["nausea", "anxiety", "insomnia", "fatigue"],
        "info": {
            "description": "Common gastrointestinal disorder affecting the large intestine, marked by cramping, gas, bloating, and altered bowel habits.",
            "severity": "Low to Moderate",
            "specialist": "Gastroenterologist",
            "precautions": [
                "Keep a food diary to identify personal FODMAP food triggers.",
                "Eat regular, balanced meals slowly without rushing.",
                "Incorporate dietary fiber gradually and drink plenty of water.",
                "Practice stress management techniques like regular exercise and meditation."
            ],
            "emergency_warning": "Unexplained weight loss, nocturnal diarrhea, rectal bleeding, or persistent fever."
        }
    },
    "Gastroesophageal Reflux Disease (GERD)": {
        "primary": ["acid_reflux", "heartburn", "chest_pain", "indigestion"],
        "secondary": ["nausea", "sore_throat", "cough", "bloating"],
        "info": {
            "description": "Digestive disease where stomach acid frequently flows back into the tube connecting mouth and stomach.",
            "severity": "Low to Moderate",
            "specialist": "Gastroenterologist",
            "precautions": [
                "Eat smaller, more frequent meals instead of heavy dinners.",
                "Avoid lying down for at least 3 hours after eating.",
                "Elevate the head of your bed by 6-8 inches.",
                "Avoid trigger foods like spicy, fatty, acidic foods, caffeine, and alcohol."
            ],
            "emergency_warning": "Difficulty swallowing, severe burning chest pain radiating to back/arm, or black stool."
        }
    },

    # --- OTHER GENERAL & SYSTEMIC DISEASES ---
    "COVID-19": {
        "primary": ["fever", "cough", "fatigue", "loss_of_taste_smell", "shortness_of_breath"],
        "secondary": ["headache", "sore_throat", "muscle_ache", "chills"],
        "info": {
            "description": "Contagious respiratory illness caused by SARS-CoV-2 coronavirus affecting lungs and upper respiratory tract.",
            "severity": "High",
            "specialist": "Pulmonologist / Infectious Disease Specialist",
            "precautions": [
                "Isolate yourself in a well-ventilated room to protect family members.",
                "Monitor oxygen levels (SpO2) and body temperature regularly.",
                "Stay well hydrated with electrolytes and oral fluids.",
                "Seek emergency medical help if breathlessness or severe chest pain occurs."
            ],
            "emergency_warning": "SpO2 below 94%, severe shortness of breath, persistent chest tightness, or confusion."
        }
    },
    "Influenza (Flu)": {
        "primary": ["fever", "cough", "fatigue", "chills", "muscle_ache"],
        "secondary": ["headache", "sore_throat", "runny_nose", "sweating"],
        "info": {
            "description": "Viral infection attacking the respiratory system causing sudden fever, body aches, and fatigue.",
            "severity": "Moderate",
            "specialist": "General Physician / Internal Medicine",
            "precautions": [
                "Get plenty of bed rest to allow your immune system to recover.",
                "Drink warm fluids like herbal tea, broth, and water.",
                "Take over-the-counter antipyretics for fever as advised by a doctor.",
                "Wash hands frequently and avoid close contact with others."
            ],
            "emergency_warning": "Difficulty breathing, persistent high fever (>103°F/39.4°C), or severe dizziness."
        }
    },
    "Common Cold": {
        "primary": ["runny_nose", "sneezing", "sore_throat", "cough"],
        "secondary": ["fatigue", "headache", "itchy_eyes"],
        "info": {
            "description": "Mild viral upper respiratory tract infection characterized by nasal congestion, sneezing, and sore throat.",
            "severity": "Low",
            "specialist": "General Physician",
            "precautions": [
                "Use saline nasal drops or steam inhalation for nasal relief.",
                "Stay warm and rest adequately.",
                "Gargle with warm salt water for throat comfort.",
                "Consume vitamin C rich fruits and stay hydrated."
            ],
            "emergency_warning": "Symptoms persisting longer than 10-14 days or high fever onset."
        }
    },
    "Pneumonia": {
        "primary": ["fever", "cough", "shortness_of_breath", "chest_pain", "chills"],
        "secondary": ["fatigue", "sweating", "nausea", "confusion"],
        "info": {
            "description": "Infection that inflames air sacs in one or both lungs, which may fill with fluid or pus.",
            "severity": "High",
            "specialist": "Pulmonologist",
            "precautions": [
                "Consult a physician immediately for prescription antibiotic or antiviral treatment.",
                "Avoid smoking or exposure to secondhand smoke and air pollutants.",
                "Rest in an elevated head position to ease breathing.",
                "Complete the full course of prescribed medications strictly."
            ],
            "emergency_warning": "Bluish lips/fingertips, severe difficulty breathing, confusion in elderly patients."
        }
    },
    "Acute Bronchitis": {
        "primary": ["cough", "wheezing", "chest_pain", "fatigue"],
        "secondary": ["sore_throat", "fever", "shortness_of_breath"],
        "info": {
            "description": "Inflammation of the lining of bronchial tubes, often following a cold or viral illness.",
            "severity": "Moderate",
            "specialist": "Pulmonologist / General Physician",
            "precautions": [
                "Use a cool-mist humidifier in your sleeping area.",
                "Drink warm liquids to soothe chest and thin mucous secretions.",
                "Avoid respiratory irritants like tobacco smoke and harsh fumes.",
                "Get ample rest and avoid intense strenuous physical exercise."
            ],
            "emergency_warning": "Coughing up blood, wheezing causing respiratory distress, or fever above 101°F."
        }
    },
    "Asthma": {
        "primary": ["shortness_of_breath", "wheezing", "chest_pain", "cough"],
        "secondary": ["anxiety", "fatigue", "insomnia"],
        "info": {
            "description": "Chronic condition in which airways narrow, swell, and produce extra mucus, making breathing difficult.",
            "severity": "Moderate to High",
            "specialist": "Pulmonologist / Allergist",
            "precautions": [
                "Always keep your prescribed quick-relief inhaler (bronchodilator) accessible.",
                "Identify and stay away from personal asthma triggers (pollen, cold air, dust).",
                "Monitor peak expiratory flow rate if advised by your physician.",
                "Follow an asthma action plan agreed upon with your pulmonologist."
            ],
            "emergency_warning": "Inhaler fails to provide relief, severe chest constriction, or inability to speak full sentences."
        }
    },
    "Migraine": {
        "primary": ["headache", "nausea", "blurred_vision", "dizziness"],
        "secondary": ["fatigue", "vomiting", "anxiety", "insomnia"],
        "info": {
            "description": "Neurological condition causing severe throbbing headache, usually on one side of the head, often with nausea.",
            "severity": "Moderate",
            "specialist": "Neurologist",
            "precautions": [
                "Rest in a quiet, dark, and cool room during an attack.",
                "Apply cold compresses to the forehead or back of the neck.",
                "Maintain a headache diary to identify trigger foods, stress, or sleep lack.",
                "Stay consistently hydrated and avoid skipping meals."
            ],
            "emergency_warning": "Sudden explosive 'thunderclap' headache, fever with neck stiffness, or weakness on one side of body."
        }
    },
    "Hypertension (High Blood Pressure)": {
        "primary": ["high_blood_pressure", "headache", "dizziness"],
        "secondary": ["blurred_vision", "chest_pain", "palpitations", "anxiety"],
        "info": {
            "description": "Long-term medical condition in which blood pressure in arteries is persistently elevated.",
            "severity": "Moderate",
            "specialist": "Cardiologist / Internal Medicine",
            "precautions": [
                "Adopt a low-sodium (DASH) diet rich in vegetables, fruits, and lean protein.",
                "Engage in regular moderate aerobic exercise (e.g. 30 min daily walking).",
                "Monitor blood pressure regularly at home and keep a log.",
                "Take prescribed antihypertensive medications consistently without missing doses."
            ],
            "emergency_warning": "BP reading >180/120 mmHg, severe chest pain, shortness of breath, or numbness."
        }
    },
    "Type 2 Diabetes": {
        "primary": ["frequent_urination", "increased_thirst", "fatigue", "weight_loss"],
        "secondary": ["blurred_vision", "muscle_ache", "dizziness"],
        "info": {
            "description": "Chronic metabolic disorder characterized by high blood sugar level, insulin resistance, and relative insulin deficiency.",
            "severity": "Moderate",
            "specialist": "Endocrinologist",
            "precautions": [
                "Monitor blood glucose levels as recommended by your physician.",
                "Follow a balanced low-glycemic index dietary plan.",
                "Engage in regular physical activity to boost insulin sensitivity.",
                "Inspect feet daily for cuts, sores, or numbness."
            ],
            "emergency_warning": "Extremely high blood sugar (>300 mg/dL), confusion, fruity-scented breath, or persistent vomiting."
        }
    },
    "Malaria": {
        "primary": ["fever", "chills", "sweating", "headache", "fatigue"],
        "secondary": ["nausea", "vomiting", "muscle_ache", "joint_pain"],
        "info": {
            "description": "Mosquito-borne infectious disease caused by Plasmodium parasite, leading to high fever cycles and shaking chills.",
            "severity": "High",
            "specialist": "Infectious Disease Specialist / General Physician",
            "precautions": [
                "Seek immediate blood diagnostic testing (blood smear/RDT) for malaria parasites.",
                "Take prescribed antimalarial medications strictly as directed.",
                "Use insect repellent, mosquito nets, and wear long sleeves.",
                "Maintain adequate fluid intake to combat high fever fluid loss."
            ],
            "emergency_warning": "High cyclical fever, jaundice (yellow eyes), confusion, or severe anemia."
        }
    },
    "Dengue Fever": {
        "primary": ["fever", "headache", "joint_pain", "muscle_ache", "skin_rash"],
        "secondary": ["nausea", "vomiting", "fatigue", "swollen_lymph_nodes"],
        "info": {
            "description": "Mosquito-borne viral disease causing severe flu-like illness, severe muscle/joint aches ('breakbone fever'), and rash.",
            "severity": "High",
            "specialist": "Infectious Disease Specialist / Hematologist",
            "precautions": [
                "Monitor platelet count regularly via complete blood count (CBC).",
                "Rest completely and drink copious amounts of fluids/electrolyte solutions.",
                "Avoid NSAIDs (like ibuprofen/aspirin) as they increase bleeding risk; use paracetamol for fever.",
                "Protect against mosquito bites using screens and repellents."
            ],
            "emergency_warning": "Severe abdominal pain, persistent vomiting, bleeding gums/nose, or extreme lethargy."
        }
    },
    "Typhoid Fever": {
        "primary": ["fever", "headache", "abdominal_pain", "fatigue", "loss_of_appetite"],
        "secondary": ["constipation", "diarrhea", "skin_rash", "muscle_ache", "stomach_cramps"],
        "info": {
            "description": "Bacterial infection caused by Salmonella typhi, transmitted through contaminated food or water.",
            "severity": "High",
            "specialist": "Infectious Disease Specialist / Internal Medicine",
            "precautions": [
                "Consult doctor for confirmation via Widal test or blood culture and start prescribed antibiotics.",
                "Drink only boiled or bottled water and eat thoroughly cooked fresh food.",
                "Maintain strict personal sanitation and hand washing.",
                "Avoid street food and unpasteurized beverages."
            ],
            "emergency_warning": "Prolonged step-ladder high fever, severe abdominal swelling, delirium, or intestinal bleeding."
        }
    },
    "Urinary Tract Infection (UTI)": {
        "primary": ["burning_urination", "frequent_urination", "abdominal_pain"],
        "secondary": ["fever", "chills", "back_pain", "fatigue"],
        "info": {
            "description": "Infection in any part of the urinary system (kidneys, bladder, or urethra), most commonly involving lower tract.",
            "severity": "Moderate",
            "specialist": "Urologist / General Physician",
            "precautions": [
                "Drink plenty of water to flush bacteria out of your urinary tract.",
                "Urinate frequently and do not hold urine for long periods.",
                "Complete the full prescribed course of antibiotics.",
                "Avoid irritating beverages like coffee, alcohol, and citrus juices during infection."
            ],
            "emergency_warning": "High fever with shaking chills, severe lower back/flank pain, or blood in urine (kidney involvement)."
        }
    },
    "Rheumatoid Arthritis": {
        "primary": ["joint_pain", "swollen_joints", "stiff_neck", "fatigue"],
        "secondary": ["muscle_ache", "fever", "weight_loss"],
        "info": {
            "description": "Chronic inflammatory autoimmune disorder affecting joints, causing painful swelling and morning stiffness.",
            "severity": "Moderate to High",
            "specialist": "Rheumatologist",
            "precautions": [
                "Apply gentle warm compresses to ease morning joint stiffness.",
                "Perform gentle range-of-motion exercises recommended by a physical therapist.",
                "Follow disease-modifying antirheumatic drug (DMARD) regimens as prescribed.",
                "Maintain a balanced anti-inflammatory diet rich in omega-3 fatty acids."
            ],
            "emergency_warning": "Severe joint deformity, inability to bear weight, or systemic fever spikes."
        }
    },
    "Allergic Rhinitis (Hay Fever)": {
        "primary": ["sneezing", "runny_nose", "itchy_eyes", "sore_throat"],
        "secondary": ["headache", "fatigue", "cough"],
        "info": {
            "description": "Allergic response to airborne allergens (pollen, dust mites, pet dander) causing nasal congestion and itchy eyes.",
            "severity": "Low",
            "specialist": "Allergist / ENT Specialist",
            "precautions": [
                "Identify and minimize exposure to known environmental allergens.",
                "Use HEPA air purifiers indoors and keep windows closed during high pollen counts.",
                "Use OTC antihistamines or corticosteroid nasal sprays as advised.",
                "Rinse nasal passages with saline water rinse."
            ],
            "emergency_warning": "Wheezing onset or severe breathing difficulty (asthma crossover)."
        }
    },
    "Anxiety & Panic Disorder": {
        "primary": ["anxiety", "palpitations", "chest_pain", "shortness_of_breath"],
        "secondary": ["dizziness", "sweating", "insomnia", "fatigue"],
        "info": {
            "description": "Mental health condition marked by intense, excessive, and persistent worry, fear, or sudden panic attacks.",
            "severity": "Moderate",
            "specialist": "Psychiatrist / Clinical Psychologist",
            "precautions": [
                "Practice slow diaphragmatic deep breathing (box breathing 4-4-4-4).",
                "Limit caffeine, nicotine, and alcohol intake which trigger anxiety spikes.",
                "Engage in daily mindfulness meditation, yoga, or physical walk.",
                "Consider cognitive behavioral therapy (CBT) with a specialist."
            ],
            "emergency_warning": "Thoughts of self-harm, severe chest pain (must rule out cardiac event)."
        }
    },
    "Viral Hepatitis": {
        "primary": ["jaundice", "fatigue", "nausea", "abdominal_pain", "loss_of_appetite"],
        "secondary": ["fever", "joint_pain", "vomiting"],
        "info": {
            "description": "Inflammation of the liver caused by viral infection (Hepatitis A, B, C), leading to liver dysfunction.",
            "severity": "High",
            "specialist": "Hepatologist / Gastroenterologist",
            "precautions": [
                "Avoid alcohol and hepatotoxic drugs completely.",
                "Get liver function tests (LFTs) and viral panel blood diagnostics.",
                "Eat a nutrient-rich, low-fat diet with small meals.",
                "Ensure strict personal hygiene and safe water/food practices."
            ],
            "emergency_warning": "Yellowing of eyes/skin, confusion/slurred speech, dark urine, or abdominal swelling."
        }
    },
    "Tuberculosis (TB)": {
        "primary": ["cough", "fever", "sweating", "weight_loss", "fatigue"],
        "secondary": ["chest_pain", "shortness_of_breath", "chills", "loss_of_appetite"],
        "info": {
            "description": "Potentially serious infectious bacterial disease mainly affecting lungs (Mycobacterium tuberculosis).",
            "severity": "High",
            "specialist": "Pulmonologist / Infectious Disease",
            "precautions": [
                "Undergo sputum test, chest X-ray, and Mantoux tuberculin skin test.",
                "Comply strictly with anti-TB treatment (DOTS) without missing any dose for 6+ months.",
                "Wear masks and ensure room ventilation during initial infectious phase.",
                "Maintain high-protein nutrition to rebuild strength."
            ],
            "emergency_warning": "Coughing up blood (hemoptysis), dramatic weight loss, or severe breathlessness."
        }
    },
    "Anemia": {
        "primary": ["fatigue", "dizziness", "shortness_of_breath", "palpitations"],
        "secondary": ["headache", "blurred_vision", "muscle_ache"],
        "info": {
            "description": "Condition in which blood lacks adequate healthy red blood cells or hemoglobin to carry oxygen to tissues.",
            "severity": "Moderate",
            "specialist": "Hematologist / General Physician",
            "precautions": [
                "Get complete blood count (CBC), iron profile, and Vitamin B12 tested.",
                "Consume iron-rich foods (spinach, legumes, red meat, fortified cereals).",
                "Take vitamin C along with iron supplements to enhance absorption.",
                "Avoid drinking tea or coffee immediately with meals."
            ],
            "emergency_warning": "Fainting, rapid pounding pulse at rest, or extreme shortness of breath."
        }
    },
    "Sinusitis": {
        "primary": ["headache", "runny_nose", "fever", "sore_throat"],
        "secondary": ["cough", "fatigue", "loss_of_taste_smell"],
        "info": {
            "description": "Inflammation or swelling of the tissue lining the sinuses, causing facial pressure and nasal discharge.",
            "severity": "Low to Moderate",
            "specialist": "ENT Specialist / General Physician",
            "precautions": [
                "Use warm moist compresses over eyes, cheeks, and nose.",
                "Practice warm saline nasal irrigation daily.",
                "Stay well hydrated to thin mucus secretions.",
                "Avoid sudden temperature changes and tobacco smoke."
            ],
            "emergency_warning": "High fever, vision changes, eye swelling, or severe stiff neck."
        }
    },
    "Hypothyroidism": {
        "primary": ["fatigue", "constipation", "muscle_ache", "joint_pain"],
        "secondary": ["weight_loss", "insomnia", "anxiety", "blurred_vision"],
        "info": {
            "description": "Underactive thyroid gland where the body doesn't produce enough crucial thyroid hormones.",
            "severity": "Moderate",
            "specialist": "Endocrinologist",
            "precautions": [
                "Get Thyroid Function Test (TSH, Free T3, Free T4) done.",
                "Take prescribed levothyroxine medication on an empty stomach every morning.",
                "Wait at least 30-60 minutes before eating breakfast or drinking coffee after thyroid medication.",
                "Have thyroid blood levels re-checked periodically as directed."
            ],
            "emergency_warning": "Severe cold intolerance, extreme lethargy, slow heart rate, or facial swelling."
        }
    }
}

# 3. Synthetic Dataset Generation Engine (Generate 1800 realistic patient records)
def generate_dataset(num_samples_per_disease=60):
    records = []
    np.random.seed(42)

    for disease_name, profile in DISEASE_PROFILES.items():
        primary = profile["primary"]
        secondary = profile["secondary"]

        for _ in range(num_samples_per_disease):
            record = {}
            for s in SYMPTOM_IDS:
                if s in primary:
                    record[s] = 1 if np.random.rand() < 0.90 else 0
                elif s in secondary:
                    record[s] = 1 if np.random.rand() < 0.45 else 0
                else:
                    record[s] = 1 if np.random.rand() < 0.03 else 0

            present_primary = [s for s in primary if record[s] == 1]
            if len(present_primary) < 2:
                for p_sym in np.random.choice(primary, size=2, replace=False):
                    record[p_sym] = 1

            record["disease"] = disease_name
            records.append(record)

    df = pd.DataFrame(records)
    return df

# 4. Train Model Pipeline
print("Generating enhanced dataset with Stomach Pain & Loose Motion diseases...")
df = generate_dataset()
dataset_path = os.path.join(MODEL_DIR, "disease_dataset.csv")
df.to_csv(dataset_path, index=False)
print(f"Dataset saved to {dataset_path} with {len(df)} samples across {len(DISEASE_PROFILES)} diseases.")

# Save Disease Meta Knowledgebase
info_path = os.path.join(MODEL_DIR, "disease_info.json")
with open(info_path, "w") as f:
    disease_meta = {d: profile["info"] for d, profile in DISEASE_PROFILES.items()}
    json.dump({"diseases": disease_meta, "symptoms": SYMPTOMS_CATALOG}, f, indent=2)
print(f"Disease knowledgebase saved to {info_path}")

# Prepare X and y
X = df[SYMPTOM_IDS]
y = df["disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

base_rf = RandomForestClassifier(
    n_estimators=160,
    max_depth=18,
    min_samples_split=3,
    class_weight="balanced",
    random_state=42
)

model = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Enhanced Model Accuracy on Test Set: {acc * 100:.2f}%")

cv_scores = cross_val_score(model, X, y, cv=5)
print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

model_path = os.path.join(MODEL_DIR, "model.joblib")
encoder_path = os.path.join(MODEL_DIR, "symptom_encoder.joblib")
joblib.dump(model, model_path, compress=3)
joblib.dump(SYMPTOM_IDS, encoder_path)
print("Model and symptom encoder re-saved successfully with compression!")

