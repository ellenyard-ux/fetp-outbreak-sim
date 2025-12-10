import streamlit as st
import anthropic
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import json

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="FETP Sim: Sidero Valley JE Outbreak",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a5276 0%, #2e86ab 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .alert-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 0 5px 5px 0;
    }
    .transcript-box {
        background-color: #f8f9fa;
        border-left: 4px solid #dc3545;
        padding: 15px;
        font-family: 'Georgia', serif;
        margin-bottom: 15px;
        font-style: italic;
    }
    .handwritten-note {
        font-family: 'Comic Sans MS', 'Chalkboard SE', cursive;
        font-size: 15px;
        background: linear-gradient(to bottom, #fdf6e3 0%, #fef9e7 100%);
        color: #2c3e50;
        padding: 12px 15px;
        border: 1px solid #d5dbdb;
        box-shadow: 3px 3px 8px rgba(0,0,0,0.15);
        margin-bottom: 12px;
        transform: rotate(-0.5deg);
        line-height: 1.6;
    }
    .lab-result {
        font-family: 'Courier New', monospace;
        background-color: #f4f6f7;
        border: 1px solid #aab7b8;
        padding: 10px;
        margin: 5px 0;
    }
    .resource-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .competency-tag {
        background-color: #e8f6f3;
        color: #1abc9c;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 5px;
    }
    .village-card {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .village-card:hover {
        border-color: #3498db;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .case-severe { color: #e74c3c; font-weight: bold; }
    .case-mild { color: #f39c12; }
    .case-recovered { color: #27ae60; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    defaults = {
        # Navigation
        'current_view': 'briefing',
        'investigation_day': 1,
        'game_phase': 'initial',  # initial, field, analysis, report
        
        # Resources
        'budget': 3000,
        'lab_credits': 15,
        'field_days': 14,
        
        # Data Access Flags
        'hospital_data_accessed': True,  # Public health mandate
        'health_center_nalu_unlocked': False,
        'health_center_kabwe_unlocked': False,
        'community_trust_nalu': 0,
        'community_trust_kabwe': 0,
        'community_trust_tamu': 0,
        
        # Interview System
        'interview_history': {},
        'current_character': None,
        'clues_discovered': set(),
        
        # Investigation Progress
        'case_definition_written': False,
        'case_definition_text': "",
        'hypothesis_written': False,
        'hypothesis_text': "",
        'epi_curve_built': False,
        'spot_map_created': False,
        
        # Lab System
        'samples_collected': [],
        'lab_results': [],
        'pending_labs': [],
        
        # Field Work
        'sites_inspected': [],
        'environmental_samples': [],
        'household_surveys_done': 0,
        
        # Case Registry (what trainee has found)
        'confirmed_cases': [],
        'suspected_cases': [],
        'manually_entered_cases': [],
        
        # Study Design
        'study_type': None,
        'questionnaire_deployed': False,
        'field_data_collected': None,
        
        # Scoring/Debrief
        'actions_log': [],
        'critical_findings': set(),
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# GROUND TRUTH DATA GENERATION (JE OUTBREAK MODEL)
# ============================================================================
@st.cache_data
def generate_outbreak_data():
    """
    Generate a realistic JE outbreak following the epidemiological framework.
    This is the TRUTH - trainees must discover it through investigation.
    """
    np.random.seed(42)
    
    # ----- VILLAGES -----
    villages = pd.DataFrame({
        'village_id': ['V1', 'V2', 'V3'],
        'village_name': ['Nalu Village', 'Kabwe Village', 'Tamu Village'],
        'has_rice_paddies': [True, True, False],
        'pig_density': ['High', 'Moderate', 'Low'],
        'distance_to_wetland_km': [0.4, 1.2, 2.5],
        'JE_vacc_coverage': [0.22, 0.40, 0.55],
        'population_size': [480, 510, 390],
        'baseline_risk': [0.18, 0.07, 0.02],
        'zone_description': [
            'Dense rice paddies, large pig cooperative near school, wetlands nearby',
            'Scattered pig farms, intermittent irrigation canals',
            'Dryland farming, fewer pigs, seasonal pools only'
        ]
    })
    
    # ----- HOUSEHOLDS -----
    households = []
    hh_id = 1
    
    for _, village in villages.iterrows():
        n_households = village['population_size'] // 4  # ~4 people per household
        
        for _ in range(n_households):
            # Pig ownership varies by village
            if village['pig_density'] == 'High':
                pigs = np.random.choice([0, 2, 4, 6, 8, 10], p=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05])
                pig_distance = np.random.choice([10, 15, 20, 30, 50]) if pigs > 0 else None
            elif village['pig_density'] == 'Moderate':
                pigs = np.random.choice([0, 1, 2, 3, 5], p=[0.35, 0.25, 0.20, 0.15, 0.05])
                pig_distance = np.random.choice([20, 30, 40, 60]) if pigs > 0 else None
            else:
                pigs = np.random.choice([0, 1, 2], p=[0.70, 0.20, 0.10])
                pig_distance = np.random.choice([50, 80, 100]) if pigs > 0 else None
            
            # Mosquito net use inversely related to vaccination coverage (proxy for health access)
            net_use = np.random.random() < (0.3 + village['JE_vacc_coverage'])
            
            # Rice field distance
            if village['has_rice_paddies']:
                rice_dist = np.random.choice([30, 50, 80, 120, 200])
            else:
                rice_dist = np.random.choice([300, 500, 800])
            
            # Children
            n_children = np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.25, 0.30, 0.20, 0.10])
            
            # Child vaccination (correlated with village coverage)
            child_vacc = np.random.choice(
                ['none', 'partial', 'full'],
                p=[1 - village['JE_vacc_coverage'], village['JE_vacc_coverage'] * 0.6, village['JE_vacc_coverage'] * 0.4]
            )
            
            households.append({
                'hh_id': f'HH{hh_id:03d}',
                'village_id': village['village_id'],
                'village_name': village['village_name'],
                'pigs_owned': pigs,
                'pig_pen_distance_m': pig_distance,
                'uses_mosquito_nets': net_use,
                'rice_field_distance_m': rice_dist,
                'n_children_under_15': n_children,
                'child_vaccination_status': child_vacc,
                'gps_lat': 8.5 + np.random.uniform(-0.05, 0.05),
                'gps_lon': -12.3 + np.random.uniform(-0.05, 0.05)
            })
            hh_id += 1
    
    households_df = pd.DataFrame(households)
    
    # ----- INDIVIDUALS -----
    individuals = []
    person_id = 1
    
    for _, hh in households_df.iterrows():
        # Generate household members
        n_adults = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])
        n_children = hh['n_children_under_15']
        
        # Adults
        for i in range(n_adults):
            age = np.random.randint(18, 65)
            sex = 'M' if i == 0 and np.random.random() < 0.6 else np.random.choice(['M', 'F'])
            occupation = np.random.choice(['farmer', 'trader', 'teacher', 'healthcare', 'other'],
                                         p=[0.50, 0.20, 0.10, 0.05, 0.15])
            
            # JE vaccination (adults less likely)
            vaccinated = np.random.random() < (villages[villages['village_id'] == hh['village_id']]['JE_vacc_coverage'].values[0] * 0.5)
            
            # Evening outdoor exposure (farmers high, others moderate)
            evening_outdoor = np.random.random() < (0.8 if occupation == 'farmer' else 0.4)
            
            individuals.append({
                'person_id': f'P{person_id:04d}',
                'hh_id': hh['hh_id'],
                'village_id': hh['village_id'],
                'village_name': hh['village_name'],
                'age': age,
                'sex': sex,
                'occupation': occupation,
                'JE_vaccinated': vaccinated,
                'evening_outdoor_exposure': evening_outdoor,
                'pigs_near_home': hh['pigs_owned'] > 0 and (hh['pig_pen_distance_m'] or 100) < 30,
                'uses_nets': hh['uses_mosquito_nets'],
                'rice_field_nearby': hh['rice_field_distance_m'] < 100
            })
            person_id += 1
        
        # Children
        for i in range(n_children):
            age = np.random.randint(1, 15)
            sex = np.random.choice(['M', 'F'])
            
            # Child vaccination status
            if hh['child_vaccination_status'] == 'full':
                vaccinated = True
            elif hh['child_vaccination_status'] == 'partial':
                vaccinated = np.random.random() < 0.5
            else:
                vaccinated = False
            
            # Children often play outside in evenings
            evening_outdoor = np.random.random() < 0.7
            
            individuals.append({
                'person_id': f'P{person_id:04d}',
                'hh_id': hh['hh_id'],
                'village_id': hh['village_id'],
                'village_name': hh['village_name'],
                'age': age,
                'sex': sex,
                'occupation': 'child' if age < 6 else 'student',
                'JE_vaccinated': vaccinated,
                'evening_outdoor_exposure': evening_outdoor,
                'pigs_near_home': hh['pigs_owned'] > 0 and (hh['pig_pen_distance_m'] or 100) < 30,
                'uses_nets': hh['uses_mosquito_nets'],
                'rice_field_nearby': hh['rice_field_distance_m'] < 100
            })
            person_id += 1
    
    individuals_df = pd.DataFrame(individuals)
    
    # ----- ASSIGN JE INFECTIONS AND DISEASE -----
    # Get village baseline risks
    village_risk = dict(zip(villages['village_id'], villages['baseline_risk']))
    
    # Calculate individual risk
    def calculate_risk(row):
        base = village_risk[row['village_id']]
        
        # Risk modifiers
        if row['pigs_near_home']:
            base += 0.08
        if not row['uses_nets']:
            base += 0.05
        if row['rice_field_nearby']:
            base += 0.04
        if row['evening_outdoor_exposure']:
            base += 0.03
        if row['JE_vaccinated']:
            base *= 0.15  # 85% vaccine efficacy
        
        return min(base, 0.4)  # Cap at 40%
    
    individuals_df['infection_risk'] = individuals_df.apply(calculate_risk, axis=1)
    individuals_df['true_JE_infected'] = individuals_df['infection_risk'].apply(lambda x: np.random.random() < x)
    
    # Symptomatic disease (only among infected)
    def assign_symptomatic(row):
        if not row['true_JE_infected']:
            return False
        # Children much more likely to be symptomatic
        p_symp = 0.15 if row['age'] < 15 else 0.05
        return np.random.random() < p_symp
    
    individuals_df['symptomatic_AES'] = individuals_df.apply(assign_symptomatic, axis=1)
    
    # Severe neurological disease (among symptomatic)
    individuals_df['severe_neuro'] = individuals_df['symptomatic_AES'].apply(
        lambda x: np.random.random() < 0.25 if x else False
    )
    
    # Onset dates (clustered by village)
    def assign_onset(row):
        if not row['symptomatic_AES']:
            return None
        
        base_date = datetime(2024, 6, 1)
        if row['village_id'] == 'V1':  # Nalu - first cluster
            offset = np.random.randint(2, 8)
        elif row['village_id'] == 'V2':  # Kabwe - second cluster
            offset = np.random.randint(5, 12)
        else:  # Tamu - sporadic
            offset = np.random.randint(4, 14)
        
        return base_date + timedelta(days=offset)
    
    individuals_df['onset_date'] = individuals_df.apply(assign_onset, axis=1)
    
    # Outcomes
    def assign_outcome(row):
        if not row['symptomatic_AES']:
            return None
        if row['severe_neuro']:
            r = np.random.random()
            if r < 0.20:
                return 'died'
            elif r < 0.65:
                return 'recovered_sequelae'
            else:
                return 'recovered_full'
        else:
            return 'recovered_full' if np.random.random() < 0.95 else 'recovered_sequelae'
    
    individuals_df['outcome'] = individuals_df.apply(assign_outcome, axis=1)
    
    # Clinical presentation
    def assign_symptoms(row):
        if not row['symptomatic_AES']:
            return None
        
        base_symptoms = ['fever', 'headache']
        
        if row['severe_neuro']:
            base_symptoms.extend(['seizures', 'altered_consciousness'])
            if np.random.random() < 0.5:
                base_symptoms.append('neck_stiffness')
            if np.random.random() < 0.3:
                base_symptoms.append('tremors')
            if np.random.random() < 0.4:
                base_symptoms.append('paralysis')
        else:
            if np.random.random() < 0.6:
                base_symptoms.append('vomiting')
            if np.random.random() < 0.3:
                base_symptoms.append('mild_confusion')
        
        return ', '.join(base_symptoms)
    
    individuals_df['symptoms'] = individuals_df.apply(assign_symptoms, axis=1)
    
    # ----- ENVIRONMENTAL SITES -----
    env_sites = pd.DataFrame({
        'site_id': ['ES01', 'ES02', 'ES03', 'ES04', 'ES05', 'ES06'],
        'site_type': ['rice_paddy', 'pig_cooperative', 'irrigation_canal', 'seasonal_pool', 'pig_farm', 'wetland'],
        'village_id': ['V1', 'V1', 'V2', 'V3', 'V2', 'V1'],
        'breeding_index': ['high', 'high', 'medium', 'low', 'medium', 'high'],
        'culex_present': [True, True, True, True, True, True],
        'JEV_positive_mosquitoes': [True, True, True, False, True, True],
        'description': [
            'Flooded rice paddies 200m from school, expanded 2 months ago',
            'New pig cooperative with 80+ pigs, built near residential area',
            'Irrigation canal system, stagnant water sections',
            'Seasonal rain pools, dry most of year',
            'Small pig farm, 15 pigs, proper drainage',
            'Natural wetland area, traditional fishing spot'
        ]
    })
    
    # ----- LAB SAMPLES (Ground Truth) -----
    # These exist but trainee must collect them
    lab_truth = pd.DataFrame({
        'sample_id': ['L001', 'L002', 'L003', 'L004', 'L005', 'L006', 'L101', 'L102', 'L201', 'L202', 'L203'],
        'sample_type': ['human_CSF', 'human_serum', 'human_CSF', 'human_serum', 'human_CSF', 'human_serum',
                       'pig_serum', 'pig_serum', 'mosquito_pool', 'mosquito_pool', 'mosquito_pool'],
        'source_village': ['V1', 'V1', 'V1', 'V2', 'V2', 'V3', 'V1', 'V2', 'V1', 'V2', 'V3'],
        'true_JEV_positive': [True, True, True, True, True, False, True, True, True, True, False],
        'notes': [
            'Child case, severe', 'Child case, mild', 'Adult case, severe',
            'Child case, severe', 'Child case, mild', 'Febrile illness, not JE',
            'Pig cooperative sample', 'Scattered farm sample',
            'Rice paddy collection', 'Canal collection', 'Seasonal pool'
        ]
    })
    
    return {
        'villages': villages,
        'households': households_df,
        'individuals': individuals_df,
        'env_sites': env_sites,
        'lab_truth': lab_truth
    }

# Load the ground truth
TRUTH = generate_outbreak_data()

# ============================================================================
# WHAT TRAINEES CAN SEE (DISCOVERED DATA)
# ============================================================================
def get_hospital_cases():
    """
    Initial hospital line list - severe cases that presented to district hospital.
    This is always available (public health reporting).
    """
    # Get severe cases from truth data
    severe = TRUTH['individuals'][
        (TRUTH['individuals']['severe_neuro'] == True) & 
        (TRUTH['individuals']['onset_date'].notna())
    ].copy()
    
    # Hospital only has partial info
    hospital_list = []
    for _, case in severe.head(8).iterrows():  # First 8 severe cases
        hospital_list.append({
            'case_id': f"DH-{len(hospital_list)+1:02d}",
            'age': case['age'],
            'sex': case['sex'],
            'village': case['village_name'],
            'onset_date': case['onset_date'].strftime('%b %d') if case['onset_date'] else 'Unknown',
            'symptoms': case['symptoms'],
            'outcome': case['outcome'].replace('_', ' ').title() if case['outcome'] else 'Unknown',
            'occupation': case['occupation']
        })
    
    return pd.DataFrame(hospital_list)

def get_health_center_notes(village_id):
    """
    Health center handwritten notes - mild cases not reported to district.
    Must be unlocked by interviewing the right person.
    """
    mild_cases = TRUTH['individuals'][
        (TRUTH['individuals']['symptomatic_AES'] == True) & 
        (TRUTH['individuals']['severe_neuro'] == False) &
        (TRUTH['individuals']['village_id'] == village_id)
    ]
    
    notes = []
    for _, case in mild_cases.iterrows():
        onset = case['onset_date'].strftime('%b %d') if case['onset_date'] else '???'
        age = case['age']
        sex = case['sex']
        symptoms = case['symptoms'].replace('_', ' ') if case['symptoms'] else 'fever'
        
        # Handwritten style notes
        note_templates = [
            f"{onset}. {sex}, {age}y. {symptoms.split(',')[0]}. Sent home.",
            f"{onset} - {age}yo {sex}. Came w/ {symptoms.split(',')[0]}. Gave paracetamol.",
            f"{onset}: Child {age}y ({sex}). Mother worried - {symptoms}. Advised rest.",
        ]
        notes.append(np.random.choice(note_templates))
    
    return notes

# ============================================================================
# CHARACTERS FOR INTERVIEWS
# ============================================================================
CHARACTERS = {
    # Hospital/Health System
    "dr_okonkwo": {
        "name": "Dr. Okonkwo",
        "role": "District Medical Officer",
        "avatar": "👨‍⚕️",
        "cost": 0,  # Free - your supervisor
        "location": "District Hospital",
        "personality": "Methodical, concerned about outbreak response capacity",
        "knowledge": [
            "Has the official hospital line list",
            "Knows there are 8 severe AES cases, mostly children",
            "Worried about lab capacity - only 3 CSF samples sent to national lab",
            "Heard rumors of similar cases in villages not coming to hospital",
            "Suspects mosquito-borne illness but waiting for lab confirmation"
        ],
        "data_access": "hospital_cases",
        "unlocks": None,
        "trust_threshold": 0
    },
    
    "nurse_fatima": {
        "name": "Nurse Fatima",
        "role": "Nalu Health Center",
        "avatar": "👩‍⚕️",
        "cost": 100,
        "location": "Nalu Village",
        "personality": "Overworked, protective of community, keeps detailed notes",
        "knowledge": [
            "Has seen 6+ mild fever cases with headaches in past 2 weeks",
            "Notes that many cases are children who play near the new pig cooperative",
            "Knows the cooperative was built 2 months ago near the school",
            "Has noticed more mosquitoes since rice paddies expanded",
            "Doesn't always report mild cases - 'they recover anyway'"
        ],
        "data_access": "health_center_nalu",
        "unlocks": "health_center_nalu_unlocked",
        "trust_threshold": 0
    },
    
    "health_worker_joseph": {
        "name": "Joseph",
        "role": "Kabwe Community Health Worker",
        "avatar": "🧑‍⚕️",
        "cost": 100,
        "location": "Kabwe Village",
        "personality": "Enthusiastic, young, good with data",
        "knowledge": [
            "Has a notebook with 4 fever cases from Kabwe",
            "Noticed cases started a few days after Nalu",
            "Says some Kabwe farmers bought pigs from Nalu cooperative",
            "Knows which households have pigs and which use bed nets",
            "Can help with household surveys if asked"
        ],
        "data_access": "health_center_kabwe",
        "unlocks": "health_center_kabwe_unlocked",
        "trust_threshold": 0
    },
    
    # Community Members
    "mama_adama": {
        "name": "Mama Adama",
        "role": "Mother of Deceased Child",
        "avatar": "👵",
        "cost": 50,
        "location": "Nalu Village",
        "personality": "Grieving, angry at lack of response, wants answers",
        "knowledge": [
            "Her 6-year-old daughter Ama died on June 8",
            "Ama played every evening near the pig pens with friends",
            "Three of Ama's playmates are also sick",
            "Family does not use mosquito nets - 'too hot'",
            "Heard from neighbors that children who got 'the injection' are okay"
        ],
        "data_access": None,
        "unlocks": None,
        "trust_threshold": 0
    },
    
    "chief_boateng": {
        "name": "Chief Boateng",
        "role": "Nalu Village Chief",
        "avatar": "👑",
        "cost": 150,
        "location": "Nalu Village",
        "personality": "Diplomatic, worried about village reputation, protective",
        "knowledge": [
            "Approved the pig cooperative 3 months ago - big investment",
            "Knows which families are affected but doesn't want panic",
            "Can authorize household surveys if approached respectfully",
            "Remembers a similar outbreak '10 years ago' - some children died",
            "Will mention that Tamu village has fewer problems - 'they have the clinic program'"
        ],
        "data_access": None,
        "unlocks": "community_trust_nalu",
        "trust_threshold": 0
    },
    
    "mr_owusu": {
        "name": "Mr. Owusu",
        "role": "Pig Cooperative Manager",
        "avatar": "🐷",
        "cost": 200,
        "location": "Nalu Village",
        "personality": "Defensive about pigs, business-focused, worried about blame",
        "knowledge": [
            "Cooperative has 85 pigs, built 2 months ago",
            "Located 150m from the primary school",
            "Some pigs have been 'off their feed' lately - didn't report it",
            "Sold 12 pigs to Kabwe farmers 3 weeks ago",
            "Will initially deny any connection to illness",
            "If pressed, admits a veterinarian mentioned something about 'pig encephalitis'"
        ],
        "data_access": None,
        "unlocks": None,
        "trust_threshold": 0
    },
    
    "teacher_grace": {
        "name": "Teacher Grace",
        "role": "Nalu Primary School",
        "avatar": "📚",
        "cost": 50,
        "location": "Nalu Village",
        "personality": "Observant, cares deeply about students, keeps attendance records",
        "knowledge": [
            "Has attendance records - 12 students absent in past 2 weeks",
            "Noticed pattern: absences cluster among students from homes near pig cooperative",
            "Students who were vaccinated last year seem fine",
            "School is next to rice paddies - 'mosquitoes terrible at dismissal time'",
            "Children play in the fields near pig pens after school"
        ],
        "data_access": "school_attendance",
        "unlocks": None,
        "trust_threshold": 0
    },
    
    # Technical Experts
    "dr_mensah": {
        "name": "Dr. Mensah",
        "role": "District Veterinary Officer",
        "avatar": "🐄",
        "cost": 150,
        "location": "District Office",
        "personality": "Technical, One Health advocate, frustrated at lack of coordination",
        "knowledge": [
            "Has been tracking febrile illness in pigs for 3 weeks",
            "Suspects JE based on pig symptoms and seasonality",
            "Knows Culex mosquitoes breed in rice paddies - peak season now",
            "Can collect pig serum samples for JE testing",
            "Emphasizes pigs are amplifying hosts, not direct source",
            "Knows JE vaccine exists and is effective"
        ],
        "data_access": "vet_surveillance",
        "unlocks": None,
        "trust_threshold": 0
    },
    
    "entomologist_kwame": {
        "name": "Mr. Kwame",
        "role": "District Entomologist",
        "avatar": "🦟",
        "cost": 150,
        "location": "District Office",
        "personality": "Enthusiastic about vectors, data-driven",
        "knowledge": [
            "Has mosquito surveillance data showing Culex tritaeniorhynchus increase",
            "Peak breeding in rice paddies - expansion made it worse",
            "Can do mosquito pool testing for JE virus",
            "Knows the vector ecology - evening/night biting",
            "Recommends bed net distribution and source reduction"
        ],
        "data_access": "vector_surveillance",
        "unlocks": None,
        "trust_threshold": 0
    }
}

# ============================================================================
# AI INTERVIEW FUNCTION
# ============================================================================
def get_ai_response(char_key, user_input, history):
    """Generate character response using Claude API."""
    char = CHARACTERS[char_key]
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    
    if not api_key:
        return "⚠️ API Key not configured. Please add ANTHROPIC_API_KEY to your Streamlit secrets."
    
    # Build knowledge context
    knowledge_text = "\n".join([f"- {k}" for k in char['knowledge']])
    
    # Get any data this character can share
    data_context = ""
    if char['data_access'] == 'hospital_cases':
        data_context = f"\nHOSPITAL DATA YOU CAN SHARE:\n{get_hospital_cases().to_string()}"
    elif char['data_access'] == 'health_center_nalu' and st.session_state.health_center_nalu_unlocked:
        notes = get_health_center_notes('V1')
        data_context = f"\nYOUR CLINIC NOTES:\n" + "\n".join(notes[:5])
    elif char['data_access'] == 'health_center_kabwe' and st.session_state.health_center_kabwe_unlocked:
        notes = get_health_center_notes('V2')
        data_context = f"\nYOUR NOTEBOOK:\n" + "\n".join(notes[:4])
    
    system_prompt = f"""You are roleplaying as {char['name']}, {char['role']} in Sidero Valley district.

PERSONALITY: {char['personality']}
LOCATION: {char['location']}

YOUR KNOWLEDGE (share naturally when relevant):
{knowledge_text}
{data_context}

CRITICAL INSTRUCTIONS:
1. Stay in character at all times
2. Share information gradually - don't dump everything at once
3. Respond naturally to questions - expand on relevant knowledge
4. If asked about something you don't know, say so realistically
5. Be helpful but realistic - you're a real person with concerns and opinions
6. Keep responses concise (2-4 sentences typically)
7. If the investigator builds rapport, share more details
8. Reference specific data ONLY if you have it in your knowledge/data sections

SETTING: This is a disease outbreak investigation. The interviewer is from the FETP (Field Epidemiology Training Program) investigating acute encephalitis cases in your area. It is currently mid-June.
"""
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": user_input})
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=system_prompt,
            messages=messages
        )
        
        # Check for unlocks
        if char.get('unlocks'):
            st.session_state[char['unlocks']] = True
        
        return response.content[0].text
    
    except Exception as e:
        return f"[Interview connection error: {str(e)}]"

# ============================================================================
# LABORATORY SYSTEM
# ============================================================================
def process_lab_sample(sample_type, source, village_id):
    """
    Process a laboratory sample and return results based on ground truth.
    Incorporates realistic test performance (sensitivity/specificity).
    """
    # Find matching truth data
    truth = TRUTH['lab_truth']
    matching = truth[
        (truth['sample_type'] == sample_type) & 
        (truth['source_village'] == village_id)
    ]
    
    if len(matching) == 0:
        # No matching sample in truth - return negative
        true_status = False
    else:
        true_status = matching.iloc[0]['true_JEV_positive']
    
    # Apply test characteristics
    test_chars = {
        'human_CSF': {'sensitivity': 0.85, 'specificity': 0.98, 'test': 'JE IgM ELISA'},
        'human_serum': {'sensitivity': 0.80, 'specificity': 0.95, 'test': 'JE IgM ELISA'},
        'pig_serum': {'sensitivity': 0.90, 'specificity': 0.95, 'test': 'JE IgG/IgM'},
        'mosquito_pool': {'sensitivity': 0.95, 'specificity': 0.98, 'test': 'JE RT-PCR'}
    }
    
    chars = test_chars.get(sample_type, {'sensitivity': 0.80, 'specificity': 0.95, 'test': 'Standard'})
    
    # Generate result based on test performance
    if true_status:
        # True positive or false negative
        result_positive = np.random.random() < chars['sensitivity']
    else:
        # True negative or false positive
        result_positive = np.random.random() > chars['specificity']
    
    return {
        'sample_id': f"LAB-{len(st.session_state.lab_results) + 1:03d}",
        'sample_type': sample_type,
        'source': source,
        'village': village_id,
        'test_performed': chars['test'],
        'result': 'POSITIVE' if result_positive else 'NEGATIVE',
        'turnaround_days': np.random.randint(2, 5),
        'true_status': true_status  # Hidden from trainee
    }

# ============================================================================
# MAIN APPLICATION UI
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🦟 Sidero Valley Outbreak Investigation</h1>
    <p style="margin:0; opacity:0.9;">FETP Intermediate 2.0 | Acute Encephalitis Syndrome Cluster</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 📊 Investigation Dashboard")
    
    # Resources
    st.markdown("**Resources**")
    col1, col2 = st.columns(2)
    col1.metric("Budget", f"${st.session_state.budget:,}")
    col2.metric("Lab Credits", st.session_state.lab_credits)
    
    st.metric("Investigation Day", st.session_state.investigation_day)
    
    st.markdown("---")
    
    # Learning Objectives (collapsible)
    with st.expander("🎯 Learning Objectives"):
        st.markdown("""
        **By completing this simulation, you will practice:**
        
        1. Developing a case definition for AES
        2. Conducting key informant interviews
        3. Building and interpreting an epidemic curve
        4. Formulating and testing hypotheses
        5. Integrating One Health data sources
        6. Designing a case-control study
        7. Interpreting laboratory results
        """)
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    
    nav_options = {
        'briefing': '📞 Briefing',
        'interviews': '👥 Interviews',
        'linelist': '📋 Line List',
        'epicurve': '📈 Epi Curve',
        'map': '🗺️ Field Map',
        'laboratory': '🧪 Laboratory',
        'study_design': '🔬 Study Design',
        'debrief': '📝 Debrief'
    }
    
    for key, label in nav_options.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.current_view = key
            st.rerun()
    
    st.markdown("---")
    
    # Progress Tracking
    with st.expander("📋 Progress Checklist"):
        checks = {
            'Case definition written': st.session_state.case_definition_written,
            'Hospital data reviewed': st.session_state.hospital_data_accessed,
            'Nalu health center accessed': st.session_state.health_center_nalu_unlocked,
            'Kabwe health center accessed': st.session_state.health_center_kabwe_unlocked,
            'Epi curve built': st.session_state.epi_curve_built,
            'Lab samples collected': len(st.session_state.lab_results) > 0,
            'Study deployed': st.session_state.questionnaire_deployed
        }
        
        for item, done in checks.items():
            icon = "✅" if done else "⬜"
            st.markdown(f"{icon} {item}")

# ============================================================================
# MAIN CONTENT VIEWS
# ============================================================================

if st.session_state.current_view == 'briefing':
    st.markdown("### 🚨 Incoming Alert")
    
    st.markdown("""
    <div class="transcript-box">
    <strong>From:</strong> District Health Officer<br>
    <strong>Date:</strong> June 12, 2024<br>
    <strong>Subject:</strong> URGENT - AES Cluster in Sidero Valley<br><br>
    
    "We have 8 confirmed cases of acute encephalitis syndrome, including 2 deaths. 
    Most cases are children under 15. The first cases appeared about 10 days ago in 
    Nalu Village. There may be more cases that haven't reached us.<br><br>
    
    Your mission: Investigate this outbreak, identify the source, and recommend 
    control measures. You have 2 weeks and a budget of $3,000.<br><br>
    
    Start by talking to Dr. Okonkwo at the District Hospital. Good luck."
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📍 Sidero Valley Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Simple map showing villages
        fig = go.Figure()
        
        # Village markers
        villages_display = [
            {'name': 'Nalu Village', 'x': 100, 'y': 300, 'cases': 'Multiple cases', 'color': 'red'},
            {'name': 'Kabwe Village', 'x': 250, 'y': 200, 'cases': 'Some cases', 'color': 'orange'},
            {'name': 'Tamu Village', 'x': 350, 'y': 100, 'cases': 'Few cases', 'color': 'yellow'},
            {'name': 'District Hospital', 'x': 300, 'y': 350, 'cases': 'You are here', 'color': 'blue'}
        ]
        
        for v in villages_display:
            fig.add_trace(go.Scatter(
                x=[v['x']], y=[v['y']],
                mode='markers+text',
                marker=dict(size=25, color=v['color'], symbol='circle'),
                text=[v['name']],
                textposition='top center',
                name=v['name'],
                hovertext=v['cases']
            ))
        
        # Rice paddies (green areas)
        fig.add_shape(type="rect", x0=50, y0=250, x1=150, y1=350,
                     fillcolor="rgba(144,238,144,0.4)", line_width=0)
        fig.add_annotation(x=100, y=340, text="Rice Paddies", showarrow=False, font=dict(size=10))
        
        # River
        fig.add_trace(go.Scatter(
            x=[0, 400], y=[150, 200],
            mode='lines',
            line=dict(color='blue', width=3),
            name='River',
            showlegend=False
        ))
        
        fig.update_layout(
            height=400,
            xaxis=dict(visible=False, range=[0, 400]),
            yaxis=dict(visible=False, range=[0, 400]),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Quick Facts:**")
        st.markdown("""
        - **Population:** ~1,400 across 3 villages
        - **Main livelihoods:** Rice farming, pig rearing
        - **Season:** Rainy season (peak mosquito breeding)
        - **Health facilities:** 1 district hospital, 2 health centers
        """)
        
        st.markdown("**Your First Steps:**")
        st.markdown("""
        1. Review hospital case data
        2. Talk to Dr. Okonkwo
        3. Develop initial hypothesis
        4. Plan field investigation
        """)
    
    st.markdown("---")
    
    # Case Definition Exercise
    st.markdown("### 📝 Develop Your Case Definition")
    st.info("Before investigating further, write a working case definition for this outbreak.")
    
    with st.form("case_definition_form"):
        st.markdown("**Clinical Criteria:**")
        clinical = st.text_area(
            "What symptoms/signs define a case?",
            placeholder="e.g., Acute onset of fever AND altered mental status OR seizures...",
            height=80
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Person:**")
            person = st.text_input("Who? (age, other characteristics)", placeholder="e.g., Any age, resident of...")
        
        with col2:
            st.markdown("**Place:**")
            place = st.text_input("Where?", placeholder="e.g., Sidero Valley district")
        
        st.markdown("**Time:**")
        time_period = st.text_input("When?", placeholder="e.g., Onset between June 1-30, 2024")
        
        if st.form_submit_button("Save Case Definition"):
            full_def = f"Clinical: {clinical}\nPerson: {person}\nPlace: {place}\nTime: {time_period}"
            st.session_state.case_definition_text = full_def
            st.session_state.case_definition_written = True
            st.success("✅ Case definition saved! You can refine it as you learn more.")

elif st.session_state.current_view == 'interviews':
    st.markdown("### 👥 Key Informant Interviews")
    st.info(f"Budget: ${st.session_state.budget:,} | Each interview has a travel/time cost.")
    
    # Group characters by location
    locations = {
        'District Hospital/Office': ['dr_okonkwo', 'dr_mensah', 'entomologist_kwame'],
        'Nalu Village': ['nurse_fatima', 'mama_adama', 'chief_boateng', 'mr_owusu', 'teacher_grace'],
        'Kabwe Village': ['health_worker_joseph']
    }
    
    for location, char_keys in locations.items():
        st.markdown(f"#### 📍 {location}")
        
        cols = st.columns(len(char_keys))
        for i, char_key in enumerate(char_keys):
            char = CHARACTERS[char_key]
            with cols[i]:
                # Check if already interviewed
                interviewed = char_key in st.session_state.interview_history
                
                st.markdown(f"**{char['avatar']} {char['name']}**")
                st.caption(char['role'])
                st.caption(f"Cost: ${char['cost']}")
                
                if interviewed:
                    st.success("✓ Interviewed")
                
                if st.button(f"Talk to {char['name'].split()[0]}", key=f"btn_{char_key}"):
                    if st.session_state.budget >= char['cost']:
                        st.session_state.budget -= char['cost']
                        st.session_state.current_character = char_key
                        if char_key not in st.session_state.interview_history:
                            st.session_state.interview_history[char_key] = []
                        st.rerun()
                    else:
                        st.error("Insufficient funds!")
        
        st.markdown("---")
    
    # Active Interview
    if st.session_state.current_character:
        char = CHARACTERS[st.session_state.current_character]
        
        st.markdown(f"### 💬 Interviewing {char['name']}")
        st.caption(f"{char['role']} | {char['location']}")
        
        if st.button("🔙 End Interview"):
            st.session_state.current_character = None
            st.rerun()
        
        # Display conversation history
        history = st.session_state.interview_history[st.session_state.current_character]
        
        for msg in history:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.write(msg['content'])
            else:
                with st.chat_message("assistant", avatar=char['avatar']):
                    st.write(msg['content'])
        
        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            with st.chat_message("user"):
                st.write(prompt)
            history.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant", avatar=char['avatar']):
                with st.spinner("..."):
                    response = get_ai_response(
                        st.session_state.current_character,
                        prompt,
                        history[:-1]
                    )
                    st.write(response)
            
            history.append({"role": "assistant", "content": response})
            st.rerun()

elif st.session_state.current_view == 'linelist':
    st.markdown("### 📋 Case Line List")
    
    # Hospital Data (always available)
    st.markdown("#### 🏥 District Hospital Cases")
    st.caption("Severe AES cases reported to district surveillance")
    
    hospital_df = get_hospital_cases()
    st.dataframe(hospital_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Health Center Data (must unlock)
    st.markdown("#### 🏠 Community Health Center Records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Nalu Village Health Center**")
        if st.session_state.health_center_nalu_unlocked:
            st.success("✅ Access granted by Nurse Fatima")
            notes = get_health_center_notes('V1')
            for note in notes:
                st.markdown(f'<div class="handwritten-note">{note}</div>', unsafe_allow_html=True)
            
            st.caption("💡 These cases weren't reported to district. Consider abstracting them to your line list.")
        else:
            st.warning("🔒 Locked - Interview Nurse Fatima to access")
    
    with col2:
        st.markdown("**Kabwe Village Records**")
        if st.session_state.health_center_kabwe_unlocked:
            st.success("✅ Access granted by Joseph")
            notes = get_health_center_notes('V2')
            for note in notes:
                st.markdown(f'<div class="handwritten-note">{note}</div>', unsafe_allow_html=True)
        else:
            st.warning("🔒 Locked - Interview Joseph to access")
    
    st.markdown("---")
    
    # Manual case entry
    st.markdown("#### ✏️ Add Cases to Your Working Line List")
    st.caption("Abstract cases from health center notes or community reports")
    
    with st.form("add_case_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            case_id = st.text_input("Case ID", placeholder="e.g., C-01")
            age = st.number_input("Age", min_value=0, max_value=100, value=5)
        with col2:
            sex = st.selectbox("Sex", ["M", "F"])
            village = st.selectbox("Village", ["Nalu", "Kabwe", "Tamu", "Unknown"])
        with col3:
            onset = st.date_input("Onset Date")
            symptoms = st.text_input("Symptoms", placeholder="fever, seizures...")
        with col4:
            outcome = st.selectbox("Outcome", ["Alive", "Died", "Unknown"])
            classification = st.selectbox("Classification", ["Suspected", "Probable", "Confirmed"])
        
        if st.form_submit_button("Add Case"):
            new_case = {
                'case_id': case_id, 'age': age, 'sex': sex, 'village': village,
                'onset_date': onset.strftime('%b %d'), 'symptoms': symptoms,
                'outcome': outcome, 'classification': classification
            }
            st.session_state.manually_entered_cases.append(new_case)
            st.success(f"Added case {case_id}")
            st.rerun()
    
    if st.session_state.manually_entered_cases:
        st.markdown("**Your Working Line List:**")
        st.dataframe(pd.DataFrame(st.session_state.manually_entered_cases), use_container_width=True)

elif st.session_state.current_view == 'epicurve':
    st.markdown("### 📈 Epidemic Curve Builder")
    
    # Get all known cases
    hospital_df = get_hospital_cases()
    
    st.markdown("#### Step 1: Review Your Case Data")
    st.dataframe(hospital_df[['case_id', 'onset_date', 'village', 'outcome']], use_container_width=True)
    
    st.markdown("#### Step 2: Build the Epidemic Curve")
    
    # Parse dates and create epi curve
    date_counts = hospital_df['onset_date'].value_counts().sort_index()
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(date_counts.index),
        y=list(date_counts.values),
        marker_color='#e74c3c',
        name='Cases'
    ))
    
    fig.update_layout(
        title='Epidemic Curve: AES Cases by Date of Onset',
        xaxis_title='Date of Symptom Onset',
        yaxis_title='Number of Cases',
        height=400,
        bargap=0.1
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.session_state.epi_curve_built = True
    
    # Interpretation exercise
    st.markdown("#### Step 3: Interpret the Curve")
    
    with st.form("epicurve_interpretation"):
        st.markdown("**What type of outbreak pattern does this suggest?**")
        pattern = st.radio(
            "Select one:",
            ["Point source (single exposure)", 
             "Continuous common source",
             "Propagated (person-to-person)",
             "Mixed pattern"],
            index=None
        )
        
        st.markdown("**What is the likely incubation period?**")
        incubation = st.text_input("Your estimate:", placeholder="e.g., 5-15 days")
        
        st.markdown("**What does the village distribution suggest?**")
        village_interp = st.text_area("Your interpretation:", height=80)
        
        if st.form_submit_button("Submit Interpretation"):
            st.success("Interpretation recorded!")
            
            # Feedback
            with st.expander("💡 Facilitator Feedback"):
                st.markdown("""
                **Pattern:** This appears to be a **propagated/continuous source** outbreak, 
                consistent with vector-borne transmission. Cases occur over multiple weeks 
                with no sharp peak.
                
                **Incubation:** JE typically has a 5-15 day incubation period, which fits 
                the temporal spread observed.
                
                **Village clustering:** Cases started in Nalu and spread to Kabwe, suggesting 
                a geographic focus that could relate to a common environmental exposure 
                (e.g., breeding sites, animal reservoirs).
                """)

elif st.session_state.current_view == 'map':
    st.markdown("### 🗺️ Field Investigation Map")
    
    # Village cards with environmental details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="village-card" style="border-color: #e74c3c;">
            <h4>🔴 Nalu Village</h4>
            <p><strong>Population:</strong> 480</p>
            <p><strong>Features:</strong> Rice paddies, pig cooperative, wetlands nearby</p>
            <p><strong>JE Vaccine Coverage:</strong> 22%</p>
            <p><strong>Status:</strong> High case count</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="village-card" style="border-color: #f39c12;">
            <h4>🟠 Kabwe Village</h4>
            <p><strong>Population:</strong> 510</p>
            <p><strong>Features:</strong> Scattered pig farms, irrigation canals</p>
            <p><strong>JE Vaccine Coverage:</strong> 40%</p>
            <p><strong>Status:</strong> Moderate case count</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="village-card" style="border-color: #27ae60;">
            <h4>🟢 Tamu Village</h4>
            <p><strong>Population:</strong> 390</p>
            <p><strong>Features:</strong> Dryland farming, few pigs, seasonal pools</p>
            <p><strong>JE Vaccine Coverage:</strong> 55%</p>
            <p><strong>Status:</strong> Low case count</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Site Inspection
    st.markdown("#### 🔍 Environmental Site Inspection")
    st.caption("Send a team to investigate potential exposure sites. Cost: $100 per site")
    
    env_sites = TRUTH['env_sites']
    
    cols = st.columns(3)
    for i, (_, site) in enumerate(env_sites.iterrows()):
        with cols[i % 3]:
            inspected = site['site_id'] in st.session_state.sites_inspected
            
            st.markdown(f"**{site['site_type'].replace('_', ' ').title()}**")
            st.caption(f"Location: {site['village_id']} | Breeding Index: {site['breeding_index']}")
            
            if inspected:
                st.success("✓ Inspected")
                st.markdown(f"*{site['description']}*")
                if site['culex_present']:
                    st.warning("🦟 Culex mosquitoes present")
            else:
                if st.button(f"Inspect ($100)", key=f"inspect_{site['site_id']}"):
                    if st.session_state.budget >= 100:
                        st.session_state.budget -= 100
                        st.session_state.sites_inspected.append(site['site_id'])
                        st.rerun()
                    else:
                        st.error("Insufficient funds!")

elif st.session_state.current_view == 'laboratory':
    st.markdown("### 🧪 Laboratory Investigation")
    st.info(f"Lab Credits: {st.session_state.lab_credits} | Each test costs 1 credit")
    
    # Sample collection
    st.markdown("#### Collect & Submit Samples")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Human Samples**")
        with st.form("human_sample"):
            sample_type = st.selectbox("Sample Type", ["human_CSF", "human_serum"])
            source = st.text_input("Source (Case ID or description)")
            village = st.selectbox("Village", ["V1", "V2", "V3"])
            
            if st.form_submit_button("Submit to Lab (1 credit)"):
                if st.session_state.lab_credits >= 1:
                    st.session_state.lab_credits -= 1
                    result = process_lab_sample(sample_type, source, village)
                    st.session_state.lab_results.append(result)
                    st.success(f"Sample submitted! Results in {result['turnaround_days']} days.")
                    st.rerun()
                else:
                    st.error("No lab credits remaining!")
    
    with col2:
        st.markdown("**Environmental/Animal Samples**")
        with st.form("env_sample"):
            sample_type = st.selectbox("Sample Type", ["mosquito_pool", "pig_serum"])
            source = st.text_input("Collection Site")
            village = st.selectbox("Location", ["V1", "V2", "V3"], key="env_village")
            
            if st.form_submit_button("Submit to Lab (1 credit)"):
                if st.session_state.lab_credits >= 1:
                    st.session_state.lab_credits -= 1
                    result = process_lab_sample(sample_type, source, village)
                    st.session_state.lab_results.append(result)
                    st.success(f"Sample submitted! Results in {result['turnaround_days']} days.")
                    st.rerun()
                else:
                    st.error("No lab credits remaining!")
    
    st.markdown("---")
    
    # Lab results
    st.markdown("#### 📋 Laboratory Results")
    
    if st.session_state.lab_results:
        for result in st.session_state.lab_results:
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.markdown(f"**{result['sample_id']}** - {result['sample_type'].replace('_', ' ').title()}")
                    st.caption(f"Source: {result['source']} | Village: {result['village']}")
                with col2:
                    st.markdown(f"Test: {result['test_performed']}")
                with col3:
                    if result['result'] == 'POSITIVE':
                        st.error(f"🔴 {result['result']}")
                    else:
                        st.success(f"🟢 {result['result']}")
                st.markdown("---")
    else:
        st.caption("No laboratory results yet. Submit samples to begin testing.")
    
    # Lab interpretation guide
    with st.expander("📖 Laboratory Test Reference"):
        st.markdown("""
        | Test | Sample | Sensitivity | Specificity | Interpretation |
        |------|--------|-------------|-------------|----------------|
        | JE IgM ELISA | CSF | 85-90% | >95% | Best for confirming acute infection |
        | JE IgM ELISA | Serum | 75-85% | >95% | Good screening test |
        | JE RT-PCR | Mosquito pool | >95% | >98% | Confirms virus circulation |
        | JE IgG | Pig serum | High | High | Shows amplification, not direct transmission |
        
        **Note:** Negative results don't rule out JE, especially if sample collected too early or too late.
        """)

elif st.session_state.current_view == 'study_design':
    st.markdown("### 🔬 Analytical Study Design")
    
    st.markdown("""
    Based on your investigation so far, design a study to test your hypothesis 
    about risk factors for illness in this outbreak.
    """)
    
    with st.form("study_design_form"):
        st.markdown("#### 1. Study Type")
        study_type = st.selectbox(
            "What type of study will you conduct?",
            ["Case-control study", "Cohort study", "Cross-sectional survey"],
            index=0
        )
        
        st.markdown("#### 2. Hypothesis")
        hypothesis = st.text_area(
            "State your primary hypothesis:",
            placeholder="e.g., Living near pig pens is associated with increased risk of AES",
            height=80
        )
        
        st.markdown("#### 3. Case & Control Definitions")
        col1, col2 = st.columns(2)
        with col1:
            case_def = st.text_area("Case definition:", height=100)
        with col2:
            control_def = st.text_area("Control definition:", height=100)
        
        st.markdown("#### 4. Exposures to Measure")
        st.multiselect(
            "Select exposures to include in your questionnaire:",
            ["Proximity to pig pens (<30m)",
             "Mosquito net use",
             "Distance to rice paddies",
             "Evening outdoor activities",
             "JE vaccination status",
             "Water source",
             "Occupation",
             "Recent travel"]
        )
        
        st.markdown("#### 5. Sample Size")
        col1, col2 = st.columns(2)
        with col1:
            n_cases = st.number_input("Number of cases to enroll", min_value=5, max_value=50, value=15)
        with col2:
            n_controls = st.number_input("Controls per case", min_value=1, max_value=4, value=2)
        
        cost = (n_cases + n_cases * n_controls) * 20  # $20 per interview
        st.caption(f"Estimated cost: ${cost} ({n_cases + n_cases * n_controls} interviews × $20)")
        
        if st.form_submit_button(f"Deploy Field Team (${cost})"):
            if st.session_state.budget >= cost:
                st.session_state.budget -= cost
                st.session_state.questionnaire_deployed = True
                st.session_state.hypothesis_text = hypothesis
                st.success("✅ Field team deployed! Data collection in progress...")
                
                # Generate mock results
                st.markdown("---")
                st.markdown("### 📊 Preliminary Results")
                
                # Simulated 2x2 table
                st.markdown("**Exposure: Living within 30m of pig pens**")
                
                data = {
                    '': ['Exposed (near pigs)', 'Unexposed', 'Total'],
                    'Cases': [12, 3, 15],
                    'Controls': [8, 22, 30],
                    'Total': [20, 25, 45]
                }
                st.table(pd.DataFrame(data))
                
                # Calculate OR
                OR = (12 * 22) / (3 * 8)
                st.metric("Crude Odds Ratio", f"{OR:.1f}")
                st.caption("95% CI: 2.8 - 39.2 | p < 0.001")
                
            else:
                st.error("Insufficient funds!")

elif st.session_state.current_view == 'debrief':
    st.markdown("### 📝 Investigation Debrief")
    
    st.markdown("#### Your Investigation Summary")
    
    # Progress summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Budget Used", f"${3000 - st.session_state.budget:,}")
        st.metric("Interviews Conducted", len(st.session_state.interview_history))
    
    with col2:
        st.metric("Lab Tests Done", len(st.session_state.lab_results))
        st.metric("Sites Inspected", len(st.session_state.sites_inspected))
    
    with col3:
        st.metric("Data Sources Unlocked", 
                 sum([st.session_state.health_center_nalu_unlocked,
                      st.session_state.health_center_kabwe_unlocked, 1]))
    
    st.markdown("---")
    
    # Key findings checklist
    st.markdown("#### 🔑 Key Findings Assessment")
    
    critical_clues = {
        "Identified pigs as amplifying host": st.session_state.health_center_nalu_unlocked,
        "Connected outbreak to pig cooperative": 'mr_owusu' in st.session_state.interview_history,
        "Identified Culex mosquitoes as vector": 'entomologist_kwame' in st.session_state.interview_history,
        "Found community cases not in hospital data": st.session_state.health_center_nalu_unlocked,
        "Tested environmental samples": any(r['sample_type'] in ['mosquito_pool', 'pig_serum'] for r in st.session_state.lab_results),
        "Inspected pig cooperative site": 'ES02' in st.session_state.sites_inspected,
        "Built epidemic curve": st.session_state.epi_curve_built,
        "Viewed spot map": st.session_state.spot_map_viewed,
        "Deployed analytical study": st.session_state.study_deployed
    }
    
    score = sum(findings.values())
    
    for finding, achieved in findings.items():
        icon = "✅" if achieved else "❌"
        st.markdown(f"{icon} {finding}")
    
    st.markdown(f"### Score: {score}/{len(findings)} critical findings")
    
    # Progress bar
    st.progress(score / len(findings))
    
    st.markdown("---")
    
    # Final diagnosis
    st.markdown("#### 🎯 Final Assessment")
    
    with st.form("final_report"):
        diagnosis = st.selectbox(
            "Most likely etiology:",
            ["Select...", "Japanese Encephalitis", "Bacterial meningitis", "Cerebral malaria",
             "Rabies", "Nipah virus", "Unknown viral encephalitis"]
        )
        
        transmission = st.multiselect(
            "Transmission route:",
            ["Mosquito-borne (vector)", "Direct contact with pigs", "Person-to-person",
             "Contaminated water", "Airborne", "Unknown"]
        )
        
        st.markdown("**Top 3 Control Recommendations:**")
        rec1 = st.text_input("Recommendation 1:")
        rec2 = st.text_input("Recommendation 2:")
        rec3 = st.text_input("Recommendation 3:")
        
        if st.form_submit_button("Submit Final Report"):
            if diagnosis == "Japanese Encephalitis":
                st.balloons()
                st.success("🎉 Correct diagnosis! Japanese Encephalitis.")
            else:
                st.warning(f"The correct diagnosis was Japanese Encephalitis. You selected: {diagnosis}")
            
            log_action("final_diagnosis", diagnosis)
    
    st.markdown("---")
    
    # Facilitator section
    if st.session_state.facilitator_mode:
        st.markdown("""
        <div class="facilitator-mode">
        <h4>🎓 FACILITATOR MODE - GROUND TRUTH</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Disease Model")
        st.markdown("""
        **Diagnosis:** Japanese Encephalitis (JE)
        
        **Transmission cycle:**
        - Pigs serve as amplifying hosts (high viremia)
        - Culex tritaeniorhynchus mosquitoes are the primary vector
        - Mosquitoes breed in rice paddies (recently expanded)
        - Humans are dead-end hosts (infected by mosquito bite)
        
        **Key risk factors in this outbreak:**
        1. Living <30m from pig pens (OR ~11)
        2. Not using mosquito nets (OR ~4)
        3. Evening outdoor activities (OR ~3)
        4. No JE vaccination (OR ~6)
        5. Living near rice paddies (OR ~3)
        
        **Timeline:**
        - Pig cooperative built 3 months ago
        - Rice paddies expanded 2 months ago
        - First human cases: June 3, 2025
        - Peak: June 5-8, 2025
        """)
        
        st.markdown("#### Ground Truth Data")
        
        tab1, tab2, tab3 = st.tabs(["Individuals", "Households", "Villages"])
        
        with tab1:
            individuals = TRUTH['individuals']
            cases = individuals[individuals['symptomatic_AES'] == True]
            st.markdown(f"**Total symptomatic cases:** {len(cases)}")
            st.markdown(f"**Severe cases:** {cases['severe_neuro'].sum()}")
            st.markdown(f"**Deaths:** {len(cases[cases['outcome'] == 'died'])}")
            st.dataframe(cases[['person_id', 'age', 'sex', 'village_id', 'onset_date', 'severe_neuro', 'outcome']])
        
        with tab2:
            st.dataframe(TRUTH['households'].head(20))
        
        with tab3:
            st.dataframe(TRUTH['villages'])
        
        st.markdown("#### Recommended Control Measures")
        st.markdown("""
        **Immediate (Week 1):**
        1. Emergency JE vaccination campaign for children <15 in Nalu and Kabwe
        2. Distribute insecticide-treated bed nets to affected villages
        3. Indoor residual spraying in high-risk households
        
        **Short-term (Weeks 2-4):**
        4. Larviciding of rice paddies and breeding sites
        5. Health education: evening protective measures, symptom recognition
        6. Enhanced surveillance for new AES cases
        
        **Long-term:**
        7. Integrate JE vaccine into routine immunization program
        8. Establish pig serosurveillance system
        9. Improve drainage around pig cooperative
        10. Create buffer zone between pig farms and residential areas
        """)
    
    # Action log
    with st.expander("📜 Complete Action Log"):
        if st.session_state.actions_log:
            for action in st.session_state.actions_log:
                st.caption(f"Day {action['day']}: {action['action']} - {action['details']}")
        else:
            st.caption("No actions logged yet.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("FETP Intermediate 2.0 | JE Outbreak Simulation v2.0 | For training purposes only")