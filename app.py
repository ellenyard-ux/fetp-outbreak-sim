import streamlit as st
import anthropic
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
FACILITATOR_PASSWORD = "fetp2025"  # Change this for your deployment

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
# SESSION STATE INITIALIZATION - 5-DAY STRUCTURE
# ============================================================================
def init_session_state():
    defaults = {
        # Navigation & Day Structure
        'current_view': 'briefing',
        'investigation_day': 1,  # Days 1-5
        'day_phase': 'morning',  # morning, afternoon, evening
        'exercise_started': False,
        
        # Resources
        'budget': 3000,
        'lab_credits': 15,
        
        # Day-based Injects (shown when day advances)
        'inject_shown': {},  # Track which injects have been displayed
        'current_inject': None,
        
        # NPC Access Control (Day-based unlocking)
        'npcs_unlocked': ['dr_chen', 'nurse_joy', 'mama_kofi', 'foreman_rex', 'teacher_grace'],  # Day 1 NPCs
        'one_health_triggered': False,  # True when trainee asks about animals/environment
        'vet_unlocked': False,
        'env_officer_unlocked': False,
        'healer_unlocked': False,
        
        # Data Access Flags
        'hospital_data_accessed': True,
        'health_center_nalu_unlocked': False,
        'health_center_kabwe_unlocked': False,
        'vet_data_unlocked': False,
        'entomology_data_unlocked': False,
        
        # Interview System
        'interview_history': {},
        'current_character': None,
        'clues_discovered': set(),
        'questions_asked_about': set(),  # Track topics asked (animals, mosquitoes, etc.)
        
        # Day 1: Case Definition & Descriptive Epi
        'case_definition_written': False,
        'case_definition_text': "",
        'case_definition_components': {},  # clinical, person, place, time
        'initial_hypotheses': [],
        'descriptive_epi_done': False,
        
        # Day 2: Interviews, Hypotheses, Study Design
        'hypotheses_documented': [],
        'study_design_chosen': None,  # case-control, cohort, cross-sectional
        'questionnaire_variables': [],
        'questionnaire_submitted': False,
        
        # Day 3: Analysis
        'dataset_received': False,
        'data_cleaning_done': False,
        'descriptive_analysis_done': False,
        'analytic_results': None,
        'odds_ratios_calculated': {},
        
        # Day 4: Lab & Environmental
        'lab_samples_submitted': [],
        'lab_results': [],
        'lab_queue': [],
        'environmental_samples': [],
        'mosquito_traps_set': [],
        'pig_samples_collected': False,
        'triangulation_complete': False,
        
        # Day 5: Briefing & Consequences
        'briefing_prepared': False,
        'recommendations_submitted': [],
        'final_diagnosis': None,
        'consequence_outcome': None,
        
        # Field Work
        'sites_inspected': [],
        'field_observations': {},
        
        # Case Registry
        'confirmed_cases': [],
        'manually_entered_cases': [],
        
        # Epi Curve & Spot Map
        'epi_curve_built': False,
        'spot_map_viewed': False,
        
        # Scoring & Tracking
        'actions_log': [],
        'mistakes_made': [],
        'good_decisions': [],
        
        # Facilitator Mode
        'facilitator_mode': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# DAY STRUCTURE & INJECTS
# ============================================================================
DAY_STRUCTURE = {
    1: {
        'title': 'Day 1: Detect, Confirm, Describe',
        'objectives': [
            'Establish working case definition',
            'Summarize descriptive epidemiology',
            'Identify data gaps',
            'Propose initial hypotheses'
        ],
        'inject': {
            'title': '🚨 INITIAL ALERT',
            'source': 'District Health Office',
            'message': '''We are receiving reports of several children with seizures, plus one miner with sudden confusion and coma. 
            
**Three deaths were reported in the past 48 hours.**

All cases come from villages near Sidero Valley. We need your team in Nalu immediately.

Your tasks today:
1. Review initial cases
2. Develop a working case definition  
3. Begin interviews to understand the situation
4. Document initial hypotheses'''
        },
        'npcs_available': ['dr_chen', 'nurse_joy', 'mama_kofi', 'foreman_rex', 'teacher_grace'],
        'data_visible': ['hospital_linelist', 'map'],
        'data_hidden': ['pig_restlessness', 'household_clustering', 'mosquito_positives', 'vaccination_rates']
    },
    
    2: {
        'title': 'Day 2: Interviews, Hypotheses, Study Design',
        'objectives': [
            'Conduct hypothesis-generating interviews',
            'Develop specific hypotheses',
            'Choose study design',
            'Draft questionnaire'
        ],
        'inject': {
            'title': '🏥 ADDITIONAL AES ALERTS',
            'source': 'District Hospital',
            'message': '''Four new AES cases from Nalu and Kabwe admitted overnight. 

**Two of the new cases are siblings from the same household.**

Updated line list is available. You should:
1. Continue key informant interviews
2. Finalize your hypotheses
3. Design your analytical study
4. Prepare your questionnaire'''
        },
        'npcs_available': ['dr_chen', 'nurse_joy', 'mama_kofi', 'foreman_rex', 'teacher_grace', 
                          'healer_marcus', 'auntie_ama'],  # + conditional unlocks
        'unlock_conditions': {
            'vet_amina': 'Asked about animals/pigs',
            'mr_osei': 'Asked about mosquitoes/environment'
        }
    },
    
    3: {
        'title': 'Day 3: Data Analysis',
        'objectives': [
            'Clean your dataset',
            'Run descriptive epidemiology',
            'Conduct analytic study',
            'Calculate measures of association'
        ],
        'inject': {
            'title': '📊 YOUR DATA HAVE ARRIVED',
            'source': 'Field Team',
            'message': '''Your field team has completed data collection based on your study design and questionnaire.

**Note:** The dataset may contain:
- Missing values
- Inconsistent coding
- Data entry errors

This is realistic field data. Clean it carefully before analysis.

Today you should:
1. Review and clean the dataset
2. Run descriptive statistics
3. Calculate odds ratios (case-control) or risk ratios (cohort)
4. Interpret your findings'''
        },
        'analysis_mode': True
    },
    
    4: {
        'title': 'Day 4: Laboratory & Environmental Sampling',
        'objectives': [
            'Order appropriate diagnostic tests',
            'Collect environmental samples',
            'Sample animal reservoirs',
            'Triangulate all evidence'
        ],
        'inject': {
            'title': '🔬 LABORATORY ACCESS GRANTED',
            'source': 'Regional Laboratory',
            'message': '''You now have authority to order diagnostic tests.

**Available sample types:**
- Human CSF (IgM ELISA)
- Human serum (IgM ELISA)  
- Pig serum (JEV antibodies)
- Mosquito pools (JEV PCR)
- Water samples (bacterial culture)
- Food samples (culture)

**Budget remaining:** ${budget}
**Lab credits remaining:** {lab_credits}

Choose wisely - some tests are more useful than others.'''
        },
        'lab_unlocked': True
    },
    
    5: {
        'title': 'Day 5: MOH Briefing & Recommendations',
        'objectives': [
            'Prepare verbal briefing',
            'Present findings to Director',
            'Recommend interventions',
            'Receive outcome feedback'
        ],
        'inject': {
            'title': '🏛️ BRIEFING TIME',
            'source': 'District Director of Health',
            'message': '''The District Director of Health will see you now.

You must present:
1. Summary of the outbreak
2. Your conclusion on the likely cause
3. Key risk factors identified
4. Recommended interventions
5. Resources needed
6. One Health considerations

**Your recommendations will determine what happens next in the outbreak.**'''
        },
        'briefing_mode': True
    }
}

# ============================================================================
# CONSEQUENCE ENGINE
# ============================================================================
def calculate_consequences():
    """
    Calculate outbreak consequences based on trainee decisions.
    Returns outcome narrative and statistics.
    """
    score = 0
    outcomes = []
    
    # Check for correct diagnosis
    if st.session_state.final_diagnosis == 'Japanese Encephalitis':
        score += 20
        outcomes.append("✅ Correct diagnosis: Japanese Encephalitis identified")
    else:
        score -= 10
        outcomes.append("❌ Incorrect diagnosis led to delayed response")
    
    # Check if One Health approach was used
    if 'vet_amina' in st.session_state.interview_history:
        score += 15
        outcomes.append("✅ One Health: Veterinary officer consulted")
    else:
        score -= 5
        outcomes.append("⚠️ Missed One Health opportunity - pig surveillance delayed")
    
    if st.session_state.env_officer_unlocked or 'mr_osei' in st.session_state.interview_history:
        score += 10
        outcomes.append("✅ Environmental assessment completed")
    else:
        outcomes.append("⚠️ Environmental risk factors not fully assessed")
    
    # Check lab strategy
    pig_samples = any('pig' in s.get('sample_type', '') for s in st.session_state.lab_samples_submitted)
    mosquito_samples = any('mosquito' in s.get('sample_type', '') for s in st.session_state.lab_samples_submitted)
    human_samples = any('human' in s.get('sample_type', '') for s in st.session_state.lab_samples_submitted)
    
    if pig_samples and mosquito_samples and human_samples:
        score += 20
        outcomes.append("✅ Comprehensive sampling strategy (human + animal + vector)")
    elif human_samples and (pig_samples or mosquito_samples):
        score += 10
        outcomes.append("⚡ Partial One Health sampling")
    else:
        score -= 5
        outcomes.append("❌ Limited sampling missed key evidence")
    
    # Check questionnaire quality
    good_variables = ['pig', 'mosquito', 'net', 'vaccination', 'evening', 'outdoor', 'dusk']
    variables_included = sum(1 for v in good_variables if any(v in q.lower() for q in st.session_state.questionnaire_variables))
    
    if variables_included >= 5:
        score += 15
        outcomes.append("✅ Excellent questionnaire captured key risk factors")
    elif variables_included >= 3:
        score += 8
        outcomes.append("⚡ Adequate questionnaire - some risk factors captured")
    else:
        score -= 5
        outcomes.append("❌ Poor questionnaire missed key exposures")
    
    # Check recommendations
    recommendations = ' '.join(st.session_state.recommendations_submitted).lower()
    
    good_recs = {
        'vaccination': ['vaccine', 'vaccination', 'immunization', 'immunize'],
        'vector_control': ['bed net', 'bednet', 'mosquito net', 'larvicid', 'spray'],
        'pig_management': ['pig', 'relocat', 'distance', 'pen'],
        'surveillance': ['surveillance', 'monitor', 'report'],
        'education': ['education', 'awareness', 'community']
    }
    
    recs_score = 0
    for category, keywords in good_recs.items():
        if any(kw in recommendations for kw in keywords):
            recs_score += 1
    
    if recs_score >= 4:
        score += 20
        outcomes.append("✅ Comprehensive intervention package recommended")
    elif recs_score >= 2:
        score += 10
        outcomes.append("⚡ Partial interventions recommended")
    else:
        score -= 10
        outcomes.append("❌ Weak interventions - outbreak likely to continue")
    
    # Determine final outcome
    if score >= 70:
        final_outcome = {
            'status': 'SUCCESS',
            'narrative': '''**OUTBREAK CONTROLLED**
            
Your evidence-based recommendations were implemented:
- Emergency JE vaccination campaign reached 2,500 children
- Bed nets distributed to high-risk households
- Pig cooperative relocated 50m from school
- Mosquito breeding sites treated

**Result:** New cases dropped to zero within 2 weeks. 
The District Director praised your team's One Health approach.''',
            'new_cases': 0,
            'color': 'success'
        }
    elif score >= 40:
        final_outcome = {
            'status': 'PARTIAL SUCCESS',
            'narrative': '''**OUTBREAK PARTIALLY CONTROLLED**
            
Some of your recommendations were implemented, but gaps remained:
- Vaccination campaign was delayed
- Vector control was incomplete

**Result:** Cases continued for another week before declining.
Two additional children were hospitalized but survived.''',
            'new_cases': 2,
            'color': 'warning'
        }
    else:
        final_outcome = {
            'status': 'OUTBREAK CONTINUES',
            'narrative': '''**OUTBREAK NOT CONTROLLED**
            
Your recommendations missed key interventions:
- No vaccination campaign initiated
- Pig cooperative unchanged
- Mosquito breeding sites remain

**Result:** Cases spread to Tamu village. 
Three more children died. Regional investigation launched.
Mayor blamed the investigation team for "incomplete response."''',
            'new_cases': 8,
            'color': 'error'
        }
    
    final_outcome['score'] = score
    final_outcome['outcomes'] = outcomes
    
    return final_outcome

def check_npc_unlock_triggers(question_text):
    """
    Check if a question should unlock additional NPCs.
    Called after each interview question.
    """
    question_lower = question_text.lower()
    
    # Animal/pig triggers → unlock Vet Amina
    animal_triggers = ['animal', 'pig', 'livestock', 'farm animal', 'cattle', 'pigs', 'swine']
    if any(trigger in question_lower for trigger in animal_triggers):
        st.session_state.questions_asked_about.add('animals')
        if not st.session_state.vet_unlocked:
            st.session_state.vet_unlocked = True
            st.session_state.npcs_unlocked.append('vet_amina')
            st.session_state.one_health_triggered = True
            return "🔓 **NEW NPC UNLOCKED:** Vet Amina (District Veterinary Officer) - Your question about animals triggered a One Health connection!"
    
    # Mosquito/environment triggers → unlock Mr. Osei
    env_triggers = ['mosquito', 'mosquitoes', 'vector', 'breeding', 'standing water', 'environment', 'rice paddy', 'irrigation']
    if any(trigger in question_lower for trigger in env_triggers):
        st.session_state.questions_asked_about.add('mosquitoes')
        if not st.session_state.env_officer_unlocked:
            st.session_state.env_officer_unlocked = True
            st.session_state.npcs_unlocked.append('mr_osei')
            return "🔓 **NEW NPC UNLOCKED:** Mr. Osei (Environmental Health Officer) - Your question about mosquitoes/environment was insightful!"
    
    # Healer triggers (alternative medicine, early cases)
    healer_triggers = ['traditional', 'healer', 'clinic', 'private', 'early case', 'first case', 'before hospital']
    if any(trigger in question_lower for trigger in healer_triggers):
        if not st.session_state.healer_unlocked:
            st.session_state.healer_unlocked = True
            st.session_state.npcs_unlocked.append('healer_marcus')
            return "🔓 **NEW NPC UNLOCKED:** Healer Marcus (Private Clinic) - He saw cases before the hospital!"
    
    return None

def advance_day():
    """Advance to the next day with inject and unlocks."""
    current_day = st.session_state.investigation_day
    
    if current_day < 5:
        st.session_state.investigation_day += 1
        new_day = st.session_state.investigation_day
        
        # Show inject for new day
        st.session_state.current_inject = DAY_STRUCTURE[new_day]['inject']
        st.session_state.inject_shown[new_day] = False
        
        # Day 2: Add conditional NPCs if triggers were met
        if new_day == 2:
            if st.session_state.one_health_triggered:
                if 'vet_amina' not in st.session_state.npcs_unlocked:
                    st.session_state.npcs_unlocked.append('vet_amina')
        
        # Day 3: Generate analysis dataset
        if new_day == 3:
            st.session_state.dataset_received = True
        
        # Day 4: Unlock lab
        if new_day == 4:
            pass  # Lab already available
        
        # Process pending lab results
        new_results = []
        remaining_queue = []
        for sample in st.session_state.lab_queue:
            sample['days_remaining'] = sample.get('days_remaining', 1) - 1
            if sample['days_remaining'] <= 0:
                new_results.append(sample)
            else:
                remaining_queue.append(sample)
        
        st.session_state.lab_queue = remaining_queue
        st.session_state.lab_results.extend(new_results)
        
        return True
    return False

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
# CHARACTERS FOR INTERVIEWS - FULL NPC TRUTH DOCUMENTS
# ============================================================================
CHARACTERS = {
    # -------------------------------------------------------------------------
    # DR. CHEN — Hospital Director
    # -------------------------------------------------------------------------
    "dr_chen": {
        "name": "Dr. Chen",
        "role": "Hospital Director",
        "avatar": "👨‍⚕️",
        "cost": 0,  # Free - your supervisor
        "location": "District Hospital",
        "personality": "Precise and factual. A bit impatient. Trusts lab data more than interviews. Will not speculate early on.",
        
        # TRUTH THE NPC KNOWS (always available)
        "knowledge": [
            "Has full hospital AES line list: ages 2-12 years primarily, 3 fatalities",
            "All cases from Nalu & Kabwe villages",
            "Onsets between June 3-10",
            "No adult AES cases EXCEPT one miner",
            "Has seen several children with seizures, fever, confusion",
            "All hospitalized AES patients are from households within 1 km of rice paddies or pig farms",
            "No shared meal or event identified",
            "JE vaccination status poorly recorded, but 'most parents say no vaccines recently'",
            "One 9-year-old died 24 hours after admission",
            "They ruled out bacterial meningitis in early cases",
            "They could not run JE IgM yet - samples sent to regional lab, awaiting results"
        ],
        
        # HIDDEN CLUES - only revealed if DIRECTLY asked about these topics
        "hidden_clues": {
            "animals": "Two families reported pigs 'acting strangely' - restless, vocalizing at night",
            "pigs": "Two families reported pigs 'acting strangely' - restless, vocalizing at night",
            "mosquito nets": "One severe case slept without mosquito nets",
            "nets": "One severe case slept without mosquito nets",
            "miner": "The dead miner lived right beside new irrigation ponds",
            "irrigation": "The dead miner lived right beside new irrigation ponds",
            "children activities": "Nurses noticed children often play outdoors at dusk",
            "dusk": "Nurses noticed children often play outdoors at dusk",
            "evening": "Nurses noticed children often play outdoors at dusk"
        },
        
        # RED HERRINGS - may mention these but they're wrong
        "red_herrings": [
            "Possibly 'viral meningitis outbreak from school' - but no event links",
            "Maybe 'heat stroke or toxins from the mine' - but she doubts it"
        ],
        
        # WHAT THEY DON'T KNOW
        "does_not_know": [
            "Pig infection status",
            "Mosquito pool results", 
            "Any environmental surveillance data",
            "True JE infection mechanism"
        ],
        
        "data_access": "hospital_cases",
        "unlocks": None
    },
    
    # -------------------------------------------------------------------------
    # NURSE JOY — Triage Nurse
    # -------------------------------------------------------------------------
    "nurse_joy": {
        "name": "Nurse Joy",
        "role": "Triage Nurse",
        "avatar": "🩺",
        "cost": 50,
        "location": "District Hospital",
        "personality": "Exhausted, overwhelmed. Kind but disorganized. Gives 'too much' detail sometimes. Good observer.",
        
        "knowledge": [
            "Has incoming case logs with symptoms and onset dates",
            "Children arrived in clusters on June 3, 6, 7, 9",
            "Symptoms progress: fever → headache → confusion → seizures",
            "No rash observed in any cases",
            "Several parents mentioned living 'near the big rice fields'",
            "Parents mentioned many mosquitoes at dusk"
        ],
        
        "hidden_clues": {
            "animals": "Two parents mentioned 'pigs screaming at night'",
            "pigs": "Two parents mentioned 'pigs screaming at night'",
            "mosquitoes": "Yes, 'bites everywhere, especially around the ankles'",
            "bites": "Yes, 'bites everywhere, especially around the ankles'",
            "vaccination": "Many parents said vaccines 'ran out last year'",
            "vaccines": "Many parents said vaccines 'ran out last year'"
        },
        
        "red_herrings": [
            "Thinks maybe 'bad drinking water' is causing illness",
            "Mentions a rumor about 'spirits in the forest' - completely irrelevant"
        ],
        
        "does_not_know": [
            "JE vaccination schedules",
            "Lab results",
            "Pig farms' infection status"
        ],
        
        "data_access": "triage_logs",
        "unlocks": None
    },
    
    # -------------------------------------------------------------------------
    # HEALER MARCUS — Private Clinic / Traditional Practitioner  
    # -------------------------------------------------------------------------
    "healer_marcus": {
        "name": "Healer Marcus",
        "role": "Private Clinic Practitioner",
        "avatar": "🌿",
        "cost": 150,
        "location": "Nalu Village",
        "personality": "Suspicious of government. Proud, mystical tone. Sometimes helpful, sometimes evasive. Wants recognition.",
        
        "knowledge": [
            "Has clinic notes of ~10 children with fever + tremors (AES-like)",
            "Saw early cases BEFORE hospital did",
            "Notices many patients from households near pigs",
            "Several children from pig-farming households came sick",
            "Symptoms began after increased mosquito activity",
            "He believes 'bad air from the rice paddies' is to blame"
        ],
        
        "hidden_clues": {
            "animals": "Pigs restless at night... biting insects around them",
            "pigs": "Pigs restless at night... biting insects around them",
            "environment": "New irrigation canal brings still water",
            "irrigation": "New irrigation canal brings still water",
            "water": "New irrigation canal brings still water",
            "timing": "He saw first AES case 3 days before hospital admission",
            "first case": "He saw first AES case 3 days before hospital admission"
        },
        
        "red_herrings": [
            "Might mention herbal contamination as possible cause",
            "Suggests 'the mines poison the wind'"
        ],
        
        "does_not_know": [
            "Vector transmission mechanism",
            "Lab results",
            "JE-specific information"
        ],
        
        "data_access": "private_clinic",
        "unlocks": "health_center_nalu_unlocked"
    },
    
    # -------------------------------------------------------------------------
    # MAMA KOFI — Mother of Case
    # -------------------------------------------------------------------------
    "mama_kofi": {
        "name": "Mama Kofi",
        "role": "Mother of Sick Child",
        "avatar": "👵",
        "cost": 50,
        "location": "Nalu Village", 
        "personality": "Emotional, distressed. Highly motivated to talk. But gives unclear sequences unless asked carefully.",
        
        "knowledge": [
            "Her house is 10-20 meters from pig pen",
            "Family does not use mosquito nets",
            "Children play outside at dusk with neighbors",
            "Recently, pigs have been 'hot, restless, noisy'",
            "Her daughter's symptoms started after an evening with intense mosquito biting"
        ],
        
        "hidden_clues": {
            "vaccination": "They had no JE vaccine available last year",
            "vaccine": "They had no JE vaccine available last year",
            "water": "The irrigation pump broke, causing standing water near homes",
            "irrigation": "The irrigation pump broke, causing standing water near homes",
            "standing water": "The irrigation pump broke, causing standing water near homes",
            "birds": "She saw 'many herons' this season around the paddies",
            "herons": "She saw 'many herons' this season around the paddies"
        },
        
        "red_herrings": [
            "Thought it might be 'witchcraft from jealous neighbors'"
        ],
        
        "does_not_know": [
            "Medical terminology",
            "Other families' situations in detail",
            "Lab results"
        ],
        
        "data_access": None,
        "unlocks": None
    },
    
    # -------------------------------------------------------------------------
    # TEACHER GRACE — School Principal
    # -------------------------------------------------------------------------
    "teacher_grace": {
        "name": "Teacher Grace",
        "role": "School Principal",
        "avatar": "📚",
        "cost": 50,
        "location": "Nalu Village",
        "personality": "Observant. Data-driven. Cooperative. Worried about school reputation.",
        
        "knowledge": [
            "Three AES cases attend the same school",
            "Children walk home past the pig co-op",
            "Evening tutoring sessions mean children are outdoors at dusk",
            "Mosquitoes inside classrooms after heavy rains"
        ],
        
        "hidden_clues": {
            "vaccination": "She has records showing low JE vaccination rates among students",
            "vaccine": "She has records showing low JE vaccination rates among students",
            "mosquitoes": "Parents complained about mosquito breeding in rice paddies",
            "rice": "Parents complained about mosquito breeding in rice paddies",
            "food": "No shared school meals recently",
            "meals": "No shared school meals recently"
        },
        
        "red_herrings": [
            "Wonders whether the new school latrine project 'brought disease'"
        ],
        
        "does_not_know": [
            "Medical details",
            "Pig farm operations",
            "Lab results"
        ],
        
        "data_access": "school_attendance",
        "unlocks": None
    },
    
    # -------------------------------------------------------------------------
    # FOREMAN REX — Mine Manager
    # -------------------------------------------------------------------------
    "foreman_rex": {
        "name": "Foreman Rex",
        "role": "Mine Manager",
        "avatar": "⛏️",
        "cost": 200,
        "location": "Mining Area",
        "personality": "Defensive. Doesn't want blame. Limited knowledge about community. Direct, curt.",
        
        "knowledge": [
            "One miner is hospitalized with AES",
            "Mine expanded irrigation ponds recently",
            "Standing water collects near worker housing"
        ],
        
        "hidden_clues": {
            "water": "New water pits were created 6 weeks ago",
            "ponds": "New water pits were created 6 weeks ago",
            "irrigation": "New water pits were created 6 weeks ago",
            "mosquitoes": "Mosquitoes worsened after the new pits were dug",
            "workers": "Workers live near pig-farming villages"
        },
        
        "red_herrings": [
            "Thinks maybe 'chemical exposure' is causing the illness",
            "Claims 'mosquitoes aren't a real issue' - but they are"
        ],
        
        "does_not_know": [
            "Community health details",
            "Pig farm status",
            "Medical information"
        ],
        
        "data_access": None,
        "unlocks": None
    },
    
    # -------------------------------------------------------------------------
    # AUNTIE AMA — Market Vendor
    # -------------------------------------------------------------------------
    "auntie_ama": {
        "name": "Auntie Ama",
        "role": "Market Vendor",
        "avatar": "🍎",
        "cost": 50,
        "location": "Central Market",
        "personality": "Knows everyone. Gossip-heavy. Colorful but unreliable narrator. Good geographic insight.",
        
        "knowledge": [
            "Cases cluster near Nalu & Kabwe villages",
            "She heard pig farmers complaining about 'mosquitoes eating pigs alive'",
            "She noticed many sick children from families near rice paddies"
        ],
        
        "hidden_clues": {
            "pigs": "She saw dead piglets at a farm last week",
            "piglets": "She saw dead piglets at a farm last week",
            "dead animals": "She saw dead piglets at a farm last week",
            "geography": "She knows which households cluster near irrigation canal",
            "canal": "She knows which households cluster near irrigation canal",
            "location": "She knows which households cluster near irrigation canal"
        },
        
        "red_herrings": [
            "Might claim 'bad batch of fruit' caused illness",
            "Invents rumors for dramatic effect"
        ],
        
        "does_not_know": [
            "Medical details",
            "Official data",
            "Lab results"
        ],
        
        "data_access": None,
        "unlocks": None
    },
    
    # -------------------------------------------------------------------------
    # VET AMINA — District Animal Health Officer (MOST ACCURATE NPC)
    # -------------------------------------------------------------------------
    "vet_amina": {
        "name": "Vet Amina",
        "role": "District Veterinary Officer",
        "avatar": "🐄",
        "cost": 150,
        "location": "District Office",
        "personality": "Highly technical. Calm and smart. Very helpful if trainees know to ask. Underutilized unless trainees use One Health lens.",
        
        "knowledge": [
            "Pigs are amplifying hosts for Japanese Encephalitis",
            "She has noticed: restlessness in pigs, fever in piglets, sows miscarrying",
            "Mosquito abundance is very high near pig pens",
            "Wild herons and egrets are common this season - they are natural JE reservoirs",
            "JE is transmitted by Culex mosquitoes between pigs and humans",
            "Humans are dead-end hosts - they don't transmit further"
        ],
        
        "hidden_clues": {
            "testing": "Pigs near Nalu tested JEV IgM-positive last year",
            "previous": "Pigs near Nalu tested JEV IgM-positive last year",
            "last year": "Pigs near Nalu tested JEV IgM-positive last year",
            "funding": "She warned district officials previously but got no funding for surveillance",
            "warning": "She warned district officials previously but got no funding for surveillance",
            "vaccine": "JE vaccination for pigs is not routinely available in this district"
        },
        
        "red_herrings": [],  # She is the most accurate NPC - no red herrings
        
        "does_not_know": [
            "Human lab results",
            "Timing of rice paddy expansion"
        ],
        
        "data_access": "vet_surveillance",
        "unlocks": "vet_data_unlocked"
    },
    
    # -------------------------------------------------------------------------
    # MR. OSEI — Environmental Health Officer
    # -------------------------------------------------------------------------
    "mr_osei": {
        "name": "Mr. Osei",
        "role": "Environmental Health Officer",
        "avatar": "🌊",
        "cost": 150,
        "location": "District Office",
        "personality": "Practical. Focused on water, sanitation, and environment. Notices physical clues others miss.",
        
        "knowledge": [
            "Rice paddy expansion led to more standing water",
            "Irrigation canal leaks are causing water pooling",
            "Pig waste accumulating near homes in Nalu",
            "High mosquito burden near fields & pig farms"
        ],
        
        "hidden_clues": {
            "tamu": "Seasonal pools in Tamu also breed mosquitoes but at lower intensity",
            "other villages": "Seasonal pools in Tamu also breed mosquitoes but at lower intensity",
            "pig cooperative": "Pig co-op near school violates safe distance guidelines",
            "school": "Pig co-op near school violates safe distance guidelines",
            "distance": "Pig co-op near school violates safe distance guidelines",
            "mosquito counts": "Evening mosquito counts are extremely high - he has data",
            "surveillance": "Evening mosquito counts are extremely high - he has data"
        },
        
        "red_herrings": [
            "May suspect waterborne toxin early on",
            "Mentions latrine overflow - irrelevant to JE"
        ],
        
        "does_not_know": [
            "Medical details",
            "Lab results",
            "Pig infection status"
        ],
        
        "data_access": "environmental_data",
        "unlocks": "entomology_data_unlocked"
    },
    
    # -------------------------------------------------------------------------
    # MAYOR SIMON — District Politician
    # -------------------------------------------------------------------------
    "mayor_simon": {
        "name": "Mayor Simon",
        "role": "District Politician",
        "avatar": "🏛️",
        "cost": 100,
        "location": "District Office",
        "personality": "Worries about money & reputation. Downplays outbreaks. Pushes for quick explanations. Politically sensitive.",
        
        "knowledge": [
            "He approved the rice paddy expansion project",
            "He denied funding for mosquito control last budget cycle",
            "Streetlights near paddies broke months ago - meaning more darkness at dusk"
        ],
        
        "hidden_clues": {
            "pig cooperative": "He knows pig co-op was placed too close to school - approved it anyway",
            "location": "He knows pig co-op was placed too close to school - approved it anyway",
            "siting": "He knows pig co-op was placed too close to school - approved it anyway",
            "vaccination": "He knows vaccination coverage is low but doesn't want it public",
            "coverage": "He knows vaccination coverage is low but doesn't want it public"
        },
        
        "red_herrings": [
            "Suggests 'school must be the source'",
            "Claims 'miners brought disease' - baseless"
        ],
        
        "does_not_know": [
            "Medical details",
            "Technical disease information",
            "Lab results"
        ],
        
        "data_access": None,
        "unlocks": None
    }
}

# ============================================================================
# AI INTERVIEW FUNCTION - WITH HIDDEN CLUES AND GUARDRAILS
# ============================================================================
def get_ai_response(char_key, user_input, history):
    """
    Generate character response using Claude API.
    Implements the NPC truth document system with:
    - Base knowledge (always available)
    - Hidden clues (only revealed if directly asked)
    - Red herrings (may mislead)
    - Guardrails (prevent hallucination)
    """
    char = CHARACTERS[char_key]
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    
    if not api_key:
        return "⚠️ API Key not configured. Please add ANTHROPIC_API_KEY to your Streamlit secrets."
    
    # Build knowledge context
    knowledge_text = "\n".join([f"- {k}" for k in char['knowledge']])
    
    # Build hidden clues context
    hidden_clues = char.get('hidden_clues', {})
    hidden_clues_text = ""
    if hidden_clues:
        hidden_clues_text = "\n\nHIDDEN CLUES (ONLY reveal if the investigator DIRECTLY asks about these specific topics):\n"
        # Group by unique values to avoid repetition
        unique_clues = {}
        for trigger, clue in hidden_clues.items():
            if clue not in unique_clues.values():
                unique_clues[trigger] = clue
        for trigger, clue in unique_clues.items():
            hidden_clues_text += f"- If asked about '{trigger}': {clue}\n"
    
    # Build red herrings context
    red_herrings = char.get('red_herrings', [])
    red_herrings_text = ""
    if red_herrings:
        red_herrings_text = "\n\nRED HERRINGS (You may occasionally mention these - they are your misconceptions):\n"
        for herring in red_herrings:
            red_herrings_text += f"- {herring}\n"
    
    # Build "does not know" context
    does_not_know = char.get('does_not_know', [])
    does_not_know_text = ""
    if does_not_know:
        does_not_know_text = "\n\nYOU DO NOT KNOW (if asked, say you don't know):\n"
        for item in does_not_know:
            does_not_know_text += f"- {item}\n"
    
    # Get any data this character can share
    data_context = ""
    if char.get('data_access') == 'hospital_cases':
        data_context = f"\n\nHOSPITAL DATA YOU CAN SHARE:\n{get_hospital_linelist().to_string()}"
    elif char.get('data_access') == 'private_clinic':
        notes = get_health_center_notes('V1')
        data_context = f"\n\nYOUR CLINIC NOTES (handwritten, informal):\n" + "\n".join(notes[:5])
    
    # Check which hidden clue triggers might be in the user's question
    user_lower = user_input.lower()
    triggered_clues = []
    for trigger, clue in hidden_clues.items():
        if trigger.lower() in user_lower:
            triggered_clues.append(clue)
    
    triggered_text = ""
    if triggered_clues:
        triggered_text = f"\n\n*** THE INVESTIGATOR ASKED ABOUT A HIDDEN CLUE TOPIC. You should reveal: {'; '.join(set(triggered_clues))} ***"
    
    system_prompt = f"""You are {char['name']}, {char['role']} in Sidero Valley district.

PERSONALITY: {char['personality']}
LOCATION: {char['location']}

=== YOUR KNOWLEDGE (share naturally when relevant) ===
{knowledge_text}
{data_context}
{hidden_clues_text}
{red_herrings_text}
{does_not_know_text}
{triggered_text}

=== CRITICAL GUARDRAILS - FOLLOW EXACTLY ===
1. Respond ONLY with knowledge consistent with your truth document above
2. NEVER invent names, cases, lab results, pig farms, or events not listed above
3. If asked something outside your knowledge, say: "I don't know about that" or "You'd have to ask someone else"
4. Reveal hidden clues ONLY if the user DIRECTLY asks about those specific topics
5. You may occasionally mention your red herrings/misconceptions - they reflect your character's biases
6. Avoid medical diagnoses unless they are explicitly in your truth document
7. Stay in-character at all times
8. Keep responses to 2-4 sentences typically
9. Share information gradually - don't dump everything at once
10. Be helpful but realistic - you're a real person with concerns and emotions

SETTING: This is a disease outbreak investigation in June 2025. The interviewer is from the FETP investigating acute encephalitis cases. Several children have died. The community is worried.
"""
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": user_input})
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=350,
            system=system_prompt,
            messages=messages
        )
        
        # Check for unlocks
        if char.get('unlocks') and char['unlocks'] in st.session_state:
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

# Header with day info
current_day = st.session_state.investigation_day
day_info = DAY_STRUCTURE.get(current_day, {})

st.markdown(f"""
<div class="main-header">
    <h1>🦟 Sidero Valley Outbreak Investigation</h1>
    <p style="margin:0; opacity:0.9;">FETP Intermediate 2.0 | {day_info.get('title', 'Day ' + str(current_day))}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SHOW DAILY INJECT (if not yet shown)
# ============================================================================
if current_day in DAY_STRUCTURE and not st.session_state.inject_shown.get(current_day, False):
    inject = DAY_STRUCTURE[current_day]['inject']
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%); 
                border-left: 5px solid #ffc107; 
                padding: 20px; 
                border-radius: 0 10px 10px 0;
                margin-bottom: 20px;">
        <h3>{inject['title']}</h3>
        <p><strong>From:</strong> {inject['source']}</p>
        <div style="white-space: pre-line;">{inject['message'].format(
            budget=st.session_state.budget,
            lab_credits=st.session_state.lab_credits
        )}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("✅ Acknowledge & Continue", key="ack_inject"):
        st.session_state.inject_shown[current_day] = True
        st.rerun()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    # Day Counter - prominent display
    st.markdown(f"""
    <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; margin-bottom: 15px; color: white;">
        <div style="font-size: 36px; font-weight: bold;">Day {current_day}</div>
        <div style="font-size: 14px; opacity: 0.9;">{day_info.get('title', '').replace('Day ' + str(current_day) + ': ', '')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Day objectives
    if 'objectives' in day_info:
        with st.expander(f"🎯 Day {current_day} Objectives", expanded=True):
            for obj in day_info['objectives']:
                st.markdown(f"• {obj}")
    
    # Resources
    st.markdown("### 💰 Resources")
    col1, col2 = st.columns(2)
    col1.metric("Budget", f"${st.session_state.budget:,}")
    col2.metric("Lab Credits", st.session_state.lab_credits)
    
    st.markdown("---")
    
    # Navigation - day-appropriate options
    st.markdown("### 🧭 Navigation")
    
    # Base navigation always available
    nav_options = [
        ('briefing', '📞 Briefing/Inject'),
        ('interviews', '👥 Interviews'),
        ('linelist', '📋 Line List'),
        ('epicurve', '📈 Epi Curve'),
        ('spotmap', '📍 Spot Map'),
        ('map', '🗺️ Field Sites'),
    ]
    
    # Day 2+: Study design
    if current_day >= 2:
        nav_options.append(('study_design', '🔬 Study Design'))
    
    # Day 3+: Analysis
    if current_day >= 3:
        nav_options.append(('analysis', '📊 Analysis'))
    
    # Day 4+: Laboratory
    if current_day >= 4:
        nav_options.append(('laboratory', '🧪 Laboratory'))
    
    # Day 5: Briefing
    if current_day >= 5:
        nav_options.append(('final_briefing', '🏛️ MOH Briefing'))
    
    # Always show debrief
    nav_options.append(('debrief', '📝 Debrief'))
    
    for key, label in nav_options:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.current_view = key
            st.rerun()
    
    st.markdown("---")
    
    # Progress Tracking by Day
    with st.expander("📋 Progress Checklist"):
        st.markdown("**Day 1: Detect & Describe**")
        st.markdown(f"{'✅' if st.session_state.case_definition_written else '⬜'} Case definition written")
        st.markdown(f"{'✅' if st.session_state.epi_curve_built else '⬜'} Epi curve reviewed")
        st.markdown(f"{'✅' if len(st.session_state.interview_history) > 0 else '⬜'} Initial interviews")
        
        if current_day >= 2:
            st.markdown("**Day 2: Hypotheses & Design**")
            st.markdown(f"{'✅' if len(st.session_state.hypotheses_documented) > 0 else '⬜'} Hypotheses documented")
            st.markdown(f"{'✅' if st.session_state.study_design_chosen else '⬜'} Study design chosen")
            st.markdown(f"{'✅' if st.session_state.questionnaire_submitted else '⬜'} Questionnaire submitted")
        
        if current_day >= 3:
            st.markdown("**Day 3: Analysis**")
            st.markdown(f"{'✅' if st.session_state.data_cleaning_done else '⬜'} Data cleaned")
            st.markdown(f"{'✅' if st.session_state.descriptive_analysis_done else '⬜'} Descriptive analysis")
            st.markdown(f"{'✅' if st.session_state.analytic_results else '⬜'} Analytic results")
        
        if current_day >= 4:
            st.markdown("**Day 4: Lab & Environment**")
            st.markdown(f"{'✅' if len(st.session_state.lab_samples_submitted) > 0 else '⬜'} Samples submitted")
            st.markdown(f"{'✅' if st.session_state.pig_samples_collected else '⬜'} Pig samples collected")
            st.markdown(f"{'✅' if len(st.session_state.mosquito_traps_set) > 0 else '⬜'} Mosquito pools tested")
        
        if current_day >= 5:
            st.markdown("**Day 5: Briefing**")
            st.markdown(f"{'✅' if st.session_state.briefing_prepared else '⬜'} Briefing prepared")
            st.markdown(f"{'✅' if st.session_state.final_diagnosis else '⬜'} Diagnosis submitted")
    
    # One Health Progress
    with st.expander("🌍 One Health Integration"):
        st.markdown(f"{'✅' if st.session_state.vet_unlocked else '🔒'} Veterinary Officer consulted")
        st.markdown(f"{'✅' if st.session_state.env_officer_unlocked else '🔒'} Environmental Officer consulted")
        st.markdown(f"{'✅' if st.session_state.pig_samples_collected else '🔒'} Animal samples collected")
        st.markdown(f"{'✅' if len(st.session_state.mosquito_traps_set) > 0 else '🔒'} Vector sampling done")
    
    st.markdown("---")
    
    # Advance Day button - moved to bottom
    if current_day < 5:
        if st.button(f"⏭️ Complete Day {current_day} → Day {current_day + 1}", use_container_width=True, type="primary"):
            if advance_day():
                st.rerun()
    else:
        st.success("📋 Final Day - Complete your briefing!")
    
    st.markdown("---")
    
    # Facilitator mode
    with st.expander("🔐 Facilitator Mode"):
        password = st.text_input("Password:", type="password", key="fac_pw")
        if password == FACILITATOR_PASSWORD:
            st.session_state.facilitator_mode = True
            st.success("✅ Facilitator mode enabled")
        elif password:
            st.error("Incorrect password")

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
        - **Season:** Rainy season
        - **Health facilities:** 1 district hospital, 2 health centers
        """)
        
        st.markdown("**Your First Steps:**")
        st.markdown("""
        1. Review initial cases
        2. Develop a working case definition
        3. Begin interviews to understand the situation
        4. Document initial hypotheses
        """)
    
    st.markdown("---")
    
    # Case Definition Exercise
    st.markdown("### 📝 Develop Your Case Definition")
    
    with st.form("case_definition_form"):
        st.markdown("**Clinical Criteria:**")
        clinical = st.text_area(
            "What symptoms/signs define a case?",
            height=80
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Person:**")
            person = st.text_input("Who? (age, other characteristics)")
        
        with col2:
            st.markdown("**Place:**")
            place = st.text_input("Where?")
        
        st.markdown("**Time:**")
        time_period = st.text_input("When?")
        
        if st.form_submit_button("Save Case Definition"):
            full_def = f"Clinical: {clinical}\nPerson: {person}\nPlace: {place}\nTime: {time_period}"
            st.session_state.case_definition_text = full_def
            st.session_state.case_definition_written = True
            st.success("✅ Case definition saved! You can refine it as you learn more.")
    
    st.markdown("---")
    
    # Initial Hypotheses Section
    st.markdown("### 💡 Document Initial Hypotheses")
    
    with st.form("hypotheses_form"):
        st.markdown("Based on what you know so far, what are possible causes of this outbreak?")
        
        hypothesis1 = st.text_input("Hypothesis 1:")
        hypothesis2 = st.text_input("Hypothesis 2:")
        hypothesis3 = st.text_input("Hypothesis 3:")
        hypothesis4 = st.text_input("Hypothesis 4 (optional):")
        
        if st.form_submit_button("Save Hypotheses"):
            hypotheses = [h for h in [hypothesis1, hypothesis2, hypothesis3, hypothesis4] if h.strip()]
            st.session_state.hypotheses_documented = hypotheses
            st.session_state.initial_hypotheses = hypotheses
            st.success(f"✅ {len(hypotheses)} hypotheses documented!")

elif st.session_state.current_view == 'interviews':
    st.markdown("### 👥 Key Informant Interviews")
    st.info(f"💰 Budget: ${st.session_state.budget:,} | Day {st.session_state.investigation_day}")
    
    # Show unlock notifications
    if st.session_state.one_health_triggered and not st.session_state.get('one_health_notified'):
        st.success("🌍 **One Health Approach Activated!** You've unlocked additional expert NPCs by asking about animals/environment.")
        st.session_state.one_health_notified = True
    
    # Active Interview takes priority
    if st.session_state.current_character:
        char = CHARACTERS[st.session_state.current_character]
        
        st.markdown(f"### 💬 Interviewing {char['avatar']} {char['name']}")
        st.caption(f"{char['role']} | 📍 {char['location']}")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔙 End"):
                st.session_state.current_character = None
                st.rerun()
        
        st.markdown("---")
        
        # Display conversation history
        history = st.session_state.interview_history.get(st.session_state.current_character, [])
        
        for msg in history:
            if msg['role'] == 'user':
                with st.chat_message("user"):
                    st.write(msg['content'])
            else:
                with st.chat_message("assistant", avatar=char['avatar']):
                    st.write(msg['content'])
        
        # Chat input
        if prompt := st.chat_input("Ask a question..."):
            # Check for NPC unlock triggers BEFORE showing response
            unlock_msg = check_npc_unlock_triggers(prompt)
            
            with st.chat_message("user"):
                st.write(prompt)
            
            if st.session_state.current_character not in st.session_state.interview_history:
                st.session_state.interview_history[st.session_state.current_character] = []
            
            st.session_state.interview_history[st.session_state.current_character].append({
                "role": "user", "content": prompt
            })
            
            with st.chat_message("assistant", avatar=char['avatar']):
                with st.spinner("..."):
                    response = get_ai_response(
                        st.session_state.current_character,
                        prompt,
                        st.session_state.interview_history[st.session_state.current_character][:-1]
                    )
                    st.write(response)
            
            st.session_state.interview_history[st.session_state.current_character].append({
                "role": "assistant", "content": response
            })
            
            # Show unlock notification if triggered
            if unlock_msg:
                st.info(unlock_msg)
            
            st.rerun()
    
    else:
        # NPC Selection Grid with unlock status
        st.markdown("#### Select someone to interview:")
        
        # Group by location with unlock status
        locations = {
            'District Hospital': ['dr_chen', 'nurse_joy'],
            'District Office': ['vet_amina', 'mr_osei', 'mayor_simon'],
            'Nalu Village': ['healer_marcus', 'mama_kofi', 'teacher_grace'],
            'Other Locations': ['foreman_rex', 'auntie_ama']
        }
        
        for location, char_keys in locations.items():
            st.markdown(f"##### 📍 {location}")
            
            cols = st.columns(min(len(char_keys), 4))
            for i, char_key in enumerate(char_keys):
                if char_key not in CHARACTERS:
                    continue
                    
                char = CHARACTERS[char_key]
                is_unlocked = char_key in st.session_state.npcs_unlocked
                interviewed = char_key in st.session_state.interview_history
                
                with cols[i % 4]:
                    # Show locked/unlocked status
                    if not is_unlocked:
                        st.markdown(f"**🔒 {char['name']}**")
                        st.caption(f"{char['role']}")
                        st.caption("*Not yet available*")
                    else:
                        st.markdown(f"**{char['avatar']} {char['name']}**")
                        st.caption(char['role'])
                        st.caption(f"Cost: ${char['cost']}")
                        
                        if interviewed:
                            st.success("✓ Interviewed")
                        
                        btn_label = "Continue" if interviewed else "Talk"
                        if st.button(f"{btn_label}", key=f"btn_{char_key}"):
                            cost = 0 if interviewed else char['cost']
                            if st.session_state.budget >= cost:
                                if not interviewed:
                                    st.session_state.budget -= cost
                                st.session_state.current_character = char_key
                                if char_key not in st.session_state.interview_history:
                                    st.session_state.interview_history[char_key] = []
                                st.rerun()
                            else:
                                st.error("Insufficient funds!")
            
            st.markdown("---")

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

elif st.session_state.current_view == 'spotmap':
    st.markdown("### 📍 Spot Map - Geographic Distribution of Cases")
    
    st.session_state.spot_map_viewed = True
    
    # Get case data
    cases = TRUTH['individuals'][TRUTH['individuals']['symptomatic_AES'] == True].copy()
    
    # Assign coordinates with jitter for visualization
    village_coords = {
        'V1': {'lat': 5.55, 'lon': -0.20, 'name': 'Nalu'},
        'V2': {'lat': 5.52, 'lon': -0.15, 'name': 'Kabwe'},
        'V3': {'lat': 5.58, 'lon': -0.12, 'name': 'Tamu'}
    }
    
    # Add coordinates with jitter
    np.random.seed(42)
    cases['lat'] = cases['village_id'].apply(
        lambda v: village_coords.get(v, {}).get('lat', 5.55) + np.random.uniform(-0.015, 0.015)
    )
    cases['lon'] = cases['village_id'].apply(
        lambda v: village_coords.get(v, {}).get('lon', -0.18) + np.random.uniform(-0.015, 0.015)
    )
    cases['village_name'] = cases['village_id'].map(lambda v: village_coords.get(v, {}).get('name', 'Unknown'))
    
    # Color by severity
    cases['severity'] = cases['severe_neuro'].map({True: 'Severe', False: 'Mild'})
    cases['color'] = cases['outcome'].map({
        'died': 'red',
        'recovered_sequelae': 'orange', 
        'recovered': 'yellow'
    }).fillna('yellow')
    
    # Create map
    fig = px.scatter_mapbox(
        cases,
        lat='lat',
        lon='lon',
        color='severity',
        color_discrete_map={'Severe': '#e74c3c', 'Mild': '#f39c12'},
        size_max=15,
        hover_data=['age', 'sex', 'village_name', 'onset_date', 'outcome'],
        zoom=11,
        height=500
    )
    
    # Add village markers
    for vid, coords in village_coords.items():
        fig.add_trace(go.Scattermapbox(
            lat=[coords['lat']],
            lon=[coords['lon']],
            mode='markers+text',
            marker=dict(size=20, color='blue', opacity=0.5),
            text=[coords['name']],
            textposition='top center',
            name=coords['name'],
            showlegend=False
        ))
    
    # Add pig cooperative marker
    fig.add_trace(go.Scattermapbox(
        lat=[5.545],
        lon=[-0.195],
        mode='markers+text',
        marker=dict(size=15, symbol='circle', color='brown'),
        text=['🐷 Pig Co-op'],
        textposition='bottom center',
        name='Pig Cooperative',
        showlegend=False
    ))
    
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("---")
    st.markdown("#### 📊 Geographic Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        nalu_cases = len(cases[cases['village_id'] == 'V1'])
        st.metric("Nalu (V1)", f"{nalu_cases} cases")
    
    with col2:
        kabwe_cases = len(cases[cases['village_id'] == 'V2'])
        st.metric("Kabwe (V2)", f"{kabwe_cases} cases")
    
    with col3:
        tamu_cases = len(cases[cases['village_id'] == 'V3'])
        st.metric("Tamu (V3)", f"{tamu_cases} cases")
    
    # Interpretation prompts
    with st.expander("🤔 Spot Map Interpretation Questions"):
        st.markdown("""
        Consider these questions as you review the geographic distribution:
        
        1. **Clustering:** Do cases cluster in specific areas? What might explain this?
        2. **Village comparison:** Why might Nalu have more cases than Tamu?
        3. **Environmental features:** What is located near the case clusters?
        4. **Hypothesis generation:** What geographic exposures might explain this pattern?
        """)

elif st.session_state.current_view == 'analysis':
    st.markdown("### 📊 Data Analysis - Day 3")
    
    if not st.session_state.questionnaire_submitted:
        st.warning("⚠️ You need to complete study design and deploy your field team first (Study Design tab).")
    else:
        st.success("✅ Your field team has returned with data!")
        
        st.session_state.descriptive_analysis_done = True
        
        # Generate a realistic dataset based on ground truth
        cases = TRUTH['individuals'][TRUTH['individuals']['symptomatic_AES'] == True].copy()
        
        # Sample controls (non-cases from same villages)
        non_cases = TRUTH['individuals'][
            (TRUTH['individuals']['symptomatic_AES'] == False) &
            (TRUTH['individuals']['village_id'].isin(['V1', 'V2']))
        ].sample(n=min(30, len(TRUTH['individuals'])), random_state=42)
        
        # Combine into study dataset
        study_df = pd.concat([
            cases.assign(case_status=1),
            non_cases.assign(case_status=0)
        ])
        
        # Add household-level variables
        hh_lookup = TRUTH['households'].set_index('hh_id').to_dict('index')
        
        study_df['pigs_nearby'] = study_df['hh_id'].apply(
            lambda hh: hh_lookup.get(hh, {}).get('pigs_owned', 0) > 0 and 
                       (hh_lookup.get(hh, {}).get('pig_pen_distance_m') or 100) < 30
        )
        study_df['uses_nets'] = study_df['hh_id'].apply(
            lambda hh: hh_lookup.get(hh, {}).get('uses_mosquito_nets', True)
        )
        
        # Inject some missing values for realism
        np.random.seed(42)
        for col in ['JE_vaccinated', 'evening_outdoor_exposure']:
            mask = np.random.random(len(study_df)) < 0.08
            study_df.loc[mask, col] = np.nan
        
        st.markdown("#### 📋 Your Study Dataset")
        st.caption(f"n = {len(study_df)} ({len(cases)} cases, {len(non_cases)} controls)")
        
        # Show data
        display_cols = ['person_id', 'age', 'sex', 'village_id', 'case_status', 
                       'JE_vaccinated', 'evening_outdoor_exposure', 'pigs_nearby', 'uses_nets']
        st.dataframe(study_df[display_cols], use_container_width=True, hide_index=True)
        
        # Download button
        csv = study_df[display_cols].to_csv(index=False)
        st.download_button("📥 Download Dataset (CSV)", csv, "study_data.csv", "text/csv")
        
        st.markdown("---")
        st.markdown("#### 📈 Descriptive Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Age Distribution by Case Status**")
            fig = px.box(study_df, x='case_status', y='age', 
                        labels={'case_status': 'Case Status (1=Case, 0=Control)', 'age': 'Age (years)'},
                        color='case_status',
                        color_discrete_map={1: '#e74c3c', 0: '#3498db'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Sex Distribution**")
            sex_table = pd.crosstab(study_df['sex'], study_df['case_status'], margins=True)
            sex_table.columns = ['Controls', 'Cases', 'Total']
            sex_table.index = ['Female', 'Male', 'Total']
            st.table(sex_table)
        
        st.markdown("---")
        st.markdown("#### 🔬 Analytical Results (2×2 Tables)")
        
        # Function to calculate OR and display 2x2 table
        def show_2x2_table(df, exposure_col, exposure_name):
            # Filter out missing values
            df_clean = df.dropna(subset=[exposure_col])
            
            # Create 2x2 counts
            a = len(df_clean[(df_clean[exposure_col] == True) & (df_clean['case_status'] == 1)])
            b = len(df_clean[(df_clean[exposure_col] == True) & (df_clean['case_status'] == 0)])
            c = len(df_clean[(df_clean[exposure_col] == False) & (df_clean['case_status'] == 1)])
            d = len(df_clean[(df_clean[exposure_col] == False) & (df_clean['case_status'] == 0)])
            
            table_data = {
                '': ['Exposed', 'Unexposed', 'Total'],
                'Cases': [a, c, a+c],
                'Controls': [b, d, b+d],
                'Total': [a+b, c+d, a+b+c+d]
            }
            
            st.markdown(f"**{exposure_name}**")
            st.table(pd.DataFrame(table_data))
            
            # Calculate OR
            if c > 0 and b > 0:
                OR = (a * d) / (b * c)
                st.metric("Odds Ratio", f"{OR:.2f}")
                if OR > 2:
                    st.caption("⚠️ Strong positive association")
                elif OR < 0.5:
                    st.caption("🛡️ Protective association")
            else:
                st.caption("Cannot calculate OR (zero cell)")
            
            return a, b, c, d
        
        tab1, tab2, tab3, tab4 = st.tabs(["Pig Proximity", "Mosquito Nets", "Vaccination", "Evening Exposure"])
        
        with tab1:
            show_2x2_table(study_df, 'pigs_nearby', 'Living within 30m of pig pens')
        
        with tab2:
            # Invert for risk (not using nets = exposed)
            study_df['no_nets'] = ~study_df['uses_nets']
            show_2x2_table(study_df, 'no_nets', 'NOT using mosquito nets')
        
        with tab3:
            # Invert for risk (not vaccinated = exposed)
            study_df['not_vaccinated'] = study_df['JE_vaccinated'].apply(lambda x: not x if pd.notna(x) else np.nan)
            show_2x2_table(study_df, 'not_vaccinated', 'NOT vaccinated against JE')
        
        with tab4:
            show_2x2_table(study_df, 'evening_outdoor_exposure', 'Evening outdoor activities')
        
        st.markdown("---")
        with st.expander("📖 Interpreting Your Results"):
            st.markdown("""
            **Odds Ratio Interpretation:**
            - OR = 1.0: No association
            - OR > 1.0: Exposure associated with increased odds of disease
            - OR < 1.0: Exposure associated with decreased odds (protective)
            
            **Statistical Significance:**
            - Look for OR confidence intervals that don't cross 1.0
            - Consider biological plausibility
            - Triangulate with other evidence (interviews, environmental, lab)
            
            **Confounding:**
            - Could age or village explain the association?
            - Consider stratified analysis
            """)

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
                    st.session_state.lab_samples_submitted.append({
                        'sample_type': sample_type,
                        'village': village
                    })
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
                    st.session_state.lab_samples_submitted.append({
                        'sample_type': sample_type,
                        'village': village
                    })
                    # Track specific sample types
                    if sample_type == 'pig_serum':
                        st.session_state.pig_samples_collected = True
                    if sample_type == 'mosquito_pool':
                        st.session_state.mosquito_traps_set.append(village)
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
        selected_exposures = st.multiselect(
            "Select exposures to include in your questionnaire:",
            ["Proximity to pig pens (<30m)",
             "Mosquito net use",
             "Distance to rice paddies",
             "Evening outdoor activities",
             "JE vaccination status",
             "Water source",
             "Occupation",
             "Recent travel"],
            default=["Proximity to pig pens (<30m)", "Mosquito net use", "JE vaccination status"]
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
                st.session_state.questionnaire_submitted = True
                st.session_state.study_design_chosen = study_type
                st.session_state.hypothesis_text = hypothesis
                st.session_state.questionnaire_variables = selected_exposures
                st.success("✅ Field team deployed! Data collection in progress. Go to Analysis tab on Day 3 to see results.")
            else:
                st.error("Insufficient funds!")

# ============================================================================
# DAY 5: MOH BRIEFING VIEW
# ============================================================================
elif st.session_state.current_view == 'final_briefing':
    st.markdown("### 🏛️ MOH Briefing - Day 5")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); 
                color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin:0;">Present Your Findings to the District Director of Health</h3>
        <p style="margin:10px 0 0 0; opacity:0.9;">Your recommendations will determine the outbreak response.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.consequence_outcome:
        # Already submitted - show results
        outcome = st.session_state.consequence_outcome
        
        if outcome['status'] == 'SUCCESS':
            st.success(f"## ✅ {outcome['status']}")
        elif outcome['status'] == 'PARTIAL SUCCESS':
            st.warning(f"## ⚠️ {outcome['status']}")
        else:
            st.error(f"## ❌ {outcome['status']}")
        
        st.markdown(outcome['narrative'])
        
        st.markdown("---")
        st.markdown("### 📊 Performance Assessment")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Score", f"{outcome['score']}/100")
            st.metric("New Cases After Intervention", outcome['new_cases'])
        
        with col2:
            st.markdown("**Key Decisions:**")
            for item in outcome['outcomes']:
                st.markdown(item)
    
    else:
        # Briefing form
        st.markdown("### Your Briefing")
        
        with st.form("moh_briefing"):
            st.markdown("#### 1. Outbreak Summary")
            summary = st.text_area(
                "Summarize the outbreak (who, what, where, when):",
                height=100,
                placeholder="e.g., Since June 3, 2025, we have identified X cases of acute encephalitis syndrome..."
            )
            
            st.markdown("#### 2. Your Diagnosis")
            diagnosis = st.selectbox(
                "What is the most likely cause of this outbreak?",
                ["Select diagnosis...", "Japanese Encephalitis", "Bacterial meningitis", 
                 "Cerebral malaria", "Rabies", "Nipah virus", "Chemical/toxic exposure",
                 "Waterborne pathogen", "Unknown viral encephalitis"]
            )
            st.session_state.final_diagnosis = diagnosis
            
            st.markdown("#### 3. Key Risk Factors Identified")
            risk_factors = st.multiselect(
                "Select the risk factors your investigation identified:",
                ["Proximity to pig pens", "Lack of mosquito net use", "Evening outdoor activities",
                 "Low JE vaccination coverage", "Proximity to rice paddies", "Contaminated water",
                 "School exposure", "Mine chemical exposure", "Food contamination"]
            )
            
            st.markdown("#### 4. Evidence Supporting Your Diagnosis")
            evidence = st.text_area(
                "What evidence supports your conclusion?",
                height=100,
                placeholder="e.g., Lab results showed..., Epidemiologic analysis revealed..., Environmental sampling confirmed..."
            )
            
            st.markdown("#### 5. Recommended Interventions")
            st.markdown("*List your top recommendations for controlling this outbreak:*")
            
            rec1 = st.text_input("Recommendation 1 (highest priority):", 
                                placeholder="e.g., Emergency JE vaccination campaign")
            rec2 = st.text_input("Recommendation 2:", 
                                placeholder="e.g., Bed net distribution")
            rec3 = st.text_input("Recommendation 3:", 
                                placeholder="e.g., Mosquito breeding site control")
            rec4 = st.text_input("Recommendation 4:", 
                                placeholder="e.g., Pig farm relocation")
            rec5 = st.text_input("Recommendation 5:", 
                                placeholder="e.g., Community health education")
            
            st.markdown("#### 6. One Health Considerations")
            one_health = st.text_area(
                "How should human, animal, and environmental health sectors coordinate?",
                height=80,
                placeholder="e.g., Joint surveillance between health and veterinary services..."
            )
            
            if st.form_submit_button("📤 Submit Briefing to Director", type="primary"):
                # Store recommendations
                st.session_state.recommendations_submitted = [rec1, rec2, rec3, rec4, rec5]
                st.session_state.final_diagnosis = diagnosis
                st.session_state.briefing_prepared = True
                
                # Calculate consequences
                outcome = calculate_consequences()
                st.session_state.consequence_outcome = outcome
                
                st.rerun()
        
        # Pre-briefing checklist
        st.markdown("---")
        st.markdown("### 📋 Pre-Briefing Checklist")
        
        checks = [
            ("Completed Day 1-4 activities", st.session_state.investigation_day >= 5),
            ("Interviewed key informants", len(st.session_state.interview_history) >= 3),
            ("Built epidemic curve", st.session_state.epi_curve_built),
            ("Submitted lab samples", len(st.session_state.lab_samples_submitted) > 0),
            ("Consulted veterinary officer (One Health)", 'vet_amina' in st.session_state.interview_history),
        ]
        
        for item, done in checks:
            st.markdown(f"{'✅' if done else '⬜'} {item}")

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
        "Identified pigs as amplifying host": 'vet_amina' in st.session_state.interview_history,
        "Connected outbreak to pig cooperative": 'healer_marcus' in st.session_state.interview_history or 'mama_kofi' in st.session_state.interview_history,
        "Identified Culex mosquitoes as vector": 'mr_osei' in st.session_state.interview_history or st.session_state.entomology_data_unlocked,
        "Found community cases not in hospital data": st.session_state.health_center_nalu_unlocked,
        "Consulted veterinary officer (One Health)": 'vet_amina' in st.session_state.interview_history,
        "Tested environmental samples": any(r['sample_type'] in ['mosquito_pool', 'pig_serum'] for r in st.session_state.lab_results),
        "Inspected pig cooperative site": 'ES02' in st.session_state.sites_inspected,
        "Built epidemic curve": st.session_state.epi_curve_built,
        "Viewed spot map": st.session_state.spot_map_viewed,
        "Deployed analytical study": st.session_state.questionnaire_submitted
    }
    
    score = sum(critical_clues.values())
    
    for finding, achieved in critical_clues.items():
        icon = "✅" if achieved else "❌"
        st.markdown(f"{icon} {finding}")
    
    st.markdown(f"### Score: {score}/{len(critical_clues)} critical findings")
    
    # Progress bar
    st.progress(score / len(critical_clues))
    
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