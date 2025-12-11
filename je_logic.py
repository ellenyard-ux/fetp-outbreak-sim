"""
je_logic.py - Japanese Encephalitis Outbreak Simulation Logic

This module contains the core logic for:
- Loading truth data
- Generating populations from seed data
- Case definition → dataset generation
- Lab test simulation
- Consequence engine

Separating logic from UI enables:
- Easier testing
- Plug-and-play scenarios
- Cleaner code organization
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# DATA LOADING
# ============================================================================

def load_truth_data(data_dir: str = "data"):
    """
    Load all truth tables from CSV/JSON files.
    Returns a dictionary of DataFrames and the NPC truth dict.
    """
    data_path = Path(data_dir)
    
    truth = {
        'villages': pd.read_csv(data_path / "villages.csv"),
        'households_seed': pd.read_csv(data_path / "households_seed.csv"),
        'individuals_seed': pd.read_csv(data_path / "individuals_seed.csv"),
        'lab_samples': pd.read_csv(data_path / "lab_samples.csv"),
        'environment_sites': pd.read_csv(data_path / "environment_sites.csv"),
    }
    
    with open(data_path / "npc_truth.json") as f:
        truth['npc_truth'] = json.load(f)
    
    return truth


def generate_full_population(villages_df, households_seed, individuals_seed, random_seed=42):
    """
    Generate a complete population from seed data + generation rules.
    
    Uses:
    - Poisson distribution for pig ownership (λ=3 Nalu, λ=1 Kabwe, 0-1 Tamu)
    - Village-specific net use rates (30%, 50%, 70%)
    - Risk-based infection assignment
    """
    np.random.seed(random_seed)
    
    # Start with seed data
    all_households = [households_seed.copy()]
    all_individuals = [individuals_seed.copy()]
    
    # Generation parameters
    village_params = {
        'V1': {'pig_lambda': 3, 'net_rate': 0.30, 'rice_dist': (20, 150), 'proportion': 0.40},
        'V2': {'pig_lambda': 1, 'net_rate': 0.50, 'rice_dist': (80, 200), 'proportion': 0.40},
        'V3': {'pig_lambda': 0.2, 'net_rate': 0.70, 'rice_dist': (200, 500), 'proportion': 0.20}
    }
    
    target_households = 350
    hh_counter = 300
    person_counter = 3000
    
    # Track existing IDs
    existing_hh_ids = set(households_seed['hh_id'].tolist())
    
    # Generate additional households
    for village_id, params in village_params.items():
        n_hh = int(target_households * params['proportion'])
        village_row = villages_df[villages_df['village_id'] == village_id].iloc[0]
        
        for _ in range(n_hh):
            hh_id = f'HH{hh_counter:03d}'
            hh_counter += 1
            
            # Pig ownership (Poisson)
            pigs = min(np.random.poisson(params['pig_lambda']), 12)
            pig_dist = np.random.uniform(5, 50) if pigs > 0 else None
            
            # Mosquito nets
            nets = np.random.random() < params['net_rate']
            
            # Rice field distance
            rice_dist = np.random.uniform(*params['rice_dist'])
            
            # Children
            n_children = min(np.random.poisson(1.8), 5)
            
            # Child vaccination
            vacc_coverage = village_row['JE_vacc_coverage']
            vacc_probs = [
                1 - vacc_coverage,  # none
                vacc_coverage * 0.4,  # low
                vacc_coverage * 0.35,  # medium
                vacc_coverage * 0.25   # high
            ]
            child_vacc = np.random.choice(['none', 'low', 'medium', 'high'], p=vacc_probs)
            
            all_households.append(pd.DataFrame([{
                'hh_id': hh_id,
                'village_id': village_id,
                'pigs_owned': pigs,
                'pig_pen_distance_m': pig_dist,
                'uses_mosquito_nets': nets,
                'rice_field_distance_m': rice_dist,
                'children_under_15': n_children,
                'JE_vaccination_children': child_vacc
            }]))
            
            # Generate household members
            n_adults = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])
            
            for i in range(n_adults):
                age = np.random.randint(18, 65)
                sex = 'M' if i == 0 and np.random.random() < 0.6 else np.random.choice(['M', 'F'])
                occupation = np.random.choice(
                    ['farmer', 'trader', 'teacher', 'healthcare', 'other'],
                    p=[0.50, 0.20, 0.10, 0.05, 0.15]
                )
                vaccinated = np.random.random() < (vacc_coverage * 0.5)
                evening_outdoor = np.random.random() < (0.8 if occupation == 'farmer' else 0.4)
                
                all_individuals.append(pd.DataFrame([{
                    'person_id': f'P{person_counter:04d}',
                    'hh_id': hh_id,
                    'village_id': village_id,
                    'age': age,
                    'sex': sex,
                    'occupation': occupation,
                    'JE_vaccinated': vaccinated,
                    'evening_outdoor_exposure': evening_outdoor,
                    'true_je_infection': False,
                    'symptomatic_AES': False,
                    'severe_neuro': False,
                    'onset_date': None,
                    'outcome': None,
                    'name_hint': None
                }]))
                person_counter += 1
            
            # Generate children
            for i in range(n_children):
                age = np.random.randint(1, 15)
                sex = np.random.choice(['M', 'F'])
                occupation = 'child' if age < 6 else 'student'
                
                if child_vacc == 'high':
                    vaccinated = np.random.random() < 0.85
                elif child_vacc == 'medium':
                    vaccinated = np.random.random() < 0.50
                elif child_vacc == 'low':
                    vaccinated = np.random.random() < 0.20
                else:
                    vaccinated = False
                
                evening_outdoor = np.random.random() < 0.7
                
                all_individuals.append(pd.DataFrame([{
                    'person_id': f'P{person_counter:04d}',
                    'hh_id': hh_id,
                    'village_id': village_id,
                    'age': age,
                    'sex': sex,
                    'occupation': occupation,
                    'JE_vaccinated': vaccinated,
                    'evening_outdoor_exposure': evening_outdoor,
                    'true_je_infection': False,
                    'symptomatic_AES': False,
                    'severe_neuro': False,
                    'onset_date': None,
                    'outcome': None,
                    'name_hint': None
                }]))
                person_counter += 1
    
    households_df = pd.concat(all_households, ignore_index=True)
    individuals_df = pd.concat(all_individuals, ignore_index=True)
    
    # Assign infections using risk model (skip seed individuals)
    individuals_df = assign_infections(individuals_df, households_df)
    
    return households_df, individuals_df


def assign_infections(individuals_df, households_df):
    """
    Assign JE infections based on risk model.
    Preserves seed individual status.
    """
    # Base risk by village
    base_risk = {'V1': 0.18, 'V2': 0.07, 'V3': 0.02}
    
    # Create household lookup
    hh_lookup = households_df.set_index('hh_id').to_dict('index')
    
    def calculate_risk(row):
        # Seed individuals keep their status
        if row['person_id'].startswith('P0') or row['person_id'].startswith('P1') or row['person_id'].startswith('P2'):
            if len(row['person_id']) <= 5:  # P0001, P1001, etc.
                return row['true_je_infection']
        
        risk = base_risk.get(row['village_id'], 0.02)
        
        hh = hh_lookup.get(row['hh_id'], {})
        if hh:
            if hh.get('pigs_owned', 0) >= 3:
                risk += 0.08
            if pd.notna(hh.get('pig_pen_distance_m')) and hh.get('pig_pen_distance_m', 100) < 20:
                risk += 0.05
            if not hh.get('uses_mosquito_nets', True):
                risk += 0.05
            if hh.get('rice_field_distance_m', 200) < 100:
                risk += 0.04
        
        if row.get('JE_vaccinated', False):
            risk *= 0.15
        
        return np.random.random() < min(risk, 0.4)
    
    individuals_df['true_je_infection'] = individuals_df.apply(calculate_risk, axis=1)
    
    # Symptomatic AES
    def assign_symptomatic(row):
        if row['person_id'].startswith('P0') or row['person_id'].startswith('P1') or row['person_id'].startswith('P2'):
            if len(row['person_id']) <= 5:
                return row['symptomatic_AES']
        if not row['true_je_infection']:
            return False
        p_symp = 0.15 if row['age'] < 15 else 0.05
        return np.random.random() < p_symp
    
    individuals_df['symptomatic_AES'] = individuals_df.apply(assign_symptomatic, axis=1)
    
    # Severe neuro
    def assign_severe(row):
        if row['person_id'].startswith('P0') or row['person_id'].startswith('P1') or row['person_id'].startswith('P2'):
            if len(row['person_id']) <= 5:
                return row['severe_neuro']
        if not row['symptomatic_AES']:
            return False
        return np.random.random() < 0.25
    
    individuals_df['severe_neuro'] = individuals_df.apply(assign_severe, axis=1)
    
    # Onset dates
    def assign_onset(row):
        if pd.notna(row['onset_date']):
            return row['onset_date']
        if not row['symptomatic_AES']:
            return None
        
        base = datetime(2025, 6, 1)
        if row['village_id'] == 'V1':
            offset = np.random.randint(2, 8)  # June 3-7
        elif row['village_id'] == 'V2':
            offset = np.random.randint(5, 11)  # June 6-10
        else:
            offset = np.random.randint(4, 13)  # June 5-12
        
        return (base + timedelta(days=offset)).strftime('%Y-%m-%d')
    
    individuals_df['onset_date'] = individuals_df.apply(assign_onset, axis=1)
    
    # Outcomes
    def assign_outcome(row):
        if pd.notna(row['outcome']):
            return row['outcome']
        if not row['symptomatic_AES']:
            return None
        if row['severe_neuro']:
            r = np.random.random()
            if r < 0.20:
                return 'died'
            elif r < 0.65:
                return 'recovered_sequelae'
            else:
                return 'recovered'
        return 'recovered' if np.random.random() < 0.95 else 'recovered_sequelae'
    
    individuals_df['outcome'] = individuals_df.apply(assign_outcome, axis=1)
    
    return individuals_df


# ============================================================================
# CASE DEFINITION & DATASET GENERATION
# ============================================================================

def apply_case_definition(individuals_df, criteria):
    """
    Filter individuals based on case definition criteria.
    
    criteria example:
    {
        "clinical_AES": True,
        "min_onset_date": "2025-06-01",
        "max_onset_date": "2025-06-30",
        "villages": ["V1", "V2"],
        "min_age": 0,
        "max_age": 100
    }
    """
    df = individuals_df.copy()
    
    if criteria.get("clinical_AES", True):
        df = df[df["symptomatic_AES"] == True]
    
    if criteria.get("min_onset_date"):
        df = df[df["onset_date"] >= criteria["min_onset_date"]]
    
    if criteria.get("max_onset_date"):
        df = df[df["onset_date"] <= criteria["max_onset_date"]]
    
    if criteria.get("villages"):
        df = df[df["village_id"].isin(criteria["villages"])]
    
    if "min_age" in criteria:
        df = df[df["age"] >= criteria["min_age"]]
    
    if "max_age" in criteria:
        df = df[df["age"] <= criteria["max_age"]]
    
    return df


def generate_study_dataset(individuals_df, households_df, decisions, random_seed=42):
    """
    Generate a trainee-visible dataset based on their study design and questionnaire.
    
    decisions = {
        "case_definition": {...},
        "study_design": {"type": "case_control", "controls_per_case": 2},
        "mapped_columns": ["age", "sex", "JE_vaccinated", "evening_outdoor_exposure", ...],
        "sample_size": {"cases": 15, "controls_per_case": 2}
    }
    """
    np.random.seed(random_seed)
    
    # Get cases based on case definition
    case_criteria = decisions.get("case_definition", {"clinical_AES": True})
    cases_pool = apply_case_definition(individuals_df, case_criteria)
    
    study_type = decisions.get("study_design", {}).get("type", "case_control")
    
    if study_type == "case_control":
        # Sample cases
        n_cases = decisions.get("sample_size", {}).get("cases", min(15, len(cases_pool)))
        cases = cases_pool.sample(n=min(n_cases, len(cases_pool)), random_state=random_seed)
        
        # Sample controls (non-cases from same villages)
        controls_ratio = decisions.get("sample_size", {}).get("controls_per_case", 2)
        case_villages = cases["village_id"].unique().tolist()
        
        noncases_pool = individuals_df[
            (~individuals_df["person_id"].isin(cases["person_id"])) &
            (individuals_df["village_id"].isin(case_villages)) &
            (individuals_df["symptomatic_AES"] == False)
        ]
        
        n_controls = len(cases) * controls_ratio
        controls = noncases_pool.sample(n=min(n_controls, len(noncases_pool)), random_state=random_seed+1)
        
        # Combine
        study_df = pd.concat([
            cases.assign(case_status=1),
            controls.assign(case_status=0)
        ])
    
    elif study_type == "cohort":
        # Define cohort (e.g., all children in affected villages)
        cohort = individuals_df[
            (individuals_df["age"] <= 15) &
            (individuals_df["village_id"].isin(["V1", "V2"]))
        ].copy()
        cohort["case_status"] = cohort["symptomatic_AES"].astype(int)
        study_df = cohort
    
    else:
        # Cross-sectional
        sample_size = decisions.get("sample_size", {}).get("total", 100)
        study_df = individuals_df.sample(n=min(sample_size, len(individuals_df)), random_state=random_seed)
        study_df["case_status"] = study_df["symptomatic_AES"].astype(int)
    
    # Add household-level variables
    hh_lookup = households_df.set_index('hh_id').to_dict('index')
    
    study_df['pigs_near_home'] = study_df['hh_id'].apply(
        lambda hh: hh_lookup.get(hh, {}).get('pigs_owned', 0) > 0 and 
                   (hh_lookup.get(hh, {}).get('pig_pen_distance_m') or 100) < 30
    )
    study_df['uses_mosquito_nets'] = study_df['hh_id'].apply(
        lambda hh: hh_lookup.get(hh, {}).get('uses_mosquito_nets', True)
    )
    study_df['rice_field_nearby'] = study_df['hh_id'].apply(
        lambda hh: hh_lookup.get(hh, {}).get('rice_field_distance_m', 200) < 100
    )
    
    # Keep only mapped columns + essentials
    mapped_cols = decisions.get("mapped_columns", [])
    
    # Column name mapping for questionnaire items
    col_mapping = {
        "pig proximity": "pigs_near_home",
        "pigs": "pigs_near_home",
        "mosquito net": "uses_mosquito_nets",
        "bed net": "uses_mosquito_nets",
        "nets": "uses_mosquito_nets",
        "rice": "rice_field_nearby",
        "paddy": "rice_field_nearby",
        "vaccination": "JE_vaccinated",
        "vaccine": "JE_vaccinated",
        "evening": "evening_outdoor_exposure",
        "outdoor": "evening_outdoor_exposure",
        "dusk": "evening_outdoor_exposure"
    }
    
    # Resolve mapped columns
    available_cols = ['person_id', 'village_id', 'age', 'sex', 'case_status']
    for col in mapped_cols:
        col_lower = col.lower()
        if col in study_df.columns:
            available_cols.append(col)
        else:
            for key, mapped in col_mapping.items():
                if key in col_lower and mapped in study_df.columns:
                    available_cols.append(mapped)
                    break
    
    available_cols = list(set(available_cols))
    final_df = study_df[[c for c in available_cols if c in study_df.columns]].copy()
    
    # Inject realistic noise
    final_df = inject_data_noise(final_df)
    
    return final_df


def inject_data_noise(df, missing_rate=0.08, error_rate=0.02, random_seed=42):
    """
    Inject realistic data quality issues:
    - Random missing values (5-10%)
    - Occasional coding errors
    - Inconsistent formatting
    """
    np.random.seed(random_seed)
    df = df.copy()
    
    n_rows = len(df)
    
    for col in df.columns:
        if col in ['person_id', 'case_status']:  # Don't corrupt key fields
            continue
        
        # Random missingness
        n_missing = int(n_rows * missing_rate)
        missing_idx = np.random.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_idx, col] = np.nan
        
        # For boolean columns, occasionally flip values
        if df[col].dtype == bool:
            n_errors = int(n_rows * error_rate)
            error_idx = np.random.choice(df.index, size=n_errors, replace=False)
            df.loc[error_idx, col] = ~df.loc[error_idx, col]
    
    return df


# ============================================================================
# LABORATORY SIMULATION
# ============================================================================

LAB_TESTS = {
    "JE_IgM_CSF": {"sensitivity": 0.85, "specificity": 0.98, "cost": 2, "days": 3},
    "JE_IgM_serum": {"sensitivity": 0.80, "specificity": 0.95, "cost": 1, "days": 3},
    "JE_PCR_CSF": {"sensitivity": 0.40, "specificity": 0.99, "cost": 3, "days": 4},
    "JE_PCR_mosquito": {"sensitivity": 0.95, "specificity": 0.98, "cost": 2, "days": 5},
    "JE_IgG_pig": {"sensitivity": 0.90, "specificity": 0.95, "cost": 1, "days": 4},
    "bacterial_culture": {"sensitivity": 0.70, "specificity": 0.99, "cost": 1, "days": 3},
    "water_quality": {"sensitivity": 0.95, "specificity": 0.90, "cost": 1, "days": 2}
}


def process_lab_order(order, lab_samples_truth, random_seed=None):
    """
    Process a lab order and return results based on truth + test performance.
    
    order = {
        "sample_type": "human_CSF",
        "village_id": "V1",
        "test": "JE_IgM_CSF",
        "source_description": "Case DH-01"
    }
    """
    if random_seed:
        np.random.seed(random_seed)
    
    test_params = LAB_TESTS.get(order["test"], {"sensitivity": 0.80, "specificity": 0.95})
    
    # Find matching truth
    matching = lab_samples_truth[
        (lab_samples_truth["sample_type"] == order["sample_type"]) &
        (lab_samples_truth["linked_village_id"] == order["village_id"])
    ]
    
    if len(matching) > 0:
        true_positive = matching.iloc[0]["true_JEV_positive"]
    else:
        # Default based on sample type and village
        if order["village_id"] in ["V1", "V2"]:
            true_positive = order["sample_type"] in ["human_CSF", "human_serum", "pig_serum", "mosquito_pool"]
        else:
            true_positive = False
    
    # Apply test performance
    if true_positive:
        result_positive = np.random.random() < test_params["sensitivity"]
    else:
        result_positive = np.random.random() > test_params["specificity"]
    
    return {
        "sample_id": f"LAB-{np.random.randint(1000, 9999)}",
        "sample_type": order["sample_type"],
        "village_id": order["village_id"],
        "test": order["test"],
        "result": "POSITIVE" if result_positive else "NEGATIVE",
        "true_status": true_positive,  # Hidden from trainee
        "cost": test_params["cost"],
        "days_to_result": test_params["days"]
    }


# ============================================================================
# CONSEQUENCE ENGINE
# ============================================================================

def evaluate_interventions(decisions, interview_history):
    """
    Calculate outbreak consequences based on trainee decisions.
    
    Returns outcome category and narrative.
    """
    score = 0
    outcomes = []
    
    # Diagnosis
    diagnosis = decisions.get("final_diagnosis", "")
    if "japanese encephalitis" in diagnosis.lower() or "je" in diagnosis.lower():
        score += 20
        outcomes.append("✅ Correct diagnosis: Japanese Encephalitis")
    else:
        score -= 10
        outcomes.append(f"❌ Incorrect diagnosis: {diagnosis}")
    
    # One Health approach
    if "vet_amina" in interview_history:
        score += 15
        outcomes.append("✅ Consulted veterinary officer (One Health)")
    else:
        score -= 5
        outcomes.append("⚠️ Did not consult veterinary officer")
    
    if "mr_osei" in interview_history:
        score += 10
        outcomes.append("✅ Environmental assessment completed")
    else:
        outcomes.append("⚠️ Environmental factors not fully assessed")
    
    # Lab strategy
    lab_orders = decisions.get("lab_orders", [])
    sample_types = [o.get("sample_type", "") for o in lab_orders]
    
    has_human = any("human" in s for s in sample_types)
    has_pig = any("pig" in s for s in sample_types)
    has_mosquito = any("mosquito" in s for s in sample_types)
    
    if has_human and has_pig and has_mosquito:
        score += 20
        outcomes.append("✅ Comprehensive sampling (human + animal + vector)")
    elif has_human and (has_pig or has_mosquito):
        score += 10
        outcomes.append("⚡ Partial One Health sampling")
    elif has_human:
        score += 5
        outcomes.append("⚠️ Human samples only - missed animal/vector evidence")
    
    # Questionnaire quality
    questionnaire_vars = decisions.get("mapped_columns", [])
    good_vars = ["pig", "mosquito", "net", "vaccin", "evening", "outdoor", "dusk"]
    
    matches = sum(1 for v in questionnaire_vars for g in good_vars if g in v.lower())
    
    if matches >= 5:
        score += 15
        outcomes.append("✅ Excellent questionnaire design")
    elif matches >= 3:
        score += 8
        outcomes.append("⚡ Adequate questionnaire")
    else:
        score -= 5
        outcomes.append("❌ Poor questionnaire - missed key risk factors")
    
    # Recommendations
    recommendations = ' '.join(decisions.get("recommendations", [])).lower()
    
    rec_scores = {
        "vaccination": any(w in recommendations for w in ["vaccin", "immuniz"]),
        "vector_control": any(w in recommendations for w in ["bed net", "bednet", "mosquito net", "larvicid", "spray"]),
        "pig_management": any(w in recommendations for w in ["pig", "relocat", "pen"]),
        "surveillance": any(w in recommendations for w in ["surveill", "monitor"]),
        "education": any(w in recommendations for w in ["educat", "awareness"])
    }
    
    recs_count = sum(rec_scores.values())
    
    if recs_count >= 4:
        score += 20
        outcomes.append("✅ Comprehensive intervention package")
    elif recs_count >= 2:
        score += 10
        outcomes.append("⚡ Partial interventions recommended")
    else:
        score -= 10
        outcomes.append("❌ Weak interventions")
    
    # Penalize wrong approaches
    if any(w in recommendations for w in ["water", "chlorin", "borehole"]):
        score -= 5
        outcomes.append("❌ Water interventions are not relevant to JE")
    
    if any(w in recommendations for w in ["close school", "close market"]):
        score -= 3
        outcomes.append("⚠️ Closures not evidence-based for vector-borne disease")
    
    # Determine outcome category
    if score >= 70:
        status = "SUCCESS"
        narrative = """**OUTBREAK CONTROLLED**

Your evidence-based recommendations were implemented:
- Emergency JE vaccination campaign reached 2,500 children
- Bed nets distributed to high-risk households  
- Pig cooperative relocated 50m from school
- Mosquito breeding sites treated

**Result:** New cases dropped to zero within 2 weeks.
The District Director praised your team's One Health approach."""
        new_cases = 0
        
    elif score >= 40:
        status = "PARTIAL SUCCESS"
        narrative = """**OUTBREAK PARTIALLY CONTROLLED**

Some recommendations were implemented, but gaps remained:
- Vaccination campaign was delayed
- Vector control was incomplete
- Not all risk factors addressed

**Result:** Cases continued for another week before declining.
Two additional children were hospitalized but survived."""
        new_cases = 2
        
    else:
        status = "OUTBREAK CONTINUES"
        narrative = """**OUTBREAK NOT CONTROLLED**

Your recommendations missed key interventions:
- No vaccination campaign initiated
- Pig cooperative unchanged  
- Mosquito breeding sites remain

**Result:** Cases spread to Tamu village.
Three more children died. Regional investigation launched.
Mayor blamed the investigation team for "incomplete response"."""
        new_cases = 8
    
    return {
        "status": status,
        "narrative": narrative,
        "score": score,
        "max_score": 100,
        "new_cases": new_cases,
        "outcomes": outcomes
    }


# ============================================================================
# DAY PREREQUISITES
# ============================================================================

def check_day_prerequisites(current_day, session_state):
    """
    Check if prerequisites are met to advance to next day.
    Returns (can_advance: bool, missing: list of strings)
    
    session_state can be either a dict or streamlit session_state object
    """
    missing = []
    
    # Helper to safely get values from either dict or session_state
    def get_val(key, default=None):
        if hasattr(session_state, 'get'):
            return session_state.get(key, default)
        else:
            return getattr(session_state, key, default)
    
    if current_day == 1:
        # Day 1 → Day 2: Need case definition + hypotheses + at least 2 interviews
        if not get_val("case_definition_written", False):
            missing.append("Write a case definition")
        if not get_val("hypotheses_documented", False):
            missing.append("Document initial hypotheses")
        interview_history = get_val("interview_history", {})
        if len(interview_history) < 2:
            missing.append("Complete at least 2 interviews")
    
    elif current_day == 2:
        # Day 2 → Day 3: Need study design + questionnaire
        decisions = get_val("decisions", {})
        if not decisions.get("study_design"):
            missing.append("Choose a study design")
        if not get_val("questionnaire_submitted", False):
            missing.append("Submit questionnaire")
    
    elif current_day == 3:
        # Day 3 → Day 4: Need to complete analysis
        if not get_val("descriptive_analysis_done", False):
            missing.append("Complete descriptive analysis")
    
    elif current_day == 4:
        # Day 4 → Day 5: Need lab samples
        lab_samples = get_val("lab_samples_submitted", [])
        if len(lab_samples) < 1:
            missing.append("Submit at least one lab sample")
    
    return len(missing) == 0, missing
