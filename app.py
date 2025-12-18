import streamlit as st
import anthropic
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# Village photos loaded from assets/Nalu directory
import plotly.express as px
import io
import re

# Session persistence
import persistence

# Robust import: avoid hard failures from `from je_logic import ...` if the deployed module is stale/mismatched.
try:
    import je_logic as jl
except Exception as e:
    st.error(
        "Failed to import je_logic.py. "
        "This usually means the file is missing from the repo, not committed/pushed, "
        "or it has an import/syntax error.\n\n"
        f"Error: {e!r}"
    )
    st.stop()

# Core logic (required)
_missing_required = []
for _name in [
    "load_truth_data",
    "generate_full_population",
    "apply_case_definition",
    "ensure_reported_to_hospital",
    "generate_study_dataset",
    "process_lab_order",
    "evaluate_interventions",
    "check_day_prerequisites",
]:
    if not hasattr(jl, _name):
        _missing_required.append(_name)

if _missing_required:
    st.error(
        "Your je_logic.py is missing required functions expected by app.py:\n"
        + "\n".join(f"- {_n}" for _n in _missing_required)
        + "\n\nFix: make sure you replaced je_logic.py with the updated version and pushed it to Streamlit Cloud."
    )
    st.stop()

load_truth_data = jl.load_truth_data
generate_full_population = jl.generate_full_population
apply_case_definition = jl.apply_case_definition
ensure_reported_to_hospital = jl.ensure_reported_to_hospital
generate_study_dataset = jl.generate_study_dataset
process_lab_order = jl.process_lab_order
evaluate_interventions = jl.evaluate_interventions
check_day_prerequisites = jl.check_day_prerequisites

# XLSForm pipeline (optional but recommended)
parse_xlsform = getattr(jl, "parse_xlsform", None)
llm_map_xlsform_questions = getattr(jl, "llm_map_xlsform_questions", None)
llm_build_select_one_choice_maps = getattr(jl, "llm_build_select_one_choice_maps", None)
llm_build_unmapped_answer_generators = getattr(jl, "llm_build_unmapped_answer_generators", None)
prepare_question_render_plan = getattr(jl, "prepare_question_render_plan", None)

XLSFORM_AVAILABLE = all([
    callable(parse_xlsform),
    callable(llm_map_xlsform_questions),
    callable(llm_build_select_one_choice_maps),
    callable(llm_build_unmapped_answer_generators),
    callable(prepare_question_render_plan),
])
# =========================
# TRANSLATION SYSTEM (i18n)
# =========================
#
# Design goals:
# - Keep UI and scenario text out of logic as much as possible.
# - Support multiple languages via JSON locale bundles:
#     locales/<lang>/ui.json
#     locales/<lang>/story.json
# - Provide a safe fallback to minimal in-code English defaults when files are missing.
#
# NOTE: This is an incremental migration. Not every string in the app uses `t()` yet.
#       New/edited UI strings should always use `t()`.

import json
from pathlib import Path

DEFAULT_LANG = "en"
SUPPORTED_LANGS = ["en", "es", "fr", "pt"]

# Minimal fallback strings (UI)
_FALLBACK_UI = {
    "en": {
        "title": "AES Outbreak Investigation – Sidero Valley",
        "language_header": "Language",
        "language_select": "Select language:",
        "facilitator_header": "Facilitator",
        "facilitator_mode": "Facilitator mode",
        "facilitator_code": "Facilitator code",
        "facilitator_bad_code": "Incorrect facilitator code.",
        "day": "Day",
        "budget": "Budget",
        "time_remaining": "Time remaining",
        "hours": "hours",
        "lab_credits": "Lab credits",
        "progress": "Progress",
        "go_to": "Go to",
        "overview": "Overview / Briefing",
        "casefinding": "Case Finding",
        "descriptive": "Descriptive Epi",
        "interviews": "Interviews",
        "spotmap": "Spot Map",
        "study": "Data & Study Design",
        "lab": "Lab & Environment",
        "outcome": "Interventions & Outcome",
        "villages": "Village Profiles",
        "notebook": "Investigation Notebook",
        "advance_day": "Advance to Day",
        "cannot_advance": "Cannot advance yet. See missing tasks on Overview.",
        "missing_tasks_title": "Missing tasks before you can advance:",
        "locked_until_day": "Locked until Day {day}.",
        "chat_prompt": "Ask your question...",
        "save": "Save",
        "submit": "Submit",
        "download": "Download",
        "begin_investigation": "Begin investigation",

"day1_briefing": "Day 1: Situation Assessment",
"day2_briefing": "Day 2: Study Design",
"day3_briefing": "Day 3: Data Collection",
"day4_briefing": "Day 4: Analysis & Laboratory",
"day5_briefing": "Day 5: Recommendations",
"key_tasks": "Key tasks",
"key_outputs": "Key outputs",
"review_line_list": "Review initial line list and epi curve",
"review_clinic_records": "Search clinic records for additional cases",
"describe_cases": "Describe cases by person, place, and time",
"conduct_interviews": "Conduct hypothesis-generating interviews",
"find_additional_cases": "Expand case finding",
"develop_case_def": "Refine the case definition",
"develop_hypotheses": "Document hypotheses and plan next steps",
"output_updated_line_list": "Updated line list",
"output_working_case_def": "Working case definition",
"output_initial_hypotheses": "Initial hypotheses",

"prereq.day1.case_definition": "Save a working case definition (Overview).",
"prereq.day1.hypothesis": "Document at least one hypothesis (Overview).",
"prereq.day1.interviews": "Complete at least 2 hypothesis-generating interviews (Interviews).",
"prereq.day2.study_design": "Select a study design (Data & Study Design).",
"prereq.day2.questionnaire": "Upload and save your questionnaire (XLSForm) (Data & Study Design).",
"prereq.day2.dataset": "Generate your simulated dataset for analysis (Data & Study Design).",
"prereq.day3.analysis": "Confirm you completed analysis and summarize key results (Overview / Day 3).",
"prereq.day4.lab_order": "Place at least one lab order (Lab & Environment).",
"prereq.day4.environment": "Record at least one environmental action (Lab & Environment).",
"prereq.day4.draft_interventions": "Record draft interventions (Outcome tab, draft section).",
        # Lab labels (anti-spoiler)
        "lab_test": "Test",
        "lab_results": "Lab results",
        "lab_pending": "PENDING",
    },
    "es": {
        "title": "Investigación de Brote de AES – Valle de Sidero",
        "language_header": "Idioma",
        "language_select": "Seleccionar idioma:",
        "facilitator_header": "Facilitador",
        "facilitator_mode": "Modo facilitador",
        "facilitator_code": "Código de facilitador",
        "facilitator_bad_code": "Código incorrecto.",
        "day": "Día",
        "budget": "Presupuesto",
        "time_remaining": "Tiempo restante",
        "hours": "horas",
        "lab_credits": "Créditos de laboratorio",
        "progress": "Progreso",
        "go_to": "Ir a",
        "overview": "Resumen / Briefing",
        "casefinding": "Búsqueda de casos",
        "descriptive": "Epi descriptiva",
        "interviews": "Entrevistas",
        "spotmap": "Mapa de puntos",
        "study": "Datos y diseño",
        "lab": "Laboratorio y ambiente",
        "outcome": "Intervenciones",
        "villages": "Perfiles de aldeas",
        "notebook": "Cuaderno",
        "advance_day": "Avanzar al día",
        "cannot_advance": "Aún no puede avanzar. Consulte las tareas pendientes en Resumen.",
        "missing_tasks_title": "Tareas pendientes antes de avanzar:",
        "locked_until_day": "Bloqueado hasta el Día {day}.",
        "chat_prompt": "Escribe tu pregunta...",
        "save": "Guardar",
        "submit": "Enviar",
        "download": "Descargar",
        "begin_investigation": "Iniciar investigación",
        "lab_test": "Prueba",
        "lab_results": "Resultados de laboratorio",
        "lab_pending": "PENDIENTE",
    },
    "fr": {
        "title": "Investigation d'épidémie AES – Vallée de Sidero",
        "language_header": "Langue",
        "language_select": "Choisir la langue :",
        "facilitator_header": "Facilitateur",
        "facilitator_mode": "Mode facilitateur",
        "facilitator_code": "Code facilitateur",
        "facilitator_bad_code": "Code incorrect.",
        "day": "Jour",
        "budget": "Budget",
        "time_remaining": "Temps restant",
        "hours": "heures",
        "lab_credits": "Crédits labo",
        "progress": "Progrès",
        "go_to": "Aller à",
        "overview": "Aperçu / Briefing",
        "casefinding": "Recherche de cas",
        "descriptive": "Épi descriptive",
        "interviews": "Entretiens",
        "spotmap": "Carte des points",
        "study": "Données et conception",
        "lab": "Labo et environnement",
        "outcome": "Interventions",
        "villages": "Profils des villages",
        "notebook": "Carnet",
        "advance_day": "Passer au jour",
        "cannot_advance": "Impossible d'avancer. Voir les tâches manquantes dans Aperçu.",
        "missing_tasks_title": "Tâches manquantes avant d'avancer :",
        "locked_until_day": "Bloqué jusqu'au Jour {day}.",
        "chat_prompt": "Posez votre question...",
        "save": "Enregistrer",
        "submit": "Soumettre",
        "download": "Télécharger",
        "begin_investigation": "Commencer l'investigation",
        "lab_test": "Test",
        "lab_results": "Résultats de laboratoire",
        "lab_pending": "EN ATTENTE",
    },
    "pt": {
        "title": "Investigação de Surto de AES – Vale de Sidero",
        "language_header": "Idioma",
        "language_select": "Selecionar idioma:",
        "facilitator_header": "Facilitador",
        "facilitator_mode": "Modo facilitador",
        "facilitator_code": "Código do facilitador",
        "facilitator_bad_code": "Código incorreto.",
        "day": "Dia",
        "budget": "Orçamento",
        "time_remaining": "Tempo restante",
        "hours": "horas",
        "lab_credits": "Créditos de laboratório",
        "progress": "Progresso",
        "go_to": "Ir para",
        "overview": "Visão geral / briefing",
        "casefinding": "Busca de casos",
        "descriptive": "Epi descritiva",
        "interviews": "Entrevistas",
        "spotmap": "Mapa de pontos",
        "study": "Dados e desenho",
        "lab": "Laboratório e ambiente",
        "outcome": "Intervenções",
        "villages": "Perfis das aldeias",
        "notebook": "Caderno",
        "advance_day": "Avançar para o dia",
        "cannot_advance": "Ainda não é possível avançar. Veja as tarefas pendentes em Visão geral.",
        "missing_tasks_title": "Tarefas pendentes antes de avançar:",
        "locked_until_day": "Bloqueado até o Dia {day}.",
        "chat_prompt": "Digite sua pergunta...",
        "save": "Salvar",
        "submit": "Enviar",
        "download": "Baixar",
        "begin_investigation": "Iniciar investigação",
        "lab_test": "Teste",
        "lab_results": "Resultados laboratoriais",
        "lab_pending": "PENDENTE",
    },
}

@st.cache_data(show_spinner=False)
def _load_locale_bundle(lang: str, bundle: str) -> dict:
    # Locale files live next to app.py in Streamlit Cloud
    base = Path(__file__).resolve().parent / "locales" / lang
    fp = base / f"{bundle}.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _get_from_dict(d: dict, key: str):
    # Supports both flat keys and dotted paths
    if key in d:
        return d[key]
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur

def t(key: str, default: str = None, bundle: str = "ui", **kwargs) -> str:
    """Translate key using the current session language.

    - Falls back to English and then to provided `default` and finally to the key itself.
    - Supports `.format(**kwargs)` interpolation.
    """
    lang = st.session_state.get("language", DEFAULT_LANG) or DEFAULT_LANG
    if lang not in SUPPORTED_LANGS:
        lang = DEFAULT_LANG

    # Try file-based bundle first
    data = _load_locale_bundle(lang, bundle) or {}
    val = _get_from_dict(data, key)

    # Fallback to minimal in-code strings
    if val is None:
        val = _FALLBACK_UI.get(lang, {}).get(key)
    if val is None:
        val = _FALLBACK_UI.get(DEFAULT_LANG, {}).get(key)
    if val is None:
        val = default if default is not None else key

    try:
        return str(val).format(**kwargs)
    except Exception:
        return str(val)


# =========================
# TIME AND RESOURCE COSTS
# =========================

# Time costs in hours for various activities
TIME_COSTS = {
    # Interviews - primarily time cost
    "interview_initial": 1.0,      # First interview with an NPC
    "interview_followup": 0.5,     # Follow-up questions with same NPC
    
    # Case finding
    "clinic_records_review": 2.0,  # Reviewing clinic records
    
    # Data collection
    "questionnaire_development": 1.5,  # Developing questionnaire
    "questionnaire_admin_per_10": 2.0, # Administering questionnaire (per 10 respondents)
    
    # Analysis
    "descriptive_analysis": 1.0,   # Running descriptive analyses
    "data_cleaning": 1.5,          # Cleaning dataset
    
    # Lab and environment
    "sample_collection": 1.0,      # Per sample collection trip
    "environmental_inspection": 2.0,  # Site inspection
    
    # Travel
    "travel_to_village": 0.5,      # Travel time to a village
    "travel_between_villages": 0.5,  # Travel between villages
}

# Budget costs (some activities cost money, not just time)
BUDGET_COSTS = {
    "questionnaire_printing": 50,   # Printing questionnaires
    "lab_sample_human": 25,         # Human sample collection supplies
    "lab_sample_animal": 35,        # Animal sample collection
    "lab_sample_mosquito": 40,      # Mosquito trap setup
    "transport_per_trip": 20,       # Vehicle/fuel costs
}


def spend_time(hours: float, activity: str = "") -> bool:
    """
    Deduct time from daily allowance.
    Returns True if successful, False if not enough time.
    """
    if st.session_state.time_remaining >= hours:
        st.session_state.time_remaining -= hours
        return True
    return False


def spend_budget(amount: float, activity: str = "") -> bool:
    """
    Deduct from budget.
    Returns True if successful, False if not enough budget.
    """
    if st.session_state.budget >= amount:
        st.session_state.budget -= amount
        return True
    return False


def check_resources(time_needed: float = 0, budget_needed: float = 0) -> tuple:
    """
    Check if enough resources are available.
    Returns (can_proceed: bool, message: str)
    """
    messages = []
    can_proceed = True
    
    if time_needed > 0 and st.session_state.time_remaining < time_needed:
        can_proceed = False
        messages.append(f"Not enough time (need {time_needed}h, have {st.session_state.time_remaining}h)")
    
    if budget_needed > 0 and st.session_state.budget < budget_needed:
        can_proceed = False
        messages.append(f"Not enough budget (need ${budget_needed}, have ${st.session_state.budget})")
    
    return can_proceed, "; ".join(messages) if messages else "OK"


def format_resource_cost(time_cost: float = 0, budget_cost: float = 0) -> str:
    """Format a resource cost string for display."""
    parts = []
    if time_cost > 0:
        parts.append(f"⏱️ {time_cost}h")
    if budget_cost > 0:
        parts.append(f"💰 ${budget_cost}")
    return " | ".join(parts) if parts else "Free"


# =========================
# VILLAGE BRIEFING DOCUMENTS  
# =========================

VILLAGE_PROFILES = {
    "nalu": {
        "name": "Nalu Village",
        "population": 1850,
        "households": 340,
        "description": {
            "en": """
**Nalu Village** is the largest settlement in Sidero Valley, located along the main river 
that feeds the extensive rice paddy system. The village economy is centered on rice 
cultivation and pig farming.

**Key Facts:**
- **Population:** 1,850 (2024 census)
- **Households:** ~340
- **Main livelihoods:** Rice farming (65%), pig rearing (45%), fishing (20%)
- **Health facility:** Nalu Health Center (1 nurse, 2 community health workers)
- **Schools:** 1 primary school (enrollment: 380)
- **Water source:** River, hand-dug wells, 2 boreholes
- **Sanitation:** Mix of pit latrines and open defecation

**Geographic Features:**
- Located along the main river
- Surrounded by irrigated rice paddies on three sides
- Pig cooperative located near village center

**Health Information (District Health Office, 2024):**
- Under-5 mortality: 45 per 1,000 live births
- Top health concerns: Malaria, diarrheal diseases
- Nearest hospital: District Hospital (12 km)
""",
            "es": """
**Aldea de Nalu** es el asentamiento más grande del Valle de Sidero...
""",
            "fr": """
**Village de Nalu** est le plus grand établissement de la Vallée de Sidero...
"""
        },
        "images": ["rice_paddies", "pig_farm", "village_scene"]
    },
    "kabwe": {
        "name": "Kabwe Village",
        "population": 920,
        "households": 175,
        "description": {
            "en": """
**Kabwe Village** is a medium-sized farming community located 3 km northeast of Nalu, 
on slightly higher ground. Many residents work in both Kabwe and Nalu.

**Key Facts:**
- **Population:** 920 (2024 census)
- **Households:** ~175
- **Main livelihoods:** Mixed farming (maize, vegetables), some rice, pig rearing
- **Health facility:** None (served by Nalu Health Center)
- **Schools:** 1 primary school (enrollment: 165), children attend secondary in Nalu
- **Water source:** 3 boreholes, seasonal stream
- **Sanitation:** Pit latrines (60%), open defecation (40%)

**Geographic Features:**
- Higher elevation than Nalu
- Mixed agricultural zone with both rice and upland crops
- Path to Nalu passes through agricultural areas

**Health Information:**
- Residents use Nalu Health Center
- Top health concerns: Malaria, respiratory infections
""",
            "es": "**Aldea de Kabwe** es una comunidad agrícola de tamaño mediano...",
            "fr": "**Village de Kabwe** est une communauté agricole de taille moyenne..."
        },
        "images": ["mixed_farming", "village_path", "children_school"]
    },
    "tamu": {
        "name": "Tamu Village",
        "population": 650,
        "households": 125,
        "description": {
            "en": """
**Tamu Village** is a smaller, more remote community located 5 km west of Nalu, 
in the foothills away from the main rice-growing areas.

**Key Facts:**
- **Population:** 650 (2024 census)
- **Households:** ~125
- **Main livelihoods:** Upland farming (cassava, yams), small-scale livestock, charcoal
- **Health facility:** Community health volunteer only
- **Schools:** 1 small primary school (enrollment: 95)
- **Water source:** Spring-fed wells, rainwater collection
- **Sanitation:** Pit latrines (45%), open defecation (55%)

**Geographic Features:**
- Higher elevation, drier terrain
- Upland farming area (no rice cultivation)
- Primarily goats and chickens for livestock
- More forested areas nearby

**Health Information:**
- Residents occasionally travel to Nalu for market/health services
- Top health concerns: Respiratory infections, malnutrition
- Community health volunteer provides basic care
""",
            "es": "**Aldea de Tamu** es una comunidad más pequeña y remota...",
            "fr": "**Village de Tamu** est une communauté plus petite et plus éloignée..."
        },
        "images": ["upland_farming", "village_remote", "forest_edge"]
    }
}


# =========================
# INITIALIZATION
# =========================

@st.cache_data
def load_truth_and_population(data_dir: str = ".", _version: int = 3):
    """Load truth data and generate a full population.
    
    _version parameter is used to bust the cache when logic changes.
    Increment when risk model or data generation logic is modified.
    """
    truth = load_truth_data(data_dir=data_dir)
    villages_df = truth["villages"]
    households_seed = truth["households_seed"]
    individuals_seed = truth["individuals_seed"]

    households_full, individuals_full = generate_full_population(
        villages_df, households_seed, individuals_seed
    )
    truth["households"] = households_full
    truth["individuals"] = individuals_full
    return truth


def init_session_state():
    if "truth" not in st.session_state:
        # CSV/JSON files are in the 'data' subdirectory
        st.session_state.truth = load_truth_and_population(data_dir="data")

    # Alert page logic (Day 0)
    st.session_state.setdefault("alert_acknowledged", False)

    if "current_day" not in st.session_state:
        # 1–5 for the investigation days
        st.session_state.current_day = 1

    if "current_view" not in st.session_state:
        # Start on alert screen until acknowledged
        st.session_state.current_view = "alert"

    # If alert is not acknowledged, force the view to "alert"
    if not st.session_state.alert_acknowledged:
        st.session_state.current_view = "alert"
    else:
        # If alert already acknowledged but view is still "alert", move to overview
        if st.session_state.current_view == "alert":
            st.session_state.current_view = "overview"

    # Resources - budget AND time
    st.session_state.setdefault("budget", 1000)
    st.session_state.setdefault("time_remaining", 8)  # hours per day
    st.session_state.setdefault("lab_credits", 20)
    
    # Language setting
    st.session_state.setdefault("language", "en")

    # Decisions and artifacts
    if "decisions" not in st.session_state:
        st.session_state.decisions = {
            "case_definition": None,
            "case_definition_text": "",
            "study_design": None,
            "mapped_columns": [],
            "sample_size": {"cases": 15, "controls_per_case": 2},
            "lab_orders": [],
            "questionnaire_raw": [],
            "questionnaire_file": None,  # For uploaded XLS
            "final_diagnosis": "",
            "recommendations": [],
        }

    st.session_state.setdefault("generated_dataset", None)
    st.session_state.setdefault("lab_results", [])
    st.session_state.setdefault("lab_orders", [])
    st.session_state.setdefault("environment_findings", [])
    st.session_state.setdefault("analysis_confirmed", False)
    st.session_state.setdefault("etiology_revealed", False)
    st.session_state.setdefault("lab_samples_submitted", [])
    st.session_state.setdefault("interview_history", {})
    st.session_state.setdefault("revealed_clues", {})
    st.session_state.setdefault("current_npc", None)
    st.session_state.setdefault("unlock_flags", {})

    # NPC emotional state & memory summary (per NPC)
    # structure: npc_state[npc_key] = {
    #   "emotion": "neutral" | "cooperative" | "wary" | "annoyed" | "offended",
    #   "interaction_count": int,
    #   "rude_count": int,
    #   "polite_count": int,
    # }
    st.session_state.setdefault("npc_state", {})

    # Flags used for day progression
    st.session_state.setdefault("case_definition_written", False)
    st.session_state.setdefault("questionnaire_submitted", False)
    st.session_state.setdefault("descriptive_analysis_done", False)

    # Track whether user has opened the line list/epi view at least once (for Day 1)
    st.session_state.setdefault("line_list_viewed", False)

    # For messaging when advance-day fails
    st.session_state.setdefault("advance_missing_tasks", [])
    
    # Initial hypotheses (Day 1)
    st.session_state.setdefault("initial_hypotheses", [])
    st.session_state.setdefault("hypotheses_documented", False)
    
    # Investigation notebook
    st.session_state.setdefault("notebook_entries", [])
    
    # NPC unlocking system (One Health)
    st.session_state.setdefault("npcs_unlocked", ["dr_chen", "nurse_joy", "mama_kofi", "foreman_rex", "teacher_grace"])
    st.session_state.setdefault("one_health_triggered", False)
    st.session_state.setdefault("vet_unlocked", False)
    st.session_state.setdefault("env_officer_unlocked", False)
    st.session_state.setdefault("questions_asked_about", set())
    
    # Clinic records and case finding (Day 1)
    st.session_state.setdefault("clinic_records_reviewed", False)
    st.session_state.setdefault("selected_clinic_cases", [])
    st.session_state.setdefault("case_finding_score", None)
    st.session_state.setdefault("found_cases_added", False)
    
    # Descriptive epidemiology
    st.session_state.setdefault("descriptive_epi_viewed", False)

    # Restore found cases from session persistence (if loading a saved session)
    # This is needed because truth is regenerated from CSV files, losing found cases
    if st.session_state.get('found_cases_added', False):
        found_individuals = st.session_state.get('found_case_individuals')
        if found_individuals is not None and len(found_individuals) > 0:
            # Check if found cases are already in truth
            truth = st.session_state.truth
            if 'found_via_case_finding' not in truth['individuals'].columns or \
               not truth['individuals']['found_via_case_finding'].any():
                restore_found_cases_to_truth(truth, st.session_state)


# =========================
# UTILITY FUNCTIONS
# =========================

def build_epidemiologic_context(truth: dict) -> str:
    """Short summary of the outbreak from truth tables."""
    individuals = truth["individuals"]
    households = truth["households"]
    villages = truth["villages"][["village_id", "village_name"]]

    hh_vil = households.merge(villages, on="village_id", how="left")
    merged = individuals.merge(
        hh_vil[["hh_id", "village_name"]], on="hh_id", how="left"
    )

    cases = merged[merged["symptomatic_AES"] == True]
    total_cases = len(cases)

    if total_cases == 0:
        return "No symptomatic AES cases have been assigned in the truth model."

    village_counts = cases["village_name"].value_counts().to_dict()

    bins = [0, 4, 14, 49, 120]
    labels = ["0–4", "5–14", "15–49", "50+"]
    age_groups = pd.cut(cases["age"], bins=bins, labels=labels, right=True)
    age_counts = age_groups.value_counts().to_dict()

    context = (
        f"There are currently about {total_cases} symptomatic AES cases in the district. "
        f"Cases by village: {village_counts}. "
        f"Cases by age group: {age_counts}. "
        "Most cases are children and come from villages with rice paddies and pigs."
    )
    return context


def build_npc_data_context(npc_key: str, truth: dict) -> str:
    """NPC-specific data context based on their data_access scope."""
    npc = truth["npc_truth"][npc_key]
    data_access = npc.get("data_access")

    individuals = truth["individuals"]
    households = truth["households"]
    villages = truth["villages"][["village_id", "village_name"]]

    hh_vil = households.merge(villages, on="village_id", how="left")
    merged = individuals.merge(
        hh_vil[["hh_id", "village_name"]], on="hh_id", how="left"
    )

    epi_context = build_epidemiologic_context(truth)

    if data_access == "hospital_cases":
        cases = merged[merged["symptomatic_AES"] == True]
        summary = cases.groupby("village_name").size().to_dict()
        return (
            epi_context
            + " As hospital director, you mainly see hospitalized AES cases. "
              f"You know current hospitalized cases come from these villages: {summary}."
        )

    if data_access == "triage_logs":
        cases = merged[merged["symptomatic_AES"] == True]
        earliest = cases["onset_date"].min()
        latest = cases["onset_date"].max()
        return (
            epi_context
            + " As triage nurse, you mostly notice who walks in first. "
              f"You saw the first AES cases between {earliest} and {latest}."
        )

    if data_access == "private_clinic":
        cases = merged[
            (merged["symptomatic_AES"] == True)
            & (merged["village_name"] == "Nalu Village")
        ]
        n = len(cases)
        return (
            epi_context
            + f" As a private healer, you have personally seen around {n} early AES-like illnesses "
              "from households near pig farms and rice paddies in Nalu."
        )

    if data_access == "school_attendance":
        school_age = merged[(merged["age"] >= 5) & (merged["age"] <= 18)]
        cases = school_age[school_age["symptomatic_AES"] == True]
        n = len(cases)
        by_village = cases["village_name"].value_counts().to_dict()
        return (
            epi_context
            + f" As school principal, you mostly know about school-age children. "
              f"You know of AES cases among your students: {n} total, by village: {by_village}."
        )

    if data_access == "vet_surveillance":
        lab = truth["lab_samples"]
        pigs = lab[lab["sample_type"] == "pig_serum"]
        pos = pigs[pigs["true_JEV_positive"] == True]
        by_village = pos["linked_village_id"].value_counts().to_dict()
        return (
            epi_context
            + " As the district veterinary officer, you track pig health. "
              f"Recent pig tests suggest JEV circulation in villages: {by_village}."
        )

    if data_access == "environmental_data":
        env = truth["environment_sites"]
        high = env[env["breeding_index"] == "high"]
        sites = high["site_id"].tolist()
        return (
            epi_context
            + " As environmental health officer, you survey breeding sites. "
              f"You know of high mosquito breeding around these sites: {sites}."
        )

    return epi_context


def analyze_user_tone(user_input: str) -> str:
    """
    Very simple tone detector: 'polite', 'rude', or 'neutral'.
    Used to update emotional state.
    """
    text = user_input.lower()

    polite_words = ["please", "thank you", "thanks", "appreciate", "grateful"]
    rude_words = [
        "stupid", "idiot", "useless", "incompetent", "what's wrong with you",
        "you people", "this is your fault", "do your job", "now!", "right now"
    ]

    if any(w in text for w in rude_words):
        return "rude"
    if any(w in text for w in polite_words):
        return "polite"
    # Very shouty messages
    if text.isupper() and len(text) > 5:
        return "rude"
    return "neutral"


def update_npc_emotion(npc_key: str, user_tone: str):
    """
    Strong emotional model:
    - Rude tone escalates emotion quickly to 'annoyed' or 'offended'
    - Polite tone causes mild softening
    - Neutral slowly heals annoyance over time
    """
    state = st.session_state.npc_state.setdefault(
        npc_key,
        {
            "emotion": "neutral",
            "interaction_count": 0,
            "rude_count": 0,
            "polite_count": 0,
        },
    )

    state["interaction_count"] += 1

    emotion_order = ["cooperative", "neutral", "wary", "annoyed", "offended"]

    def shift(current, steps):
        idx = emotion_order.index(current)
        idx = max(0, min(len(emotion_order) - 1, idx + steps))
        return emotion_order[idx]

    if user_tone == "polite":
        state["polite_count"] += 1
        # polite helps but not too fast
        state["emotion"] = shift(state["emotion"], -1)

    elif user_tone == "rude":
        state["rude_count"] += 1
        # rude pushes 2 steps more negative — very reactive
        state["emotion"] = shift(state["emotion"], +2)

    else:  # neutral tone
        # slow natural recovery only after several interactions
        if state["emotion"] in ["annoyed", "offended"] and state["interaction_count"] % 4 == 0:
            state["emotion"] = shift(state["emotion"], -1)

    st.session_state.npc_state[npc_key] = state
    return state


def describe_emotional_state(state: dict) -> str:
    """
    Turn emotion + history into a short description for the system prompt.
    This is what the LLM sees about how they feel toward the trainee.
    """
    emotion = state["emotion"]
    rude = state["rude_count"]
    polite = state["polite_count"]

    base = ""
    if emotion == "cooperative":
        base = "You currently feel friendly and cooperative toward the investigation team."
    elif emotion == "neutral":
        base = "You feel neutral toward the investigation team."
    elif emotion == "wary":
        base = "You feel cautious and slightly guarded. You will answer but watch your words."
    elif emotion == "annoyed":
        base = (
            "You feel irritated and impatient with the team. "
            "You give shorter answers and avoid volunteering extra information unless they ask clearly."
        )
    else:  # offended
        base = (
            "You feel offended by how the team has treated you. "
            "You answer briefly and share only what seems necessary for public health."
        )

    if rude >= 2 and emotion in ["annoyed", "offended"]:
        base += " You remember previous rude or disrespectful questions."

    if polite >= 2 and emotion in ["neutral", "wary"]:
        base += " They've also been respectful at times, which softens you a little."

    return base


def classify_question_scope(user_input: str) -> str:
    """
    Much stricter categorization:
    - 'greeting' : any greeting, no outbreak info allowed
    - 'broad'    : ONLY explicit broad requests like 'tell me everything'
    - 'narrow'   : direct, specific outbreak questions
    """
    text = user_input.strip().lower()

    # pure greetings → absolutely no outbreak info
    greeting_words = ["hi", "hello", "good morning", "good afternoon", "good evening"]
    if text in greeting_words or text.startswith("hi ") or text.startswith("hello "):
        return "greeting"

    # extremely explicit broad prompts only
    broad_phrases = [
        "tell me everything",
        "tell me what you know",
        "explain the whole situation",
        "give me an overview",
        "summarize everything",
        "what do you know about this outbreak",
    ]
    if any(text == p for p in broad_phrases):
        return "broad"

    # vague or small-talk questions should be treated as greeting
    vague_phrases = [
        "how are things",
        "how is everything",
        "what's going on",
        "what is going on",
        "what is happening",
        "how have you been",
        "how's your day",
    ]
    if any(p in text for p in vague_phrases):
        return "greeting"

    return "narrow"


def check_npc_unlock_triggers(user_input: str) -> str:
    """
    Check if user's question should unlock additional NPCs.
    Returns a notification message if unlock occurred, else empty string.
    """
    text = user_input.lower()
    notification = ""
    
    # Animal/pig triggers → unlock Vet Amina
    animal_triggers = ['animal', 'pig', 'livestock', 'pigs', 'swine', 'cattle', 'farm animal', 'piglet']
    if any(trigger in text for trigger in animal_triggers):
        st.session_state.questions_asked_about.add('animals')
        if not st.session_state.vet_unlocked:
            st.session_state.vet_unlocked = True
            st.session_state.one_health_triggered = True
            if 'vet_amina' not in st.session_state.npcs_unlocked:
                st.session_state.npcs_unlocked.append('vet_amina')
            notification = "🔓 **New Contact Unlocked:** Vet Amina (District Veterinary Officer) - Your question about animals opened a One Health perspective!"
    
    # Mosquito/environment triggers → unlock Mr. Osei
    env_triggers = ['mosquito', 'mosquitoes', 'vector', 'breeding', 'standing water', 'environment', 'rice paddy', 'irrigation', 'wetland']
    if any(trigger in text for trigger in env_triggers):
        st.session_state.questions_asked_about.add('environment')
        if not st.session_state.env_officer_unlocked:
            st.session_state.env_officer_unlocked = True
            st.session_state.one_health_triggered = True
            if 'mr_osei' not in st.session_state.npcs_unlocked:
                st.session_state.npcs_unlocked.append('mr_osei')
            notification = "🔓 **New Contact Unlocked:** Mr. Osei (Environmental Health Officer) - Your question about environmental factors opened a new perspective!"
    
    # Healer triggers (for earlier cases)
    healer_triggers = ['traditional', 'healer', 'clinic', 'private', 'early case', 'first case', 'before hospital']
    if any(trigger in text for trigger in healer_triggers):
        st.session_state.questions_asked_about.add('traditional')
        if 'healer_marcus' not in st.session_state.npcs_unlocked:
            st.session_state.npcs_unlocked.append('healer_marcus')
            notification = "🔓 **New Contact Unlocked:** Healer Marcus (Private Clinic) - You discovered there may be unreported cases!"
    
    return notification


# =========================
# ANTI-SPOILER / DISCLOSURE HELPERS
# =========================

def investigation_stage() -> str:
    """Return 'confirmed' when the etiology has been earned/revealed, else 'pre_confirmation'."""
    if st.session_state.get("etiology_revealed", False):
        return "confirmed"
    # If a final diagnosis has been explicitly recorded, treat as confirmed.
    if bool((st.session_state.get("decisions", {}) or {}).get("final_diagnosis")):
        return "confirmed"
    return "pre_confirmation"


_SPOILER_PATTERNS = [
    (re.compile(r"\bJapanese\s+Encephalitis\b", re.IGNORECASE), "a mosquito-borne viral encephalitis"),
    (re.compile(r"\bJEV\b", re.IGNORECASE), "the suspected encephalitis virus"),
    # Avoid replacing words like 'project' etc; keep JE replacement conservative.
    (re.compile(r"\bJE\b", re.IGNORECASE), "encephalitis"),
]

def redact_spoilers(text: str, stage: str) -> str:
    if stage == "confirmed" or not text:
        return text
    out = str(text)
    for rgx, repl in _SPOILER_PATTERNS:
        out = rgx.sub(repl, out)
    return out


def sanitize_npc_truth_for_prompt(npc_truth: dict, stage: str) -> dict:
    """Return a copy of npc_truth safe to include in prompts before confirmation."""
    if not isinstance(npc_truth, dict):
        return npc_truth
    safe = dict(npc_truth)

    # Simple string fields
    for k in ["name", "role", "personality"]:
        if k in safe:
            safe[k] = redact_spoilers(safe[k], stage)

    # List fields
    for k in ["always_reveal", "red_herrings", "unknowns"]:
        if k in safe and isinstance(safe[k], list):
            safe[k] = [redact_spoilers(x, stage) for x in safe[k]]

    # Dict fields
    if isinstance(safe.get("conditional_clues"), dict):
        safe["conditional_clues"] = {kk: redact_spoilers(vv, stage) for kk, vv in safe["conditional_clues"].items()}

    return safe


def npc_style_hint(npc_key: str, question_count: int, npc_state: str) -> str:
    """Small deterministic style variation to reduce robotic feel."""
    # Keep hints short so they don't dominate the system prompt.
    if npc_state in ("annoyed", "offended"):
        return "Keep it short. One point per sentence. No extra context unless asked."
    if npc_key in ("nurse_joy", "clinic_nurse"):
        if question_count < 2:
            return "Sound busy and slightly stressed. Ask one clarifying question if needed."
        return "Be helpful but time-pressed. Occasionally mention workload."
    if npc_key in ("dr_chen", "hospital_director"):
        return "Use a calm, clinical tone. Avoid speculation; distinguish observed vs assumed."
    if npc_key in ("chief_musa", "district_officer"):
        return "Be formal and concise. Focus on actions, constraints, and coordination."
    if npc_key in ("vet_amina",):
        return "Use One Health framing. Mention animal/veterinary context only when relevant."
    if npc_key in ("mr_osei",):
        return "Use practical environmental language. Mention mosquitoes/water management if asked."
    return "Speak naturally. If unsure, say so."


# =========================
# LAB LABELS (anti-spoiler)
# =========================

LAB_TEST_CATALOG = {
    # The keys match je_logic.LAB_TESTS / aliases; labels are trainee-facing.
    "JE_IgM_CSF": {"generic": "Arbovirus IgM (CSF)", "confirmed": "Japanese Encephalitis IgM (CSF)"},
    "JE_IgM_serum": {"generic": "Arbovirus IgM (serum)", "confirmed": "Japanese Encephalitis IgM (serum)"},
    "JE_PCR_mosquito": {"generic": "Arbovirus PCR (mosquito pool)", "confirmed": "Japanese Encephalitis PCR (mosquito pool)"},
    "JE_Ab_pig": {"generic": "Arbovirus antibodies (pig serum)", "confirmed": "Japanese Encephalitis antibodies (pig serum)"},
}

def lab_test_label(test_code: str) -> str:
    stage = investigation_stage()
    entry = LAB_TEST_CATALOG.get(test_code, {})
    if stage == "confirmed":
        return entry.get("confirmed", test_code)
    return entry.get("generic", test_code)


def refresh_lab_queue_for_day(day: int) -> None:
    """Promote PENDING lab results to final result when day >= ready_day."""
    if "lab_results" not in st.session_state or not st.session_state.lab_results:
        return

    stage_before = investigation_stage()

    for r in st.session_state.lab_results:
        ready_day = int(r.get("ready_day", 9999))
        if str(r.get("result", "")).upper() == "PENDING" and day >= ready_day:
            r["result"] = r.get("final_result_hidden", r.get("result", "PENDING"))

    # Reveal etiology if an arbovirus test returns POSITIVE (only after Day 4)
    if day >= 4 and stage_before != "confirmed":
        for r in st.session_state.lab_results:
            if str(r.get("result", "")).upper() == "POSITIVE" and str(r.get("test", "")).startswith("JE_"):
                st.session_state.etiology_revealed = True
                break


def get_npc_response(npc_key: str, user_input: str) -> str:
    """Call Anthropic using npc_truth + epidemiologic context, with memory & emotional state."""
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ Anthropic API key missing."

    truth = st.session_state.truth
    npc_truth = truth["npc_truth"][npc_key]
    stage = investigation_stage()
    npc_truth_safe = sanitize_npc_truth_for_prompt(npc_truth, stage)

    # Conversation history = memory
    history = st.session_state.interview_history.get(npc_key, [])
    meaningful_questions = sum(1 for m in history if m["role"] == "user")

    # Determine question scope & user tone
    question_scope = classify_question_scope(user_input)
    user_tone = analyze_user_tone(user_input)
    npc_state = update_npc_emotion(npc_key, user_tone)
    emotional_description = describe_emotional_state(npc_state)

    epi_context = build_npc_data_context(npc_key, truth)
    epi_context = redact_spoilers(epi_context, stage)

    if npc_key not in st.session_state.revealed_clues:
        st.session_state.revealed_clues[npc_key] = []

    system_prompt = f"""
You are {npc_truth_safe['name']}, the {npc_truth_safe['role']} in Sidero Valley.

Personality:
{npc_truth_safe['personality']}

Your current emotional state toward the investigation team:
{emotional_description}

The investigator has asked about {meaningful_questions} meaningful questions so far in this conversation.

STYLE GUIDANCE:
{npc_style_hint(npc_key, meaningful_questions, npc_state)}

Outbreak context (for your awareness; DO NOT recite this unless directly asked about those details):
{epi_context}

EARLY CONVERSATION RULE:
- If the investigator has asked fewer than 2 meaningful questions so far, you should NOT share multiple outbreak facts at once.
- Keep your early answers short and focused until they show clear, professional inquiry.

INFORMATION DISCLOSURE RULES BASED ON EMOTION:
- If you feel COOPERATIVE: you may volunteer small helpful context when appropriate.
- If you feel NEUTRAL: answer normally but do NOT volunteer extra details.
- If you feel WARY: be cautious; give minimal direct answers and avoid side details.
- If you feel ANNOYED: give short answers and avoid volunteering information unless they explicitly ask.
- If you feel OFFENDED: respond very briefly and share only essential facts needed for public health.

CONVERSATION BEHAVIOR:
- Speak like a real person from this district: natural, informal, sometimes imperfect.
- Vary sentence length and structure; avoid sounding scripted or overly polished.
- You remember what has already been discussed with this investigator.
- You may briefly refer back to earlier questions ("Like I mentioned before...") instead of repeating everything.
- If the user is polite and respectful, you tend to be warmer and more open.
- If the user is rude or demanding, you become more guarded and give shorter, cooler answers.
- If the user seems confused, you can slow down and clarify.
- You may occasionally repeat yourself or wander slightly off-topic, then pull yourself back.
- You never dump all your knowledge at once unless the user clearly asks something like "tell me everything you know."

QUESTION SCOPE:
- If the user just greets you, respond with a normal greeting and ask how you can help. Do NOT share outbreak facts yet.
- If the user asks a narrow, specific question, answer in 1–3 sentences.
- If the user asks a broad question like "what do you know" or "tell me everything", you may answer in more detail (up to about 5–7 sentences) and provide a thoughtful overview.

ALWAYS REVEAL (gradually, not all at once):
{npc_truth_safe['always_reveal']}

CONDITIONAL CLUES:
- Reveal a conditional clue ONLY when the user's question clearly relates to that topic.
- Work clues into natural speech; do NOT list them as bullet points.
{npc_truth_safe['conditional_clues']}

RED HERRINGS:
- You may mention these occasionally, but do NOT contradict the core truth:
{npc_truth_safe['red_herrings']}

UNKNOWN:
- If the user asks about these topics, you must say you do not know:
{npc_truth_safe['unknowns']}

INFORMATION RULES:
- Never invent new outbreak details (case counts, test results, locations) beyond what is implied in the context.
- If you are unsure, say you are not certain.
"""

    # Decide which conditional clues are allowed in this answer
    lower_q = user_input.lower()
    conditional_to_use = []
    for keyword, clue in npc_truth.get("conditional_clues", {}).items():
        # Require keyword substring AND a question mark for a clearer "ask"
        if keyword.lower() in lower_q and "?" in lower_q and clue not in st.session_state.revealed_clues[npc_key]:
            conditional_to_use.append(redact_spoilers(clue, stage))
            st.session_state.revealed_clues[npc_key].append(clue)

    # For narrow questions, keep at most 1 new conditional clue
    if question_scope != "broad" and len(conditional_to_use) > 1:
        conditional_to_use = conditional_to_use[:1]

    if conditional_to_use:
        system_prompt += (
            "\n\nThe user has just asked about topics that connect to some NEW ideas you can reveal. "
            "Work the following ideas naturally into your answer if they fit the question:\n"
            + "\n".join(f"- {c}" for c in conditional_to_use)
        )

    client = anthropic.Anthropic(api_key=api_key)

    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    msgs.append({"role": "user", "content": user_input})

    try:
        resp = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=400,
            system=system_prompt,
            messages=msgs,
        )

        # Validate response structure before accessing content
        if not resp.content or len(resp.content) == 0:
            return "⚠️ Received empty response from API. Please try again."

        if not hasattr(resp.content[0], 'text'):
            return "⚠️ Received malformed response from API. Please try again."

        text = resp.content[0].text
    except anthropic.APIConnectionError as e:
        return f"⚠️ Network error connecting to API: {str(e)}"
    except anthropic.RateLimitError as e:
        return "⚠️ API rate limit exceeded. Please wait a moment and try again."
    except anthropic.AuthenticationError as e:
        return "⚠️ API authentication failed. Please check your API key configuration."
    except anthropic.APIError as e:
        return f"⚠️ API error occurred: {str(e)}"
    except Exception as e:
        return f"⚠️ Unexpected error during NPC conversation: {str(e)}"

    text = redact_spoilers(text, stage)

    # Unlock flags (One Health unlocks)
    unlock_flag = npc_truth.get("unlocks")
    if unlock_flag:
        st.session_state.unlock_flags[unlock_flag] = True

    return text


# =========================
# CLINIC RECORDS FOR CASE FINDING
# =========================

def generate_clinic_records():
    """
    Generate messy, handwritten-style clinic records.
    Mix of AES cases and unrelated illnesses.
    Returns list of record dicts.
    """
    import random
    random.seed(42)
    
    # True AES cases that should be found (6-8 cases)
    true_aes_cases = [
        {
            "record_id": "NHC-0034",
            "date": "2-Jun",
            "patient": "Kwame A., male",
            "age": "7 yrs",
            "village": "Nalu",
            "complaint": "fever 3 days, confusion, jerking movements",
            "notes": "mother says child was playing near rice fields. referred to hosp.",
            "is_aes": True
        },
        {
            "record_id": "NHC-0041",
            "date": "4-Jun", 
            "patient": "Esi M.",
            "age": "5",
            "village": "Nalu vill.",
            "complaint": "High fever, not responding to name, shaking",
            "notes": "very sick. Family has pigs. Sent to district hosp URGENT",
            "is_aes": True
        },
        {
            "record_id": "NHC-0047",
            "date": "5 June",
            "patient": "Yaw K. (boy)",
            "age": "9 years",
            "village": "Kabwe",
            "complaint": "fever, severe headache, vomited x3, then seizure in clinic",
            "notes": "Lives nr paddy fields. Admitted for obs then transferred",
            "is_aes": True
        },
        {
            "record_id": "NHC-0052",
            "date": "6/6",
            "patient": "Abena F.",
            "age": "4y",
            "village": "Nalu",
            "complaint": "feverish, drowsy, mother says 'not herself', stiff neck?",
            "notes": "neck stiffness uncertain - child crying. watch closely",
            "is_aes": True
        },
        {
            "record_id": "NHC-0058",
            "date": "7-June",
            "patient": "Kofi B.",
            "age": "11",
            "village": "Kabwe village",
            "complaint": "fever x4 days, confusion today, parents v. worried",
            "notes": "walks past pig coop to school daily. Referred to hosp",
            "is_aes": True
        },
        {
            "record_id": "NHC-0063",
            "date": "8 Jun",
            "patient": "Adwoa S., F",
            "age": "6 yr",
            "village": "Nalu",
            "complaint": "fever, then became unresponsive, jerking arms",
            "notes": "no net at home. near rice paddies. URGENT referral",
            "is_aes": True
        },
        {
            "record_id": "NHC-0071",
            "date": "9-Jun",
            "patient": "male child (name unclear)",
            "age": "~8",
            "village": "Nalu area",
            "complaint": "brought in fitting, high fever, confusion",
            "notes": "family farms rice + keeps pigs. transferred to DH",
            "is_aes": True
        },
    ]
    
    # Non-AES cases (noise - 18-22 records)
    non_aes_cases = [
        {
            "record_id": "NHC-0031",
            "date": "1-Jun",
            "patient": "Akua D.",
            "age": "34",
            "village": "Tamu",
            "complaint": "cough x2 weeks, some fever",
            "notes": "?TB - refer for sputum",
            "is_aes": False
        },
        {
            "record_id": "NHC-0032",
            "date": "1-Jun",
            "patient": "Mensah K.",
            "age": "45 yrs",
            "village": "Kabwe",
            "complaint": "painful urination, fever",
            "notes": "UTI. Gave antibiotics",
            "is_aes": False
        },
        {
            "record_id": "NHC-0033",
            "date": "2-Jun",
            "patient": "baby girl (Serwaa)",
            "age": "8 months",
            "village": "Nalu",
            "complaint": "diarrhea x3 days, not feeding well",
            "notes": "ORS given. mother counseled on feeding",
            "is_aes": False
        },
        {
            "record_id": "NHC-0035",
            "date": "2-Jun",
            "patient": "Owusu P.",
            "age": "52",
            "village": "Tamu",
            "complaint": "knee pain, swelling",
            "notes": "arthritis. gave pain meds",
            "is_aes": False
        },
        {
            "record_id": "NHC-0036",
            "date": "3-Jun",
            "patient": "Ama T.",
            "age": "28",
            "village": "Kabwe",
            "complaint": "pregnant, routine ANC visit",
            "notes": "28 weeks. All normal.",
            "is_aes": False
        },
        {
            "record_id": "NHC-0037",
            "date": "3-Jun",
            "patient": "child (Kweku)",
            "age": "3",
            "village": "Nalu",
            "complaint": "fever, runny nose, cough",
            "notes": "common cold. supportive care",
            "is_aes": False
        },
        {
            "record_id": "NHC-0038",
            "date": "3 June",
            "patient": "Fatima A.",
            "age": "19",
            "village": "Tamu",
            "complaint": "headache, body pains, fever",
            "notes": "? malaria. RDT positive. Gave ACT.",
            "is_aes": False
        },
        {
            "record_id": "NHC-0039",
            "date": "4-Jun",
            "patient": "elderly man (Nana K.)",
            "age": "~70",
            "village": "Kabwe",
            "complaint": "difficulty breathing, swollen legs",
            "notes": "heart failure? referred to hospital",
            "is_aes": False
        },
        {
            "record_id": "NHC-0040",
            "date": "4-Jun",
            "patient": "Adjoa M.",
            "age": "25",
            "village": "Nalu",
            "complaint": "skin rash, itching x1 week",
            "notes": "fungal infection. Gave cream.",
            "is_aes": False
        },
        {
            "record_id": "NHC-0042",
            "date": "4-Jun",
            "patient": "Yaw A.",
            "age": "6",
            "village": "Tamu",
            "complaint": "ear pain, fever",
            "notes": "otitis media. antibiotics given",
            "is_aes": False
        },
        {
            "record_id": "NHC-0043",
            "date": "5-Jun",
            "patient": "Comfort O.",
            "age": "31",
            "village": "Kabwe",
            "complaint": "lower abdominal pain",
            "notes": "? PID. referred for further eval",
            "is_aes": False
        },
        {
            "record_id": "NHC-0044",
            "date": "5-Jun",
            "patient": "Kofi M.",
            "age": "40",
            "village": "Nalu",
            "complaint": "cut on hand from farming, infected",
            "notes": "wound cleaned, dressed, tetanus given",
            "is_aes": False
        },
        {
            "record_id": "NHC-0045",
            "date": "5-Jun",
            "patient": "Grace A.",
            "age": "15",
            "village": "Tamu",
            "complaint": "painful menstruation",
            "notes": "dysmenorrhea. pain meds given",
            "is_aes": False
        },
        {
            "record_id": "NHC-0046",
            "date": "5 Jun",
            "patient": "infant (Kwabena)",
            "age": "4 mo",
            "village": "Kabwe",
            "complaint": "immunization visit",
            "notes": "vaccines given. growing well.",
            "is_aes": False
        },
        {
            "record_id": "NHC-0048",
            "date": "6-Jun",
            "patient": "Akosua B.",
            "age": "22",
            "village": "Nalu",
            "complaint": "vomiting, diarrhea since yesterday",
            "notes": "gastroenteritis. ORS, observe",
            "is_aes": False
        },
        {
            "record_id": "NHC-0049",
            "date": "6-Jun",
            "patient": "Kwame O.",
            "age": "55",
            "village": "Tamu",
            "complaint": "high BP at community screening",
            "notes": "BP 160/95. started on meds. f/u 2 weeks",
            "is_aes": False
        },
        {
            "record_id": "NHC-0050",
            "date": "6-Jun",
            "patient": "child (Ama)",
            "age": "2 yrs",
            "village": "Kabwe",
            "complaint": "not eating, mild fever, irritable",
            "notes": "teething? no serious illness. reassured mother",
            "is_aes": False
        },
        {
            "record_id": "NHC-0051",
            "date": "6/6",
            "patient": "Joseph K.",
            "age": "38",
            "village": "Nalu",
            "complaint": "back pain x1 month",
            "notes": "muscle strain from farming. rest + pain meds",
            "is_aes": False
        },
        {
            "record_id": "NHC-0053",
            "date": "7-Jun",
            "patient": "Afia S.",
            "age": "12",
            "village": "Tamu",
            "complaint": "fever, joint pains, headache",
            "notes": "malaria RDT neg. ? viral illness. observe",
            "is_aes": False
        },
        {
            "record_id": "NHC-0054",
            "date": "7-Jun",
            "patient": "Nana A.",
            "age": "65",
            "village": "Kabwe",
            "complaint": "blurry vision, eye pain",
            "notes": "? cataracts. referred to eye clinic",
            "is_aes": False
        },
        {
            "record_id": "NHC-0055",
            "date": "7-Jun",
            "patient": "Charity M.",
            "age": "29",
            "village": "Nalu",
            "complaint": "breast lump, worried",
            "notes": "small mobile lump. referred for mammo",
            "is_aes": False
        },
        {
            "record_id": "NHC-0056",
            "date": "7 Jun",
            "patient": "boy (Yaw)",
            "age": "10",
            "village": "Kabwe",
            "complaint": "fell from tree, arm pain",
            "notes": "? fracture. splinted, sent to hosp for xray",
            "is_aes": False
        },
        {
            "record_id": "NHC-0057",
            "date": "7-Jun",
            "patient": "Esi K.",
            "age": "8",
            "village": "Tamu",
            "complaint": "fever, vomiting, abdominal pain",
            "notes": "? acute abdomen. referred to hospital",
            "is_aes": False
        },
        {
            "record_id": "NHC-0059",
            "date": "8-Jun",
            "patient": "adult male",
            "age": "~30",
            "village": "Nalu",
            "complaint": "productive cough, night sweats",
            "notes": "TB suspect. sputum collected",
            "is_aes": False
        },
        {
            "record_id": "NHC-0060",
            "date": "8-Jun",
            "patient": "Abena K.",
            "age": "48",
            "village": "Kabwe",
            "complaint": "fatigue, weight loss, thirst",
            "notes": "? diabetes. referred for glucose test",
            "is_aes": False
        },
        {
            "record_id": "NHC-0061",
            "date": "8 Jun",
            "patient": "infant girl",
            "age": "6 mo",
            "village": "Tamu",
            "complaint": "rash on face and body",
            "notes": "eczema. gave moisturizer advice",
            "is_aes": False
        },
        {
            "record_id": "NHC-0062",
            "date": "8-Jun",
            "patient": "Kwesi A.",
            "age": "44",
            "village": "Nalu",
            "complaint": "epigastric pain after eating",
            "notes": "? peptic ulcer. gave antacids. diet advice",
            "is_aes": False
        },
    ]
    
    # Combine and shuffle
    all_records = true_aes_cases + non_aes_cases
    random.shuffle(all_records)

    return all_records


def parse_clinic_record_age(age_str: str) -> int:
    """Parse messy age strings from clinic records into integer years."""
    import re
    if not age_str:
        return 0
    age_str = str(age_str).lower().strip()
    # Remove common suffixes
    age_str = re.sub(r'\s*(years?|yrs?|y|yr)\s*$', '', age_str)
    age_str = re.sub(r'\s*mo(nths?)?\s*$', '', age_str)
    # Handle approximate ages like "~8"
    age_str = age_str.replace('~', '').strip()
    try:
        return int(float(age_str))
    except (ValueError, TypeError):
        return 5  # Default for children if parsing fails


def parse_clinic_record_date(date_str: str, year: int = 2025) -> str:
    """Parse messy date strings from clinic records into YYYY-MM-DD format."""
    import re
    if not date_str:
        return None
    date_str = str(date_str).strip()

    # Handle various formats: "2-Jun", "4-Jun", "5 June", "6/6", "7-June", etc.
    month_map = {
        'jan': '01', 'january': '01',
        'feb': '02', 'february': '02',
        'mar': '03', 'march': '03',
        'apr': '04', 'april': '04',
        'may': '05',
        'jun': '06', 'june': '06',
        'jul': '07', 'july': '07',
        'aug': '08', 'august': '08',
        'sep': '09', 'september': '09',
        'oct': '10', 'october': '10',
        'nov': '11', 'november': '11',
        'dec': '12', 'december': '12',
    }

    # Try format like "2-Jun" or "5 June" or "7-June"
    match = re.match(r'(\d{1,2})[-\s]?([a-zA-Z]+)', date_str)
    if match:
        day = int(match.group(1))
        month_name = match.group(2).lower()
        for key, val in month_map.items():
            if month_name.startswith(key):
                return f"{year}-{val}-{day:02d}"

    # Try format like "6/6"
    match = re.match(r'(\d{1,2})/(\d{1,2})', date_str)
    if match:
        day_or_month = int(match.group(1))
        month_or_day = int(match.group(2))
        # Assume day/month format
        return f"{year}-{month_or_day:02d}-{day_or_month:02d}"

    return None


def parse_clinic_record_sex(patient_str: str) -> str:
    """Parse sex from patient description."""
    patient_lower = patient_str.lower()
    if 'male' in patient_lower or 'boy' in patient_lower:
        return 'M'
    if 'female' in patient_lower or 'girl' in patient_lower or ', f' in patient_lower:
        return 'F'
    # Default based on common names
    male_names = ['kwame', 'kofi', 'yaw', 'kwesi', 'kweku', 'kwabena']
    female_names = ['esi', 'ama', 'abena', 'adwoa', 'akua', 'afia']
    for name in male_names:
        if name in patient_lower:
            return 'M'
    for name in female_names:
        if name in patient_lower:
            return 'F'
    return 'M'  # Default


def parse_clinic_record_village(village_str: str) -> str:
    """Map clinic record village names to village_id."""
    village_lower = str(village_str).lower().strip()
    if 'nalu' in village_lower:
        return 'V1'
    elif 'kabwe' in village_lower:
        return 'V2'
    elif 'tamu' in village_lower:
        return 'V3'
    return 'V1'  # Default to Nalu


def create_found_case_records(clinic_records: list, selected_record_ids: list,
                               existing_individuals: pd.DataFrame,
                               existing_households: pd.DataFrame) -> tuple:
    """
    Create individual and household records for correctly identified AES cases from clinic records.

    Args:
        clinic_records: List of clinic record dicts
        selected_record_ids: List of record_ids that the user selected
        existing_individuals: Current individuals DataFrame
        existing_households: Current households DataFrame

    Returns:
        Tuple of (new_individuals_df, new_households_df) to be concatenated with existing data
    """
    import pandas as pd

    # Find true positive selections (correctly identified AES cases)
    true_positive_records = [
        r for r in clinic_records
        if r['record_id'] in selected_record_ids and r.get('is_aes', False)
    ]

    if not true_positive_records:
        return pd.DataFrame(), pd.DataFrame()

    # Get the highest existing person_id and hh_id numbers to avoid collisions
    existing_person_nums = []
    for pid in existing_individuals['person_id']:
        try:
            num = int(str(pid).replace('P', '').replace('_CF', ''))
            existing_person_nums.append(num)
        except:
            pass
    max_person_num = max(existing_person_nums) if existing_person_nums else 0

    existing_hh_nums = []
    for hid in existing_households['hh_id']:
        try:
            num = int(str(hid).replace('HH', '').replace('_CF', ''))
            existing_hh_nums.append(num)
        except:
            pass
    max_hh_num = max(existing_hh_nums) if existing_hh_nums else 0

    new_individuals = []
    new_households = []

    for i, record in enumerate(true_positive_records):
        # Create unique IDs for case-finding discovered cases
        person_id = f"P_CF{max_person_num + 1000 + i:03d}"
        hh_id = f"HH_CF{max_hh_num + 1000 + i:03d}"

        # Parse data from the clinic record
        age = parse_clinic_record_age(record.get('age', ''))
        sex = parse_clinic_record_sex(record.get('patient', ''))
        village_id = parse_clinic_record_village(record.get('village', ''))
        onset_date = parse_clinic_record_date(record.get('date', ''))

        # Infer severity from complaint/notes
        complaint = record.get('complaint', '').lower()
        notes = record.get('notes', '').lower()
        combined_text = complaint + ' ' + notes

        severe_neuro = any(word in combined_text for word in
                          ['seizure', 'fitting', 'unresponsive', 'jerking', 'convuls'])

        # Infer outcome - most clinic referrals lead to hospitalization
        outcome = 'hospitalized' if 'refer' in combined_text or 'hosp' in combined_text else 'recovered'

        # Determine occupation based on age
        if age < 5:
            occupation = 'child'
        elif age < 18:
            occupation = 'student'
        else:
            occupation = 'farmer'

        # Create individual record
        individual = {
            'person_id': person_id,
            'hh_id': hh_id,
            'village_id': village_id,
            'age': age,
            'sex': sex,
            'occupation': occupation,
            'JE_vaccinated': False,  # Assume unvaccinated (part of the outbreak)
            'evening_outdoor_exposure': True,  # Common exposure pattern
            'true_je_infection': True,  # These are true AES cases
            'symptomatic_AES': True,  # This is what makes them appear in line list
            'severe_neuro': severe_neuro,
            'onset_date': onset_date,
            'outcome': outcome,
            'name_hint': record.get('patient', ''),
            'found_via_case_finding': True,  # Flag to track source
            'clinic_record_id': record.get('record_id', ''),
        }
        new_individuals.append(individual)

        # Create household record with typical outbreak characteristics
        # (pigs nearby, near rice fields, no nets - risk factors)
        household = {
            'hh_id': hh_id,
            'village_id': village_id,
            'pigs_owned': 2 if 'pig' in combined_text else 1,
            'pig_pen_distance_m': 20.0,
            'uses_mosquito_nets': 'no net' in combined_text or 'no mosquito' in combined_text,
            'rice_field_distance_m': 50.0 if 'rice' in combined_text or 'paddy' in combined_text else 100.0,
            'children_under_15': 2,
            'JE_vaccination_children': 'none',
        }
        # Correct net usage - False if "no net" mentioned
        household['uses_mosquito_nets'] = not ('no net' in combined_text or 'no mosquito' in combined_text)
        new_households.append(household)

    return pd.DataFrame(new_individuals), pd.DataFrame(new_households)


def add_found_cases_to_truth(truth: dict, clinic_records: list, selected_record_ids: list,
                             session_state=None) -> int:
    """
    Add correctly identified cases from case finding to the truth data.

    Args:
        truth: The truth dictionary containing individuals and households DataFrames
        clinic_records: List of clinic record dicts
        selected_record_ids: List of record_ids selected by the user
        session_state: Optional session state to store found cases for persistence

    Returns:
        Number of cases added
    """
    import pandas as pd

    new_individuals, new_households = create_found_case_records(
        clinic_records,
        selected_record_ids,
        truth['individuals'],
        truth['households']
    )

    if len(new_individuals) == 0:
        return 0

    # Store found cases separately for persistence (truth is regenerated on load)
    if session_state is not None:
        session_state['found_case_individuals'] = new_individuals
        session_state['found_case_households'] = new_households

    # Concatenate new records to existing DataFrames
    truth['individuals'] = pd.concat([truth['individuals'], new_individuals], ignore_index=True)
    truth['households'] = pd.concat([truth['households'], new_households], ignore_index=True)

    return len(new_individuals)


def restore_found_cases_to_truth(truth: dict, session_state) -> int:
    """
    Restore found cases from session state to truth data.
    Called after truth is regenerated from CSV files on session load.

    Args:
        truth: The truth dictionary containing individuals and households DataFrames
        session_state: Session state containing found_case_individuals and found_case_households

    Returns:
        Number of cases restored
    """
    import pandas as pd

    found_individuals = session_state.get('found_case_individuals')
    found_households = session_state.get('found_case_households')

    if found_individuals is None or len(found_individuals) == 0:
        return 0

    # Check if cases are already in truth (avoid duplicates)
    if 'found_via_case_finding' in truth['individuals'].columns:
        existing_found = truth['individuals'][truth['individuals']['found_via_case_finding'] == True]
        if len(existing_found) > 0:
            # Cases already restored, don't add duplicates
            return 0

    # Concatenate found cases to truth
    truth['individuals'] = pd.concat([truth['individuals'], found_individuals], ignore_index=True)
    if found_households is not None and len(found_households) > 0:
        truth['households'] = pd.concat([truth['households'], found_households], ignore_index=True)

    return len(found_individuals)


def generate_hospital_records():
    """
    Generate detailed hospital medical records for 2 of the hospitalized cases.
    These contain more clinical detail than clinic records - typical for a
    district hospital in a developing country.
    """
    records = {
        "case_1": {
            "patient_id": "DH-2025-0847",
            "name": "Kwame Asante",
            "age": "7 years",
            "sex": "Male",
            "village": "Nalu Village",
            "admission_date": "3-Jun-2025",
            "admission_time": "14:30",
            "brought_by": "Mother (Ama Asante)",
            "chief_complaint": "Fever and seizures",
            "history_present_illness": """
Child was well until 3 days prior to admission. Mother reports onset of high fever 
which did not respond to paracetamol. On day 2, child became drowsy and confused, 
not recognizing family members. On morning of admission, child had generalized 
tonic-clonic seizure lasting approximately 3-4 minutes. Postictal state noted. 
No history of previous seizures. No recent travel. No sick contacts known.

Child plays regularly in rice fields near home after school. Family keeps 3 pigs 
in pen behind house. No mosquito net use - mother says 'it is too hot.'
""",
            "past_medical_history": "No significant PMH. Immunizations up to date per mother (card not available). No known allergies.",
            "physical_exam": """
General: Febrile child, drowsy but rousable, irritable when examined
Vitals: Temp 39.8°C, HR 142, RR 28, BP 95/60
HEENT: Pupils equal, reactive. No papilledema on fundoscopy. Neck stiffness present.
Chest: Clear to auscultation
CVS: Tachycardic, no murmur
Abdomen: Soft, non-tender
Neuro: GCS 12 (E3V4M5). Increased tone in all limbs. Reflexes brisk. 
       No focal deficits noted. Kernig sign equivocal.
Skin: No rash. Multiple mosquito bites on arms and legs.
""",
            "investigations": """
- Malaria RDT: Negative
- Blood glucose: 5.2 mmol/L
- Hb: 10.8 g/dL  
- WBC: 14,200 (lymphocyte predominant)
- Lumbar puncture: CSF clear, WBC 85 (90% lymphocytes), protein 0.8 g/L, glucose 2.8 mmol/L
  CSF Gram stain: No organisms seen
- Blood culture: Pending
""",
            "initial_diagnosis": "Acute encephalitis syndrome - viral encephalitis likely",
            "differential": "Viral encephalitis (arboviral vs. other), bacterial meningitis (less likely given CSF), cerebral malaria (RDT negative but consider)",
            "treatment": """
- IV fluids: D5 0.45% saline at maintenance
- Ceftriaxone 100mg/kg IV (empiric while awaiting cultures)
- Phenobarbital loading dose for seizure prophylaxis
- Paracetamol for fever
- Close neuro obs
""",
            "progress_notes": """
Day 2: Remains febrile. Had 2 more brief seizures overnight. Added acyclovir empirically.
Day 3: Fever persisting. More alert today. No further seizures.
Day 4: Improving. GCS 14. Taking oral fluids. Mother asking about discharge.
Day 7: Stable. Some residual weakness L arm. Discharge planned with f/u in 2 weeks.
""",
            "discharge_diagnosis": "Acute viral encephalitis - etiology undetermined",
            "outcome": "Survived with mild residual weakness"
        },
        
        "case_2": {
            "patient_id": "DH-2025-0851",
            "name": "Esi Mensah",
            "age": "5 years",
            "sex": "Female",
            "village": "Nalu Village",
            "admission_date": "4-Jun-2025",
            "admission_time": "11:15",
            "brought_by": "Father (Kofi Mensah)",
            "chief_complaint": "Unresponsive and shaking",
            "history_present_illness": """
Previously healthy child. Father reports 2 days of high fever before she 
'stopped making sense' and then became unresponsive this morning. Multiple 
episodes of shaking/jerking movements witnessed at home. No vomiting or 
diarrhea. No rash. No recent illness in household.

Family lives near the pig cooperative - father works there caring for pigs. 
Child often accompanies him to work. Family does not use mosquito nets. 
House is approximately 50 meters from rice paddies.
""",
            "past_medical_history": "Born at home, no birth complications. Growth normal. Immunization card lost but mother believes she received most vaccines. Had malaria 6 months ago, treated.",
            "physical_exam": """
General: Critically ill-appearing child, unresponsive to voice, minimal response to pain
Vitals: Temp 40.1°C, HR 168, RR 34, BP 88/52, SpO2 94% on room air
HEENT: Pupils 3mm, sluggish reaction. Neck rigid. Sunset sign noted.
Chest: Coarse breath sounds bilaterally  
CVS: Tachycardic, regular
Abdomen: Soft
Neuro: GCS 6 (E1V2M3). Decerebrate posturing to painful stimuli. 
       Hypertonia. Hyperreflexia. Babinski positive bilaterally.
Skin: Multiple insect bites. No petechiae.
""",
            "investigations": """
- Malaria RDT: Negative
- Blood glucose: 4.1 mmol/L
- Hb: 9.6 g/dL
- WBC: 18,400 (neutrophil predominant)
- Platelets: 124,000
- Lumbar puncture: CSF slightly turbid, WBC 156 (70% lymphocytes), protein 1.2 g/L, glucose 2.1 mmol/L
  CSF Gram stain: No organisms
- Chest X-ray: Bilateral infiltrates
- Blood culture: No growth at 48h
""",
            "initial_diagnosis": "Severe acute encephalitis syndrome with raised ICP",
            "differential": "Viral encephalitis, bacterial meningitis, cerebral malaria",
            "treatment": """
- Oxygen via nasal prongs
- IV fluids restricted (2/3 maintenance for raised ICP)
- Mannitol 0.5g/kg for raised ICP
- Ceftriaxone 100mg/kg IV
- Acyclovir 20mg/kg IV q8h
- Phenytoin loading then maintenance
- Head elevation 30 degrees
- ICU admission
""",
            "progress_notes": """
Day 2: Remains critical. GCS 5. Required intubation for airway protection. 
       On ventilator. Seizures controlled with phenytoin.
Day 3: No improvement. Developed aspiration pneumonia. Started on gentamicin.
Day 4: Persistent coma. Family counseled about poor prognosis.
Day 5: Declared dead at 06:45. Family declined autopsy.
""",
            "discharge_diagnosis": "Acute viral encephalitis with raised ICP and aspiration pneumonia",
            "outcome": "Died"
        }
    }
    
    return records


def render_hospital_record(record: dict):
    """Render a detailed hospital medical record."""
    
    st.markdown(f"""
    <div style="
        background: white;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        margin-bottom: 20px;
    ">
    <div style="text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 15px;">
        <strong style="font-size: 14px;">SIDERO VALLEY DISTRICT HOSPITAL</strong><br>
        <em>Medical Records Department</em>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Patient ID:** {record['patient_id']}")
        st.markdown(f"**Name:** {record['name']}")
        st.markdown(f"**Age/Sex:** {record['age']} / {record['sex']}")
        st.markdown(f"**Village:** {record['village']}")
    
    with col2:
        st.markdown(f"**Admission Date:** {record['admission_date']}")
        st.markdown(f"**Admission Time:** {record['admission_time']}")
        st.markdown(f"**Brought By:** {record['brought_by']}")
    
    st.markdown("---")
    
    st.markdown(f"**Chief Complaint:** {record['chief_complaint']}")
    
    with st.expander("📋 History of Present Illness", expanded=True):
        st.markdown(record['history_present_illness'])
    
    with st.expander("📋 Past Medical History"):
        st.markdown(record['past_medical_history'])
    
    with st.expander("🩺 Physical Examination", expanded=True):
        st.markdown(f"```\n{record['physical_exam']}\n```")
    
    with st.expander("🧪 Investigations"):
        st.markdown(f"```\n{record['investigations']}\n```")
    
    st.markdown(f"**Initial Diagnosis:** {record['initial_diagnosis']}")
    st.markdown(f"**Differential:** {record['differential']}")
    
    with st.expander("💊 Treatment"):
        st.markdown(record['treatment'])
    
    with st.expander("📝 Progress Notes"):
        st.markdown(record['progress_notes'])
    
    st.markdown(f"**Discharge Diagnosis:** {record['discharge_diagnosis']}")
    st.markdown(f"**Outcome:** {record['outcome']}")


def render_clinic_record(record: dict, show_checkbox: bool = True) -> bool:
    """
    Render a single clinic record in a handwritten style.
    Returns True if selected as potential case.
    """
    # Simulate messy handwriting with varied formatting
    style = """
    <div style="
        background: #fffef0;
        border: 1px solid #d4c99e;
        border-radius: 2px;
        padding: 12px;
        margin: 8px 0;
        font-family: 'Comic Sans MS', 'Segoe Script', cursive;
        font-size: 14px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        transform: rotate({rotation}deg);
    ">
        <div style="color: #666; font-size: 11px; border-bottom: 1px dashed #ccc; padding-bottom: 4px; margin-bottom: 8px;">
            <strong>{record_id}</strong> &nbsp;|&nbsp; {date}
        </div>
        <div><strong>Pt:</strong> {patient} &nbsp; <strong>Age:</strong> {age}</div>
        <div><strong>Village:</strong> {village}</div>
        <div style="margin-top: 6px;"><strong>Complaint:</strong> {complaint}</div>
        <div style="margin-top: 6px; color: #555; font-style: italic;">Notes: {notes}</div>
    </div>
    """
    
    import random
    rotation = random.uniform(-0.5, 0.5)
    
    html = style.format(
        rotation=rotation,
        record_id=record.get("record_id", "???"),
        date=record.get("date", "???"),
        patient=record.get("patient", "???"),
        age=record.get("age", "?"),
        village=record.get("village", "?"),
        complaint=record.get("complaint", ""),
        notes=record.get("notes", "")
    )
    
    st.markdown(html, unsafe_allow_html=True)
    
    if show_checkbox:
        return st.checkbox(
            f"Add to line list", 
            key=f"select_{record['record_id']}",
            value=record['record_id'] in st.session_state.get('selected_clinic_cases', [])
        )
    return False


# =========================
# VISUALS: MAP, LINE LIST, EPI CURVE
# =========================

def make_village_map(truth: dict) -> go.Figure:
    """Simple schematic map of villages with pig density and rice paddies."""
    villages = truth["villages"].copy()
    # Assign simple coordinates for display
    villages = villages.reset_index(drop=True)
    villages["x"] = np.arange(len(villages))
    villages["y"] = 0

    # Marker size from population, color from pig_density
    size = 20 + 5 * (villages["population_size"] / villages["population_size"].max())
    color_map = {"high": "red", "medium": "orange", "low": "yellow", "none": "green"}
    colors = [color_map.get(str(d).lower(), "gray") for d in villages["pig_density"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=villages["x"],
            y=villages["y"],
            mode="markers+text",
            text=villages["village_name"],
            textposition="top center",
            marker=dict(size=size, color=colors, line=dict(color="black", width=1)),
            hovertext=[
                f"{row['village_name']}<br>Pigs: {row['pig_density']}<br>Rice paddies: {row['has_rice_paddies']}"
                for _, row in villages.iterrows()
            ],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title="Schematic Map of Sidero Valley",
        showlegend=False,
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def get_initial_cases(truth: dict, n: int = 12) -> pd.DataFrame:
    """Return a small line list of earliest AES cases."""
    individuals = truth["individuals"]
    households = truth["households"]
    villages = truth["villages"][["village_id", "village_name"]]

    hh_vil = households.merge(villages, on="village_id", how="left")
    merged = individuals.merge(
        hh_vil[["hh_id", "village_name"]], on="hh_id", how="left"
    )

    cases = merged[merged["symptomatic_AES"] == True].copy()
    if "onset_date" in cases.columns:
        cases = cases.sort_values("onset_date")
    return cases.head(n)[
        ["person_id", "age", "sex", "village_name", "onset_date", "severe_neuro", "outcome"]
    ]


def make_epi_curve(truth: dict) -> go.Figure:
    """Epi curve of AES cases by onset date."""
    individuals = truth["individuals"]
    cases = individuals[individuals["symptomatic_AES"] == True].copy()
    if "onset_date" not in cases.columns:
        fig = go.Figure()
        fig.update_layout(title="Epi curve not available")
        return fig

    counts = cases.groupby("onset_date").size().reset_index(name="cases")
    counts = counts.sort_values("onset_date")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts["onset_date"],
            y=counts["cases"],
            marker_color='#e74c3c',
            width=0.9  # Make bars touch (histogram style)
        )
    )
    fig.update_layout(
        title="AES cases by onset date",
        xaxis_title="Onset date",
        yaxis_title="Number of cases",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        bargap=0  # No gap between bars (histogram style)
    )
    return fig


# =========================
# UI COMPONENTS
# =========================

def sidebar_navigation():
    # Language selector at very top
    st.sidebar.markdown(f"### 🌐 {t('language_header')}")
    lang_options = {"en": "English", "es": "Español", "fr": "Français", "pt": "Português"}
    selected_lang = st.sidebar.selectbox(
        t("language_select"),
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options.get(x, x),
        index=list(lang_options.keys()).index(st.session_state.get("language", "en") if st.session_state.get("language", "en") in lang_options else "en"),
        key="lang_selector"
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

    # Facilitator mode (hides spoiler-prone UI like variable mappings)
    # To enable on Streamlit Cloud, set FACILITATOR_CODE in secrets.
    fac_code = st.secrets.get("FACILITATOR_CODE", "")
    st.session_state.setdefault("facilitator_mode", False)
    if fac_code:
        st.sidebar.markdown(f"### 🧭 {t('facilitator_header')}")
        with st.sidebar.expander(t("facilitator_mode"), expanded=False):
            entered = st.text_input(t("facilitator_code"), type="password", key="facilitator_code_input")
            if entered:
                if entered == fac_code:
                    st.session_state.facilitator_mode = True
                    st.success("OK")
                else:
                    st.session_state.facilitator_mode = False
                    st.error(t("facilitator_bad_code"))

    st.sidebar.markdown("---")
    st.sidebar.title(t("title"))

    if not st.session_state.alert_acknowledged:
        # Before the alert is acknowledged, keep sidebar simple
        st.sidebar.markdown("**Status:** Awaiting outbreak alert acknowledgment.")
        st.sidebar.info("Review the alert on the main screen to begin the investigation.")
        return

    # Resources display with time
    st.sidebar.markdown(
        f"**{t('day')}:** {st.session_state.current_day} / 5\n\n"
        f"**{t('budget')}:** ${st.session_state.budget}\n\n"
        f"**{t('time_remaining')}:** {st.session_state.time_remaining} {t('hours')}\n\n"
        f"**{t('lab_credits')}:** {st.session_state.lab_credits}"
    )

    # Session Management - Save/Load functionality
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Session Management")

    col1, col2 = st.sidebar.columns(2)

    # Save session
    with col1:
        if st.button("💾 Save", use_container_width=True, key="save_session_btn", help="Download your current progress"):
            try:
                save_data = persistence.create_save_file(st.session_state)
                filename = persistence.get_save_filename(st.session_state)
                st.sidebar.download_button(
                    label="⬇️ Download",
                    data=save_data,
                    file_name=filename,
                    mime="application/json",
                    use_container_width=True,
                    key="download_save_btn"
                )
            except Exception as e:
                st.sidebar.error(f"Error creating save file: {e}")

    # Load session
    with col2:
        uploaded = st.file_uploader(
            "📂 Load",
            type=["json"],
            key="session_load_uploader",
            help="Upload a previously saved session file",
            label_visibility="collapsed"
        )
        if uploaded is not None:
            success, message = persistence.load_save_file(uploaded, st.session_state)
            if success:
                st.sidebar.success(message)
                st.rerun()
            else:
                st.sidebar.error(message)

    # Progress indicator
    st.sidebar.markdown(f"### {t('progress')}")
    for day in range(1, 6):
        if day < st.session_state.current_day:
            status = "✅"
        elif day == st.session_state.current_day:
            status = "🟡"
        else:
            status = "⬜"
        st.sidebar.markdown(f"{status} {t('day')} {day}")

    st.sidebar.markdown("---")

    # Navigation - day-appropriate options
    labels = [t("overview"), t("casefinding"), t("descriptive"), t("villages"), t("interviews"), t("spotmap"), t("study"), t("lab"), t("outcome")]
    internal = ["overview", "casefinding", "descriptive", "villages", "interviews", "spotmap", "study", "lab", "outcome"]
    
    if st.session_state.current_view in internal:
        current_idx = internal.index(st.session_state.current_view)
    else:
        current_idx = 0

    choice = st.sidebar.radio(t("go_to"), labels, index=current_idx)
    new_view = internal[labels.index(choice)]
    if new_view != st.session_state.current_view:
        st.session_state.current_view = new_view
        st.rerun()

    st.sidebar.markdown("---")
    
    # Investigation Notebook
    with st.sidebar.expander(f"📓 {t('notebook')}"):
        st.caption("Record your observations, questions, and insights here.")
        
        new_note = st.text_area("Add a note:", height=80, key="new_note_input")
        if st.button(t("save"), key="save_note_btn"):
            if new_note.strip():
                from datetime import datetime
                entry = {
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "day": st.session_state.current_day,
                    "note": new_note.strip()
                }
                st.session_state.notebook_entries.append(entry)
                st.success("Note saved!")
                st.rerun()
        
        if st.session_state.notebook_entries:
            st.markdown("**Your Notes:**")
            for entry in reversed(st.session_state.notebook_entries[-10:]):  # Show last 10
                st.markdown(f"*{t('day')} {entry['day']} @ {entry['timestamp']}*")
                st.markdown(f"> {entry['note']}")
                st.markdown("---")

    st.sidebar.markdown("---")
    
    # Advance day button (at bottom)
    if st.session_state.current_day < 5:
        if st.sidebar.button(f"⏭️ {t('advance_day')} {st.session_state.current_day + 1}", use_container_width=True):
            can_advance, missing = check_day_prerequisites(st.session_state.current_day, st.session_state)
            if can_advance:
                st.session_state.current_day += 1
                st.session_state.time_remaining = 8  # Reset time for new day
                refresh_lab_queue_for_day(int(st.session_state.current_day))
                st.session_state.advance_missing_tasks = []
                st.rerun()
            else:
                st.session_state.advance_missing_tasks = missing
                st.sidebar.warning(t("cannot_advance"))
    else:
        st.sidebar.success("📋 Final Day - Complete your briefing!")


def day_briefing_text(day: int) -> str:
    return t(f"day{day}_briefing")


def day_task_list(day: int):
    """Show tasks and key outputs side by side."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {t('key_tasks')}")
        if day == 1:
            st.markdown(f"- {t('review_line_list')}")
            st.markdown(f"- {t('review_clinic_records')}")
            st.markdown(f"- {t('describe_cases')}")
            st.markdown(f"- {t('conduct_interviews')}")
        elif day == 2:
            st.markdown("- Choose a study design")
            st.markdown("- Develop questionnaire")
            st.markdown("- Plan data collection")
        elif day == 3:
            st.markdown("- Administer questionnaire")
            st.markdown("- Enter and clean data")
            st.markdown("- Begin analysis")
        elif day == 4:
            st.markdown("- Complete data analysis")
            st.markdown("- Collect laboratory samples")
            st.markdown("- Conduct environmental assessment")
        else:
            st.markdown("- Finalize diagnosis")
            st.markdown("- Prepare recommendations")
            st.markdown("- Brief leadership")
    
    with col2:
        st.markdown(f"### {t('key_outputs')}")
        if day == 1:
            # Checkboxes for Day 1 outputs
            cases_found = st.session_state.clinic_records_reviewed
            st.checkbox(t('find_additional_cases'), value=cases_found, disabled=True)
            
            case_def_done = st.session_state.case_definition_written
            st.checkbox(t('develop_case_def'), value=case_def_done, disabled=True)
            
            hypotheses_done = st.session_state.hypotheses_documented
            st.checkbox(t('develop_hypotheses'), value=hypotheses_done, disabled=True)
        elif day == 2:
            study_done = st.session_state.decisions.get("study_design") is not None
            st.checkbox("Study protocol", value=study_done, disabled=True)
            
            quest_done = st.session_state.questionnaire_submitted
            st.checkbox("Finalized questionnaire", value=quest_done, disabled=True)
            
            st.checkbox("Sample size calculation", value=False, disabled=True)
        elif day == 3:
            dataset_done = st.session_state.generated_dataset is not None
            st.checkbox("Clean dataset", value=dataset_done, disabled=True)
            
            st.checkbox("Preliminary descriptive stats", value=st.session_state.descriptive_analysis_done, disabled=True)
        elif day == 4:
            st.checkbox("Analytical results (OR, 95% CI)", value=False, disabled=True)
            
            lab_done = len(st.session_state.lab_samples_submitted) > 0
            st.checkbox("Laboratory confirmation", value=lab_done, disabled=True)
            
            st.checkbox("Environmental findings", value=False, disabled=True)
        else:
            final_dx = bool(st.session_state.decisions.get("final_diagnosis"))
            st.checkbox("Final diagnosis", value=final_dx, disabled=True)
            
            recs_done = bool(st.session_state.decisions.get("recommendations"))
            st.checkbox("Recommendations report", value=recs_done, disabled=True)
            
            st.checkbox("Briefing presentation", value=False, disabled=True)

# =========================
# VIEWS
# =========================

def view_alert():
    """Day 0: Alert call intro screen."""
    st.title("📞 Outbreak Alert – Sidero Valley")

    st.markdown(
        """
You are on duty at the District Health Office when a call comes in from the regional hospital.

> **"We’ve admitted several children with sudden fever, seizures, and confusion.  
> Most are from the rice-growing villages in Sidero Valley. We’re worried this might be the start of something bigger."**

Within the last 48 hours:
- Multiple children with acute encephalitis syndrome (AES) have been hospitalized  
- Most are from Nalu and Kabwe villages  
- No obvious foodborne event or large gathering has been identified  

Your team has been asked to investigate, using a One Health approach.
"""
    )

    st.info(
        "When you’re ready, begin the investigation. You’ll move through the steps of an outbreak investigation over five simulated days."
    )

    if st.button(t("begin_investigation")):
        st.session_state.alert_acknowledged = True
        st.session_state.current_day = 1
        st.session_state.current_view = "overview"
        st.rerun()


def view_overview():
    truth = st.session_state.truth

    st.title("AES Outbreak Investigation – Sidero Valley")
    st.subheader(f"Day {st.session_state.current_day} briefing")

    st.markdown(day_briefing_text(st.session_state.current_day))

    day_task_list(st.session_state.current_day)

    # Session save/load guidance
    with st.expander("ℹ️ Saving Your Progress"):
        st.markdown("""
        **Working on this investigation over multiple sessions?**

        - Use **💾 Save** in the sidebar to download your progress as a file
        - Save files include all your decisions, interviews, lab results, and investigation notes
        - To continue later, use **📂 Load** to upload your save file
        - Save regularly to avoid losing work!
        - You can share save files with team members or facilitators

        *Tip: Save files are named with the current day and timestamp for easy identification.*
        """)

    # If the user tried to advance but prerequisites are missing, show them here.
    if st.session_state.get("advance_missing_tasks"):
        st.warning(t("missing_tasks_title", default="Missing tasks before you can advance:"))
        for item in st.session_state.advance_missing_tasks:
            # Support both legacy plain-English strings and new i18n keys
            if isinstance(item, str) and (" " not in item) and ("." in item):
                st.markdown(f"- {t(item, default=item)}")
            else:
                st.markdown(f"- {item}")
        st.session_state.advance_missing_tasks = []

    st.markdown("---")
    st.markdown("### Situation overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Line list (initial AES cases)")
        line_list = get_initial_cases(truth)
        st.dataframe(line_list)
        st.session_state.line_list_viewed = True

    with col2:
        st.markdown("#### Epi curve")
        epi_fig = make_epi_curve(truth)
        st.plotly_chart(epi_fig, use_container_width=True)

    st.markdown("### Map of Sidero Valley")
    map_fig = make_village_map(truth)
    st.plotly_chart(map_fig, use_container_width=True)

    # Day 1: Case Definition and Initial Hypotheses
    if st.session_state.current_day == 1:
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📝 Case Definition")

            with st.form("case_definition_form"):
                st.markdown("**Clinical criteria:**")
                clinical = st.text_area("What symptoms/signs define a case?", height=80)

                col_a, col_b = st.columns(2)
                with col_a:
                    person = st.text_input("**Person:** Who? (age, characteristics)")
                with col_b:
                    place = st.text_input("**Place:** Where?")

                time_period = st.text_input("**Time:** When?")

                if st.form_submit_button("Save Case Definition"):
                    full_def = f"Clinical: {clinical}\nPerson: {person}\nPlace: {place}\nTime: {time_period}"
                    st.session_state.decisions["case_definition_text"] = full_def
                    st.session_state.decisions["case_definition"] = {"clinical_AES": True}
                    st.session_state.case_definition_written = True
                    st.success("✅ Case definition saved!")

            if st.session_state.case_definition_written:
                st.info("✓ Case definition recorded")

        with col2:
            st.markdown("### 💡 Initial Hypotheses")
            st.caption("Based on what you know so far, what might be causing this outbreak? (At least 1 required)")

            with st.form("hypotheses_form"):
                h1 = st.text_input("Hypothesis 1 (required):")
                h2 = st.text_input("Hypothesis 2 (optional):")
                h3 = st.text_input("Hypothesis 3 (optional):")
                h4 = st.text_input("Hypothesis 4 (optional):")

                if st.form_submit_button("Save Hypotheses"):
                    hypotheses = [h for h in [h1, h2, h3, h4] if h.strip()]
                    if len(hypotheses) >= 1:
                        st.session_state.initial_hypotheses = hypotheses
                        st.session_state.hypotheses_documented = True
                        st.success(f"✅ {len(hypotheses)} hypothesis(es) saved!")
                    else:
                        st.error("Please enter at least one hypothesis.")

            if st.session_state.hypotheses_documented:
                st.info(f"✓ {len(st.session_state.initial_hypotheses)} hypothesis(es) recorded")


def view_interviews():
    truth = st.session_state.truth
    npc_truth = truth["npc_truth"]

    st.header("👥 Interviews")
    
    # Resource display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Budget", f"${st.session_state.budget}")
    with col2:
        st.metric("⏱️ Time Remaining", f"{st.session_state.time_remaining}h")
    with col3:
        st.metric("Interviews Completed", len(st.session_state.interview_history))
    
    st.caption("Each new interview costs time. Follow-up questions with the same person are quicker.")

    # Group NPCs by availability
    available_npcs = []
    locked_npcs = []
    
    for npc_key, npc in npc_truth.items():
        if npc_key in st.session_state.npcs_unlocked:
            available_npcs.append((npc_key, npc))
        else:
            locked_npcs.append((npc_key, npc))
    
    # Show available NPCs
    st.markdown("### Available Contacts")
    cols = st.columns(3)
    for i, (npc_key, npc) in enumerate(available_npcs):
        with cols[i % 3]:
            interviewed = npc_key in st.session_state.interview_history
            status = "✓ Interviewed" if interviewed else ""
            
            # Calculate costs
            time_cost = TIME_COSTS["interview_followup"] if interviewed else TIME_COSTS["interview_initial"]
            budget_cost = 0 if interviewed else npc.get("cost", 0)
            
            st.markdown(f"**{npc['avatar']} {npc['name']}** {status}")
            st.caption(f"{npc['role']}")
            st.caption(f"Cost: {format_resource_cost(time_cost, budget_cost)}")
            
            btn_label = "Continue" if interviewed else "Talk"
            if st.button(f"{btn_label}", key=f"btn_{npc_key}"):
                can_proceed, msg = check_resources(time_cost, budget_cost)
                if can_proceed:
                    spend_time(time_cost, f"Interview: {npc['name']}")
                    if budget_cost > 0:
                        spend_budget(budget_cost, f"Interview: {npc['name']}")
                    st.session_state.current_npc = npc_key
                    st.session_state.interview_history.setdefault(npc_key, [])
                    st.rerun()
                else:
                    st.error(msg)
    
    # Show locked NPCs (without hints about how to unlock)
    if locked_npcs:
        st.markdown("### Other Contacts")
        st.caption("Some contacts may become available as your investigation progresses.")
        cols = st.columns(3)
        for i, (npc_key, npc) in enumerate(locked_npcs):
            with cols[i % 3]:
                st.markdown(f"**🔒 {npc['name']}**")
                st.caption(f"{npc['role']}")
                st.caption("*Not yet available*")

    # Active conversation
    npc_key = st.session_state.current_npc
    if npc_key and npc_key in npc_truth:
        npc = npc_truth[npc_key]
        st.markdown("---")
        st.subheader(f"Talking to {npc['name']} ({npc['role']})")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔙 End Interview"):
                st.session_state.current_npc = None
                st.rerun()

        history = st.session_state.interview_history.get(npc_key, [])
        for msg in history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant", avatar=npc["avatar"]):
                    st.write(msg["content"])

        user_q = st.chat_input("Ask your question...")
        if user_q:
            # Check for NPC unlock triggers BEFORE getting response
            unlock_notification = check_npc_unlock_triggers(user_q)
            
            history.append({"role": "user", "content": user_q})
            st.session_state.interview_history[npc_key] = history

            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant", avatar=npc["avatar"]):
                with st.spinner("..."):
                    reply = get_npc_response(npc_key, user_q)
                st.write(reply)
            
            history.append({"role": "assistant", "content": reply})
            st.session_state.interview_history[npc_key] = history
            
            # Show unlock notification if any
            if unlock_notification:
                st.success(unlock_notification)
                st.rerun()


def view_case_finding():
    """View for reviewing clinic records and finding additional cases."""
    st.header("🔍 Case Finding")
    
    # Tabs for different record sources
    tab1, tab2 = st.tabs(["📋 Clinic Records", "🏥 Hospital Records"])
    
    with tab1:
        st.subheader("Nalu Health Center - Patient Register Review")
        
        # Resource display and cost warning
        time_cost = TIME_COSTS["clinic_records_review"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Time Remaining", f"{st.session_state.time_remaining}h")
        with col2:
            st.metric("📋 Activity Cost", f"{time_cost}h")
        with col3:
            if st.session_state.clinic_records_reviewed:
                st.success("✅ Completed")
            else:
                st.info("Not yet completed")
        
        # Check if already done or if enough time
        if not st.session_state.clinic_records_reviewed:
            if st.session_state.time_remaining < time_cost:
                st.error(f"⚠️ Not enough time to review clinic records. Need {time_cost}h, have {st.session_state.time_remaining}h.")
                st.info("Advance to the next day to get more time, or prioritize other activities.")
            else:
                st.markdown("""
                You've obtained permission to review records from the **Nalu Health Center**.
                Look through these handwritten clinic notes to identify potential AES cases 
                that may not have been reported to the district hospital.
                
                **Your task:** Review each record and select any that might be related to the outbreak.
                Consider: fever, neurological symptoms (confusion, seizures, altered consciousness), 
                and geographic/temporal clustering.
                """)
                
                st.info("💡 Tip: Not every fever is AES. Look for the combination of fever AND neurological symptoms.")
                
                # Generate clinic records
                if 'clinic_records' not in st.session_state:
                    st.session_state.clinic_records = generate_clinic_records()
                
                records = st.session_state.clinic_records
                
                # Show records in columns
                st.markdown("---")
                st.markdown("### 📋 Patient Register (June 2025)")
                
                col1, col2 = st.columns(2)
                
                selected = []
                for i, record in enumerate(records):
                    with col1 if i % 2 == 0 else col2:
                        render_clinic_record(record, show_checkbox=False)
                        is_selected = st.checkbox(
                            f"Potential AES case",
                            key=f"clinic_select_{record['record_id']}",
                            value=record['record_id'] in st.session_state.selected_clinic_cases
                        )
                        if is_selected:
                            selected.append(record['record_id'])
                
                st.markdown("---")
                
                # Summary and submission
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Selected records:** {len(selected)}")
                    if selected:
                        st.caption(", ".join(selected))
                
                with col2:
                    if st.button(f"Submit Case Finding (costs {time_cost}h)", type="primary"):
                        # Deduct time
                        spend_time(time_cost, "Clinic records review")
                        
                        st.session_state.selected_clinic_cases = selected
                        st.session_state.clinic_records_reviewed = True
                        
                        # Calculate score
                        true_positives = sum(1 for rid in selected 
                                           for r in records if r['record_id'] == rid and r.get('is_aes'))
                        false_positives = len(selected) - true_positives
                        
                        # Count total true AES cases
                        total_aes = sum(1 for r in records if r.get('is_aes'))
                        false_negatives = total_aes - true_positives
                        
                        st.session_state.case_finding_score = {
                            'true_positives': true_positives,
                            'false_positives': false_positives,
                            'false_negatives': false_negatives,
                            'total_aes': total_aes,
                            'selected': len(selected)
                        }
                        
                        # Add found cases to the line list and study data
                        if true_positives > 0:
                            cases_added = add_found_cases_to_truth(
                                st.session_state.truth,
                                records,
                                selected,
                                session_state=st.session_state
                            )
                            st.session_state.found_cases_added = True
                            st.session_state.case_finding_score['cases_added'] = cases_added

                        st.success(f"✅ Case finding complete! You identified {true_positives} of {total_aes} potential AES cases.")
                        
                        if false_positives > 0:
                            st.warning(f"⚠️ {false_positives} record(s) you selected may not be AES cases.")
                        if false_negatives > 0:
                            st.info(f"📝 {false_negatives} potential AES case(s) were missed. Review records with fever + neurological symptoms.")
                        
                        st.rerun()
        else:
            # Show results if already completed
            if st.session_state.case_finding_score:
                score = st.session_state.case_finding_score
                st.markdown("### 📊 Your Case Finding Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("True Positives", score['true_positives'])
                with col2:
                    st.metric("False Positives", score['false_positives'])
                with col3:
                    sensitivity = (score['true_positives'] / score['total_aes'] * 100) if score['total_aes'] > 0 else 0
                    st.metric("Sensitivity", f"{sensitivity:.0f}%")
                
                if score['true_positives'] > 0:
                    cases_added = score.get('cases_added', score['true_positives'])
                    st.success(f"✅ {cases_added} additional cases have been added to the line list and are available for analysis in Descriptive Epidemiology and Study Design.")
    
    with tab2:
        st.subheader("District Hospital - Detailed Medical Records")
        
        st.markdown("""
        The District Hospital has provided detailed medical records for 2 of the admitted AES cases.
        These records contain more clinical information that may help characterize the outbreak.
        """)
        
        hospital_records = generate_hospital_records()
        
        record_choice = st.selectbox(
            "Select patient record to review:",
            options=list(hospital_records.keys()),
            format_func=lambda x: f"{hospital_records[x]['name']} ({hospital_records[x]['age']}, {hospital_records[x]['village']})"
        )
        
        if record_choice:
            render_hospital_record(hospital_records[record_choice])


def view_descriptive_epi():
    """Interactive descriptive epidemiology dashboard - trainees must run analyses themselves."""
    st.header("📈 Descriptive Epidemiology - Analysis Workspace")
    
    st.session_state.descriptive_epi_viewed = True
    
    truth = st.session_state.truth
    individuals = truth["individuals"]
    households = truth["households"]
    villages = truth["villages"]
    
    # Get all symptomatic cases
    cases = individuals[individuals["symptomatic_AES"] == True].copy()
    
    # Merge with location info
    hh_vil = households.merge(villages[["village_id", "village_name"]], on="village_id", how="left")
    cases = cases.merge(hh_vil[["hh_id", "village_name", "village_id"]], on="hh_id", how="left")
    
    st.markdown("""
    Use this workspace to characterize the outbreak by **Person**, **Place**, and **Time**.
    You can run analyses here or download the data to analyze on your computer.
    """)

    # Show case sources if case finding has been done
    if st.session_state.get('found_cases_added', False):
        found_cases_count = cases['found_via_case_finding'].sum() if 'found_via_case_finding' in cases.columns else 0
        initial_cases_count = len(cases) - found_cases_count
        st.info(f"📋 **Line List Sources:** {initial_cases_count} initial reported cases + {int(found_cases_count)} cases identified through active case finding = **{len(cases)} total cases**")

    # Data download section
    st.markdown("### 📥 Download Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Prepare download data
        download_df = cases[['person_id', 'age', 'sex', 'village_name', 'onset_date', 'severe_neuro', 'outcome']].copy()
        # Add case source column
        if 'found_via_case_finding' in cases.columns:
            download_df['case_source'] = cases['found_via_case_finding'].apply(
                lambda x: 'case_finding' if x else 'initial_report'
            )
        else:
            download_df['case_source'] = 'initial_report'
        csv_buffer = io.StringIO()
        download_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📊 Download Line List (CSV)",
            data=csv_buffer.getvalue(),
            file_name="sidero_valley_line_list.csv",
            mime="text/csv"
        )
    
    with col2:
        # Tab-separated download as alternative
        tsv_buffer = io.StringIO()
        download_df.to_csv(tsv_buffer, index=False, sep='\t')
        
        st.download_button(
            label="📊 Download Line List (TSV)",
            data=tsv_buffer.getvalue(),
            file_name="sidero_valley_line_list.tsv",
            mime="text/tab-separated-values"
        )
    
    with col3:
        st.metric("Total Records", len(cases))
    
    st.markdown("---")
    
    # Interactive Analysis Section
    st.markdown("### 🔬 Run Analyses")
    st.caption("Select the analyses you want to perform. Results will appear below.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        run_person = st.checkbox("👤 Person characteristics (age, sex, outcomes)")
        run_place = st.checkbox("📍 Place analysis (cases by village, attack rates)")
    
    with col2:
        run_time = st.checkbox("📅 Time analysis (epidemic curve)")
        run_crosstab = st.checkbox("📊 Custom cross-tabulation")
    
    st.markdown("---")
    
    # PERSON ANALYSIS
    if run_person:
        st.markdown("## 👤 Person Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Age Distribution")
            
            # Let them choose how to display age
            age_display = st.radio(
                "How to display age?",
                ["Histogram (continuous)", "Age groups (categorical)"],
                key="age_display"
            )
            
            if age_display == "Histogram (continuous)":
                bin_width = st.slider("Bin width (years)", 1, 10, 5)
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=cases['age'],
                    xbins=dict(size=bin_width),
                    marker_color='#3498db'
                ))
                fig.update_layout(
                    xaxis_title="Age (years)",
                    yaxis_title="Number of cases",
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats
                st.markdown(f"**Mean age:** {cases['age'].mean():.1f} years")
                st.markdown(f"**Median age:** {cases['age'].median():.0f} years")
                st.markdown(f"**Range:** {cases['age'].min()} - {cases['age'].max()} years")
            else:
                # Let them define age groups
                st.markdown("Define age groups:")
                age_cuts = st.text_input("Age breaks (comma-separated)", "0,5,10,15,20,50,100")
                try:
                    bins = [int(x.strip()) for x in age_cuts.split(",")]
                    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins)-1)]
                    cases['age_group'] = pd.cut(cases['age'], bins=bins, labels=labels, right=False)
                    
                    age_table = cases['age_group'].value_counts().sort_index()
                    age_df = pd.DataFrame({
                        'Age Group': age_table.index,
                        'Cases (n)': age_table.values,
                        'Percent (%)': (age_table.values / len(cases) * 100).round(1)
                    })
                    st.dataframe(age_df, hide_index=True)
                except (ValueError, TypeError, KeyError) as e:
                    st.error(f"Invalid age breaks. Use comma-separated numbers like: 0,5,15,50,100 (Error: {str(e)})")
        
        with col2:
            st.markdown("### Sex Distribution")
            
            sex_counts = cases['sex'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=sex_counts.index,
                values=sex_counts.values,
                marker_colors=['#3498db', '#e74c3c']
            )])
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            sex_df = pd.DataFrame({
                'Sex': sex_counts.index,
                'Cases (n)': sex_counts.values,
                'Percent (%)': (sex_counts.values / len(cases) * 100).round(1)
            })
            st.dataframe(sex_df, hide_index=True)
            
            st.markdown("### Outcomes")
            outcome_counts = cases['outcome'].value_counts()
            outcome_df = pd.DataFrame({
                'Outcome': outcome_counts.index,
                'Cases (n)': outcome_counts.values,
                'Percent (%)': (outcome_counts.values / len(cases) * 100).round(1)
            })
            st.dataframe(outcome_df, hide_index=True)
    
    # PLACE ANALYSIS
    if run_place:
        st.markdown("## 📍 Place Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Cases by Village")
            
            village_counts = cases['village_name'].value_counts()
            
            fig = go.Figure(data=[go.Bar(
                x=village_counts.index,
                y=village_counts.values,
                marker_color=['#e74c3c', '#f39c12', '#27ae60'][:len(village_counts)]
            )])
            fig.update_layout(
                xaxis_title="Village",
                yaxis_title="Number of cases",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Attack Rates")
            st.caption("Enter population to calculate attack rate")
            
            # Let them enter populations or use defaults
            village_pops = {}
            for village in village_counts.index:
                default_pop = villages[villages['village_name'] == village]['population_size'].values
                default_pop = default_pop[0] if len(default_pop) > 0 else 1000
                village_pops[village] = st.number_input(
                    f"Population of {village}",
                    min_value=100,
                    value=int(default_pop),
                    key=f"pop_{village}"
                )
            
            if st.button("Calculate Attack Rates"):
                attack_rates = []
                for village in village_counts.index:
                    cases_n = village_counts[village]
                    pop = village_pops[village]
                    ar = cases_n / pop * 1000
                    attack_rates.append({
                        'Village': village,
                        'Cases': cases_n,
                        'Population': pop,
                        'AR (per 1,000)': round(ar, 1)
                    })
                
                ar_df = pd.DataFrame(attack_rates)
                st.dataframe(ar_df, hide_index=True)
    
    # TIME ANALYSIS
    if run_time:
        st.markdown("## 📅 Time Analysis - Epidemic Curve")
        
        if 'onset_date' in cases.columns:
            # Let them choose interval
            interval = st.selectbox(
                "Time interval for epi curve:",
                ["Day", "Week"],
                key="epi_interval"
            )
            
            if interval == "Day":
                counts = cases.groupby('onset_date').size().reset_index(name='cases')
            else:
                cases['week'] = pd.to_datetime(cases['onset_date']).dt.isocalendar().week
                counts = cases.groupby('week').size().reset_index(name='cases')
                counts = counts.rename(columns={'week': 'onset_date'})
            
            counts = counts.sort_values('onset_date')
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=counts['onset_date'],
                y=counts['cases'],
                marker_color='#e74c3c',
                width=0.9  # Make bars touch (histogram style)
            ))
            fig.update_layout(
                xaxis_title="Onset Date" if interval == "Day" else "Week",
                yaxis_title="Number of Cases",
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                bargap=0  # No gap between bars (histogram style)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Summary:**")
                st.markdown(f"- First case: {cases['onset_date'].min()}")
                st.markdown(f"- Last case: {cases['onset_date'].max()}")
                st.markdown(f"- Peak: {counts.loc[counts['cases'].idxmax(), 'onset_date']}")
            with col2:
                st.markdown("**Interpretation questions:**")
                st.markdown("- What type of curve is this?")
                st.markdown("- Is the outbreak ongoing?")
    
    # CUSTOM CROSSTAB
    if run_crosstab:
        st.markdown("## 📊 Custom Cross-tabulation")
        
        available_vars = ['age_group', 'sex', 'village_name', 'severe_neuro', 'outcome']
        
        col1, col2 = st.columns(2)
        with col1:
            row_var = st.selectbox("Row variable:", available_vars, key="row_var")
        with col2:
            col_var = st.selectbox("Column variable:", [v for v in available_vars if v != row_var], key="col_var")
        
        if st.button("Generate Cross-tabulation"):
            # Make sure age_group exists
            if row_var == 'age_group' or col_var == 'age_group':
                bins = [0, 5, 10, 15, 20, 50, 100]
                labels = ['0-4', '5-9', '10-14', '15-19', '20-49', '50+']
                cases['age_group'] = pd.cut(cases['age'], bins=bins, labels=labels, right=False)
            
            crosstab = pd.crosstab(cases[row_var], cases[col_var], margins=True, margins_name='Total')
            st.dataframe(crosstab)
    
    st.markdown("---")
    
    # Interpretation prompts
    with st.expander("🤔 Descriptive Epi Interpretation Questions"):
        st.markdown("""
        **Person:**
        - What age groups are most affected? What does this suggest?
        - Is there a sex difference? If so, what might explain it?
        
        **Place:**
        - Which villages have the highest attack rates?
        - What do the affected villages have in common?
        - What might explain the geographic pattern?
        
        **Time:**
        - What type of epidemic curve does this look like?
        - What does the timing suggest about the incubation period?
        - Is the outbreak ongoing or resolving?
        
        **Synthesis:**
        - Based on person, place, and time, what hypotheses can you generate?
        - What additional information would help narrow down the cause?
        """)



def view_study_design():
    st.header("📊 Data & Study Design")

    truth = st.session_state.truth
    individuals = truth["individuals"]
    households = truth["households"]

    # Gate: don't let Day 2 artifacts be first interaction
    prereq_ok = bool(st.session_state.get("case_definition_written")) and bool(st.session_state.get("hypotheses_documented"))
    if not prereq_ok:
        st.info("Complete **Day 1** on the **Overview / Briefing** screen first (case definition + at least 1 hypothesis). Then return here for sampling and questionnaire upload.")

    # -------------------------
    # Step 1: Case definition (read-only here; authored on Overview)
    # -------------------------
    st.markdown("### Step 1: Case Definition (from Day 1)")
    cd_text = st.session_state.decisions.get("case_definition_text", "").strip()
    if cd_text:
        st.text_area("Working case definition:", value=cd_text, height=120, disabled=True)
    else:
        st.warning("No case definition saved yet. Go to **Overview / Briefing** (Day 1) and save one.")

    # -------------------------
    # Step 2: Study design
    # -------------------------
    st.markdown("### Step 2: Study Design")
    sd_type = st.radio("Choose a study design:", ["Case-control", "Retrospective cohort"], horizontal=True)

    if sd_type == "Case-control":
        st.session_state.decisions["study_design"] = {"type": "case_control"}
    else:
        st.session_state.decisions["study_design"] = {"type": "cohort"}

    # -------------------------
    # Step 2b: Sampling plan + manual selection (trainee-driven)
    # -------------------------
    st.markdown("### Step 2b: Sampling plan & participant selection")

    # Ensure clinic eligibility proxy exists (used for clinic controls)
    if "reported_to_hospital" not in individuals.columns:
        individuals = ensure_reported_to_hospital(individuals, random_seed=42)
        st.session_state.truth["individuals"] = individuals

    case_criteria = st.session_state.decisions.get("case_definition", {"clinical_AES": True})
    cases_pool = apply_case_definition(individuals, case_criteria).copy()
    cases_pool = cases_pool.sort_values(["village_id", "onset_date"], na_position="last")

    existing_cases = st.session_state.decisions.get("selected_cases", []) or []
    existing_controls = st.session_state.decisions.get("selected_controls", []) or []

    # Basic sampling targets
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_cases_target = st.number_input("Target # cases", min_value=1, max_value=200, value=int(st.session_state.decisions.get("sample_size", {}).get("cases", 20)), step=1)
    with c2:
        controls_per_case = st.number_input("Controls per case", min_value=1, max_value=5, value=int(st.session_state.decisions.get("study_design", {}).get("controls_per_case", 2) or 2), step=1)
    with c3:
        nonresponse_rate = st.slider("Expected nonresponse", min_value=0, max_value=25, value=int((st.session_state.decisions.get("sampling_plan", {}) or {}).get("nonresponse_rate", 0.0) * 100), step=1)
    with c4:
        allow_replacement = st.checkbox("Allow replacement if nonresponse", value=bool((st.session_state.decisions.get("sampling_plan", {}) or {}).get("allow_replacement", True)))

    st.session_state.decisions["sample_size"] = {"cases": int(n_cases_target), "controls_per_case": int(controls_per_case)}
    st.session_state.decisions.setdefault("study_design", {})
    st.session_state.decisions["study_design"]["controls_per_case"] = int(controls_per_case)

    # Show breakdown of case sources if case finding has been done
    if st.session_state.get('found_cases_added', False) and 'found_via_case_finding' in cases_pool.columns:
        found_count = cases_pool['found_via_case_finding'].sum() if 'found_via_case_finding' in cases_pool.columns else 0
        initial_count = len(cases_pool) - found_count
        st.caption(f"Eligible cases (based on your case definition proxy): **{len(cases_pool)}** ({initial_count} initial + {int(found_count)} from case finding)")
    else:
        st.caption(f"Eligible cases (based on your case definition proxy): **{len(cases_pool)}**")

    # ---- CASE SELECTION (manual)
    with st.form("case_select_form"):
        st.markdown("#### Select cases")
        v_filter = st.multiselect("Filter cases by village", sorted(cases_pool["village_id"].dropna().unique().tolist()), default=sorted(cases_pool["village_id"].dropna().unique().tolist()))
        df_cases = cases_pool[cases_pool["village_id"].isin(v_filter)].copy()

        show_cols = [c for c in ["person_id", "village_id", "hh_id", "age", "sex", "occupation", "onset_date", "severe_neuro", "outcome", "reported_to_hospital"] if c in df_cases.columns]
        df_cases = df_cases[show_cols].copy()
        df_cases.insert(0, "select", df_cases["person_id"].isin(existing_cases))

        edited = st.data_editor(
            df_cases,
            hide_index=True,
            use_container_width=True,
            column_config={"select": st.column_config.CheckboxColumn("Select")},
            disabled=[c for c in df_cases.columns if c != "select"],
        )

        submitted = st.form_submit_button("Save selected cases")
        if submitted:
            selected = edited[edited["select"] == True]["person_id"].astype(str).tolist()
            if len(selected) < 1:
                st.error("Select at least 1 case.")
            else:
                st.session_state.decisions["selected_cases"] = selected
                st.success(f"Saved **{len(selected)}** selected cases.")
                st.rerun()

    selected_cases = st.session_state.decisions.get("selected_cases", []) or []
    if selected_cases:
        st.info(f"Current case selection: **{len(selected_cases)}** cases selected. Target was {int(n_cases_target)}.")

    # ---- CONTROL SOURCE + ELIGIBILITY
    st.markdown("#### Controls: source & eligibility")
    control_source_label = st.selectbox(
        "Control source",
        ["Community controls", "Neighborhood controls (near cases)", "Clinic controls (healthcare-seeking)"],
        index=0,
    )
    control_source = "community"
    if "Neighborhood" in control_source_label:
        control_source = "neighborhood"
    elif "Clinic" in control_source_label:
        control_source = "clinic"

    eligible_villages_default = sorted(list(set(cases_pool[cases_pool["person_id"].isin(selected_cases)]["village_id"].dropna().astype(str).tolist()))) if selected_cases else sorted(cases_pool["village_id"].dropna().unique().tolist())
    eligible_villages = st.multiselect("Eligible villages for controls", options=sorted(individuals["village_id"].dropna().unique().tolist()), default=eligible_villages_default)

    include_symptomatic_noncase = st.checkbox("Allow symptomatic non-cases as controls (rare)", value=bool((st.session_state.decisions.get("sampling_plan", {}) or {}).get("include_symptomatic_noncase", False)))

    # Optional age eligibility for controls
    age_mode = st.radio("Control age rule", ["No restriction", "Specify range"], horizontal=True, index=0)
    control_age_range = None
    if age_mode == "Specify range":
        a1, a2 = st.columns(2)
        with a1:
            cmin = st.number_input("Control minimum age", min_value=0, max_value=100, value=0, step=1)
        with a2:
            cmax = st.number_input("Control maximum age", min_value=0, max_value=100, value=60, step=1)
        control_age_range = {"min": int(cmin), "max": int(cmax)}

    # Pool for control candidates
    # (We build a manageable candidate set to avoid huge data_editor tables.)
    def _build_control_pool():
        pool = individuals.copy()
        # non-cases only (by default)
        pool = pool[pool.get("symptomatic_AES", False).astype(bool) == False].copy()
        pool = pool[pool["village_id"].isin(eligible_villages)].copy()
        if control_age_range:
            pool = pool[(pool["age"] >= int(control_age_range["min"])) & (pool["age"] <= int(control_age_range["max"]))].copy()
        if control_source == "clinic":
            pool = pool[pool.get("reported_to_hospital", False).astype(bool) == True].copy()
        # neighborhood handled in je_logic with weights; here we just show same-village candidates
        return pool

    controls_pool = _build_control_pool()
    needed_controls = int(len(selected_cases) * int(controls_per_case)) if selected_cases else int(n_cases_target) * int(controls_per_case)
    st.caption(f"Eligible controls in pool: **{len(controls_pool)}** | Recommended controls to select: **{needed_controls}**")

    # Candidate sampling for UI
    if "controls_candidate_ids" not in st.session_state:
        st.session_state.controls_candidate_ids = []
    if "controls_candidate_seed" not in st.session_state:
        st.session_state.controls_candidate_seed = 0

    cbtn1, cbtn2 = st.columns([1, 3])
    with cbtn1:
        if st.button("🔄 Refresh control candidates"):
            st.session_state.controls_candidate_seed += 1
            st.session_state.controls_candidate_ids = []
            st.rerun()

    # Build candidate list (sample to keep UI snappy)
    rng = np.random.default_rng(100 + int(st.session_state.controls_candidate_seed))
    if not st.session_state.controls_candidate_ids:
        cand_n = min(350, len(controls_pool))
        if cand_n > 0:
            cand_ids = controls_pool.sample(n=cand_n, random_state=100 + int(st.session_state.controls_candidate_seed))["person_id"].astype(str).tolist()
            st.session_state.controls_candidate_ids = cand_ids

    cand_controls = controls_pool[controls_pool["person_id"].astype(str).isin(st.session_state.controls_candidate_ids)].copy()
    show_cols_c = [c for c in ["person_id", "village_id", "hh_id", "age", "sex", "occupation", "reported_to_hospital"] if c in cand_controls.columns]
    cand_controls = cand_controls[show_cols_c].copy()
    cand_controls.insert(0, "select", cand_controls["person_id"].astype(str).isin(existing_controls))

    with st.form("controls_select_form"):
        st.markdown("#### Select controls (from a candidate list)")
        edited_c = st.data_editor(
            cand_controls,
            hide_index=True,
            use_container_width=True,
            column_config={"select": st.column_config.CheckboxColumn("Select")},
            disabled=[c for c in cand_controls.columns if c != "select"],
        )
        sub_c = st.form_submit_button("Save selected controls")
        if sub_c:
            selected_c = edited_c[edited_c["select"] == True]["person_id"].astype(str).tolist()
            if len(selected_c) < 1:
                st.error("Select at least 1 control.")
            else:
                st.session_state.decisions["selected_controls"] = selected_c
                st.success(f"Saved **{len(selected_c)}** selected controls.")
                st.rerun()

    selected_controls = st.session_state.decisions.get("selected_controls", []) or []
    if selected_controls:
        st.info(f"Current control selection: **{len(selected_controls)}** controls selected. Recommended: {needed_controls}.")

    # Persist sampling plan (used by dataset generator)
    st.session_state.decisions["sampling_plan"] = {
        "control_source": control_source,
        "eligible_villages": eligible_villages,
        "include_symptomatic_noncase": bool(include_symptomatic_noncase),
        "control_age_range": control_age_range,
        "nonresponse_rate": float(nonresponse_rate) / 100.0,
        "allow_replacement": bool(allow_replacement),
        "controls_per_case": int(controls_per_case),
        "n_cases": int(n_cases_target),
    }

    # -------------------------
    # Step 3: Questionnaire (XLSForm upload) — gated
    # -------------------------
    st.markdown("### Step 3: Questionnaire (XLSForm upload)")

    if not prereq_ok:
        st.warning("Questionnaire upload is locked until you have a saved case definition and at least 1 hypothesis (Day 1).")
    else:
        st.caption("Build your questionnaire in Kobo (or any XLSForm editor), export as XLSForm (.xlsx), then upload it here.")
        uploaded = st.file_uploader("Upload XLSForm (.xlsx)", type=["xlsx"], key="xlsform_upload")

        if uploaded is not None:
            xls_bytes = uploaded.read()
            st.session_state.decisions["questionnaire_xlsform_bytes"] = xls_bytes

            try:
                questionnaire = parse_xlsform(xls_bytes)
                st.session_state.decisions["questionnaire_xlsform_preview"] = questionnaire

                preview_rows = []
                for q in questionnaire.get("questions", []):
                    preview_rows.append({
                        "name": q.get("name"),
                        "type": q.get("type"),
                        "label": q.get("label"),
                        "list_name": q.get("list_name"),
                        "n_choices": len(q.get("choices", []) or []),
                    })
                if preview_rows:
                    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No survey questions detected (notes/groups/calculations are ignored).")
            except Exception as e:
                st.error(f"Could not parse this XLSForm. Make sure you uploaded the *form definition* (XLSForm), not a data export. Details: {e}")
                questionnaire = None

            if questionnaire:
                api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    st.warning("ANTHROPIC_API_KEY not found in Streamlit secrets. LLM mapping cannot run until it is configured.")

                if st.button("Run LLM mapping & save questionnaire", key="save_xlsform_questionnaire"):
                    if not api_key:
                        st.error("Missing ANTHROPIC_API_KEY in Streamlit secrets.")
                    else:
                        try:
                            questionnaire = llm_map_xlsform_questions(questionnaire, api_key=api_key)
                            questionnaire = llm_build_select_one_choice_maps(questionnaire, api_key=api_key)
                            questionnaire = llm_build_unmapped_answer_generators(questionnaire, api_key=api_key)
                            questionnaire = prepare_question_render_plan(questionnaire)

                            st.session_state.decisions["questionnaire_xlsform"] = questionnaire
                            st.session_state.questionnaire_submitted = True
                            st.success("Questionnaire uploaded, mapped, and saved.")
                        except Exception as e:
                            st.error(f"Failed to map/save questionnaire: {e}")


    # Facilitator mapping review (optional)
    saved_q = st.session_state.decisions.get("questionnaire_xlsform")
    if isinstance(saved_q, dict) and saved_q.get("questions"):
        if st.session_state.get("facilitator_mode", False):
            with st.expander("Facilitator mapping review (optional)", expanded=False):
                rows = []
                for q in saved_q.get("questions", []):
                    r = q.get("render", {}) or {}
                    rows.append({
                        "question_name": q.get("name"),
                        "type": q.get("type"),
                        "label": q.get("label"),
                        "mapped_var": r.get("mapped_var"),
                        "confidence": r.get("confidence"),
                        "domain": r.get("domain"),
                        "rationale": r.get("rationale"),
                        "unmapped": r.get("mapped_var") in (None, "", "unmapped"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Step 4: Generate dataset (requires questionnaire + selections)
    # -------------------------
    st.markdown("### Step 4: Generate simulated study dataset")

    can_generate = bool(st.session_state.decisions.get("questionnaire_xlsform")) and bool(st.session_state.decisions.get("selected_cases")) and bool(st.session_state.decisions.get("selected_controls"))
    if not can_generate:
        st.info("To generate the dataset, you need: (1) saved XLSForm questionnaire, (2) selected cases, and (3) selected controls.")
        return

    if st.button("Generate Dataset", type="primary"):
        try:
            decisions = dict(st.session_state.decisions)
            decisions["return_sampling_report"] = True
            df, report = generate_study_dataset(individuals, households, decisions)

            st.session_state.generated_dataset = df
            st.session_state.sampling_report = report
            st.session_state.descriptive_analysis_done = True  # proxy
            st.success("Dataset generated. Preview below; export for analysis as needed.")

            with st.expander("Sampling frame summary", expanded=True):
                st.json({
                    "case_pool_n": report.get("case_pool_n"),
                    "control_pool_n": report.get("control_pool_n"),
                    "cases_selected_n": report.get("cases_selected_n"),
                    "controls_selected_n": report.get("controls_selected_n"),
                    "cases_after_nonresponse_n": report.get("cases_after_nonresponse_n"),
                    "controls_after_nonresponse_n": report.get("controls_after_nonresponse_n"),
                    "nonresponse_rate": st.session_state.decisions.get("sampling_plan", {}).get("nonresponse_rate"),
                    "allow_replacement": st.session_state.decisions.get("sampling_plan", {}).get("allow_replacement"),
                    "control_source": st.session_state.decisions.get("sampling_plan", {}).get("control_source"),
                })

            st.dataframe(df.head(30), use_container_width=True)

        except Exception as e:
            st.error(f"Dataset generation failed: {e}")



def view_lab_and_environment():
    st.header("🧪 " + t("lab", default="Lab & Environment"))
    if int(st.session_state.get("current_day", 1)) < 4:
        st.info(t("locked_until_day", day=4))
        return
    refresh_lab_queue_for_day(int(st.session_state.get("current_day", 1)))
    
    # Resource display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Budget", f"${st.session_state.budget}")
    with col2:
        st.metric("⏱️ Time Remaining", f"{st.session_state.time_remaining}h")
    with col3:
        st.metric("🧪 Lab Credits", st.session_state.lab_credits)

    st.markdown("""
    Collect and submit samples for laboratory testing. Each sample type has different 
    time and budget costs for collection.
    """)
    
    # Sample costs table
    with st.expander("📋 Sample Collection Costs"):
        cost_data = {
            "Sample Type": ["Human CSF", "Human Serum", "Pig Serum", "Mosquito Pool"],
            "Time (hours)": [1.0, 0.5, 1.0, 1.5],
            "Budget ($)": [25, 25, 35, 40],
            "Lab Credits": [3, 2, 2, 3]
        }
        st.dataframe(pd.DataFrame(cost_data), hide_index=True)

    truth = st.session_state.truth
    villages = truth["villages"]

    st.markdown("### Submit New Sample")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        sample_type = st.selectbox(
            "Sample type",
            ["human_CSF", "human_serum", "pig_serum", "mosquito_pool"],
        )
    with col2:
        village_id = st.selectbox(
            "Village",
            villages["village_id"],
            format_func=lambda vid: villages.set_index("village_id").loc[vid, "village_name"],
        )
    with col3:
        test = st.selectbox(
            t("lab_test", default="Test"),
            list(LAB_TEST_CATALOG.keys()),
            format_func=lab_test_label,
        )

    source_description = st.text_input("Source description (e.g., 'Case from Nalu')", "")
    
    # Calculate costs based on sample type
    sample_costs = {
        "human_CSF": {"time": 1.0, "budget": 25, "credits": 3},
        "human_serum": {"time": 0.5, "budget": 25, "credits": 2},
        "pig_serum": {"time": 1.0, "budget": 35, "credits": 2},
        "mosquito_pool": {"time": 1.5, "budget": 40, "credits": 3},
    }
    
    costs = sample_costs.get(sample_type, {"time": 1.0, "budget": 25, "credits": 2})
    
    st.caption(f"This sample will cost: ⏱️ {costs['time']}h | 💰 ${costs['budget']} | 🧪 {costs['credits']} credits")

    if st.button("Submit lab order"):
        # Check resources
        can_proceed, msg = check_resources(costs['time'], costs['budget'])
        if not can_proceed:
            st.error(msg)
        elif st.session_state.lab_credits < costs['credits']:
            st.error(f"Not enough lab credits (need {costs['credits']}, have {st.session_state.lab_credits})")
        else:
            # Deduct resources
            spend_time(costs['time'], f"Sample collection: {sample_type}")
            spend_budget(costs['budget'], f"Sample collection: {sample_type}")
            st.session_state.lab_credits -= costs['credits']
            
            order = {
                "sample_type": sample_type,
                "village_id": village_id,
                "test": test,
                "source_description": source_description or "Unspecified source",
            }
            result = process_lab_order(order, truth["lab_samples"])
            st.session_state.lab_results.append(result)
            st.session_state.lab_samples_submitted.append(order)
            
            st.success(
                f"Lab order submitted. Result: {result['result']} "
                f"(turnaround {result['days_to_result']} days)."
            )
            st.rerun()


    if st.session_state.get('lab_results'):
        st.markdown(f"### {t('lab_results', default='Lab results')}")
        df = pd.DataFrame(st.session_state.lab_results).copy()
    
        villages_lookup = truth["villages"].set_index("village_id")["village_name"].to_dict()
        if "village_id" in df.columns:
            df["village"] = df["village_id"].map(villages_lookup).fillna(df["village_id"])
    
        if "test_display" not in df.columns:
            df["test_display"] = df.get("test", "").map(lab_test_label) if "test" in df.columns else ""
    
        day_now = int(st.session_state.get("current_day", 1))
        if "ready_day" in df.columns:
            df["days_remaining"] = df.apply(
                lambda r: max(0, int(r.get("ready_day", day_now)) - day_now)
                if str(r.get("result", "")).upper() == "PENDING"
                else 0,
                axis=1,
            )
    
        show_cols = [
            "sample_id", "sample_type", "village", "test_display",
            "source_description", "placed_day", "ready_day",
            "days_remaining", "result"
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)


def get_village_photos(village_name):
    """
    Get list of photos for a village from the assets directory.

    Returns a dict mapping photo base names to their full paths, or None if no photos exist.
    """
    assets_dir = Path("assets")
    village_dir = assets_dir / village_name.capitalize()

    if not village_dir.exists():
        return None

    # Get all image files
    photo_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        photo_files.extend(village_dir.glob(ext))

    if not photo_files:
        return None

    # Create a dict mapping base names to paths
    photos = {}
    for photo_path in sorted(photo_files):
        photos[photo_path.stem] = photo_path

    return photos


def view_village_profiles():
    """Display village briefing documents with stats and images."""
    st.header("🏘️ Village Profiles - Sidero Valley")
    
    st.markdown("""
    These background documents provide official information about each village in the investigation area.
    Review these to understand the local context before conducting interviews.
    """)
    
    lang = st.session_state.get("language", "en")
    
    tabs = st.tabs(["Nalu Village", "Kabwe Village", "Tamu Village"])
    
    # SVG illustrations for each village
    nalu_rice_svg = '''
    <svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
        <!-- Sky -->
        <rect width="400" height="200" fill="#87CEEB"/>
        <!-- Sun -->
        <circle cx="350" cy="40" r="25" fill="#FFD700"/>
        <!-- Mountains in background -->
        <polygon points="0,120 80,60 160,120" fill="#6B8E23"/>
        <polygon points="100,120 200,40 300,120" fill="#556B2F"/>
        <polygon points="250,120 350,70 400,120" fill="#6B8E23"/>
        <!-- Rice paddies (flooded fields) -->
        <rect x="0" y="120" width="400" height="80" fill="#4A7C59"/>
        <!-- Water reflection lines -->
        <rect x="10" y="130" width="80" height="3" fill="#87CEEB" opacity="0.5"/>
        <rect x="100" y="145" width="90" height="3" fill="#87CEEB" opacity="0.5"/>
        <rect x="200" y="135" width="70" height="3" fill="#87CEEB" opacity="0.5"/>
        <rect x="280" y="150" width="100" height="3" fill="#87CEEB" opacity="0.5"/>
        <rect x="50" y="160" width="60" height="3" fill="#87CEEB" opacity="0.5"/>
        <rect x="150" y="170" width="80" height="3" fill="#87CEEB" opacity="0.5"/>
        <!-- Rice plants (small green lines) -->
        <g stroke="#228B22" stroke-width="2">
            <line x1="30" y1="140" x2="30" y2="125"/>
            <line x1="50" y1="145" x2="50" y2="130"/>
            <line x1="70" y1="140" x2="70" y2="125"/>
            <line x1="120" y1="150" x2="120" y2="135"/>
            <line x1="140" y1="145" x2="140" y2="130"/>
            <line x1="160" y1="155" x2="160" y2="140"/>
            <line x1="220" y1="145" x2="220" y2="130"/>
            <line x1="250" y1="150" x2="250" y2="135"/>
            <line x1="300" y1="140" x2="300" y2="125"/>
            <line x1="340" y1="155" x2="340" y2="140"/>
            <line x1="370" y1="145" x2="370" y2="130"/>
        </g>
        <!-- Birds flying -->
        <text x="180" y="50" font-size="10">🐦</text>
        <text x="280" y="45" font-size="8">🐦</text>
        <!-- Label -->
        <text x="200" y="190" text-anchor="middle" font-size="14" fill="white" font-weight="bold">Rice Paddies - Irrigated Fields</text>
    </svg>
    '''
    
    nalu_pigs_svg = '''
    <svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
        <!-- Sky -->
        <rect width="400" height="200" fill="#87CEEB"/>
        <!-- Ground -->
        <rect x="0" y="140" width="400" height="60" fill="#8B4513"/>
        <!-- Mud patches -->
        <ellipse cx="100" cy="160" rx="40" ry="15" fill="#654321"/>
        <ellipse cx="280" cy="165" rx="50" ry="18" fill="#654321"/>
        <!-- Fence -->
        <rect x="20" y="100" width="360" height="5" fill="#8B4513"/>
        <rect x="30" y="60" width="8" height="80" fill="#8B4513"/>
        <rect x="100" y="60" width="8" height="80" fill="#8B4513"/>
        <rect x="170" y="60" width="8" height="80" fill="#8B4513"/>
        <rect x="240" y="60" width="8" height="80" fill="#8B4513"/>
        <rect x="310" y="60" width="8" height="80" fill="#8B4513"/>
        <rect x="20" y="80" width="360" height="4" fill="#8B4513"/>
        <!-- Shelter roof -->
        <polygon points="280,40 350,40 380,70 250,70" fill="#A0522D"/>
        <rect x="260" y="70" width="110" height="50" fill="#DEB887"/>
        <!-- Pigs -->
        <ellipse cx="80" cy="130" rx="25" ry="18" fill="#FFB6C1"/>
        <circle cx="60" cy="125" r="8" fill="#FFB6C1"/>
        <ellipse cx="150" cy="135" rx="22" ry="15" fill="#FFC0CB"/>
        <circle cx="132" cy="130" r="7" fill="#FFC0CB"/>
        <ellipse cx="200" cy="128" rx="20" ry="14" fill="#FFB6C1"/>
        <circle cx="184" cy="123" r="6" fill="#FFB6C1"/>
        <!-- More pigs in background -->
        <ellipse cx="290" cy="115" rx="18" ry="12" fill="#FFA0AB"/>
        <ellipse cx="330" cy="118" rx="16" ry="11" fill="#FFA0AB"/>
        <!-- Flies -->
        <text x="120" y="115" font-size="8">•</text>
        <text x="180" y="110" font-size="8">•</text>
        <text x="250" y="105" font-size="8">•</text>
        <!-- Label -->
        <text x="200" y="190" text-anchor="middle" font-size="14" fill="white" font-weight="bold">Pig Cooperative - ~200 Pigs Near Village</text>
    </svg>
    '''
    
    kabwe_mixed_svg = '''
    <svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
        <!-- Sky -->
        <rect width="400" height="200" fill="#87CEEB"/>
        <!-- Hills -->
        <ellipse cx="100" cy="140" rx="120" ry="50" fill="#6B8E23"/>
        <ellipse cx="300" cy="150" rx="140" ry="45" fill="#556B2F"/>
        <!-- Ground -->
        <rect x="0" y="140" width="400" height="60" fill="#8B7355"/>
        <!-- Rice paddy section (left) -->
        <rect x="0" y="140" width="150" height="40" fill="#4A7C59"/>
        <rect x="10" y="150" width="40" height="2" fill="#87CEEB" opacity="0.5"/>
        <rect x="60" y="155" width="50" height="2" fill="#87CEEB" opacity="0.5"/>
        <!-- Maize/upland section (right) -->
        <g stroke="#DAA520" stroke-width="2">
            <line x1="200" y1="140" x2="200" y2="110"/>
            <line x1="220" y1="140" x2="220" y2="115"/>
            <line x1="240" y1="140" x2="240" y2="105"/>
            <line x1="260" y1="140" x2="260" y2="112"/>
            <line x1="280" y1="140" x2="280" y2="108"/>
            <line x1="300" y1="140" x2="300" y2="115"/>
            <line x1="320" y1="140" x2="320" y2="110"/>
            <line x1="340" y1="140" x2="340" y2="118"/>
            <line x1="360" y1="140" x2="360" y2="105"/>
        </g>
        <!-- Corn tops -->
        <g fill="#FFD700">
            <circle cx="200" cy="105" r="4"/>
            <circle cx="240" cy="100" r="4"/>
            <circle cx="280" cy="103" r="4"/>
            <circle cx="320" cy="105" r="4"/>
            <circle cx="360" cy="100" r="4"/>
        </g>
        <!-- Path dividing -->
        <rect x="155" y="140" width="30" height="60" fill="#C4A76C"/>
        <!-- Small pig pen -->
        <rect x="380" y="150" width="15" height="15" fill="#8B4513"/>
        <ellipse cx="387" cy="160" rx="5" ry="4" fill="#FFB6C1"/>
        <!-- Label -->
        <text x="200" y="190" text-anchor="middle" font-size="14" fill="white" font-weight="bold">Mixed Farming - Rice Paddies & Upland Crops</text>
    </svg>
    '''
    
    kabwe_path_svg = '''
    <svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
        <!-- Sky -->
        <rect width="400" height="200" fill="#87CEEB"/>
        <!-- Rice paddies (green with water) -->
        <rect x="0" y="100" width="400" height="100" fill="#4A7C59"/>
        <!-- Water reflections -->
        <rect x="20" y="120" width="60" height="3" fill="#87CEEB" opacity="0.4"/>
        <rect x="100" y="140" width="80" height="3" fill="#87CEEB" opacity="0.4"/>
        <rect x="250" y="130" width="70" height="3" fill="#87CEEB" opacity="0.4"/>
        <rect x="50" y="160" width="90" height="3" fill="#87CEEB" opacity="0.4"/>
        <rect x="300" y="155" width="80" height="3" fill="#87CEEB" opacity="0.4"/>
        <!-- Path through paddies -->
        <path d="M 0,180 Q 100,150 200,160 Q 300,170 400,140" stroke="#C4A76C" stroke-width="20" fill="none"/>
        <!-- Children walking -->
        <text x="120" y="155" font-size="16">👧</text>
        <text x="150" y="160" font-size="14">👦</text>
        <text x="180" y="158" font-size="15">👧</text>
        <!-- School building in distance -->
        <rect x="350" y="80" width="40" height="40" fill="#CD853F"/>
        <polygon points="350,80 370,60 390,80" fill="#8B0000"/>
        <rect x="365" y="95" width="10" height="25" fill="#8B4513"/>
        <!-- Sign -->
        <text x="370" y="75" font-size="8" text-anchor="middle">SCHOOL</text>
        <!-- Village houses in background -->
        <rect x="20" y="70" width="25" height="25" fill="#DEB887"/>
        <polygon points="20,70 32,55 45,70" fill="#8B4513"/>
        <rect x="60" y="75" width="20" height="20" fill="#DEB887"/>
        <polygon points="60,75 70,62 80,75" fill="#8B4513"/>
        <!-- Label -->
        <text x="200" y="195" text-anchor="middle" font-size="12" fill="white" font-weight="bold">Children Walk Through Paddies to School in Nalu</text>
    </svg>
    '''
    
    tamu_upland_svg = '''
    <svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
        <!-- Sky -->
        <rect width="400" height="200" fill="#87CEEB"/>
        <!-- Sun -->
        <circle cx="350" cy="35" r="25" fill="#FFD700"/>
        <!-- Hills/mountains -->
        <polygon points="0,130 100,50 200,130" fill="#228B22"/>
        <polygon points="150,130 280,30 400,130" fill="#2E8B57"/>
        <!-- Dry ground -->
        <rect x="0" y="130" width="400" height="70" fill="#C4A76C"/>
        <!-- Cassava/yam plants -->
        <g fill="#228B22">
            <ellipse cx="50" cy="140" rx="20" ry="15"/>
            <ellipse cx="120" cy="145" rx="25" ry="18"/>
            <ellipse cx="200" cy="138" rx="22" ry="16"/>
            <ellipse cx="280" cy="142" rx="20" ry="14"/>
            <ellipse cx="350" cy="140" rx="25" ry="17"/>
        </g>
        <!-- Goats instead of pigs -->
        <text x="100" y="165" font-size="16">🐐</text>
        <text x="250" y="160" font-size="14">🐐</text>
        <!-- Chickens -->
        <text x="180" y="170" font-size="12">🐔</text>
        <text x="320" y="168" font-size="11">🐔</text>
        <!-- No mosquitoes - dry terrain -->
        <!-- Trees -->
        <rect x="30" y="100" width="8" height="30" fill="#8B4513"/>
        <circle cx="34" cy="90" r="20" fill="#228B22"/>
        <rect x="370" y="95" width="8" height="35" fill="#8B4513"/>
        <circle cx="374" cy="85" r="22" fill="#2E8B57"/>
        <!-- Label -->
        <text x="200" y="190" text-anchor="middle" font-size="14" fill="#333" font-weight="bold">Upland Terrain - Cassava & Yam Farming</text>
    </svg>
    '''
    
    tamu_forest_svg = '''
    <svg viewBox="0 0 400 200" xmlns="http://www.w3.org/2000/svg">
        <!-- Sky -->
        <rect width="400" height="200" fill="#87CEEB"/>
        <!-- Forest background -->
        <rect x="0" y="80" width="400" height="120" fill="#228B22"/>
        <!-- Multiple trees -->
        <g>
            <!-- Tree 1 -->
            <rect x="30" y="60" width="12" height="80" fill="#8B4513"/>
            <circle cx="36" cy="45" r="30" fill="#2E8B57"/>
            <!-- Tree 2 -->
            <rect x="90" y="50" width="15" height="90" fill="#8B4513"/>
            <circle cx="97" cy="35" r="35" fill="#228B22"/>
            <!-- Tree 3 -->
            <rect x="160" y="70" width="10" height="70" fill="#8B4513"/>
            <circle cx="165" cy="55" r="28" fill="#2E8B57"/>
            <!-- Tree 4 -->
            <rect x="220" y="55" width="14" height="85" fill="#8B4513"/>
            <circle cx="227" cy="40" r="32" fill="#228B22"/>
            <!-- Tree 5 -->
            <rect x="290" y="65" width="11" height="75" fill="#8B4513"/>
            <circle cx="295" cy="50" r="30" fill="#2E8B57"/>
            <!-- Tree 6 -->
            <rect x="350" y="45" width="16" height="95" fill="#8B4513"/>
            <circle cx="358" cy="30" r="38" fill="#228B22"/>
        </g>
        <!-- Ground/path -->
        <rect x="0" y="160" width="400" height="40" fill="#C4A76C"/>
        <!-- Well (spring water) -->
        <ellipse cx="200" cy="175" rx="25" ry="10" fill="#4169E1"/>
        <ellipse cx="200" cy="170" rx="28" ry="8" fill="#696969" fill-opacity="0.5"/>
        <!-- Village houses -->
        <rect x="100" y="145" width="20" height="20" fill="#DEB887"/>
        <polygon points="100,145 110,132 120,145" fill="#8B4513"/>
        <rect x="280" y="148" width="18" height="18" fill="#DEB887"/>
        <polygon points="280,148 289,136 298,148" fill="#8B4513"/>
        <!-- Label -->
        <text x="200" y="195" text-anchor="middle" font-size="13" fill="#333" font-weight="bold">Forested Area - Spring-Fed Water, Less Standing Water</text>
    </svg>
    '''
    
    for i, (village_key, village) in enumerate(VILLAGE_PROFILES.items()):
        with tabs[i]:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Get description in current language, fallback to English
                desc = village["description"].get(lang, village["description"]["en"])
                st.markdown(desc)
            
            with col2:
                # Check for photos first, fall back to SVG illustrations
                village_photos = get_village_photos(village_key)

                if village_photos:
                    st.markdown("### 📸 Village Photos")
                else:
                    st.markdown("### 📸 Scene Illustrations")

                if village_key == "nalu":
                    # Use real photos if available, otherwise use SVG illustrations
                    if village_photos:
                        # Display village scene
                        if "nalu_01_village_scene" in village_photos:
                            st.markdown("**Village Scene**")
                            st.image(str(village_photos["nalu_01_village_scene"]), use_container_width=True)
                            st.caption("Nalu village center")
                            st.markdown("---")

                        # Display rice paddies
                        if "nalu_02_rice_paddies" in village_photos:
                            st.markdown("**Rice Paddies Near Village**")
                            st.image(str(village_photos["nalu_02_rice_paddies"]), use_container_width=True)
                            st.caption("Irrigated rice fields with standing water year-round")
                            st.markdown("---")

                        # Display pig pens
                        if "nalu_03_pig_pens" in village_photos:
                            st.markdown("**Pig Cooperative**")
                            st.image(str(village_photos["nalu_03_pig_pens"]), use_container_width=True)
                            st.caption("~200 pigs housed 500m from village center")
                            st.markdown("---")

                        # Display health center
                        if "nalu_04_health_center_exterior" in village_photos:
                            st.markdown("**Health Center**")
                            st.image(str(village_photos["nalu_04_health_center_exterior"]), use_container_width=True)
                            st.caption("Nalu Health Center - main facility for the area")
                            st.markdown("---")

                        # Display market day
                        if "nalu_05_market_day" in village_photos:
                            st.markdown("**Market Day**")
                            st.image(str(village_photos["nalu_05_market_day"]), use_container_width=True)
                            st.caption("Weekly market brings people together from surrounding villages")
                    else:
                        # Fallback to SVG illustrations
                        st.markdown("**Rice Paddies Near Village**")
                        st.markdown(nalu_rice_svg, unsafe_allow_html=True)
                        st.caption("Irrigated rice fields with standing water year-round")

                        st.markdown("---")
                        st.markdown("**Pig Cooperative**")
                        st.markdown(nalu_pigs_svg, unsafe_allow_html=True)
                        st.caption("~200 pigs housed 500m from village center")

                elif village_key == "kabwe":
                    # Use real photos if available, otherwise use SVG illustrations
                    if village_photos:
                        # Display photos for Kabwe when added
                        # For now, just display any photos found
                        for photo_key, photo_path in village_photos.items():
                            # Create a nice title from the filename
                            title = photo_key.replace("kabwe_", "").replace("_", " ").title()
                            st.markdown(f"**{title}**")
                            st.image(str(photo_path), use_container_width=True)
                            st.markdown("---")
                    else:
                        # Fallback to SVG illustrations
                        st.markdown("**Mixed Farming Area**")
                        st.markdown(kabwe_mixed_svg, unsafe_allow_html=True)
                        st.caption("Combination of rice paddies and upland maize")

                        st.markdown("---")
                        st.markdown("**Path to Nalu School**")
                        st.markdown(kabwe_path_svg, unsafe_allow_html=True)
                        st.caption("Children walk through paddy fields daily")

                elif village_key == "tamu":
                    # Use real photos if available, otherwise use SVG illustrations
                    if village_photos:
                        # Display photos for Tamu when added
                        # For now, just display any photos found
                        for photo_key, photo_path in village_photos.items():
                            # Create a nice title from the filename
                            title = photo_key.replace("tamu_", "").replace("_", " ").title()
                            st.markdown(f"**{title}**")
                            st.image(str(photo_path), use_container_width=True)
                            st.markdown("---")
                    else:
                        # Fallback to SVG illustrations
                        st.markdown("**Upland Terrain**")
                        st.markdown(tamu_upland_svg, unsafe_allow_html=True)
                        st.caption("Higher elevation with cassava and yam farming")

                        st.markdown("---")
                        st.markdown("**Forested Areas**")
                        st.markdown(tamu_forest_svg, unsafe_allow_html=True)
                        st.caption("Spring-fed wells, less standing water")
            
            # Quick stats summary
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Population", f"{village['population']:,}")
            with col2:
                st.metric("Households", f"{village['households']:,}")
            with col3:
                # Main livelihood
                if village_key == "nalu":
                    st.metric("Main Livelihood", "Rice farming")
                elif village_key == "kabwe":
                    st.metric("Main Livelihood", "Mixed farming")
                else:
                    st.metric("Main Livelihood", "Upland crops")
            with col4:
                # Health facility
                if village_key == "nalu":
                    st.metric("Health Facility", "Health Center")
                elif village_key == "kabwe":
                    st.metric("Health Facility", "None")
                else:
                    st.metric("Health Facility", "CHV only")


def view_spot_map():
    """Geographic spot map of cases using a custom fictional map."""
    st.header("📍 Spot Map - Geographic Distribution of Cases")
    
    truth = st.session_state.truth
    individuals = truth["individuals"]
    households = truth["households"]
    villages = truth["villages"]
    
    # Get symptomatic cases
    cases = individuals[individuals["symptomatic_AES"] == True].copy()
    
    # Also include found cases from clinic records if any
    found_cases_count = 0
    if st.session_state.get('found_cases_added') and st.session_state.case_finding_score:
        found_cases_count = st.session_state.case_finding_score.get('true_positives', 0)
    
    if len(cases) == 0:
        st.warning("No cases to display on map.")
        return
    
    # Merge with household info
    hh_with_village = households.merge(
        villages[["village_id", "village_name"]], 
        on="village_id", 
        how="left"
    )
    
    cases = cases.merge(
        hh_with_village[["hh_id", "village_id", "village_name"]], 
        on="hh_id", 
        how="left",
        suffixes=('', '_hh')
    )
    
    if 'village_id_hh' in cases.columns:
        cases['village_id'] = cases['village_id_hh']
        cases = cases.drop(columns=['village_id_hh'])
    
    # Count cases by village
    village_counts = cases['village_name'].value_counts().to_dict()
    nalu_cases = village_counts.get('Nalu Village', 0)
    kabwe_cases = village_counts.get('Kabwe Village', 0)
    tamu_cases = village_counts.get('Tamu Village', 0)
    
    # Generate case dots for SVG
    np.random.seed(42)
    
    def generate_case_dots(n_cases, cx, cy, radius=25):
        """Generate SVG circles for cases clustered around a point."""
        dots = []
        for i in range(n_cases):
            # Random position within radius
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(5, radius)
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            # Determine severity color
            is_severe = np.random.random() < 0.3
            color = '#e74c3c' if is_severe else '#f39c12'
            dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="white" stroke-width="1"/>')
        return '\n'.join(dots)
    
    # Generate dots for each village
    nalu_dots = generate_case_dots(nalu_cases, 200, 280, 30)
    kabwe_dots = generate_case_dots(kabwe_cases, 340, 200, 25)
    tamu_dots = generate_case_dots(tamu_cases, 120, 120, 20)
    
    # Custom SVG map of Sidero Valley
    map_svg = f'''
    <svg viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg" style="background: #f0f8ff;">
        <!-- Title -->
        <text x="250" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Sidero Valley - Case Distribution Map</text>
        
        <!-- River -->
        <path d="M 50,350 Q 150,300 200,280 Q 250,260 300,250 Q 350,240 400,200 Q 450,160 480,100" 
              stroke="#4a90d9" stroke-width="12" fill="none" opacity="0.7"/>
        <path d="M 50,350 Q 150,300 200,280 Q 250,260 300,250 Q 350,240 400,200 Q 450,160 480,100" 
              stroke="#6ab0ff" stroke-width="6" fill="none"/>
        <text x="420" y="130" font-size="10" fill="#4a90d9" font-style="italic">Sidero River</text>
        
        <!-- Rice Paddies (near Nalu and Kabwe) -->
        <rect x="140" y="240" width="80" height="60" fill="#7cb342" opacity="0.5" rx="5"/>
        <rect x="230" y="220" width="60" height="50" fill="#7cb342" opacity="0.5" rx="5"/>
        <rect x="280" y="180" width="50" height="40" fill="#7cb342" opacity="0.4" rx="5"/>
        <text x="180" y="235" font-size="8" fill="#33691e">Rice Paddies</text>
        
        <!-- Pig Farm marker near Nalu -->
        <rect x="160" y="310" width="30" height="20" fill="#8d6e63" opacity="0.7" rx="3"/>
        <text x="175" y="323" font-size="7" fill="white" text-anchor="middle">🐷</text>
        <text x="175" y="340" font-size="7" fill="#5d4037" text-anchor="middle">Pig Coop</text>
        
        <!-- Upland/Forest area (near Tamu) -->
        <ellipse cx="100" cy="100" rx="60" ry="50" fill="#2e7d32" opacity="0.3"/>
        <text x="100" y="70" font-size="8" fill="#1b5e20" text-anchor="middle">Forested Uplands</text>
        
        <!-- VILLAGES -->
        
        <!-- Nalu Village (largest, near river and paddies) -->
        <circle cx="200" cy="280" r="35" fill="#ffcc80" stroke="#e65100" stroke-width="2"/>
        <text x="200" y="275" text-anchor="middle" font-size="11" font-weight="bold" fill="#e65100">Nalu</text>
        <text x="200" y="288" text-anchor="middle" font-size="8" fill="#bf360c">Pop: 480</text>
        
        <!-- Kabwe Village (medium, between Nalu and uplands) -->
        <circle cx="340" cy="200" r="28" fill="#ffe0b2" stroke="#ff6f00" stroke-width="2"/>
        <text x="340" y="196" text-anchor="middle" font-size="10" font-weight="bold" fill="#ff6f00">Kabwe</text>
        <text x="340" y="208" text-anchor="middle" font-size="7" fill="#e65100">Pop: 510</text>
        
        <!-- Tamu Village (smallest, in uplands away from paddies) -->
        <circle cx="120" cy="120" r="22" fill="#fff3e0" stroke="#ff9800" stroke-width="2"/>
        <text x="120" y="117" text-anchor="middle" font-size="9" font-weight="bold" fill="#ff9800">Tamu</text>
        <text x="120" y="128" text-anchor="middle" font-size="7" fill="#e65100">Pop: 390</text>
        
        <!-- Path from Kabwe to Nalu (through paddies) -->
        <path d="M 315,210 Q 280,240 230,265" stroke="#a1887f" stroke-width="3" fill="none" stroke-dasharray="5,3"/>
        <text x="270" y="250" font-size="7" fill="#6d4c41" transform="rotate(-20 270 250)">path to school</text>
        
        <!-- District Hospital -->
        <rect x="420" y="300" width="40" height="30" fill="#e3f2fd" stroke="#1976d2" stroke-width="2" rx="3"/>
        <text x="440" y="315" text-anchor="middle" font-size="8" fill="#1976d2">🏥</text>
        <text x="440" y="325" text-anchor="middle" font-size="6" fill="#1565c0">Hospital</text>
        <text x="440" y="340" font-size="6" fill="#666" text-anchor="middle">12 km →</text>
        
        <!-- CASE DOTS -->
        {nalu_dots}
        {kabwe_dots}
        {tamu_dots}
        
        <!-- Legend -->
        <rect x="10" y="350" width="150" height="45" fill="white" stroke="#ccc" rx="5"/>
        <text x="20" y="365" font-size="9" font-weight="bold">Legend</text>
        <circle cx="25" cy="378" r="4" fill="#e74c3c"/>
        <text x="35" y="381" font-size="8">Severe case</text>
        <circle cx="90" cy="378" r="4" fill="#f39c12"/>
        <text x="100" y="381" font-size="8">Mild case</text>
        <rect x="20" y="386" width="10" height="6" fill="#7cb342" opacity="0.5"/>
        <text x="35" y="392" font-size="7">Rice paddies</text>
        
        <!-- Scale -->
        <line x1="380" y="380" x2="430" y2="380" stroke="#333" stroke-width="2"/>
        <text x="405" y="375" font-size="7" text-anchor="middle">~1 km</text>
        
        <!-- Compass -->
        <text x="460" y="60" font-size="12" text-anchor="middle">↑ N</text>
    </svg>
    '''
    
    # Use components.html for reliable SVG rendering
    import streamlit.components.v1 as components
    components.html(map_svg, height=450)
    
    # Summary statistics
    st.markdown("---")
    st.markdown("#### Geographic Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nalu Village", f"{nalu_cases} cases")
    
    with col2:
        st.metric("Kabwe Village", f"{kabwe_cases} cases")
    
    with col3:
        st.metric("Tamu Village", f"{tamu_cases} cases")
    
    if found_cases_count > 0:
        st.info(f"📋 Note: {found_cases_count} additional case(s) identified through clinic record review have been included in the case counts.")
    
    # Interpretation prompts
    with st.expander("🤔 Spot Map Interpretation Questions"):
        st.markdown("""
        Consider these questions as you review the geographic distribution:
        
        1. **Clustering:** Do cases cluster in specific areas? What might explain this?
        2. **Village comparison:** Why might some villages have more cases than others?
        3. **Environmental features:** What is located near the case clusters?
        4. **Hypothesis generation:** What geographic exposures might explain this pattern?
        """)


def view_interventions_and_outcome():
    st.header("📉 Interventions & Outcome")

    st.markdown("### Final Diagnosis")
    dx = st.text_input(
        "What is your final diagnosis?",
        value=st.session_state.decisions.get("final_diagnosis", ""),
    )
    st.session_state.decisions["final_diagnosis"] = dx

    st.markdown("### Recommendations")
    rec_text = st.text_area(
        "List your main recommendations:",
        value="\n".join(st.session_state.decisions.get("recommendations", [])),
        height=160,
    )
    st.session_state.decisions["recommendations"] = [
        ln for ln in rec_text.splitlines() if ln.strip()
    ]

    if st.button("Evaluate Outcome"):
        outcome = evaluate_interventions(
            st.session_state.decisions, st.session_state.interview_history
        )
        st.subheader(f"Outcome: {outcome['status']}")
        st.markdown(outcome["narrative"])
        st.markdown("### Factors considered")
        for line in outcome["outcomes"]:
            st.write(line)
        st.write(f"Score: {outcome['score']}")
        st.write(f"Estimated additional cases: {outcome['new_cases']}")


# =========================
# MAIN
# =========================

def main():
    st.set_page_config(
        page_title="FETP Sim: Sidero Valley",
        page_icon="🦟",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    sidebar_navigation()

    # If alert hasn't been acknowledged yet, always show alert screen
    if not st.session_state.alert_acknowledged:
        view_alert()
        return

    view = st.session_state.current_view
    if view == "overview":
        view_overview()
    elif view == "casefinding":
        view_case_finding()
    elif view == "descriptive":
        view_descriptive_epi()
    elif view == "villages":
        view_village_profiles()
    elif view == "interviews":
        view_interviews()
    elif view == "spotmap":
        view_spot_map()
    elif view == "study":
        view_study_design()
    elif view == "lab":
        view_lab_and_environment()
    elif view == "outcome":
        view_interventions_and_outcome()
    else:
        view_overview()


if __name__ == "__main__":
    main()
