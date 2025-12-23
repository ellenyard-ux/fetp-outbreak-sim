import streamlit as st
import anthropic
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# Village photos loaded from assets/Nalu directory
import plotly.express as px
import io
import re
import time
from pathlib import Path
from PIL import Image

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

# Game state management
init_game_state = getattr(jl, "init_game_state", None)
is_location_unlocked = getattr(jl, "is_location_unlocked", None)
unlock_location = getattr(jl, "unlock_location", None)
set_game_state = getattr(jl, "set_game_state", None)

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
    Always returns True - time can go negative in Sprint 2.
    """
    st.session_state.time_remaining -= hours
    return True


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
# LOCATIONS CONFIG (Adventure-Style Navigation)
# =========================

LOCATIONS = {
    # === NALU VILLAGE ===
    "nalu_village_center": {
        "name": "Nalu Village Center",
        "area": "Nalu Village",
        "description": "The heart of Nalu Village - a bustling community with houses clustered near the main road. The weekly market brings traders from surrounding areas. You see goats wandering near homes, chickens in the yards, and water buffalo being led to the paddies.",
        "image_path": "assets/Nalu/nalu_01_village_scene.png",
        "image_thumb": "assets/Nalu/nalu_01_village_scene.png",
        "icon": "🏘️",
        "npcs": ["mama_kofi", "auntie_ama"],
        "available_actions": [],
        "travel_time": 0.5,
    },
    "nalu_health_center": {
        "name": "Nalu Health Center",
        "area": "Nalu Village",
        "description": "A small health center staffed by Nurse Mai and community health workers. The building has a faded paint exterior and a long queue of waiting patients.",
        "image_path": "assets/Nalu/nalu_04_health_center_exterior.png",
        "image_thumb": "assets/Nalu/nalu_04_health_center_exterior.png",
        "icon": "🏥",
        "npcs": ["nurse_joy"],
        "available_actions": ["review_clinic_records", "view_hospital_records", "view_nalu_child_register"],
        "travel_time": 0.5,
    },
    "nalu_pig_coop": {
        "name": "Nalu Livestock Area",
        "area": "Nalu Village",
        "description": "A farming area about 500 meters from the village center where villagers keep various livestock. You see pigs in pens, chickens roaming freely, and a few goats tethered nearby. The smell is strong and mosquitoes swarm in the evening.",
        "image_path": "assets/Nalu/nalu_03_pig_pens.png",
        "image_thumb": "assets/Nalu/nalu_03_pig_pens.png",
        "icon": "🐷",
        "npcs": [],
        "available_actions": ["collect_pig_sample"],
        "travel_time": 0.5,
    },
    "nalu_rice_paddies": {
        "name": "Nalu Rice Paddies",
        "area": "Nalu Village",
        "description": "Extensive irrigated rice fields with standing water year-round. The paddies stretch to the horizon, broken only by narrow raised paths. Water buffalo work the fields.",
        "image_path": "assets/Nalu/nalu_02_rice_paddies.png",
        "image_thumb": "assets/Nalu/nalu_02_rice_paddies.png",
        "icon": "🌾",
        "npcs": [],
        "available_actions": ["collect_water_sample"],
        "travel_time": 0.5,
    },
    "nalu_school": {
        "name": "Nalu Primary School",
        "area": "Nalu Village",
        "description": "The main primary school serving Nalu and surrounding villages. Children from Kabwe walk here daily through the rice paddies.",
        "image_path": "assets/Kabwe/kabwe_03_children_school.png",
        "image_thumb": "assets/Kabwe/kabwe_03_children_school.png",
        "icon": "🏫",
        "npcs": ["teacher_grace"],
        "available_actions": ["review_attendance_records"],
        "travel_time": 0.5,
    },
    "nalu_canal": {
        "name": "Irrigation Canal",
        "area": "Nalu Village",
        "description": "Large pumps move water. You see many water birds and mosquitoes.",
        "image_path": "assets/Nalu/canal.png",
        "icon": "💧",
        "available_actions": [],
        "travel_time": 0.5,
    },
    "nalu_patient_house": {
        "name": "Patient A's House",
        "area": "Nalu Village",
        "description": "A traditional wooden house on stilts in Nalu Village. The family's livestock - several pigs, chickens, and ducks - are kept underneath and in a small pen nearby. Water jars sit on the porch for drinking and cooking. The house belongs to the family of one of the hospitalized children.",
        "image_path": "assets/Nalu/nalu_03_pig_pens.png",
        "image_thumb": "assets/Nalu/nalu_03_pig_pens.png",
        "icon": "🏠",
        "npcs": [],
        "available_actions": ["collect_household_water_sample"],
        "travel_time": 0.5,
    },
    # === KABWE VILLAGE ===
    "kabwe_village_center": {
        "name": "Kabwe Village Center",
        "area": "Kabwe Village",
        "description": "A medium-sized village on higher ground, 3km northeast of Nalu. Mixed farming community with both rice and upland crops. Chickens, goats, and buffalo are common. A few families keep pigs.",
        "image_path": "assets/Kabwe/kabwe_01_mixed_farming.png",
        "image_thumb": "assets/Kabwe/kabwe_01_mixed_farming.png",
        "icon": "🏡",
        "npcs": [],
        "available_actions": [],
        "travel_time": 0.5,
    },
    "kabwe_school_path": {
        "name": "School Path to Nalu",
        "area": "Kabwe Village",
        "description": "The walking path through rice paddies that Kabwe children use daily to reach school in Nalu. The path passes near irrigation canals.",
        "image_path": "assets/Kabwe/kabwe_02_village_path.png",
        "image_thumb": "assets/Kabwe/kabwe_02_village_path.png",
        "icon": "🚶",
        "npcs": [],
        "available_actions": [],
        "travel_time": 0.5,
    },
    "kabwe_school": {
        "name": "Kabwe Community School",
        "area": "Kabwe Village",
        "description": "A small community school where younger children attend before walking to the main school in Nalu. Teachers observe students for signs of illness.",
        "image_path": "assets/Kabwe/kabwe_03_children_school.png",
        "image_thumb": "assets/Kabwe/kabwe_03_children_school.png",
        "icon": "🏫",
        "npcs": [],
        "available_actions": ["review_attendance_records"],
        "travel_time": 0.5,
    },
    "kabwe_health_center": {
        "name": "Kabwe Health Center",
        "area": "Kabwe Village",
        "description": "A small health center where the visiting nurse holds clinics. Day 1 hub for reviewing clinic registers. Records are kept in a binder.",
        "image_path": "assets/Kabwe/kabwe_clinic.png",
        "icon": "🏥",
        "npcs": ["nurse_kabwe"],
        "available_actions": ["review_clinic_records", "review_kabwe_records"],
        "travel_time": 0.2,
    },
    "kabwe_paddies": {
        "name": "Kabwe Rice Fields",
        "area": "Kabwe Village",
        "description": "Smaller fields than Nalu. Farmers use buffalo to plow the fields. You see chickens foraging near the field edges and occasional goats grazing on the bunds.",
        "image_path": "assets/Kabwe/fields.png",
        "icon": "🌾",
        "available_actions": [],
        "travel_time": 0.5,
    },
    # === TAMU VILLAGE ===
    "tamu_remote_upland": {
        "name": "Tamu Remote Upland",
        "area": "Tamu Village",
        "description": "A smaller, more remote community in the foothills. Upland farming with cassava and yams. Spring-fed water sources. Goats and chickens are the main livestock - the terrain is too steep for buffalo or pigs.",
        "image_path": "assets/Tamu/tamu_02_village_remote.png",
        "image_thumb": "assets/Tamu/tamu_02_village_remote.png",
        "icon": "⛰️",
        "npcs": [],
        "available_actions": [],
        "travel_time": 1.0,
    },
    "tamu_forest_edge": {
        "name": "Tamu Forest Edge",
        "area": "Tamu Village",
        "description": "The boundary between the village and the surrounding forest. Wildlife occasionally ventures near the community. Goats graze here during the day under supervision.",
        "image_path": "assets/Tamu/tamu_03_forest_edge.png",
        "image_thumb": "assets/Tamu/tamu_03_forest_edge.png",
        "icon": "🌲",
        "npcs": [],
        "available_actions": [],
        "travel_time": 1.0,
    },
    "tamu_forest": {
        "name": "Upland Forest",
        "area": "Tamu Village",
        "description": "Dry and cool. Goats are grazing on the hills. Very few mosquitoes.",
        "image_path": "assets/Tamu/forest.png",
        "icon": "🌲",
        "available_actions": [],
        "travel_time": 0.5,
    },
    "tamu_health_center": {
        "name": "Tamu Health Center",
        "area": "Tamu Village",
        "description": "Volunteer Sarah's home doubles as the village health center. She keeps the village health log here. Day 1 hub for reviewing clinic registers.",
        "icon": "📝",
        "npcs": ["chv_tamu"],
        "available_actions": ["review_clinic_records", "review_tamu_records"],
        "travel_time": 0.2,
    },
    # === DISTRICT HOSPITAL - ADMIN OFFICE ===
    "hospital_ward": {
        "name": "Hospital Ward (Triage)",
        "area": "Admin Office",
        "description": "The AES patients are being treated in this ward. Monitors beep and worried families gather in the corridor.",
        "image_path": "assets/Hospital/hospital_ward.png",
        "image_thumb": "assets/Hospital/hospital_ward.png",
        "icon": "🏥",
        "npcs": ["parent_ward", "parent_general"],
        "available_actions": ["review_hospital_records", "collect_csf_sample", "collect_serum_sample", "view_ward_registry"],
        "travel_time": 0.5,
    },
    "hospital_office": {
        "name": "Hospital Admin (Charts)",
        "area": "Admin Office",
        "description": "The hospital director's office. Charts and reports are pinned to the walls. A window overlooks the hospital courtyard. Deep-dive charts are available here showing High Fever (>39C) and Lymphocytosis patterns.",
        "image_path": "assets/Hospital/hospital_admin.png",
        "image_thumb": "assets/Hospital/hospital_admin.png",
        "icon": "📋",
        "npcs": ["dr_chen"],
        "available_actions": ["review_hospital_records", "view_deep_dive_charts"],
        "travel_time": 0.0,
        "max_deep_dive_charts": 2,
    },
    # === DISTRICT HOSPITAL - LABORATORY ===
    "hospital_lab": {
        "name": "Hospital Laboratory",
        "area": "Laboratory",
        "description": "A small but functional laboratory. Basic labs are available immediately. Complex PCR and serology tests unlock on Day 4.",
        "image_path": "assets/Hospital/hospital_lab.png",
        "image_thumb": "assets/Hospital/hospital_lab.png",
        "icon": "🔬",
        "npcs": [],
        "available_actions": ["view_lab_results", "submit_lab_samples"],
        "travel_time": 0.0,
        "unlock_day": 1,
        "pcr_serology_unlock_day": 4,
    },
    # === DISTRICT OFFICE ===
    "district_office": {
        "name": "District Health Office",
        "area": "District Office",
        "description": "The administrative hub for district health operations. Officials, technical officers, and the Environmental Officer work from here.",
        "image_path": "assets/Hospital/hospital_admin.png",
        "image_thumb": "assets/Hospital/hospital_admin.png",
        "icon": "🏛️",
        "npcs": ["vet_amina", "mr_osei", "mayor_simon", "env_officer"],
        "available_actions": ["request_data", "plan_interventions", "view_village_profile"],
        "travel_time": 0.5,
    },
    # === MINING AREA ===
    "mining_area": {
        "name": "Mining Compound",
        "area": "Mining Area",
        "description": "The mining operation has expanded recently, creating new irrigation ponds. Worker housing is nearby.",
        "image_path": "assets/Kabwe/kabwe_01_mixed_farming.png",
        "image_thumb": "assets/Kabwe/kabwe_01_mixed_farming.png",
        "icon": "⛏️",
        "npcs": ["foreman_rex"],
        "available_actions": ["collect_water_sample"],
        "travel_time": 1.0,
    },
    # === MARKET ===
    "central_market": {
        "name": "Central Market",
        "area": "Nalu Village",
        "description": "The weekly market where traders from all villages gather. A good place to hear rumors and observe community patterns.",
        "image_path": "assets/Nalu/nalu_01_village_scene.png",
        "image_thumb": "assets/Nalu/nalu_01_village_scene.png",
        "icon": "🛒",
        "npcs": ["auntie_ama"],
        "available_actions": [],
        "travel_time": 0.5,
    },
    # === HEALER ===
    "healer_clinic": {
        "name": "Traditional Healer's Clinic",
        "area": "Nalu Village",
        "description": "A small private clinic run by Healer Somchai. He saw some of the earliest cases before they went to the hospital.",
        "image_path": "assets/Nalu/nalu_01_village_scene.png",
        "image_thumb": "assets/Nalu/nalu_01_village_scene.png",
        "icon": "🌿",
        "npcs": ["healer_marcus"],
        "available_actions": ["review_healer_records"],
        "travel_time": 0.5,
    },
}

# Map areas to their sub-locations
AREA_LOCATIONS = {
    "Nalu Village": ["nalu_village_center", "nalu_health_center", "nalu_pig_coop", "nalu_rice_paddies", "nalu_school", "nalu_canal", "central_market", "healer_clinic"],
    "Kabwe Village": ["kabwe_village_center", "kabwe_school_path", "kabwe_school", "kabwe_health_center", "kabwe_paddies"],
    "Tamu Village": ["tamu_remote_upland", "tamu_forest_edge", "tamu_forest", "tamu_health_center"],
    "District Hospital": ["hospital_office", "hospital_ward", "hospital_lab"],
    "Admin Office": ["hospital_ward", "hospital_office"],
    "Laboratory": ["hospital_lab"],
    "District Office": ["district_office"],
    "Mining Area": ["mining_area"],
}

# Area metadata for visual rendering (hero images, descriptions)
AREA_METADATA = {
    "Admin Office": {
        "image_exterior": "assets/Hospital/hospital_exterior.png",
        "description": "The hospital ward and administrative office where AES patients are treated. Dr. Tran oversees triage and patient charts. Deep-dive clinical data is available here.",
        "icon": "🏥",
    },
    "Laboratory": {
        "image_exterior": "assets/Hospital/hospital_lab.png",
        "description": "The hospital laboratory. Basic tests are available immediately. Complex PCR and serology tests unlock on Day 4.",
        "icon": "🔬",
    },
    "District Hospital": {
        "image_exterior": "assets/Hospital/hospital_exterior.png",
        "description": "The district hospital where AES patients are being treated. Contains the administrative office, patient ward, and laboratory facilities. Dr. Tran oversees operations.",
        "icon": "🏥",
    },
    "Nalu Village": {
        "image_exterior": "assets/Nalu/nalu_01_village_scene.png",
        "description": "The largest settlement in Sidero Valley. The economy centers on rice cultivation and pig farming. Most AES cases come from here.",
        "icon": "🏘️",
    },
    "Kabwe Village": {
        "image_exterior": "assets/Kabwe/kabwe_01_mixed_farming.png",
        "description": "Located 3km northeast of Nalu on higher ground. Children walk through rice paddies to attend school in Nalu.",
        "icon": "🏡",
    },
    "Tamu Village": {
        "image_exterior": "assets/Tamu/tamu_02_village_remote.png",
        "description": "A smaller, more remote community in the foothills. Upland farming with less standing water.",
        "icon": "⛰️",
    },
    "District Office": {
        "image_exterior": "assets/Hospital/hospital_admin.png",
        "description": "The administrative hub of district health operations. Houses the public health, veterinary, and environmental health teams. Key officials coordinate outbreak response from here.",
        "icon": "🏛️",
    },
    "Mining Area": {
        "image_exterior": "assets/Kabwe/kabwe_01_mixed_farming.png",
        "description": "The mining operation has expanded recently, creating new irrigation ponds and disrupting local ecosystems. Worker housing and canteen facilities are located nearby.",
        "icon": "⛏️",
    },
}

# Map NPC keys to their primary location
NPC_LOCATIONS = {
    "dr_chen": "hospital_ward",
    "patient_parent": "hospital_ward",
    "ward_parent": "hospital_ward",
    "nurse_joy": "nalu_health_center",
    "healer_marcus": "healer_clinic",
    "mama_kofi": "nalu_village_center",
    "teacher_grace": "nalu_school",
    "foreman_rex": "mining_area",
    "auntie_ama": "central_market",
    "vet_amina": "district_office",
    "env_officer": "district_office",
    "mr_osei": "district_office",
    "mayor_simon": "district_office",
    "nurse_kabwe": "kabwe_health_center",
    "chv_tamu": "tamu_health_center",
}


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

    # Game state initialization (Serious Mode)
    if init_game_state:
        init_game_state(st.session_state)

    # Alert page logic (Day 0)
    st.session_state.setdefault("alert_acknowledged", False)

    if "current_day" not in st.session_state:
        # 1–5 for the investigation days
        st.session_state.current_day = 1

    if "current_view" not in st.session_state:
        # Start on alert screen until acknowledged
        st.session_state.current_view = "alert"

    # Adventure mode: current_location (None = show map, string = show location view)
    st.session_state.setdefault("current_location", None)
    st.session_state.setdefault("current_area", None)  # For area-level navigation

    # If alert is not acknowledged, force the view to "alert"
    if not st.session_state.alert_acknowledged:
        st.session_state.current_view = "alert"
    else:
        # If alert already acknowledged but view is still "alert", move to map
        if st.session_state.current_view == "alert":
            st.session_state.current_view = "map"

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
    st.session_state.setdefault("interview_context_location", None)
    st.session_state.setdefault("visited_locations", set())
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

    # SITREP and Evidence Board
    st.session_state.setdefault("sitrep_viewed", True)  # Don't show SITREP on Day 1 start
    init_evidence_board()
    st.session_state.setdefault("questions_asked_about", set())
    
    # Clinic records and case finding (Day 1)
    st.session_state.setdefault("clinic_records_reviewed", False)
    st.session_state.setdefault("selected_clinic_cases", [])
    st.session_state.setdefault("case_finding_score", None)
    st.session_state.setdefault("found_cases_added", False)
    
    # Descriptive epidemiology
    st.session_state.setdefault("descriptive_epi_viewed", False)

    # Medical Records workflow (Day 1)
    st.session_state.setdefault("line_list_cols", [])
    st.session_state.setdefault("my_case_def", {})
    st.session_state.setdefault("manual_cases", [])

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

    # Override for Dr. Tran - restrict his knowledge to prevent omniscience
    if npc_key == "dr_chen":
        return (
            "You are aware of 2 severe pediatric cases admitted in the last 48 hours. "
            "One has died. You recall seeing maybe 2-3 other similar cases over the "
            "last month, but you do not have the files in front of you. "
            "You DO NOT know the total case count in the district. "
            "You believe the mine is responsible."
        )

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
            notification = "🔓 **New Contact Unlocked:** Vet Supatra (District Veterinary Officer) - Your question about animals opened a One Health perspective!"
    
    # Mosquito/environment triggers → unlock Mr. Osei
    env_triggers = ['mosquito', 'mosquitoes', 'vector', 'breeding', 'standing water', 'environment', 'rice paddy', 'irrigation', 'wetland']
    if any(trigger in text for trigger in env_triggers):
        st.session_state.questions_asked_about.add('environment')
        if not st.session_state.env_officer_unlocked:
            st.session_state.env_officer_unlocked = True
            st.session_state.one_health_triggered = True
            if 'mr_osei' not in st.session_state.npcs_unlocked:
                st.session_state.npcs_unlocked.append('mr_osei')
            notification = "🔓 **New Contact Unlocked:** Mr. Nguyen (Environmental Health Officer) - Your question about environmental factors opened a new perspective!"
    
    # Healer triggers (for earlier cases)
    healer_triggers = ['traditional', 'healer', 'clinic', 'private', 'early case', 'first case', 'before hospital']
    if any(trigger in text for trigger in healer_triggers):
        st.session_state.questions_asked_about.add('traditional')
        if 'healer_marcus' not in st.session_state.npcs_unlocked:
            st.session_state.npcs_unlocked.append('healer_marcus')
            notification = "🔓 **New Contact Unlocked:** Healer Somchai (Private Clinic) - You discovered there may be unreported cases!"
    
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


def get_npc_avatar(npc: dict) -> str:
    """Get the avatar for an NPC - returns image path if available, otherwise emoji."""
    image_path = npc.get("image_path")
    if image_path and Path(image_path).exists():
        return image_path
    return npc.get("avatar", "🧑")


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

CORE RULES:
You are NOT an AI assistant. You are a fictional character in a training simulation.
Do not offer to help. Do not be polite unless your character personality is polite.
If the user asks a vague question, give a vague answer.
Force them to ask the right specific questions to get the information.

CRITICAL ANTI-SPOILER RULES:
- Be BRIEF and PROFESSIONAL. Keep responses to 2-4 sentences unless asked for more detail.
- DO NOT name specific pathogens (JEV, Japanese Encephalitis, arbovirus) unless you see lab confirmation.
- DO NOT jump to conclusions about the cause. Only provide RAW OBSERVATIONAL DATA.
- DO NOT volunteer diagnostic hunches. You are a witness, not a diagnostician.
- If asked "what is causing this?", say you don't know - you only see symptoms/patterns.

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

    # SPECIAL LOGIC: Dr. Chen permission granting
    if npc_key == "dr_chen":
        lower_q = user_input.lower()
        # Grant permission if user asks for it
        if any(keyword in lower_q for keyword in ["permission", "access", "records", "investigate", "allow"]):
            st.session_state.unlock_flags['records_access'] = True
            st.rerun()  # Force refresh to unlock the buttons immediately

    # SPECIAL LOGIC: Ward Parent livestock question counter
    if npc_key == "ward_parent":
        lower_q = user_input.lower()
        # Increment counter if asking about livestock/animals/pigs
        if any(keyword in lower_q for keyword in ["livestock", "animal", "pig", "cow", "chicken", "duck"]):
            if "?" in lower_q:  # Must be a question
                if 'ward_parent_livestock_asks' not in st.session_state:
                    st.session_state.ward_parent_livestock_asks = 0
                st.session_state.ward_parent_livestock_asks += 1

                # On second ask about livestock, reveal pigs
                if st.session_state.ward_parent_livestock_asks >= 2:
                    # Make sure the pig info gets included
                    if "pigs" not in text.lower() and "pig" not in text.lower():
                        text += "\n\n...Well, actually, we do keep pigs. Two of them, right near the house. They're for the New Year festival."

    return text


# =========================
# CLINIC RECORDS FOR CASE FINDING
# =========================

def generate_clinic_records(village_context="nalu"):
    """
    Generate messy, handwritten-style clinic records.
    Mix of AES cases and unrelated illnesses.
    Returns list of record dicts.

    Args:
        village_context: "nalu" (default - full messy list),
                        "kabwe" (mild fevers, cuts, 1-2 transferred AES),
                        "tamu" (coughs, snakebites, Panya traveler case)
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

    # Filter based on village context
    if village_context == "kabwe":
        # Kabwe: mild fevers, cuts, 1-2 transferred AES cases
        kabwe_records = [r for r in all_records if r.get("village") == "Kabwe"]
        # Include only 1-2 AES cases from Kabwe
        kabwe_aes = [r for r in kabwe_records if r.get("is_aes")][:2]
        kabwe_non_aes = [r for r in kabwe_records if not r.get("is_aes")]
        return kabwe_aes + kabwe_non_aes

    elif village_context == "tamu":
        # Tamu: coughs, snakebites, and the "Panya" case (7yo who traveled to Nalu)
        tamu_records = [r for r in all_records if r.get("village") == "Tamu"]
        # Add Panya's case - 7yo girl who traveled to Nalu market
        panya_case = {
            "record_id": "NHC-0080",
            "date": "10-Jun",
            "patient": "Panya",
            "age": "7",
            "village": "Tamu",
            "complaint": "fever, confusion, difficulty walking",
            "notes": "Mother says traveled to Nalu market recently. Referred to district hospital.",
            "is_aes": True
        }
        return tamu_records + [panya_case]

    # Default: nalu - return existing messy list
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
            "chief_complaint": "High fever (40.2C) and generalized seizures",
            "history_present_illness": """
Child well until 2 days ago. Sudden onset fever (40C), headache, vomiting.
Seizures began this morning. No history of previous seizures. No recent travel.
No sick contacts known.

Child plays regularly in rice fields near home after school. Family keeps 3 pigs
in pen behind house. No mosquito net use - mother says 'it is too hot.'
""",
            "past_medical_history": "No significant PMH. Immunizations up to date per mother (card not available). No known allergies.",
            "physical_exam": """
Temp 40.2C, HR 150. Unconscious. Neck stiffness positive.
No pinpoint pupils (rules out opiates/organophosphates).
No drooling or lacrimation.
""",
            "investigations": """
- WBC: 16,000 (85% Lymphocytes) -> Viral picture
- Hemoglobin: Normal
- Metabolic Panel: Normal anion gap (Rules out many toxins/metabolic causes)
- CSF: Clear, 120 WBC (Lymphocytic), Glucose Normal.
- Malaria RDT: Negative
""",
            "initial_diagnosis": "Acute Viral Encephalitis",
            "differential": "Viral Encephalitis vs Bacterial Meningitis. (Toxin unlikely due to high fever and lymphocytosis)",
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
            "chief_complaint": "Confusion and inability to walk",
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
Temp 39.5C. Ataxic gait. Tremors. No rash.
""",
            "investigations": """
WBC: 14,500 (Lymphocytic). CSF: Pleocytosis. Toxicology Screen: Negative for organophosphates.
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

    # Create display column for outcome that includes sequelae info
    if 'has_sequelae' in cases.columns:
        cases['outcome_display'] = cases.apply(
            lambda row: f"{row['outcome']} (with complications)" if row.get('has_sequelae') else row['outcome'],
            axis=1
        )
    else:
        cases['outcome_display'] = cases['outcome']

    return cases.head(n)[
        ["person_id", "age", "sex", "village_name", "onset_date", "severe_neuro", "outcome_display"]
    ].rename(columns={'outcome_display': 'outcome'})


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
    st.sidebar.markdown(f"### {t('language_header')}")
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
    time_display = f":red[{st.session_state.time_remaining}]" if st.session_state.time_remaining < 0 else str(st.session_state.time_remaining)
    st.sidebar.markdown(
        f"**{t('day')}:** {st.session_state.current_day} / 5\n\n"
        f"**{t('budget')}:** ${st.session_state.budget}\n\n"
        f"**{t('time_remaining')}:** {time_display} {t('hours')}\n\n"
        f"**{t('lab_credits')}:** {st.session_state.lab_credits}"
    )

    # Investigation Hub (Day 1 specific)
    if st.session_state.current_day == 1:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Investigation Hub")

        # My Line List
        with st.sidebar.expander("My Line List", expanded=False):
            if st.session_state.line_list_cols:
                st.caption(f"Selected columns: {', '.join(st.session_state.line_list_cols)}")

                if st.session_state.manual_cases:
                    st.caption(f"Cases added: {len(st.session_state.manual_cases)}")

                    # Show a preview of the line list
                    if st.session_state.get('clinic_records'):
                        records = st.session_state.clinic_records
                        selected_records = [r for r in records if r['record_id'] in st.session_state.manual_cases]

                        if selected_records:
                            st.markdown("**Preview:**")
                            for rec in selected_records[:3]:  # Show first 3
                                st.caption(f"- {rec['record_id']}: {rec['patient']}")
                            if len(selected_records) > 3:
                                st.caption(f"... and {len(selected_records) - 3} more")
                else:
                    st.caption("No cases added yet")
            else:
                st.caption("No line list structure defined yet")
                st.caption("Visit Medical Records to define your structure")

        # My Case Definition
        with st.sidebar.expander("My Case Definition", expanded=False):
            if st.session_state.my_case_def:
                for key, value in st.session_state.my_case_def.items():
                    st.caption(f"**{key}:** {value}")
            else:
                st.caption("No case definition saved yet")
                if st.session_state.get('decisions', {}).get('case_definition_text'):
                    st.caption("See Overview tab for case definition")

    # Session Management - Save/Load functionality
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session Management")

    col1, col2 = st.sidebar.columns(2)

    # Save session
    with col1:
        if st.button("Save", use_container_width=True, key="save_session_btn", help="Download your current progress"):
            try:
                save_data = persistence.create_save_file(st.session_state)
                filename = persistence.get_save_filename(st.session_state)
                st.sidebar.download_button(
                    label="Download",
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
            "Load",
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
            status = "[✓]"
        elif day == st.session_state.current_day:
            status = "[●]"
        else:
            status = "[ ]"
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
        if st.sidebar.button(f"{t('advance_day')} {st.session_state.current_day + 1}", use_container_width=True):
            can_advance, missing = check_day_prerequisites(st.session_state.current_day, st.session_state)
            if can_advance:
                st.session_state.current_day += 1
                st.session_state.time_remaining = 8  # Reset time for new day
                refresh_lab_queue_for_day(int(st.session_state.current_day))
                st.session_state.advance_missing_tasks = []
                # Show SITREP view for new day
                st.session_state.current_view = "sitrep"
                st.session_state.sitrep_viewed = False
                st.rerun()
            else:
                st.session_state.advance_missing_tasks = missing
                st.sidebar.warning(t("cannot_advance"))
    else:
        st.sidebar.success("Final Day - Complete your briefing!")


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

def view_intro():
    """Serious Mode: Dr. Tran phone call intro screen."""
    html_content = """
<style>
.phone-overlay {
    background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8d 100%);
    padding: 3rem;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    text-align: center;
    margin: 2rem auto;
    max-width: 600px;
}
.phone-title {
    font-size: 2.5rem;
    color: white;
    margin-bottom: 1rem;
}
.phone-subtitle {
    font-size: 1.2rem;
    color: #a8c5e4;
    margin-bottom: 2rem;
}
.phone-message {
    background: rgba(255,255,255,0.1);
    border-left: 4px solid #ff6b6b;
    padding: 1.5rem;
    margin: 2rem 0;
    border-radius: 8px;
    color: white;
    font-size: 1.1rem;
    line-height: 1.6;
    text-align: left;
}
.metrics-box {
    background: rgba(255,107,107,0.2);
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    text-align: center;
}
.metric-item {
    color: #ffeb3b;
    font-weight: bold;
    font-size: 1.3rem;
    margin: 0.5rem 0;
}
</style>
<div class="phone-overlay">
    <div class="phone-title">📞 Incoming Call</div>
    <div class="phone-subtitle">Dr. Tran - District Hospital</div>
    <div class="phone-message"><strong>Dr. Tran:</strong> "This is Dr. Tran at Sidero District Hospital. We have a critical situation here. We've admitted <strong>2 severe pediatric cases</strong> with acute encephalitis syndrome. Both children are experiencing seizures and altered consciousness.<br><br>And I'm sorry to report... we've had <strong>1 death</strong> already.<br><br>I need an FETP officer here <strong>immediately</strong> to investigate. This could be the start of something much bigger."</div>
    <div class="metrics-box">
        <p class="metric-item">🏥 2 Severe Cases</p>
        <p class="metric-item">💀 1 Death</p>
        <p class="metric-item">📅 Day 1</p>
    </div>
</div>
"""
    st.markdown(html_content, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ Accept Assignment", type="primary", use_container_width=True):
            if set_game_state:
                set_game_state('DASHBOARD', st.session_state)
            st.session_state.alert_acknowledged = True
            st.session_state.current_view = "map"
            st.rerun()


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

    # Evidence Board
    st.markdown("---")
    view_evidence_board()

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
            st.markdown("### Case Definition")

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
            st.markdown("### Initial Hypotheses")
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


def view_hospital_triage():
    st.markdown("## District Hospital Triage")

    # Intro Text
    st.info("Dr. Tran: 'Here are the patients admitted in the last 48 hours. Please review them. "
            "Mark the ones that fit your Case Definition to add them to your Line List.'")

    # Initialize State
    if 'line_list' not in st.session_state:
        st.session_state.line_list = []
    if 'parents_interviewed' not in st.session_state:
        st.session_state.parents_interviewed = []

    triage_data = jl.get_hospital_triage_list()

    # --- SECTION 1: THE CHECKLIST ---
    with st.container(border=True):
        st.markdown("### Patient Admission Log")

        # Grid Header
        c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
        c1.write("**Add?**")
        c2.write("**ID/Age**")
        c3.write("**Village**")
        c4.write("**Clinical Notes**")

        # Grid Rows
        for patient in triage_data:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 3])

            # Checkbox
            is_selected = patient['id'] in st.session_state.line_list
            if c1.checkbox("Select", key=f"chk_{patient['id']}", value=is_selected, label_visibility="collapsed"):
                if patient['id'] not in st.session_state.line_list:
                    st.session_state.line_list.append(patient['id'])
            else:
                if patient['id'] in st.session_state.line_list:
                    st.session_state.line_list.remove(patient['id'])

            c2.write(f"**{patient['id']}** ({patient['age']}/{patient['sex']})")
            c3.write(patient['village'])
            # Highlight Fever to rule out toxin
            if "Fever" in patient['symptom']:
                c4.markdown(f"**{patient['symptom']}** - *{patient['notes']}*")
            else:
                c4.write(f"{patient['symptom']} - *{patient['notes']}*")
            st.divider()

    # --- SECTION 2: PARENT INTERVIEWS ---
    if len(st.session_state.line_list) > 0:
        st.markdown("### Investigation")
        st.write(f"**Budget:** You can interview parents for **{2 - len(st.session_state.parents_interviewed)}** more cases.")

        cols = st.columns(3)
        for i, patient_id in enumerate(st.session_state.line_list):
            patient = next(p for p in triage_data if p['id'] == patient_id)

            with cols[i % 3]:
                with st.container(border=True):
                    st.write(f"**{patient_id}** ({patient['village']})")

                    if patient_id in st.session_state.parents_interviewed:
                        st.success("✅ Interviewed")
                    elif len(st.session_state.parents_interviewed) >= 2:
                        st.caption("⛔ No Budget")
                    else:
                        if st.button(f"Interview Parents", key=f"btn_{patient_id}"):
                            st.session_state.parents_interviewed.append(patient_id)

                            # Log the parent interview event
                            jl.log_event(
                                event_type='interview',
                                location_id=patient['parent_type'],
                                cost_time=0,
                                cost_budget=0,
                                payload={
                                    'patient_id': patient_id,
                                    'parent_type': patient['parent_type'],
                                    'village': patient['village']
                                }
                            )

                            # TRIGGER THE INTERVIEW
                            st.session_state.current_npc = patient['parent_type']
                            st.rerun()

    # --- SECTION 3: THE DIALOGUE (Pop-up) ---
    if 'current_npc' in st.session_state and st.session_state.current_npc in ['parent_tamu', 'parent_general']:
        npc_key = st.session_state.current_npc
        # Load the text from truth data
        npc_data = jl.load_truth_data().get(npc_key)

        st.markdown("---")
        st.warning(f"🎙️ Interviewing: {npc_data['name']}")
        st.write(f"**{npc_data['name']}:** \"{npc_data['always_reveal'][0]}\"")
        st.write(f"**{npc_data['name']}:** \"{npc_data['always_reveal'][2]}\"")

        # Special logic for Tamu
        if npc_key == 'parent_tamu':
            st.error("❗ **KEY FINDING:** Family traveled to Nalu 2 weeks ago!")

        if st.button("End Interview"):
            del st.session_state.current_npc
            st.rerun()


def view_interviews():
    truth = st.session_state.truth
    npc_truth = truth["npc_truth"]

    # Back button at the top for easy navigation
    col_back, col_spacer = st.columns([1, 5])
    with col_back:
        # If coming from a location, show "Return to Location" button
        current_loc = st.session_state.get("current_location")
        if current_loc:
            loc_data = LOCATIONS.get(current_loc, {})
            loc_name = loc_data.get("name", "Location")
            if st.button(f"🔙 Return to {loc_name}", key="return_to_loc_from_interviews"):
                st.session_state.current_view = "location"
                st.rerun()
        else:
            if st.button("🔙 Return to Map", key="return_to_map_from_interviews"):
                st.session_state.current_view = "map"
                st.rerun()

    st.header("👥 Interviews")

    # Check if we're accessing from a specific location with NPCs
    context_loc = st.session_state.get("interview_context_location")
    context_npcs = []
    if context_loc:
        loc_data = LOCATIONS.get(context_loc, {})
        for npc_key in loc_data.get("npcs", []):
            if npc_key in st.session_state.npcs_unlocked and npc_key in npc_truth:
                context_npcs.append((npc_key, npc_truth[npc_key]))

    # If there's exactly one NPC at the context location and no active conversation,
    # start the conversation immediately
    if len(context_npcs) == 1 and not st.session_state.current_npc:
        npc_key, npc = context_npcs[0]
        interviewed = npc_key in st.session_state.interview_history
        time_cost = TIME_COSTS["interview_followup"] if interviewed else TIME_COSTS["interview_initial"]
        budget_cost = 0 if interviewed else npc.get("cost", 0)

        can_proceed, msg = check_resources(time_cost, budget_cost)
        if can_proceed:
            spend_time(time_cost, f"Interview: {npc['name']}")
            if budget_cost > 0:
                spend_budget(budget_cost, f"Interview: {npc['name']}")
            st.session_state.current_npc = npc_key
            st.session_state.interview_history.setdefault(npc_key, [])
            # Clear context after auto-selecting
            st.session_state.interview_context_location = None
            st.rerun()

    # Resource display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Budget", f"${st.session_state.budget}")
    with col2:
        time_color = "normal" if st.session_state.time_remaining >= 0 else "inverse"
        st.metric("⏱️ Time Remaining", f"{st.session_state.time_remaining}h", delta_color=time_color)
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

                    # Log the interview event
                    jl.log_event(
                        event_type='interview',
                        location_id=npc_key,
                        cost_time=time_cost,
                        cost_budget=budget_cost,
                        payload={
                            'npc_name': npc['name'],
                            'npc_role': npc['role'],
                            'is_followup': interviewed
                        }
                    )

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
                with st.chat_message("assistant", avatar=get_npc_avatar(npc)):
                    st.write(msg["content"])

        user_q = st.chat_input("Ask your question...")
        if user_q:
            # Check for NPC unlock triggers BEFORE getting response
            unlock_notification = check_npc_unlock_triggers(user_q)
            
            history.append({"role": "user", "content": user_q})
            st.session_state.interview_history[npc_key] = history

            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant", avatar=get_npc_avatar(npc)):
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

    # Back button at the top for easy navigation
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔙 Return to Clinic", key="return_to_clinic_from_cf"):
            st.session_state.current_location = "nalu_health_center"
            st.session_state.current_area = "Nalu Village"
            st.session_state.current_view = "location"
            st.rerun()

    st.header("Case Finding")

    # Tabs for different record sources
    tab1, tab2 = st.tabs(["Clinic Records", "Hospital Records"])
    
    with tab1:
        st.subheader("Nalu Health Center - Patient Register Review")
        
        # Resource display and cost warning
        time_cost = TIME_COSTS["clinic_records_review"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            time_display = f":red[{st.session_state.time_remaining}h]" if st.session_state.time_remaining < 0 else f"{st.session_state.time_remaining}h"
            st.markdown(f"**⏱️ Time Remaining**  \n{time_display}")
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
                st.markdown("### Patient Register (June 2025)")
                
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

                        # Log the case finding event
                        jl.log_event(
                            event_type='case_finding',
                            location_id=None,
                            cost_time=0,
                            cost_budget=0,
                            payload={
                                'true_positives': true_positives,
                                'false_positives': false_positives,
                                'false_negatives': false_negatives,
                                'total_aes': total_aes,
                                'selected_count': len(selected),
                                'cases_added': true_positives
                            }
                        )

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
                st.markdown("### Your Case Finding Results")
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


def view_medical_records():
    """Day 1: Medical Records view - The Hub for building case definition and line list variables."""

    st.title("Medical Records")
    st.caption("Day 1: Review initial cases and build your field log structure")

    # Back button
    if st.button("Return to Map", key="return_from_medical_records"):
        st.session_state.current_view = "map"
        st.rerun()

    st.markdown("---")

    # STEP 1: Display 2 "Clipboards" side-by-side
    st.markdown("### Step 1: Review Initial Cases")
    st.caption("These are two of the cases reported to the district hospital. Review their clinical information.")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("Case 1: Index Case (First Reported)", expanded=True):
            st.markdown("**ID:** HOSP-01")
            st.markdown("**Age/Sex:** 6 years / Male")
            st.markdown("**Village:** Nalu")
            st.markdown("**Clinical Presentation:**")
            st.markdown("- Fever: Yes (39.5°C)")
            st.markdown("- Seizures: Yes (multiple episodes)")
            st.markdown("- Altered consciousness: Yes")
            st.markdown("- Vomiting: Yes")
            st.markdown("- Rash: No")
            st.markdown("**Status:** Hospitalized")

    with col2:
        with st.expander("Case 2: Panya (Deceased)", expanded=True):
            st.markdown("**ID:** HOSP-04")
            st.markdown("**Name:** Panya")
            st.markdown("**Age/Sex:** 7 years / Female")
            st.markdown("**Village:** Tamu")
            st.markdown("**Clinical Presentation:**")
            st.markdown("- Fever: Yes (40.1°C)")
            st.markdown("- Seizures: Yes (severe)")
            st.markdown("- Altered consciousness: Yes (coma)")
            st.markdown("- Vomiting: Yes")
            st.markdown("- Rash: No")
            st.markdown("**Status:** Deceased")

    st.markdown("---")

    # STEP 2: Build Line List Variables
    st.markdown("### Step 2: Build Line List Variables")
    st.caption("Select which columns you want to include in your field investigation log.")

    # Available options - including traps
    all_options = [
        "Age",
        "Sex",
        "Village",
        "Fever",
        "Seizure",
        "Rash",
        "Vomiting",
        "Pig Contact",  # Trap!
        "Rice Field",   # Trap!
    ]

    selected_cols = st.multiselect(
        "Select columns for your field log:",
        options=all_options,
        default=st.session_state.line_list_cols if st.session_state.line_list_cols else [],
        key="line_list_cols_select"
    )

    # Check for traps
    traps = []
    if "Pig Contact" in selected_cols:
        traps.append("Pig Contact")
    if "Rice Field" in selected_cols:
        traps.append("Rice Field")

    if traps:
        st.warning(f"⚠️ **Medical records do not typically contain exposure information like {', '.join(traps)}. Consider sticking to clinical signs and demographic information that would be documented in hospital records.**")

    if st.button("Save Line List Structure", type="primary"):
        st.session_state.line_list_cols = selected_cols
        st.success(f"✅ Line list structure saved with {len(selected_cols)} columns!")
        st.rerun()

    if st.session_state.line_list_cols:
        st.info(f"Current line list columns: {', '.join(st.session_state.line_list_cols)}")

    st.markdown("---")

    # Navigation hint
    st.markdown("### Next Steps")
    st.caption("Once you've defined your line list structure, proceed to the Clinic Register Scan to review additional cases.")
    if st.button("Go to Clinic Register Scan"):
        st.session_state.current_view = "clinic_register"
        st.rerun()


def view_clinic_register_scan():
    """Day 1: Clinic Register Scan - Review logbook and select suspect cases."""

    st.title("Clinic Register Scan")
    st.caption("Day 1: Review the clinic logbook and identify potential cases")

    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Return to Map", key="return_from_clinic_register"):
            st.session_state.current_view = "map"
            st.rerun()

    st.markdown("---")

    st.markdown("### Raw Logbook from Nalu Health Center")
    st.caption("Review these handwritten records and check any that you suspect might be AES cases.")

    # Generate clinic records if not already done
    if 'clinic_records' not in st.session_state:
        st.session_state.clinic_records = generate_clinic_records()

    records = st.session_state.clinic_records

    # Create a dataframe for display
    display_df = pd.DataFrame(records)

    # Add a suspect case checkbox column
    st.markdown("#### Patient Register (June 2025)")

    # Initialize manual_cases if needed
    if not isinstance(st.session_state.manual_cases, list):
        st.session_state.manual_cases = []

    # Display records with checkboxes
    for i, record in enumerate(records):
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 1, 5])

            # Checkbox
            is_checked = record['record_id'] in st.session_state.manual_cases
            check = col1.checkbox(
                "Suspect?",
                key=f"suspect_{record['record_id']}",
                value=is_checked,
                label_visibility="collapsed"
            )

            if check and not is_checked:
                st.session_state.manual_cases.append(record['record_id'])
                st.rerun()
            elif not check and is_checked:
                st.session_state.manual_cases.remove(record['record_id'])
                st.rerun()

            # Record details
            col2.write(f"**{record['record_id']}**")
            col3.write(record['date'])
            col4.write(record['age'])
            col5.write(f"{record['patient']} - {record['village']}")

            # Complaint and notes in full width
            st.caption(f"**Complaint:** {record['complaint']}")
            st.caption(f"**Notes:** {record['notes']}")
            st.divider()

    st.markdown("---")

    # Summary and action button
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"**Checked records:** {len(st.session_state.manual_cases)}")
        if st.session_state.manual_cases:
            st.caption(f"IDs: {', '.join(st.session_state.manual_cases)}")

    with col2:
        if st.button("Add Checked to Line List", type="primary"):
            if st.session_state.manual_cases:
                # Calculate scoring
                true_positives = sum(1 for rid in st.session_state.manual_cases
                                   for r in records if r['record_id'] == rid and r.get('is_aes'))
                false_positives = len(st.session_state.manual_cases) - true_positives
                total_aes = sum(1 for r in records if r.get('is_aes'))
                false_negatives = total_aes - true_positives

                st.success(f"✅ Added {len(st.session_state.manual_cases)} records to your line list!")
                st.info(f"📊 You identified {true_positives} of {total_aes} true AES cases.")

                if false_positives > 0:
                    st.warning(f"⚠️ {false_positives} selected record(s) may not be AES.")
                if false_negatives > 0:
                    st.caption(f"💡 {false_negatives} potential AES case(s) were not selected.")

                # Mark as reviewed
                st.session_state.clinic_records_reviewed = True
            else:
                st.error("No records selected!")


def view_nalu_child_register():
    """View Nalu Health Center Child Register - 38 entries with new cases."""
    from je_logic import get_nalu_child_register, get_nalu_medical_record

    st.title("Nalu Health Center - Child Register")
    st.caption("Review the child health register to identify potential cases")

    # Back button
    if st.button("Return to Map", key="return_from_nalu_register"):
        st.session_state.current_view = "map"
        st.rerun()

    st.markdown("---")

    # Get the register
    register = get_nalu_child_register()

    # Convert to DataFrame for display
    df = pd.DataFrame(register)

    # Instructions
    st.markdown("""
    ### Instructions
    - Review the child register entries below
    - Look for patterns in complaints and outcomes
    - Click "View Chart" to see detailed medical records for key patients
    - Identify potential AES cases based on clinical presentation
    """)

    st.markdown("---")

    # Display in table format with highlighting
    st.markdown("### Child Register (38 entries)")

    # Initialize unlocked charts if not exists
    if 'unlocked_nalu_charts' not in st.session_state:
        st.session_state['unlocked_nalu_charts'] = []

    # Show table with color coding
    for idx, entry in enumerate(register):
        # Determine card color based on status
        if 'Referred to Hospital' in entry['status']:
            card_color = "🔴"
        elif 'Died' in entry['status']:
            card_color = "⚫"
        elif 'stiff neck' in entry['complaint'].lower() or 'seizure' in entry['complaint'].lower():
            card_color = "🟡"
        else:
            card_color = "🟢"

        with st.expander(f"{card_color} **{entry['name']}** ({entry['age']} yrs, {entry['sex']}) - {entry['visit_date']}", expanded=False):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**ID:** {entry['id']}")
                st.markdown(f"**Village:** {entry['village']}")
                st.markdown(f"**Complaint:** {entry['complaint']}")
                st.markdown(f"**Status:** {entry['status']}")

            with col2:
                # Only show view chart button for key patients (those with medical records)
                key_patients = ['NALU-CH-001', 'NALU-CH-002', 'NALU-CH-017', 'NALU-CH-023', 'NALU-CH-015', 'NALU-CH-022']
                if entry['id'] in key_patients:
                    if st.button("View Chart", key=f"chart_{entry['id']}"):
                        st.session_state['current_chart'] = entry['id']
                        st.session_state['unlocked_nalu_charts'].append(entry['id'])
                        st.rerun()

    st.markdown("---")

    # Display medical chart if selected
    if 'current_chart' in st.session_state and st.session_state.get('current_chart'):
        chart_id = st.session_state['current_chart']
        st.markdown(f"### Medical Record: {chart_id}")

        record = get_nalu_medical_record(chart_id)

        if record:
            with st.container():
                st.markdown("#### 📋 Patient Chart")
                for key, value in record.items():
                    st.markdown(f"**{key}:** {value}")

                if st.button("Close Chart", key="close_chart"):
                    st.session_state['current_chart'] = None
                    st.rerun()
        else:
            st.warning("No detailed medical record available for this patient.")

    st.markdown("---")

    # Summary statistics
    st.markdown("### Summary")
    total = len(register)
    referrals = sum(1 for e in register if 'Referred to Hospital' in e['status'])
    deaths = sum(1 for e in register if 'Died' in e['status'])
    suspected = sum(1 for e in register if any(term in e['complaint'].lower() for term in ['fever', 'seizure', 'stiff neck', 'shaking']))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Entries", total)
    col2.metric("Hospital Referrals", referrals)
    col3.metric("Deaths", deaths)
    col4.metric("Suspected Neuro Cases", suspected)


def view_descriptive_epi():
    """Interactive descriptive epidemiology dashboard - trainees must run analyses themselves."""
    st.header("Descriptive Epidemiology - Analysis Workspace")
    
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

        # Add outcome display column with sequelae info
        if 'has_sequelae' in cases.columns:
            download_df['outcome'] = cases.apply(
                lambda row: f"{row['outcome']} (with complications)" if row.get('has_sequelae') else row['outcome'],
                axis=1
            )

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
    st.markdown("### Run Analyses")
    st.caption("Select the analyses you want to perform. Results will appear below.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        run_person = st.checkbox("Person characteristics (age, sex, outcomes)")
        run_place = st.checkbox("Place analysis (cases by village, attack rates)")
    
    with col2:
        run_time = st.checkbox("Time analysis (epidemic curve)")
        run_crosstab = st.checkbox("Custom cross-tabulation")
    
    st.markdown("---")
    
    # PERSON ANALYSIS
    if run_person:
        st.markdown("## Person Analysis")
        
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
        st.markdown("## Place Analysis")
        
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
        st.markdown("## Time Analysis - Epidemic Curve")
        
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
        st.markdown("## Custom Cross-tabulation")
        
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
    st.header("Data & Study Design")

    truth = st.session_state.truth
    individuals = truth["individuals"]
    households = truth["households"]

    # Gate: don't let Day 2 artifacts be first interaction
    prereq_ok = bool(st.session_state.get("case_definition_written")) and bool(st.session_state.get("hypotheses_documented"))
    if not prereq_ok:
        st.info("Complete **Day 1** on the **Overview / Briefing** screen first (case definition + at least 1 hypothesis). Then return here for sampling and questionnaire upload.")
        return

    # Initialize wizard state
    if "wizard_step" not in st.session_state:
        st.session_state.wizard_step = 1
    if "exposure_domains" not in st.session_state.decisions:
        st.session_state.decisions["exposure_domains"] = {
            "human_demographics": False,
            "animal_exposure": False,
            "environmental_factors": False,
            "behavioral_factors": False
        }

    # Wizard progress indicator
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        status1 = "[●]" if st.session_state.wizard_step == 1 else "[✓]" if st.session_state.wizard_step > 1 else "[ ]"
        st.markdown(f"**{status1} Step 1: Methodology**")
    with col2:
        status2 = "[●]" if st.session_state.wizard_step == 2 else "[✓]" if st.session_state.wizard_step > 2 else "[ ]"
        st.markdown(f"**{status2} Step 2: Exposure Domains**")
    with col3:
        status3 = "[●]" if st.session_state.wizard_step == 3 else "[ ]"
        st.markdown(f"**{status3} Step 3: Review**")
    st.markdown("---")

    # ========================================
    # STEP 1: METHODOLOGY
    # ========================================
    if st.session_state.wizard_step == 1:
        st.markdown("## Step 1: Study Methodology")

        # Case definition (read-only)
        st.markdown("### Case Definition (from Day 1)")
        cd_text = st.session_state.decisions.get("case_definition_text", "").strip()
        if cd_text:
            st.text_area("Working case definition:", value=cd_text, height=100, disabled=True)
        else:
            st.warning("No case definition saved yet.")

        st.markdown("### Study Design")
        sd_type = st.radio("Choose a study design:", ["Case-control", "Retrospective cohort"], horizontal=True,
                          index=0 if st.session_state.decisions.get("study_design", {}).get("type") == "case_control" else 1)

        if sd_type == "Case-control":
            st.session_state.decisions["study_design"] = {"type": "case_control"}
            st.info("**Case-control study**: Compare exposures between cases (people with AES) and controls (people without AES).")
        else:
            st.session_state.decisions["study_design"] = {"type": "cohort"}
            st.info("**Retrospective cohort study**: Follow a group back in time to compare disease occurrence between exposed and unexposed individuals.")

        # Navigation
        if st.button("Next: Exposure Domains", type="primary"):
            st.session_state.wizard_step = 2
            st.rerun()

    # ========================================
    # STEP 2: EXPOSURE DOMAINS (ONE HEALTH)
    # ========================================
    elif st.session_state.wizard_step == 2:
        st.markdown("## Step 2: Exposure Domains (One Health Approach)")

        st.markdown("""
        Select the exposure domains you want to investigate in your study.
        A comprehensive One Health investigation covers human, animal, environmental, and behavioral factors.
        """)

        # Domain checkboxes
        domains = st.session_state.decisions["exposure_domains"]

        col1, col2 = st.columns(2)
        with col1:
            domains["human_demographics"] = st.checkbox(
                "Human Demographics & Health",
                value=domains["human_demographics"],
                help="Age, sex, occupation, clinical symptoms, vaccination history"
            )
            domains["animal_exposure"] = st.checkbox(
                "Animal Exposure",
                value=domains["animal_exposure"],
                help="Contact with pigs, poultry, livestock; proximity to animal populations"
            )
        with col2:
            domains["environmental_factors"] = st.checkbox(
                "Environmental Factors",
                value=domains["environmental_factors"],
                help="Water sources, rice paddies, mosquito exposure, housing conditions"
            )
            domains["behavioral_factors"] = st.checkbox(
                "Behavioral & Social Factors",
                value=domains["behavioral_factors"],
                help="Outdoor activities, protective measures, living conditions, water use"
            )

        # Coverage Meter
        selected_count = sum(domains.values())
        total_count = len(domains)
        coverage_pct = (selected_count / total_count) * 100 if total_count > 0 else 0

        st.markdown("### One Health Coverage Meter")
        st.progress(coverage_pct / 100)
        st.caption(f"Coverage: {selected_count}/{total_count} domains ({coverage_pct:.0f}%)")

        if coverage_pct < 50:
            st.warning("Consider selecting more domains for a comprehensive One Health investigation.")
        elif coverage_pct < 100:
            st.info("Good coverage. Consider adding remaining domains for completeness.")
        else:
            st.success("Excellent! Full One Health coverage across all domains.")

        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Previous: Methodology"):
                st.session_state.wizard_step = 1
                st.rerun()
        with col2:
            if st.button("Next: Review", type="primary"):
                st.session_state.wizard_step = 3
                st.rerun()

    # ========================================
    # STEP 3: REVIEW & PARTICIPANT SELECTION
    # ========================================
    elif st.session_state.wizard_step == 3:
        st.markdown("## Step 3: Review & Finalize")

        # Review selections
        with st.expander("Review Study Design", expanded=True):
            st.markdown("**Study Type:** " + st.session_state.decisions.get("study_design", {}).get("type", "Not selected").replace("_", "-").title())
            st.markdown("**Exposure Domains Selected:**")
            domains = st.session_state.decisions["exposure_domains"]
            for key, selected in domains.items():
                status = "[✓]" if selected else "[ ]"
                label = key.replace("_", " ").title()
                st.markdown(f"{status} {label}")

            # Coverage meter (compact)
            selected_count = sum(domains.values())
            total_count = len(domains)
            coverage_pct = (selected_count / total_count) * 100
            st.progress(coverage_pct / 100)
            st.caption(f"One Health Coverage: {coverage_pct:.0f}%")

        # Continue with existing participant selection logic
        st.markdown("---")

        # Navigation back button
        if st.button("Previous: Exposure Domains"):
            st.session_state.wizard_step = 2
            st.rerun()

        # -------------------------
        # Participant Selection (only in Step 3)
        # -------------------------
        st.markdown("### Participant Selection")

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
            if st.button("Refresh control candidates"):
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
    st.header(t("lab", default="Lab & Environment"))
    if int(st.session_state.get("current_day", 1)) < 4:
        st.info(t("locked_until_day", day=4))
        return
    refresh_lab_queue_for_day(int(st.session_state.get("current_day", 1)))
    
    # Resource display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Budget", f"${st.session_state.budget}")
    with col2:
        time_display = f":red[{st.session_state.time_remaining}h]" if st.session_state.time_remaining < 0 else f"{st.session_state.time_remaining}h"
        st.markdown(f"**⏱️ Time Remaining**  \n{time_display}")
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

            # Log the lab test event
            jl.log_event(
                event_type='lab_test',
                location_id=village_id,
                cost_time=costs['time'],
                cost_budget=costs['budget'],
                payload={
                    'sample_type': sample_type,
                    'test': test,
                    'source_description': source_description or "Unspecified source",
                    'sample_id': result.get('sample_id'),
                    'placed_day': st.session_state.current_day,
                    'ready_day': result.get('ready_day'),
                    'credits_used': costs['credits']
                }
            )

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
    st.header("Village Profiles - Sidero Valley")
    
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
    st.header("Spot Map - Geographic Distribution of Cases")
    
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


def generate_field_briefing(session_state) -> str:
    """
    Generate a professional Field Briefing Note in Markdown format.
    Pulls from decisions, interview_history, and other session data.
    """
    from datetime import datetime

    decisions = session_state.get("decisions", {})
    interview_history = session_state.get("interview_history", {})
    current_day = session_state.get("current_day", 1)

    # Build the markdown document
    md = []

    # Header
    md.append("# FIELD BRIEFING NOTE")
    md.append("## AES Outbreak Investigation - Sidero Valley")
    md.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    md.append(f"**Investigation Day:** {current_day} of 5")
    md.append("")
    md.append("---")
    md.append("")

    # Executive Summary
    md.append("## Executive Summary")
    md.append("")
    final_conclusion = decisions.get("final_conclusion", "Investigation in progress")
    md.append(f"**Conclusion:** {final_conclusion}")
    md.append("")

    # Case Definition
    md.append("## Case Definition")
    md.append("")
    case_def = decisions.get("case_definition_text", "Not yet defined")
    md.append(f"{case_def}")
    md.append("")

    # Hypotheses
    if session_state.get("initial_hypotheses"):
        md.append("## Initial Hypotheses")
        md.append("")
        for i, hyp in enumerate(session_state.get("initial_hypotheses", []), 1):
            md.append(f"{i}. {hyp}")
        md.append("")

    # Study Design
    md.append("## Study Design")
    md.append("")
    study_design = decisions.get("study_design", {})
    study_type = study_design.get("type", "Not selected")
    md.append(f"**Design:** {study_type.replace('_', '-').title()}")
    md.append("")

    # One Health Domains
    exposure_domains = decisions.get("exposure_domains", {})
    if exposure_domains:
        md.append("### One Health Domains Investigated")
        md.append("")
        for domain, selected in exposure_domains.items():
            status = "✓" if selected else "✗"
            label = domain.replace("_", " ").title()
            md.append(f"- [{status}] {label}")
        md.append("")

    # Sample Size
    sample_size = decisions.get("sample_size", {})
    if sample_size:
        md.append("### Sample Size")
        md.append("")
        md.append(f"- **Cases:** {sample_size.get('cases', 'N/A')}")
        md.append(f"- **Controls per case:** {sample_size.get('controls_per_case', 'N/A')}")
        md.append("")

    # Key Findings from Interviews
    if interview_history:
        md.append("## Key Findings from Field Interviews")
        md.append("")
        for npc_key, history in interview_history.items():
            if history:
                md.append(f"### Interview: {npc_key.replace('_', ' ').title()}")
                md.append("")
                # Show first few exchanges
                for i, msg in enumerate(history[:6], 1):
                    role = "Investigator" if msg["role"] == "user" else "Respondent"
                    content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                    md.append(f"**{role}:** {content}")
                    md.append("")
                if len(history) > 6:
                    md.append(f"*... {len(history) - 6} more exchanges*")
                    md.append("")

    # Final Diagnosis
    md.append("## Final Diagnosis")
    md.append("")
    final_dx = decisions.get("final_diagnosis", "Not yet determined")
    md.append(f"{final_dx}")
    md.append("")

    # Recommendations
    recommendations = decisions.get("recommendations", [])
    if recommendations:
        md.append("## Recommendations")
        md.append("")
        for i, rec in enumerate(recommendations, 1):
            md.append(f"{i}. {rec}")
        md.append("")

    # Laboratory Results
    if session_state.get("lab_results"):
        md.append("## Laboratory Results Summary")
        md.append("")
        md.append("Laboratory testing was conducted. Results are available in the investigation database.")
        md.append("")

    # Footer
    md.append("---")
    md.append("")
    md.append("*This field briefing note was generated by the FETP Outbreak Simulation System.*")
    md.append("*For official use in training exercises only.*")

    return "\n".join(md)


def view_interventions_and_outcome():
    st.header("Interventions & Outcome")

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

    st.markdown("### Final Conclusion")
    conclusion_options = [
        "Investigation ongoing - more data needed",
        "Japanese Encephalitis outbreak confirmed",
        "Other arboviral disease - further testing required",
        "Environmental exposure identified and mitigated",
        "Outbreak contained - surveillance continuing",
        "Custom conclusion"
    ]
    conclusion_choice = st.selectbox(
        "Select your final conclusion:",
        conclusion_options,
        index=conclusion_options.index(st.session_state.decisions.get("final_conclusion", conclusion_options[0])) if st.session_state.decisions.get("final_conclusion") in conclusion_options else 0
    )

    if conclusion_choice == "Custom conclusion":
        custom_conclusion = st.text_input(
            "Enter your custom conclusion:",
            value=st.session_state.decisions.get("final_conclusion", "") if st.session_state.decisions.get("final_conclusion") not in conclusion_options else ""
        )
        st.session_state.decisions["final_conclusion"] = custom_conclusion
    else:
        st.session_state.decisions["final_conclusion"] = conclusion_choice

    # Generate and export Field Briefing Note
    st.markdown("---")
    st.markdown("### Field Briefing Note")
    st.caption("Generate a professional briefing document summarizing your investigation.")

    if st.button("Generate Field Briefing Note", type="primary"):
        briefing_md = generate_field_briefing(st.session_state)
        st.session_state.field_briefing_note = briefing_md
        st.success("Field Briefing Note generated! Preview and download below.")

    if st.session_state.get("field_briefing_note"):
        with st.expander("Preview Field Briefing Note", expanded=True):
            st.markdown(st.session_state.field_briefing_note)

        st.download_button(
            label="Download Field Briefing Note (.md)",
            data=st.session_state.field_briefing_note,
            file_name=f"field_briefing_sidero_valley_day{st.session_state.current_day}.md",
            mime="text/markdown",
            help="Download the field briefing as a Markdown file"
        )

    st.markdown("---")
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
# ADVENTURE MODE VIEWS
# =========================

# Location coordinates for interactive map (0-100 scale, 0,0 is bottom-left)
MAP_LOCATIONS = {
    "Nalu Village": {"x": 35, "y": 45, "icon": "🌾", "desc": "Large rice-farming village. Pig cooperative nearby."},
    "Kabwe Village": {"x": 65, "y": 40, "icon": "🌿", "desc": "Medium village on higher ground. Mixed farming."},
    "Tamu Village": {"x": 15, "y": 85, "icon": "⛰️", "desc": "Remote upland community. Cassava farming."},
    "Mining Area": {"x": 20, "y": 15, "icon": "⛏️", "desc": "Recent expansion. New irrigation ponds."},
    "District Hospital": {"x": 85, "y": 80, "icon": "🏥", "desc": "AES patients admitted here. Lab available."},
    "District Office": {"x": 40, "y": 35, "icon": "🏛️", "desc": "Meet officials, veterinary and environmental officers."},
}


def render_interactive_map():
    """
    Render an interactive point-and-click map of Sidero Valley using Plotly.
    Clicking a location updates st.session_state.current_area and reruns.
    """
    # Load the background map image
    map_image_path = Path("assets/map_background.png")

    if not map_image_path.exists():
        st.error("Map background image not found at assets/map_background.png")
        return

    img = Image.open(map_image_path)
    img_width, img_height = img.size

    # Create figure with the background image
    fig = go.Figure()

    # Add the background image
    fig.add_layout_image(
        dict(
            source=img,
            xref="x",
            yref="y",
            x=0,
            y=100,
            sizex=100,
            sizey=100,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # Prepare data for scatter points, separating unlocked and locked locations
    unlocked_x = []
    unlocked_y = []
    unlocked_names = []
    unlocked_descriptions = []

    locked_x = []
    locked_y = []
    locked_names = []
    locked_descriptions = []

    for loc_name, loc_data in MAP_LOCATIONS.items():
        # Check if location is unlocked
        is_unlocked = True
        if is_location_unlocked and hasattr(st.session_state, 'game_state'):
            is_unlocked = is_location_unlocked(loc_name, st.session_state)

        if is_unlocked:
            unlocked_x.append(loc_data["x"])
            unlocked_y.append(loc_data["y"])
            unlocked_names.append(loc_name)
            unlocked_descriptions.append(f"{loc_data['icon']} {loc_name}<br>{loc_data['desc']}")
        else:
            locked_x.append(loc_data["x"])
            locked_y.append(loc_data["y"])
            locked_names.append(loc_name)
            locked_descriptions.append(f"🔒 {loc_name}<br>Location locked")

    # Add clickable scatter points for UNLOCKED locations with a subtle glow effect
    if unlocked_x:
        # First add a larger, semi-transparent marker for the glow/halo effect
        fig.add_trace(go.Scatter(
            x=unlocked_x,
            y=unlocked_y,
            mode='markers',
            marker=dict(
                size=28,
                color='rgba(255, 255, 255, 0.4)',
                line=dict(width=0)
            ),
            hoverinfo='skip',
            showlegend=False
        ))

        # Add the main marker points for unlocked locations
        fig.add_trace(go.Scatter(
            x=unlocked_x,
            y=unlocked_y,
            mode='markers',
            marker=dict(
                size=18,
                color='#FF6B35',  # Orange-red color for visibility
                line=dict(width=3, color='white'),
                symbol='circle'
            ),
            text=unlocked_descriptions,
            hovertemplate='%{text}<extra></extra>',
            customdata=unlocked_names,
            showlegend=False
        ))

    # Add LOCKED locations (greyed out, not clickable)
    if locked_x:
        fig.add_trace(go.Scatter(
            x=locked_x,
            y=locked_y,
            mode='markers',
            marker=dict(
                size=18,
                color='rgba(128, 128, 128, 0.5)',  # Grey for locked locations
                line=dict(width=3, color='rgba(200, 200, 200, 0.5)'),
                symbol='circle'
            ),
            text=locked_descriptions,
            hovertemplate='%{text}<extra></extra>',
            hoverinfo='text',
            showlegend=False
        ))

    # Add text labels with shadow effect for readability
    # First add shadow/outline (slightly offset dark text)
    for loc_name, loc_data in MAP_LOCATIONS.items():
        # Shadow offset positions
        for dx, dy in [(1, -1), (-1, -1), (1, 1), (-1, 1)]:
            fig.add_annotation(
                x=loc_data["x"] + dx * 0.5,
                y=loc_data["y"] + 6 + dy * 0.5,
                text=f"<b>{loc_name}</b>",
                showarrow=False,
                font=dict(size=12, color='rgba(0,0,0,0.8)'),
                xanchor='center',
                yanchor='bottom'
            )

    # Add the main white text labels on top
    for loc_name, loc_data in MAP_LOCATIONS.items():
        fig.add_annotation(
            x=loc_data["x"],
            y=loc_data["y"] + 6,
            text=f"<b>{loc_name}</b>",
            showarrow=False,
            font=dict(size=12, color='white'),
            xanchor='center',
            yanchor='bottom'
        )

    # Configure the layout to look like a clean map
    fig.update_layout(
        xaxis=dict(
            range=[0, 100],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[0, 100],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            showline=False,
            scaleanchor="x",
            scaleratio=1,
            fixedrange=True
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        dragmode=False,
        clickmode='event+select'
    )

    # Display the map with click handling
    st.markdown("### Click a location to travel there")

    # Use plotly_events for click detection
    selected_point = st.plotly_chart(
        fig,
        use_container_width=True,
        key="interactive_map",
        on_select="rerun",
        selection_mode="points"
    )

    # Handle click/selection events (only for unlocked locations)
    if selected_point and selected_point.selection and selected_point.selection.points:
        point_data = selected_point.selection.points[0]
        point_index = point_data.get("point_index", None)
        if point_index is not None and 0 <= point_index < len(unlocked_names):
            selected_location = unlocked_names[point_index]
            # Check if location is unlocked before allowing navigation
            is_unlocked = True
            if is_location_unlocked and hasattr(st.session_state, 'game_state'):
                is_unlocked = is_location_unlocked(selected_location, st.session_state)

            if is_unlocked:
                st.session_state.current_area = selected_location
                st.session_state.current_view = "area"
                st.rerun()
            else:
                st.warning("🔒 This location is locked. Complete previous objectives to unlock.")

    # Show location legend/quick reference below the map
    st.markdown("---")
    st.markdown("**Locations:**")
    cols = st.columns(3)
    for i, (loc_name, loc_data) in enumerate(MAP_LOCATIONS.items()):
        with cols[i % 3]:
            # Check if location is unlocked
            is_unlocked = True
            if is_location_unlocked and hasattr(st.session_state, 'game_state'):
                is_unlocked = is_location_unlocked(loc_name, st.session_state)

            button_label = f"{loc_data['icon']} {loc_name}" if is_unlocked else f"🔒 {loc_name}"
            button_disabled = not is_unlocked

            if st.button(button_label, key=f"map_btn_{loc_name}", use_container_width=True, disabled=button_disabled):
                if is_unlocked:
                    st.session_state.current_area = loc_name
                    st.session_state.current_view = "area"
                    st.rerun()
            st.caption(loc_data['desc'])


def view_travel_map():
    """Main travel map showing all areas and allowing navigation."""
    st.title("Sidero Valley - Investigation Map")

    # Serious Mode: Show outbreak summary metrics prominently on Day 1
    if st.session_state.current_day == 1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #d32f2f 0%, #c62828 100%);
                    padding: 1.5rem;
                    border-radius: 10px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    margin-bottom: 1.5rem;'>
            <h3 style='color: white; margin: 0 0 1rem 0; text-align: center;'>🚨 Outbreak Summary</h3>
            <div style='display: flex; justify-content: space-around; text-align: center;'>
                <div>
                    <div style='font-size: 2.5rem; color: #ffeb3b; font-weight: bold;'>2</div>
                    <div style='color: white; font-size: 1.1rem;'>Severe Cases</div>
                </div>
                <div>
                    <div style='font-size: 2.5rem; color: #ffeb3b; font-weight: bold;'>1</div>
                    <div style='color: white; font-size: 1.1rem;'>Death</div>
                </div>
                <div>
                    <div style='font-size: 2.5rem; color: #ffeb3b; font-weight: bold;'>Day 1</div>
                    <div style='color: white; font-size: 1.1rem;'>Investigation Start</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Show current status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(f"{t('day')}", f"{st.session_state.current_day} / 5")
    with col2:
        time_display = f":red[{st.session_state.time_remaining}h]" if st.session_state.time_remaining < 0 else f"{st.session_state.time_remaining}h"
        st.markdown(f"**{t('time_remaining')}**  \n{time_display}")
    with col3:
        st.metric(f"{t('budget')}", f"${st.session_state.budget}")
    with col4:
        st.metric(f"{t('lab_credits')}", st.session_state.lab_credits)

    st.markdown("---")

    # Day briefing
    if st.session_state.current_day == 1:
        with st.expander("Day 1 Briefing - Situation Assessment", expanded=True):
            st.markdown("""
            **Your tasks today:**
            - Visit the **District Hospital** to meet Dr. Tran and review cases
            - Travel to **Nalu Village** to interview residents and review clinic records
            - Document your initial hypotheses about the outbreak source

            *Click on a location below to travel there.*
            """)

    # About Sidero District
    with st.expander("ℹ️ About Sidero District"):
        st.markdown("""
        Sidero District is a rural agricultural region known for extensive rice farming.
        Recent irrigation projects have expanded the paddy fields closer to residential areas.
        The population is approximately 15,000, spread across 3 main villages (Nalu, Tamu, Kabwe).
        Livestock farming (pigs, ducks) is common in backyard settings.
        """)

    # Render the interactive satellite map for destination selection
    render_interactive_map()

    # Quick access to data views
    st.markdown("### Investigation Tools")
    cols = st.columns(4)
    with cols[0]:
        if st.button("Epi Curve & Line List", use_container_width=True):
            st.session_state.current_view = "descriptive"
            st.rerun()
    with cols[1]:
        if st.button("Spot Map", use_container_width=True):
            st.session_state.current_view = "spotmap"
            st.rerun()
    with cols[2]:
        if st.button("Study Design", use_container_width=True):
            st.session_state.current_view = "study"
            st.rerun()
    with cols[3]:
        if st.button("Final Report", use_container_width=True):
            st.session_state.current_view = "outcome"
            st.rerun()


# =========================
# UI/UX HELPER FUNCTIONS
# =========================

def get_location_status(loc_key: str) -> dict:
    """Get the status of a location (visited, actions completed, etc.)."""
    status = {
        "visited": False,
        "clinic_reviewed": False,
        "environment_inspected": False,
        "samples_collected": False,
        "npcs_interviewed": [],
    }

    # Check if location was visited (if they went to this location view)
    visited_locations = st.session_state.get("visited_locations", set())
    status["visited"] = loc_key in visited_locations

    # Check specific action completions
    if loc_key in ["nalu_health_center"]:
        status["clinic_reviewed"] = st.session_state.get("clinic_records_reviewed", False)

    # Check environment inspections
    env_findings = st.session_state.get("environment_findings", [])
    for finding in env_findings:
        if finding.get("location") == loc_key:
            status["environment_inspected"] = True
            break

    # Check samples collected at this location
    lab_samples = st.session_state.get("lab_samples_submitted", [])
    for sample in lab_samples:
        if sample.get("location") == loc_key:
            status["samples_collected"] = True
            break

    # Check NPCs interviewed at this location
    loc = LOCATIONS.get(loc_key, {})
    for npc_key in loc.get("npcs", []):
        if npc_key in st.session_state.get("interview_history", {}):
            status["npcs_interviewed"].append(npc_key)

    return status


def render_breadcrumb(area: str = None, location: str = None):
    """Render a breadcrumb navigation bar."""
    area_meta = AREA_METADATA.get(area, {}) if area else {}
    area_icon = area_meta.get("icon", "📍")

    loc_data = LOCATIONS.get(location, {}) if location else {}
    loc_icon = loc_data.get("icon", "📍")
    loc_name = loc_data.get("name", location) if location else None

    # Build breadcrumb elements
    crumbs = []

    # Map is always first
    crumbs.append(("🗺️ Map", "map", None))

    if area:
        crumbs.append((f"{area_icon} {area}", "area", area))

    if location and loc_name:
        crumbs.append((f"{loc_icon} {loc_name}", "location", location))

    # Render breadcrumb with clickable buttons
    cols = st.columns(len(crumbs) * 2 - 1)
    col_idx = 0

    for i, (label, view_type, data) in enumerate(crumbs):
        with cols[col_idx]:
            # Don't make the last crumb clickable (it's current location)
            if i < len(crumbs) - 1:
                if st.button(label, key=f"breadcrumb_{view_type}_{i}", use_container_width=True):
                    if view_type == "map":
                        st.session_state.current_area = None
                        st.session_state.current_location = None
                        st.session_state.current_view = "map"
                    elif view_type == "area":
                        st.session_state.current_area = data
                        st.session_state.current_location = None
                        st.session_state.current_view = "area"
                    st.rerun()
            else:
                st.markdown(f"**{label}**")
        col_idx += 1

        # Add separator
        if i < len(crumbs) - 1:
            with cols[col_idx]:
                st.markdown("<div style='text-align: center; padding-top: 8px;'>›</div>", unsafe_allow_html=True)
            col_idx += 1


def travel_with_animation(destination_name: str, travel_time: float = 0.5):
    """Show a travel animation/spinner when moving to a new location."""
    with st.spinner(f"🚶 Traveling to {destination_name}..."):
        time.sleep(min(travel_time, 1.0))  # Cap animation at 1 second


def render_location_card(loc_key: str, loc: dict, npcs_here: list, npc_truth: dict, col_key: str = ""):
    """Render a styled card for a sub-location with status badges and NPC avatars."""
    from pathlib import Path

    loc_name = loc.get("name", loc_key)
    loc_icon = loc.get("icon", "📍")
    loc_desc = loc.get("description", "")
    travel_time = loc.get("travel_time", 0.5)

    # Get location status
    status = get_location_status(loc_key)

    # Build status badge HTML
    status_badges = []
    if status["clinic_reviewed"] and loc_key == "nalu_health_center":
        status_badges.append('<span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; margin-left: 5px;">✅ Records Reviewed</span>')
    if status["environment_inspected"]:
        status_badges.append('<span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; margin-left: 5px;">🔍 Inspected</span>')
    if status["samples_collected"]:
        status_badges.append('<span style="background: #8b5cf6; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; margin-left: 5px;">🧪 Sampled</span>')
    if status["npcs_interviewed"]:
        status_badges.append('<span style="background: #f59e0b; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; margin-left: 5px;">💬 Interviewed</span>')

    badge_html = " ".join(status_badges) if status_badges else ""

    # Card container with border styling
    st.markdown(f"""
    <div style="border: 1px solid #e5e7eb; border-radius: 12px; padding: 15px; margin-bottom: 10px; background: white;">
        <h4 style="margin: 0 0 5px 0;">{loc_icon} {loc_name}</h4>
        <div style="margin-bottom: 8px;">{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # Thumbnail image
    image_path = loc.get("image_thumb") or loc.get("image_path")
    if image_path:
        path = Path(image_path)
        if not path.suffix:
            for ext in ['.png', '.jpg', '.jpeg']:
                test_path = Path(str(path) + ext)
                if test_path.exists():
                    st.image(str(test_path), use_container_width=True)
                    break
        elif path.exists():
            st.image(str(path), use_container_width=True)
    else:
        # Placeholder
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    border-radius: 8px; padding: 30px; text-align: center; margin-bottom: 10px;">
            <span style="font-size: 2.5em;">{loc_icon}</span>
        </div>
        """, unsafe_allow_html=True)

    # Short description removed (caption)

    # === NPCs Present with Avatar Overlay ===
    if npcs_here:
        # Create avatar row with overlapping style
        avatar_html = '<div style="display: flex; margin: 8px 0;">'
        for idx, (npc_key, npc) in enumerate(npcs_here):
            avatar_path = npc.get("image_path")
            npc_name = npc.get("name", "Unknown")

            # Check if avatar image exists
            if avatar_path and Path(avatar_path).exists():
                # We'll render this with st.image below
                pass
            else:
                avatar_emoji = npc.get("avatar", "👤")
                avatar_html += f'''
                <div style="width: 36px; height: 36px; border-radius: 50%; background: #e5e7eb;
                            display: flex; align-items: center; justify-content: center;
                            margin-left: {-10 if idx > 0 else 0}px; border: 2px solid white;
                            font-size: 1.2em;" title="{npc_name}">
                    {avatar_emoji}
                </div>
                '''
        avatar_html += '</div>'

        st.markdown(avatar_html, unsafe_allow_html=True)

        # Show NPC names removed (caption)

    # Travel time removed (caption)

    # Go to button
    if st.button(f"Go to {loc_name}", key=f"go_{col_key}_{loc_key}", use_container_width=True):
        # Check if enough time
        if st.session_state.time_remaining >= travel_time:
            # Show travel animation
            travel_with_animation(loc_name, travel_time)

            spend_time(travel_time, f"Travel to {loc_name}")

            # Mark location as visited
            if "visited_locations" not in st.session_state:
                st.session_state.visited_locations = set()
            st.session_state.visited_locations.add(loc_key)

            # Clear chat history when changing locations
            if st.session_state.get("current_npc"):
                npc_to_clear = st.session_state.current_npc
                if npc_to_clear in st.session_state.interview_history:
                    st.session_state.interview_history[npc_to_clear] = []
                st.session_state.current_npc = None

            st.session_state.current_location = loc_key
            st.session_state.current_view = "location"
            st.rerun()
        else:
            st.error(f"Not enough time! Need {travel_time}h")

    st.markdown("---")


def render_area_hero_image(area: str) -> bool:
    """Render the hero/exterior image for an area if available."""
    area_meta = AREA_METADATA.get(area, {})
    image_path = area_meta.get("image_exterior")

    if not image_path:
        return False

    from pathlib import Path
    path = Path(image_path)

    # Try with common extensions if no extension
    if not path.suffix:
        for ext in ['.png', '.jpg', '.jpeg']:
            test_path = Path(str(path) + ext)
            if test_path.exists():
                st.image(str(test_path), use_container_width=True)
                return True
    elif path.exists():
        st.image(str(path), use_container_width=True)
        return True

    return False


def render_location_thumbnail(loc_key: str, width: int = 200) -> bool:
    """Render a thumbnail image for a sub-location if available."""
    loc = LOCATIONS.get(loc_key, {})

    # Try thumbnail first, then fall back to main image
    image_path = loc.get("image_thumb") or loc.get("image_path")

    if not image_path:
        return False

    from pathlib import Path
    path = Path(image_path)

    # Try with common extensions if no extension
    if not path.suffix:
        for ext in ['.png', '.jpg', '.jpeg']:
            test_path = Path(str(path) + ext)
            if test_path.exists():
                st.image(str(test_path), use_container_width=True)
                return True
    elif path.exists():
        st.image(str(path), use_container_width=True)
        return True

    return False


def view_area_visual(area: str):
    """Render an area with immersive visual grid layout.

    Features:
    - Breadcrumb navigation at top
    - Hero exterior image
    - 3-column grid of sub-location cards with status badges
    - Each card shows: thumbnail, name, status badges, NPCs present, Go button
    """
    from pathlib import Path

    area_meta = AREA_METADATA.get(area, {})
    area_icon = area_meta.get("icon", "📍")

    # === BREADCRUMB NAVIGATION ===
    render_breadcrumb(area=area)

    st.markdown("---")

    st.title(f"{area_icon} {area}")

    # === HERO IMAGE ===
    if not render_area_hero_image(area):
        # Show placeholder if no image available
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px; padding: 60px; text-align: center; color: white;
                    margin-bottom: 20px;">
            <h1 style="font-size: 3em; margin: 0;">{area_icon}</h1>
            <h2 style="margin: 10px 0 0 0;">{area}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Area description
    description = area_meta.get("description", "")
    if description:
        st.markdown(f"*{description}*")

    st.markdown("---")
    st.markdown("### 🚪 Locations to Explore")

    # Get locations in this area
    location_keys = AREA_LOCATIONS.get(area, [])

    if not location_keys:
        st.warning("No locations available in this area.")
        return

    # Get NPC truth data
    npc_truth = st.session_state.truth.get("npc_truth", {})

    # === SUB-LOCATION GRID (3 columns) ===
    num_cols = min(3, len(location_keys))  # Use up to 3 columns
    cols = st.columns(num_cols)

    for i, loc_key in enumerate(location_keys):
        loc = LOCATIONS.get(loc_key, {})

        # Find NPCs at this location
        npcs_here = []
        for npc_key in loc.get("npcs", []):
            if npc_key in st.session_state.npcs_unlocked and npc_key in npc_truth:
                npcs_here.append((npc_key, npc_truth[npc_key]))

        with cols[i % num_cols]:
            # Use the consolidated card rendering function
            render_location_card(loc_key, loc, npcs_here, npc_truth, col_key=f"area_{area}")


def view_area_map(area: str):
    """[DEPRECATED] Show sub-locations within an area.

    Note: This function has been deprecated in favor of view_area_visual()
    which provides a more immersive experience with hero images, status badges,
    and improved NPC avatar display. Kept for backwards compatibility.
    """
    st.title(f"{area}")

    # Return to main map button
    if st.button("Return to Main Map", key="return_to_main"):
        st.session_state.current_area = None
        st.session_state.current_view = "map"
        st.rerun()

    st.markdown("---")

    # Get locations in this area
    location_keys = AREA_LOCATIONS.get(area, [])

    if not location_keys:
        st.warning("No locations available in this area.")
        return

    # Show description for the area
    if area == "Nalu Village":
        st.markdown("""
        **Nalu Village** is the largest settlement in Sidero Valley. The economy centers on
        rice cultivation and pig farming. Most AES cases come from here.
        """)
    elif area == "Kabwe Village":
        st.markdown("""
        **Kabwe Village** is located 3km northeast of Nalu on higher ground. Children walk
        through rice paddies to attend school in Nalu.
        """)
    elif area == "Tamu Village":
        st.markdown("""
        **Tamu Village** is a smaller, more remote community in the foothills. Upland farming
        with less standing water.
        """)
    elif area == "District Hospital":
        st.markdown("""
        **District Hospital** is where the AES cases have been admitted. Dr. Tran oversees
        patient care and the laboratory can process some samples.
        """)
    elif area == "District Office":
        st.markdown("""
        **District Office** houses the public health, veterinary, and environmental health
        teams. Key officials work from here.
        """)

    st.markdown("### Locations to Visit")

    # Display locations in grid
    cols = st.columns(2)

    for i, loc_key in enumerate(location_keys):
        loc = LOCATIONS.get(loc_key, {})

        # Check if location has unlocked NPCs
        npcs_here = []
        npc_truth = st.session_state.truth.get("npc_truth", {})
        for npc_key in loc.get("npcs", []):
            if npc_key in st.session_state.npcs_unlocked and npc_key in npc_truth:
                npcs_here.append(npc_truth[npc_key])

        with cols[i % 2]:
            with st.container():
                st.markdown(f"**{loc.get('name', loc_key)}**")

                # Show NPCs available removed (caption)

                # Show available actions
                actions = loc.get("available_actions", [])
                if actions:
                    # Actions removed (caption)
                    pass

                travel_time = loc.get("travel_time", 0.5)

                if st.button(f"Go to {loc.get('name', loc_key)}", key=f"loc_{loc_key}", use_container_width=True):
                    # Check if enough time
                    if st.session_state.time_remaining >= travel_time:
                        spend_time(travel_time, f"Travel to {loc.get('name', loc_key)}")

                        # Clear chat history when changing locations
                        if st.session_state.get("current_npc"):
                            npc_to_clear = st.session_state.current_npc
                            if npc_to_clear in st.session_state.interview_history:
                                st.session_state.interview_history[npc_to_clear] = []
                            st.session_state.current_npc = None

                        st.session_state.current_location = loc_key
                        st.session_state.current_view = "location"
                        st.rerun()
                    else:
                        st.error(f"Not enough time to travel (need {travel_time}h)")

                st.markdown("---")


def render_location_image(loc_key: str):
    """Render the image for a location if available."""
    loc = LOCATIONS.get(loc_key, {})
    image_path = loc.get("image_path")

    if not image_path:
        return False

    # Try to load the image
    from pathlib import Path

    # Handle paths with or without extension
    path = Path(image_path)

    # Try with common extensions if no extension
    if not path.suffix:
        for ext in ['.png', '.jpg', '.jpeg']:
            test_path = Path(str(path) + ext)
            if test_path.exists():
                st.image(str(test_path), use_container_width=True)
                return True
    elif path.exists():
        st.image(str(path), use_container_width=True)
        return True

    return False


def view_location(loc_key: str):
    """Render a specific location with NPCs and actions."""
    loc = LOCATIONS.get(loc_key, {})

    if not loc:
        st.error("Location not found!")
        st.session_state.current_location = None
        st.session_state.current_view = "map"
        return

    area = loc.get('area', 'Unknown Area')
    loc_icon = loc.get('icon', '📍')

    # === BREADCRUMB NAVIGATION ===
    render_breadcrumb(area=area, location=loc_key)

    st.markdown("---")

    # Header with location name
    st.title(f"{loc_icon} {loc.get('name', loc_key)}")

    # Layout: Image on left, description and NPCs on right
    col1, col2 = st.columns([1, 2])

    with col1:
        # Try to render location image
        if not render_location_image(loc_key):
            # Show placeholder SVG
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 10px; padding: 40px; text-align: center; color: white;">
                <h2>📍</h2>
                <p>Location Image</p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # NPCs at this location
        npc_truth = st.session_state.truth.get("npc_truth", {})
        npcs_here = []

        for npc_key in loc.get("npcs", []):
            if npc_key in st.session_state.npcs_unlocked and npc_key in npc_truth:
                npcs_here.append((npc_key, npc_truth[npc_key]))

        if npcs_here:
            st.markdown("### 👥 People Here")
            for npc_key, npc in npcs_here:
                interviewed = npc_key in st.session_state.interview_history

                with st.container():
                    cols = st.columns([1, 3, 1])
                    with cols[0]:
                        # Show avatar image if available, otherwise emoji
                        avatar_path = npc.get("image_path")
                        if avatar_path and Path(avatar_path).exists():
                            st.image(avatar_path, width=60)
                        else:
                            st.markdown(f"## {npc['avatar']}")
                    with cols[1]:
                        status = "✓ Interviewed" if interviewed else ""
                        st.markdown(f"**{npc['name']}** {status}")
                    with cols[2]:
                        btn_label = "Continue Chat" if interviewed else "Talk"
                        if st.button(btn_label, key=f"talk_{npc_key}"):
                            st.session_state.current_npc = npc_key
                            st.session_state.interview_history.setdefault(npc_key, [])
                            st.rerun()
        else:
            st.info("No one is here to talk to right now.")

        st.markdown("---")

        # Available actions
        st.markdown("### Available Actions")
        actions = loc.get("available_actions", [])

        if actions:
            render_location_actions(loc_key, actions)

    # If we have an active NPC conversation, show it
    if st.session_state.current_npc:
        npc_key = st.session_state.current_npc
        if npc_key in npc_truth:
            st.markdown("---")
            render_npc_chat(npc_key, npc_truth[npc_key])

    # Handle action modals
    action_modal = st.session_state.get("action_modal")
    if action_modal:
        st.markdown("---")
        if action_modal == "ward_registry":
            render_ward_registry_modal()
        elif action_modal == "hospital_charts":
            render_hospital_charts_modal()
        elif action_modal == "deep_dive_charts":
            render_deep_dive_charts_modal()


def render_ward_registry_modal():
    """Render Ward Registry modal with case finding functionality."""
    st.subheader("📋 District Hospital - Ward Registry (Last 30 Days)")

    # Check permission
    if not st.session_state.get('tran_permission', False):
        st.error("⛔ Access Denied: You need Dr. Chen's permission to access hospital records.")
        st.info("💡 **Hint:** Talk to Dr. Chen and ask for 'permission' to access medical records and the laboratory.")
        if st.button("Close", key="close_ward_registry"):
            st.session_state.action_modal = None
            st.rerun()
        return

    # Generate or retrieve ward registry
    if 'ward_registry' not in st.session_state:
        st.session_state.ward_registry = jl.generate_ward_registry()

    registry = st.session_state.ward_registry

    st.markdown("""
    This is the hospital's admission log for the past 30 days. Review the Chief Complaints and
    select patients whose charts you want to examine more closely in the Office.
    """)

    if st.button("✖ Close Registry", key="close_ward_registry"):
        st.session_state.action_modal = None
        st.rerun()

    st.markdown("---")

    # Initialize unlocked charts if not exists
    if 'unlocked_charts' not in st.session_state:
        # Start with the 2 known cases
        st.session_state.unlocked_charts = ['WARD-001', 'WARD-002']

    # Display registry with checkboxes
    st.markdown("### Select Patients to Pull Charts")

    # Create a selection state
    if 'selected_charts' not in st.session_state:
        st.session_state.selected_charts = []

    # Show table with selection
    selected_ids = st.multiselect(
        "Select patients to pull charts for:",
        options=registry['Patient_ID'].tolist(),
        default=st.session_state.selected_charts,
        format_func=lambda x: f"{x} - {registry[registry['Patient_ID']==x]['Chief_Complaint'].iloc[0][:50]}...",
        key="ward_registry_select"
    )

    # Display registry table
    st.dataframe(
        registry[['Patient_ID', 'Admission_Date', 'Age', 'Sex', 'Chief_Complaint', 'Diagnosis', 'Outcome']],
        use_container_width=True,
        height=400
    )

    # Pull charts button
    if len(selected_ids) > 0:
        if st.button(f"📄 Pull {len(selected_ids)} Charts", key="pull_charts", type="primary"):
            # Add selected IDs to unlocked charts
            for pid in selected_ids:
                if pid not in st.session_state.unlocked_charts:
                    st.session_state.unlocked_charts.append(pid)

            st.session_state.selected_charts = []
            st.success(f"✅ {len(selected_ids)} charts retrieved. View them in the Hospital Office.")
            st.rerun()


def render_hospital_charts_modal():
    """Render Hospital Charts modal showing unlocked patient charts."""
    st.subheader("📄 District Hospital - Medical Charts")

    # Check permission
    if not st.session_state.get('tran_permission', False):
        st.error("⛔ Access Denied: You need Dr. Chen's permission to access hospital records.")
        st.info("💡 **Hint:** Talk to Dr. Chen and ask for 'permission' to access medical records and the laboratory.")
        if st.button("Close", key="close_charts"):
            st.session_state.action_modal = None
            st.rerun()
        return

    if st.button("✖ Close Charts", key="close_charts"):
        st.session_state.action_modal = None
        st.rerun()

    st.markdown("---")

    # Get unlocked charts
    unlocked_charts = st.session_state.get('unlocked_charts', ['WARD-001', 'WARD-002'])

    if len(unlocked_charts) == 0:
        st.info("No charts have been pulled yet. Visit the Ward Registry to select patient charts.")
        return

    st.markdown(f"**Available Charts:** {len(unlocked_charts)}")
    st.caption("These charts contain ONLY clinical data - no exposure history or risk factors.")

    # Chart selector
    selected_chart = st.selectbox(
        "Select a patient chart to review:",
        options=unlocked_charts,
        format_func=lambda x: f"Patient {x}",
        key="chart_selector"
    )

    if selected_chart:
        # Get chart text
        chart_text = jl.get_paper_chart_text(selected_chart)

        # Display chart in monospace font
        st.markdown(f"```\n{chart_text}\n```")


def render_deep_dive_charts_modal():
    """Render Deep Dive Charts modal (placeholder for future implementation)."""
    st.subheader("📊 Hospital Deep-Dive Charts")

    if st.button("✖ Close", key="close_deep_dive"):
        st.session_state.action_modal = None
        st.rerun()

    st.markdown("---")

    st.info("🚧 Deep-dive charts feature coming soon. This will show aggregate patterns across hospitalized cases.")


def render_location_actions(loc_key: str, actions: list):
    """Render action buttons for a location."""

    action_configs = {
        "review_clinic_records": {
            "label": "Review Clinic Records",
            "cost_time": TIME_COSTS.get("clinic_records_review", 2.0),
            "cost_budget": 0,
            "handler": "case_finding",
        },
        "view_hospital_records": {
            "label": "View Hospital Records",
            "cost_time": 0.5,
            "cost_budget": 0,
            "handler": "hospital_records",
        },
        "collect_pig_sample": {
            "label": "Collect Pig Serum Sample",
            "cost_time": TIME_COSTS.get("sample_collection", 1.0),
            "cost_budget": BUDGET_COSTS.get("lab_sample_animal", 35),
            "handler": "lab_sample",
            "sample_type": "pig_serum",
        },
        "collect_mosquito_sample": {
            "label": "Set Mosquito Trap",
            "cost_time": TIME_COSTS.get("sample_collection", 1.0),
            "cost_budget": BUDGET_COSTS.get("lab_sample_mosquito", 40),
            "handler": "lab_sample",
            "sample_type": "mosquito_pool",
        },
        "collect_water_sample": {
            "label": "Collect Water Sample",
            "cost_time": 0.5,
            "cost_budget": 20,
            "handler": "lab_sample",
            "sample_type": "water",
        },
        "collect_household_water_sample": {
            "label": "Collect Water Sample from Household Jar",
            "cost_time": 0.5,
            "cost_budget": 20,
            "handler": "lab_sample",
            "sample_type": "household_water",
        },
        "inspect_environment": {
            "label": "Environmental Inspection",
            "cost_time": TIME_COSTS.get("environmental_inspection", 2.0),
            "cost_budget": 0,
            "handler": "environment",
        },
        "view_village_profile": {
            "label": "View Village Profile",
            "cost_time": 0,
            "cost_budget": 0,
            "handler": "village_profile",
        },
        "collect_csf_sample": {
            "label": "Request CSF Sample",
            "cost_time": 0.5,
            "cost_budget": BUDGET_COSTS.get("lab_sample_human", 25),
            "handler": "lab_sample",
            "sample_type": "csf",
        },
        "collect_serum_sample": {
            "label": "Request Patient Serum",
            "cost_time": 0.5,
            "cost_budget": BUDGET_COSTS.get("lab_sample_human", 25),
            "handler": "lab_sample",
            "sample_type": "human_serum",
        },
        "view_lab_results": {
            "label": "View Lab Results",
            "cost_time": 0,
            "cost_budget": 0,
            "handler": "lab_results",
        },
        "submit_lab_samples": {
            "label": "Submit Samples for Testing",
            "cost_time": 0.5,
            "cost_budget": 0,
            "handler": "lab_submit",
        },
        "request_data": {
            "label": "Request Official Data",
            "cost_time": 0.5,
            "cost_budget": 0,
            "handler": "request_data",
        },
        "plan_interventions": {
            "label": "📝 Plan Interventions",
            "cost_time": 0,
            "cost_budget": 0,
            "handler": "interventions",
        },
        "view_nalu_child_register": {
            "label": "Review Nalu Child Register",
            "cost_time": 1.0,
            "cost_budget": 0,
            "handler": "nalu_child_register",
        },
        "view_ward_registry": {
            "label": "📋 View Ward Registry (30 days)",
            "cost_time": 1.0,
            "cost_budget": 0,
            "handler": "ward_registry",
        },
        "review_hospital_records": {
            "label": "📄 Review Medical Charts",
            "cost_time": 0.5,
            "cost_budget": 0,
            "handler": "hospital_charts",
        },
        "view_deep_dive_charts": {
            "label": "📊 View Deep-Dive Charts",
            "cost_time": 0.5,
            "cost_budget": 0,
            "handler": "deep_dive_charts",
        },
        "review_attendance_records": {
            "label": "📚 Review Attendance Records",
            "cost_time": 1.0,
            "cost_budget": 0,
            "handler": "attendance",
        },
        "review_tamu_records": {
            "label": "📝 Review Tamu Records",
            "cost_time": 0.5,
            "cost_budget": 0,
            "handler": "tamu_records",
        },
    }

    for action in actions:
        config = action_configs.get(action, {})
        if not config:
            continue

        label = config.get("label", action)
        cost_time = config.get("cost_time", 0)
        cost_budget = config.get("cost_budget", 0)

        # Show cost
        cost_str = format_resource_cost(cost_time, cost_budget)

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(label, key=f"action_{action}_{loc_key}", use_container_width=True):
                # Check resources
                can_proceed, msg = check_resources(cost_time, cost_budget)
                if can_proceed:
                    if cost_time > 0:
                        spend_time(cost_time, label)
                    if cost_budget > 0:
                        spend_budget(cost_budget, label)

                    # Execute handler
                    execute_location_action(action, config, loc_key)
                else:
                    st.error(msg)
        with col2:
            st.caption(cost_str)


def execute_location_action(action: str, config: dict, loc_key: str):
    """Execute a location action and show the appropriate UI."""
    handler = config.get("handler", "")

    if handler == "case_finding":
        st.session_state.current_view = "casefinding"
        st.rerun()
    elif handler == "hospital_records":
        st.session_state.action_modal = "hospital_records"
        st.rerun()
    elif handler == "lab_sample":
        sample_type = config.get("sample_type", "unknown")
        st.session_state.action_modal = f"lab_sample_{sample_type}"
        st.rerun()
    elif handler == "environment":
        st.session_state.action_modal = "environment_inspection"
        st.rerun()
    elif handler == "village_profile":
        st.session_state.current_view = "villages"
        st.rerun()
    elif handler == "lab_results":
        st.session_state.current_view = "lab"
        st.rerun()
    elif handler == "lab_submit":
        st.session_state.current_view = "lab"
        st.rerun()
    elif handler == "interventions":
        st.session_state.current_view = "outcome"
        st.rerun()
    elif handler == "request_data":
        st.session_state.current_view = "descriptive"
        st.rerun()
    elif handler == "nalu_child_register":
        # Check if nurse permits access
        from je_logic import check_nurse_rapport
        if check_nurse_rapport(st.session_state):
            st.session_state.current_view = "nalu_child_register"
            st.rerun()
        else:
            st.error("🔒 The nurse refuses access. 'Build better rapport first - show some respect for my time.'")
            st.rerun()
    elif handler == "ward_registry":
        st.session_state.action_modal = "ward_registry"
        st.rerun()
    elif handler == "hospital_charts":
        st.session_state.action_modal = "hospital_charts"
        st.rerun()
    elif handler == "deep_dive_charts":
        st.session_state.action_modal = "deep_dive_charts"
        st.rerun()
    elif handler == "attendance":
        st.session_state.action_modal = "attendance"
        st.rerun()
    elif handler == "tamu_records":
        st.session_state.action_modal = "tamu_records"
        st.rerun()


def render_npc_chat(npc_key: str, npc: dict):
    """Render chat interface for an NPC at current location."""
    st.markdown(f"### Talking to {npc['avatar']} {npc['name']}")
    st.caption(f"*{npc['role']}*")

    # End conversation button
    if st.button("End Conversation", key="end_chat"):
        st.session_state.current_npc = None
        st.rerun()

    # Show conversation history
    history = st.session_state.interview_history.get(npc_key, [])

    for msg in history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar=get_npc_avatar(npc)):
                st.write(msg["content"])

    # Special handling for Nurse Mai (nurse_joy) - Rapport mechanic
    if npc_key == "nurse_joy":
        # Import je_logic for rapport functions
        from je_logic import update_nurse_rapport, check_nurse_rapport

        # Initialize rapport and show current status
        if 'nurse_rapport' not in st.session_state:
            st.session_state['nurse_rapport'] = 0
        if 'nurse_initial_dialogue_shown' not in st.session_state:
            st.session_state['nurse_initial_dialogue_shown'] = False

        # Show rapport status
        rapport = st.session_state['nurse_rapport']
        animal_q = st.session_state.get('nurse_animal_questions', 0)

        st.info(f"**Nurse Rapport:** {rapport} | **Animal Questions Asked:** {animal_q}/3")

        # Show initial dialogue choices if first interaction
        if not st.session_state['nurse_initial_dialogue_shown'] and len(history) == 0:
            st.markdown("---")
            st.markdown("**Nurse Mai looks up from her paperwork, clearly stressed and overwhelmed.**")
            st.markdown('"Why are you here now? I have so many patients to see..."')
            st.markdown("---")
            st.markdown("**How do you respond?**")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚨 'Show me the records. Now.'", key="nurse_demand", use_container_width=True):
                    result = update_nurse_rapport('demand', st.session_state)
                    history.append({"role": "user", "content": "'Show me the records. Now.'"})
                    history.append({"role": "assistant", "content": result['message']})
                    st.session_state.interview_history[npc_key] = history
                    st.session_state['nurse_initial_dialogue_shown'] = True
                    st.rerun()

            with col2:
                if st.button("💚 'It looks busy here. Thank you for your work.'", key="nurse_empathize", use_container_width=True):
                    result = update_nurse_rapport('empathize', st.session_state)
                    history.append({"role": "user", "content": "'It looks busy here. Thank you for your work.'"})
                    history.append({"role": "assistant", "content": result['message']})
                    st.session_state.interview_history[npc_key] = history
                    st.session_state['nurse_initial_dialogue_shown'] = True
                    st.rerun()

            st.markdown("---")
            return  # Don't show chat input yet

        # Show pig clue if unlocked
        if (rapport > 20 or animal_q >= 3) and 'nurse_pig_clue_shown' not in st.session_state:
            st.success("**🐷 Nurse Mai sighs:** 'Fine. A few pig litters had abortions recently. Young farmers are careless.'")
            st.session_state['nurse_pig_clue_shown'] = True

        # Show records access status
        if rapport > 10:
            st.success("✅ **Records Access Granted** - You may now review the child register.")
        else:
            st.warning("🔒 **Records Access Denied** - Improve your rapport to access clinic records.")

    # Chat input
    user_q = st.chat_input(f"Ask {npc['name']} a question...")
    if user_q:
        # Check for NPC unlock triggers
        unlock_notification = check_npc_unlock_triggers(user_q)

        # Track animal questions for nurse_joy
        if npc_key == "nurse_joy":
            from je_logic import update_nurse_rapport
            animal_keywords = ['pig', 'pigs', 'livestock', 'animal', 'animals', 'abortion', 'abortions', 'sow', 'sows', 'litter', 'litters']
            if any(keyword in user_q.lower() for keyword in animal_keywords):
                update_nurse_rapport('animals', st.session_state)

        history.append({"role": "user", "content": user_q})
        st.session_state.interview_history[npc_key] = history

        with st.chat_message("user"):
            st.write(user_q)

        with st.chat_message("assistant", avatar=get_npc_avatar(npc)):
            with st.spinner("..."):
                reply = get_npc_response(npc_key, user_q)
            st.write(reply)

        history.append({"role": "assistant", "content": reply})
        st.session_state.interview_history[npc_key] = history

        # Show unlock notification
        if unlock_notification:
            st.success(unlock_notification)

        st.rerun()


def adventure_sidebar():
    """Minimal sidebar for adventure mode with resources and tools."""
    # Language selector
    st.sidebar.markdown(f"### {t('language_header')}")
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

    st.sidebar.markdown("---")
    st.sidebar.title("AES Investigation")

    if not st.session_state.alert_acknowledged:
        st.sidebar.info("Review the alert to begin.")
        return

    # Resources
    time_display = f":red[{st.session_state.time_remaining}h]" if st.session_state.time_remaining < 0 else f"{st.session_state.time_remaining}h"
    st.sidebar.markdown(f"""
    **{t('day')}:** {st.session_state.current_day} / 5
    **{t('time_remaining')}:** {time_display}
    **{t('budget')}:** ${st.session_state.budget}
    **{t('lab_credits')}:** {st.session_state.lab_credits}
    """)

    # Progress
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Progress")
    for day in range(1, 6):
        if day < st.session_state.current_day:
            status = "[✓]"
        elif day == st.session_state.current_day:
            status = "[●]"
        else:
            status = "[ ]"
        st.sidebar.markdown(f"{status} Day {day}")

    # Session Management
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Save", use_container_width=True, key="save_session"):
            try:
                save_data = persistence.create_save_file(st.session_state)
                filename = persistence.get_save_filename(st.session_state)
                st.sidebar.download_button(
                    label="⬇️ Download",
                    data=save_data,
                    file_name=filename,
                    mime="application/json",
                    use_container_width=True,
                    key="download_save"
                )
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    with col2:
        uploaded = st.file_uploader(
            "📂",
            type=["json"],
            key="load_session",
            label_visibility="collapsed"
        )
        if uploaded is not None:
            success, message = persistence.load_save_file(uploaded, st.session_state)
            if success:
                st.sidebar.success(message)
                st.rerun()
            else:
                st.sidebar.error(message)

    # Investigation Notebook
    st.sidebar.markdown("---")
    with st.sidebar.expander(f"📓 {t('notebook')}"):
        st.caption("Record observations and insights.")

        new_note = st.text_area("Add note:", height=60, key="new_note")
        if st.button("Save Note", key="save_note"):
            if new_note.strip():
                from datetime import datetime
                entry = {
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "day": st.session_state.current_day,
                    "note": new_note.strip()
                }
                st.session_state.notebook_entries.append(entry)
                st.success("Saved!")
                st.rerun()

        if st.session_state.notebook_entries:
            st.markdown("**Your Notes:**")
            for entry in reversed(st.session_state.notebook_entries[-5:]):
                st.markdown(f"*Day {entry['day']} @ {entry['timestamp']}*")
                st.markdown(f"> {entry['note']}")
                st.markdown("---")

    # Advance day button
    st.sidebar.markdown("---")
    if st.session_state.current_day < 5:
        if st.sidebar.button(f"{t('advance_day')} {st.session_state.current_day + 1}", use_container_width=True):
            can_advance, missing = check_day_prerequisites(st.session_state.current_day, st.session_state)
            if can_advance:
                st.session_state.current_day += 1
                st.session_state.time_remaining = 8
                refresh_lab_queue_for_day(int(st.session_state.current_day))
                st.session_state.advance_missing_tasks = []
                # Show SITREP view for new day
                st.session_state.current_view = "sitrep"
                st.session_state.sitrep_viewed = False
                st.rerun()
            else:
                st.session_state.advance_missing_tasks = missing
                st.sidebar.warning(t("cannot_advance"))
    else:
        st.sidebar.success("📋 Final Day!")


# =========================
# SITREP VIEW
# =========================

def view_sitrep():
    """Daily situation report - blocking view before advancing to next day."""
    st.title(f"Day {st.session_state.current_day} SITREP")

    st.markdown(f"""
    ### Situation Report - Day {st.session_state.current_day}

    Welcome to Day {st.session_state.current_day} of the outbreak investigation.
    """)

    # Show day briefing
    st.markdown("---")
    st.markdown("### Today's Objectives")
    day_task_list(st.session_state.current_day)

    # Activity summary from previous day (if not Day 1)
    if st.session_state.current_day > 1:
        st.markdown("---")
        st.markdown("### Activity Summary from Previous Day")

        # Show interview count
        interview_count = len(st.session_state.interview_history)
        st.markdown(f"- **Interviews completed:** {interview_count}")

        # Show locations visited
        visited_count = len(st.session_state.get("visited_locations", set()))
        st.markdown(f"- **Locations visited:** {visited_count}")

        # Show lab samples if any
        lab_count = len(st.session_state.get("lab_queue", []))
        if lab_count > 0:
            st.markdown(f"- **Lab samples submitted:** {lab_count}")

    # New admissions count
    st.markdown("---")
    st.markdown("### New Patient Admissions")
    truth = st.session_state.truth
    pop_df = truth.get("full_population", pd.DataFrame())
    if not pop_df.empty:
        new_cases = pop_df[pop_df["hospital_day"] == st.session_state.current_day]
        st.markdown(f"**{len(new_cases)} new patients** were admitted overnight.")
    else:
        st.markdown("*No new admissions recorded.*")

    # Continue button
    st.markdown("---")
    if st.button("✅ Continue to Day " + str(st.session_state.current_day), type="primary", use_container_width=True):
        st.session_state.sitrep_viewed = True
        st.session_state.current_view = "map"
        st.rerun()


# =========================
# EVIDENCE BOARD
# =========================

def init_evidence_board():
    """Initialize evidence board with 3 starting clues."""
    if "evidence_board" not in st.session_state:
        st.session_state.evidence_board = [
            {
                "clue": "Pig abortions (Viral)",
                "type": "hypothesis",
                "day_added": 1,
                "source": "Initial observation"
            },
            {
                "clue": "Dragon Fire rumor (Toxin)",
                "type": "hypothesis",
                "day_added": 1,
                "source": "Community reports"
            },
            {
                "clue": "High Fever (Weakens Toxin)",
                "type": "clinical",
                "day_added": 1,
                "source": "Hospital records"
            }
        ]

def view_evidence_board():
    """Display the evidence board."""
    st.markdown("### 🔍 Evidence Board")
    st.markdown("Track key clues and hypotheses as you investigate.")

    if not st.session_state.get("evidence_board"):
        init_evidence_board()

    for i, evidence in enumerate(st.session_state.evidence_board):
        with st.expander(f"**{evidence['clue']}** (Day {evidence['day_added']})", expanded=(i < 3)):
            st.markdown(f"**Type:** {evidence['type'].title()}")
            st.markdown(f"**Source:** {evidence['source']}")

    # Add new evidence
    st.markdown("---")
    with st.form("add_evidence"):
        new_clue = st.text_input("New clue or hypothesis")
        clue_type = st.selectbox("Type", ["hypothesis", "clinical", "environmental", "epidemiological"])
        clue_source = st.text_input("Source (optional)")

        if st.form_submit_button("Add to Evidence Board"):
            if new_clue:
                st.session_state.evidence_board.append({
                    "clue": new_clue,
                    "type": clue_type,
                    "day_added": st.session_state.current_day,
                    "source": clue_source or "Investigation team"
                })
                st.success("Added to evidence board!")
                st.rerun()


# =========================
# MAIN
# =========================

def main():
    st.set_page_config(
        page_title="FETP Sim: Sidero Valley",
        page_icon="🦟",
        layout="wide",
        initial_sidebar_state="collapsed",  # Minimal sidebar in adventure mode
    )
    init_session_state()

    # Check game state for Serious Mode
    game_state = st.session_state.get('game_state', 'INTRO')

    # INTRO state: Show Dr. Tran phone call (replaces alert screen for new sessions)
    if game_state == 'INTRO' and not st.session_state.alert_acknowledged:
        view_intro()
        return

    # Use adventure-style sidebar (now persistent across all states)
    adventure_sidebar()

    # If alert hasn't been acknowledged yet (legacy), show alert screen
    if not st.session_state.alert_acknowledged:
        view_alert()
        return

    view = st.session_state.current_view

    # SITREP view - blocking daily briefing
    if view == "sitrep":
        view_sitrep()
        return

    # Adventure mode: location-based navigation
    if view == "map" or view is None:
        view_travel_map()
    elif view == "area":
        # Show area map for selected area - use visual layout for ALL areas
        area = st.session_state.get("current_area")
        if area:
            # Use immersive visual layout with hero image and card grid for all areas
            view_area_visual(area)
        else:
            view_travel_map()
    elif view == "location":
        # Show specific location view
        loc_key = st.session_state.get("current_location")
        if loc_key:
            view_location(loc_key)
        else:
            view_travel_map()

    # Legacy views (still accessible via quick links)
    elif view == "overview":
        # Add return to map button
        if st.button("Return to Map", key="return_from_overview"):
            st.session_state.current_view = "map"
            st.rerun()
        view_overview()
    elif view == "casefinding":
        if st.button("Return to Map", key="return_from_casefinding"):
            st.session_state.current_view = "map"
            st.rerun()
        view_case_finding()
    elif view == "descriptive":
        if st.button("Return to Map", key="return_from_descriptive"):
            st.session_state.current_view = "map"
            st.rerun()
        view_descriptive_epi()
    elif view == "villages":
        if st.button("Return to Map", key="return_from_villages"):
            st.session_state.current_view = "map"
            st.rerun()
        view_village_profiles()
    elif view == "interviews":
        if st.button("Return to Map", key="return_from_interviews"):
            st.session_state.current_view = "map"
            st.rerun()
        view_interviews()
    elif view == "spotmap":
        if st.button("Return to Map", key="return_from_spotmap"):
            st.session_state.current_view = "map"
            st.rerun()
        view_spot_map()
    elif view == "study":
        if st.button("Return to Map", key="return_from_study"):
            st.session_state.current_view = "map"
            st.rerun()
        view_study_design()
    elif view == "lab":
        if st.button("Return to Map", key="return_from_lab"):
            st.session_state.current_view = "map"
            st.rerun()
        view_lab_and_environment()
    elif view == "outcome":
        if st.button("Return to Map", key="return_from_outcome"):
            st.session_state.current_view = "map"
            st.rerun()
        view_interventions_and_outcome()
    elif view == "medical_records":
        view_medical_records()
    elif view == "clinic_register":
        view_clinic_register_scan()
    elif view == "nalu_child_register":
        view_nalu_child_register()
    else:
        view_travel_map()


if __name__ == "__main__":
    main()
