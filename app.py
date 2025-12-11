import streamlit as st
import anthropic
import pandas as pd
import numpy as np

# Import core logic from je_logic
from je_logic import (
    load_truth_data,
    generate_full_population,
    generate_study_dataset,
    process_lab_order,
    evaluate_interventions,
    check_day_prerequisites,
)

# -------------------------------
# INITIAL SETUP & CACHED LOADERS
# -------------------------------

@st.cache_data
def load_truth_and_population(data_dir: str = "."):
    """
    Load static truth tables (villages, households seed, individuals seed, lab, env, npc_truth)
    and generate a full synthetic population with infection status.
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
        # NOTE: if you move CSVs/JSON to a 'data/' folder, change to data_dir="data"
        st.session_state.truth = load_truth_and_population(data_dir=".")

    if "current_day" not in st.session_state:
        st.session_state.current_day = 1

    if "current_view" not in st.session_state:
        st.session_state.current_view = "overview"

    if "budget" not in st.session_state:
        st.session_state.budget = 1000

    if "lab_credits" not in st.session_state:
        st.session_state.lab_credits = 20

    if "decisions" not in st.session_state:
        st.session_state.decisions = {
            "case_definition": None,
            "case_definition_text": "",
            "study_design": None,
            "mapped_columns": [],
            "sample_size": {"cases": 15, "controls_per_case": 2},
            "lab_orders": [],
            "interventions": [],
            "questionnaire_raw": [],
        }

    if "generated_dataset" not in st.session_state:
        st.session_state.generated_dataset = None

    if "lab_results" not in st.session_state:
        st.session_state.lab_results = []

    if "interview_history" not in st.session_state:
        st.session_state.interview_history = {}

    if "revealed_clues" not in st.session_state:
        st.session_state.revealed_clues = {}

    if "current_npc" not in st.session_state:
        st.session_state.current_npc = None

    # Unlock flags (One Health, etc.)
    if "unlock_flags" not in st.session_state:
        st.session_state.unlock_flags = {}


# -------------------------------
# NPC / INTERVIEW ENGINE
# -------------------------------

def build_epidemiologic_context(truth: dict) -> str:
    """
    Create a short summary of the outbreak from the truth tables
    that NPCs can draw on. This integrates village & household logic
    into the interviews.
    """
    individuals = truth["individuals"]
    households = truth["households"]
    villages = truth["villages"][["village_id", "village_name"]]

    # Merge individual → household → village
    hh_vil = households.merge(villages, on="village_id", how="left")
    merged = individuals.merge(hh_vil[["hh_id", "village_name"]], on="hh_id", how="left")

    # Symptomatic AES cases
    cases = merged[merged["symptomatic_AES"] == True]
    total_cases = len(cases)

    if total_cases == 0:
        return "No symptomatic AES cases have been assigned in the truth model."

    # Cases by village
    village_counts = cases["village_name"].value_counts().to_dict()

    # Cases by age group
    bins = [0, 4, 14, 49, 120]
    labels = ["0–4", "5–14", "15–49", "50+"]
    age_groups = pd.cut(cases["age"], bins=bins, labels=labels, right=True)
    age_counts = age_groups.value_counts().to_dict()

    context = (
        f"There are currently {total_cases} symptomatic AES cases in the district. "
        f"Cases by village: {village_counts}. "
        f"Cases by age group: {age_counts}. "
        "Most cases are children, and almost all come from villages with rice paddies and pigs."
    )
    return context


def build_npc_data_context(npc_key: str, truth: dict) -> str:
    """
    Provide NPC-specific data context based on data_access flags
    and the village/household-level truth tables.
    """
    npc = truth["npc_truth"][npc_key]
    data_access = npc.get("data_access")

    individuals = truth["individuals"]
    households = truth["households"]
    villages = truth["villages"][["village_id", "village_name"]]
    hh_vil = households.merge(villages, on="village_id", how="left")
    merged = individuals.merge(hh_vil[["hh_id", "village_name"]], on="hh_id", how="left")

    # Default context
    epi_context = build_epidemiologic_context(truth)

    if data_access == "hospital_cases":
        cases = merged[merged["symptomatic_AES"] == True]
        summary = cases.groupby("village_name").size().to_dict()
        return (
            epi_context
            + " As hospital director, you mainly see hospitalized AES cases. "
              f"You know that current hospitalized cases come from these villages: {summary}."
        )
    elif data_access == "triage_logs":
        cases = merged[merged["symptomatic_AES"] == True]
        earliest = cases["onset_date"].min()
        latest = cases["onset_date"].max()
        return (
            epi_context
            + " As a triage nurse, you mainly see who walks in the door first. "
              f"You have noticed the first AES cases between {earliest} and {latest}."
        )
    elif data_access == "private_clinic":
        # Assume clinic mostly sees early mild AES near Nalu
        cases = merged[
            (merged["symptomatic_AES"] == True)
            & (merged["village_name"] == "Nalu Village")
        ]
        n = len(cases)
        return (
            epi_context
            + f" As a private healer, you have personally seen about {n} early AES-like illnesses "
              "from households near pig farms and rice paddies in Nalu."
        )
    elif data_access == "school_attendance":
        school_age = merged[(merged["age"] >= 5) & (merged["age"] <= 18)]
        cases = school_age[school_age["symptomatic_AES"] == True]
        n = len(cases)
        by_village = cases["village_name"].value_counts().to_dict()
        return (
            epi_context
            + f" As school principal, you primarily know about school-age children. "
              f"You know of AES cases among your students: {n} total, by village: {by_village}."
        )
    elif data_access == "vet_surveillance":
        lab = truth["lab_samples"]
        pig_samples = lab[lab["sample_type"] == "pig_serum"]
        pos = pig_samples[pig_samples["true_JEV_status"] == True]
        by_village = pos["linked_village_id"].value_counts().to_dict()
        return (
            epi_context
            + " As the district veterinary officer, you track pig health and surveillance. "
              f"Recent pig serology suggests JEV circulation in villages: {by_village}."
        )
    elif data_access == "environmental_data":
        env = truth["environment_sites"]
        high_breeding = env[env["breeding_index"] == "high"]
        return (
            epi_context
            + " As environmental health officer, you have surveyed breeding sites. "
              f"You know of high mosquito breeding around these sites: {high_breeding['site_id'].tolist()}."
        )
    else:
        # Generic context
        return epi_context


def get_npc_response(npc_key: str, user_input: str):
    """
    Call Anthropic to generate an NPC response using:
    - npc_truth.json
    - Outbreak context from villages/households/individuals
    - Conditional clue logic
    """
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ Anthropic API key missing. Please configure secrets."

    truth = st.session_state.truth
    npc_truth = truth["npc_truth"][npc_key]

    # Build epidemiologic + NPC-specific context
    epi_context = build_npc_data_context(npc_key, truth)

    # Track revealed clues for this NPC
    if npc_key not in st.session_state.revealed_clues:
        st.session_state.revealed_clues[npc_key] = []

    # Base system prompt
    system_prompt = f"""
You are {npc_truth['name']}, the {npc_truth['role']} in Sidero Valley.

Personality:
{npc_truth['personality']}

Outbreak context (for your awareness, not to recite verbatim word-for-word):
{epi_context}

ALWAYS REVEAL (you may weave these into your answers naturally):
{npc_truth['always_reveal']}

CONDITIONAL CLUES:
Only reveal a conditional clue when the user question clearly relates to its keyword.
Conditional clues (keyword: clue):
{npc_truth['conditional_clues']}

RED HERRINGS:
You may occasionally mention these, but do NOT contradict the core truth:
{npc_truth['red_herrings']}

UNKNOWN:
If the user asks about any of these topics, explicitly say you do not know:
{npc_truth['unknowns']}

CONVERSATION RULES:
- Answer in 2–4 sentences.
- Stay in character and speak as a real person from the district.
- Do NOT invent new case counts, lab results, or locations beyond what is implied above.
- If you refer to numbers, keep them approximate (e.g., 'several children', 'a few cases').
- If you are unsure, say so.
"""

    # Determine which conditional clues are eligible based on the user question
    lower_q = user_input.lower()
    conditional_to_use = []
    for keyword, clue in npc_truth.get("conditional_clues", {}).items():
        if keyword.lower() in lower_q and clue not in st.session_state.revealed_clues[npc_key]:
            conditional_to_use.append(clue)
            st.session_state.revealed_clues[npc_key].append(clue)

    conditional_text = ""
    if conditional_to_use:
        conditional_text = (
            "\n\nYou are allowed to reveal NEW specific clues in this answer:\n"
            + "\n".join(f"- {c}" for c in conditional_to_use)
        )

    client = anthropic.Anthropic(api_key=api_key)
    history = st.session_state.interview_history.get(npc_key, [])
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    msgs.append({"role": "user", "content": user_input})

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        system=system_prompt + conditional_text,
        messages=msgs,
    )

    text = response.content[0].text

    # Handle unlocks (e.g., vet_data_unlocked, environmental_data_unlocked)
    unlock_flag = npc_truth.get("unlocks")
    if unlock_flag:
        st.session_state.unlock_flags[unlock_flag] = True

    return text


# -------------------------------
# UI COMPONENTS
# -------------------------------

def sidebar_navigation():
    st.sidebar.title("Sidero Valley JE Simulation")
    st.sidebar.markdown(
        f"**Day:** {st.session_state.current_day} / 5\n\n"
        f"**Budget:** ${st.session_state.budget}\n"
        f"**Lab Credits:** {st.session_state.lab_credits}"
    )

    # Progress indicator
    st.sidebar.markdown("### Progress")
    for day in range(1, 6):
        status = "⬜"
        if day < st.session_state.current_day:
            status = "✅"
        elif day == st.session_state.current_day:
            status = "🟡"
        st.sidebar.markdown(f"{status} Day {day}")

    st.sidebar.markdown("---")

    # Map internal view to label
    internal_to_label = {
        "overview": "Overview",
        "contacts": "Interviews",
        "study": "Data & Study Design",
        "lab": "Lab & Environment",
        "outcome": "Interventions & Outcome",
    }
    label_to_internal = {v: k for k, v in internal_to_label.items()}

    current_label = internal_to_label.get(st.session_state.current_view, "Overview")

    view_label = st.sidebar.radio(
        "Go to:",
        ["Overview", "Interviews", "Data & Study Design", "Lab & Environment", "Interventions & Outcome"],
        index=["Overview", "Interviews", "Data & Study Design", "Lab & Environment", "Interventions & Outcome"].index(
            current_label
        ),
    )

    st.session_state.current_view = label_to_internal[view_label]


def view_overview():
    st.title("Sidero Valley Outbreak Investigation")
    st.markdown(
        """
You are the district rapid response team investigating an outbreak of acute encephalitis syndrome (AES) in Sidero Valley.

Over 5 days, you will:
- Define and refine a working case definition
- Conduct hypothesis-generating interviews
- Design and analyze an epidemiologic study
- Decide which lab and environmental samples to collect
- Propose interventions to the Ministry of Health

Use the left sidebar to navigate between sections.
"""
    )


def view_interviews():
    truth = st.session_state.truth
    npc_truth = truth["npc_truth"]

    st.header("👥 Interviews")
    st.info("Each interview costs budget. Some NPCs can unlock additional data sources or One Health perspectives.")

    # NPC selection grid
    cols = st.columns(3)
    for i, (npc_key, npc) in enumerate(npc_truth.items()):
        with cols[i % 3]:
            st.markdown(f"**{npc['avatar']} {npc['name']}**")
            st.caption(f"{npc['role']} — Cost: ${npc['cost']}")
            if st.button(f"Talk to {npc['name']}", key=f"btn_{npc_key}"):
                cost = npc.get("cost", 0)
                if st.session_state.budget >= cost:
                    st.session_state.budget -= cost
                    st.session_state.current_npc = npc_key
                    if npc_key not in st.session_state.interview_history:
                        st.session_state.interview_history[npc_key] = []
                    st.experimental_rerun()
                else:
                    st.error("Insufficient budget for this interview.")

    if st.session_state.current_npc:
        npc_key = st.session_state.current_npc
        npc = npc_truth[npc_key]
        st.markdown("---")
        st.subheader(f"💬 Talking to {npc['name']} ({npc['role']})")

        history = st.session_state.interview_history.get(npc_key, [])
        for msg in history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_q = st.chat_input("Ask a question...")
        if user_q:
            # Append user question
            history.append({"role": "user", "content": user_q})
            st.session_state.interview_history[npc_key] = history

            with st.chat_message("user"):
                st.write(user_q)

            with st.chat_message("assistant", avatar=npc["avatar"]):
                reply = get_npc_response(npc_key, user_q)
                st.write(reply)
            history.append({"role": "assistant", "content": reply})
            st.session_state.interview_history[npc_key] = history


def view_study_design():
    st.header("📊 Data & Study Design")

    # Step 1: Case definition
    st.markdown("### Step 1: Case Definition (Working)")
    case_def = st.text_area(
        "Describe your current working case definition:",
        value=st.session_state.decisions.get("case_definition_text", ""),
        height=120,
        help="For example: Residents of Nalu or Kabwe with AES onset after June 1, age < 15 years, with fever and altered mental status.",
    )
    if st.button("Save Case Definition"):
        st.session_state.decisions["case_definition_text"] = case_def
        # Keep a simple structured version for the logic (you can refine this later)
        st.session_state.decisions["case_definition"] = {"clinical_AES": True}
        st.success("Case definition saved.")

    # Step 2: Study design
    st.markdown("### Step 2: Study Design")
    sd_type = st.radio("Choose a study design:", ["Case-control", "Retrospective cohort"])
    if sd_type == "Case-control":
        st.session_state.decisions["study_design"] = {"type": "case_control"}
    else:
        st.session_state.decisions["study_design"] = {"type": "cohort"}

    # Step 3: Questionnaire mapping
    st.markdown("### Step 3: Questionnaire Variables")
    st.caption("Enter key questions/variables. The simulator will map them to underlying columns based on keywords.")
    q_text = st.text_area(
        "Questionnaire content",
        value="\n".join(st.session_state.decisions.get("questionnaire_raw", [])),
        height=150,
    )

    if st.button("Save Questionnaire Variables"):
        # Simple keyword-based mapping (your coder can later replace this with an LLM mapping step)
        mapped_cols = []
        lower = q_text.lower()
        if "age" in lower:
            mapped_cols.append("age")
        if "sex" in lower or "gender" in lower:
            mapped_cols.append("sex")
        if "mosquito net" in lower or "bed net" in lower:
            mapped_cols.append("Mosquito_Net_Use")
        if "pig" in lower:
            mapped_cols.append("Pigs_Near_Home")
        if "rice" in lower or "paddy" in lower:
            mapped_cols.append("rice_field_distance_m")
        if "vaccin" in lower:
            mapped_cols.append("JE_vaccinated")
        if "evening" in lower or "dusk" in lower or "outside" in lower:
            mapped_cols.append("evening_outdoor_exposure")

        st.session_state.decisions["mapped_columns"] = list(set(mapped_cols))
        st.session_state.decisions["questionnaire_raw"] = q_text.splitlines()
        st.success(f"Mapped columns: {st.session_state.decisions['mapped_columns']}")

    # Step 4: Dataset generation
    st.markdown("### Step 4: Generate Study Dataset (for Day 3)")
    if st.button("Generate Simulated Dataset"):
        truth = st.session_state.truth
        individuals = truth["individuals"]
        households = truth["households"]

        decisions = st.session_state.decisions
        study_df = generate_study_dataset(individuals, households, decisions)
        st.session_state.generated_dataset = study_df
        st.success("Dataset generated. You can now export and analyze it outside the app.")
        st.dataframe(study_df.head())


def view_lab_and_environment():
    st.header("🧪 Lab & Environment")

    st.markdown(
        "Order lab tests and environmental investigations. "
        "In this version, the UI is a placeholder, but the backend logic lives in `je_logic.process_lab_order()`."
    )
    st.info(
        "Your coder can create forms/buttons here that build lab order dictionaries and pass them to "
        "`process_lab_order(order, truth['lab_samples'])`, then store results in `st.session_state.lab_results`."
    )


def view_interventions_and_outcome():
    st.header("📉 Interventions & Outcome")

    st.markdown("### Proposed interventions")
    interventions = st.text_area(
        "List your recommended interventions:",
        value="\n".join(st.session_state.decisions.get("interventions", [])),
        height=150,
    )

    if st.button("Save Interventions"):
        st.session_state.decisions["interventions"] = [
            line for line in interventions.splitlines() if line.strip()
        ]
        st.success("Interventions saved.")

    st.markdown("### Evaluate Outcome (Prototype)")
    if st.button("Evaluate Scenario Outcome"):
        score = evaluate_interventions(
            st.session_state.decisions, st.session_state.interview_history
        )
        st.write(f"Intervention score (prototype): {score}")
        st.info(
            "Your coder can expand this to show a projected epi curve and a narrative describing how the outbreak evolves."
        )


# -------------------------------
# MAIN APP
# -------------------------------

def main():
    st.set_page_config(page_title="JE Outbreak Simulation", layout="wide")
    init_session_state()
    sidebar_navigation()

    view_key = st.session_state.current_view

    if view_key == "overview":
        view_overview()
    elif view_key == "contacts":
        view_interviews()
    elif view_key == "study":
        view_study_design()
    elif view_key == "lab":
        view_lab_and_environment()
    elif view_key == "outcome":
        view_interventions_and_outcome()
    else:
        view_overview()


if __name__ == "__main__":
    main()
