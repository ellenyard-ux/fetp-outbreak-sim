import streamlit as st
import anthropic
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from je_logic import (
    load_truth_data,
    generate_full_population,
    generate_study_dataset,
    process_lab_order,
    evaluate_interventions,
    check_day_prerequisites,
)

# =========================
# INITIALIZATION
# =========================

@st.cache_data
def load_truth_and_population(data_dir: str = "."):
    """Load truth data and generate a full population."""
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
        # CSV/JSON files are in the repo root right now
        st.session_state.truth = load_truth_and_population(data_dir=".")

    if "current_day" not in st.session_state:
        # 1–5
        st.session_state.current_day = 1

    if "current_view" not in st.session_state:
        # Start on the briefing/overview page
        st.session_state.current_view = "overview"

    # Resources
    st.session_state.setdefault("budget", 1000)
    st.session_state.setdefault("lab_credits", 20)

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
            "final_diagnosis": "",
            "recommendations": [],
        }

    st.session_state.setdefault("generated_dataset", None)
    st.session_state.setdefault("lab_results", [])
    st.session_state.setdefault("lab_samples_submitted", [])
    st.session_state.setdefault("interview_history", {})
    st.session_state.setdefault("revealed_clues", {})
    st.session_state.setdefault("current_npc", None)
    st.session_state.setdefault("unlock_flags", {})

    # Flags used for day progression
    st.session_state.setdefault("case_definition_written", False)
    st.session_state.setdefault("questionnaire_submitted", False)
    st.session_state.setdefault("descriptive_analysis_done", False)

    # Track whether user has opened the line list/epi view at least once (for Day 1)
    st.session_state.setdefault("line_list_viewed", False)

    # For messaging when advance-day fails
    st.session_state.setdefault("advance_missing_tasks", [])


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


def get_npc_response(npc_key: str, user_input: str) -> str:
    """Call Anthropic using npc_truth + epidemiologic context."""
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ Anthropic API key missing."

    truth = st.session_state.truth
    npc_truth = truth["npc_truth"][npc_key]

    epi_context = build_npc_data_context(npc_key, truth)

    if npc_key not in st.session_state.revealed_clues:
        st.session_state.revealed_clues[npc_key] = []

    system_prompt = f"""
You are {npc_truth['name']}, the {npc_truth['role']} in Sidero Valley.

Personality:
{npc_truth['personality']}

Outbreak context (for your awareness):
{epi_context}

ALWAYS REVEAL (inevitably come up over the conversation):
{npc_truth['always_reveal']}

CONDITIONAL CLUES:
Reveal a conditional clue ONLY when the user's question clearly relates to its keyword.
Conditional clues (keyword: clue):
{npc_truth['conditional_clues']}

RED HERRINGS:
You may mention these occasionally but do NOT contradict the core truth:
{npc_truth['red_herrings']}

UNKNOWN:
If the user asks about these topics, say you do not know:
{npc_truth['unknowns']}

RULES:
- Answer in 2–4 sentences.
- Stay in character as a real person from the district.
- Do not invent new case counts, lab results, or locations beyond what is implied above.
- If unsure, say you are not sure.
"""

    # Decide which conditional clues are allowed in this answer
    lower_q = user_input.lower()
    conditional_to_use = []
    for keyword, clue in npc_truth.get("conditional_clues", {}).items():
        if keyword.lower() in lower_q and clue not in st.session_state.revealed_clues[npc_key]:
            conditional_to_use.append(clue)
            st.session_state.revealed_clues[npc_key].append(clue)

    conditional_text = ""
    if conditional_to_use:
        conditional_text = (
            "\n\nYou may reveal these NEW specific clues in this answer:\n"
            + "\n".join(f"- {c}" for c in conditional_to_use)
        )

    client = anthropic.Anthropic(api_key=api_key)

    history = st.session_state.interview_history.get(npc_key, [])
    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    msgs.append({"role": "user", "content": user_input})

    resp = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        system=system_prompt + conditional_text,
        messages=msgs,
    )

    text = resp.content[0].text

    # Unlock flags (One Health unlocks)
    unlock_flag = npc_truth.get("unlocks")
    if unlock_flag:
        st.session_state.unlock_flags[unlock_flag] = True

    return text


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
    return cases.head(n)[["person_id", "age", "sex", "village_name", "onset_date", "severe_neuro", "outcome"]]


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
        )
    )
    fig.update_layout(
        title="AES cases by onset date",
        xaxis_title="Onset date",
        yaxis_title="Number of cases",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# =========================
# UI COMPONENTS
# =========================

def sidebar_navigation():
    st.sidebar.title("Sidero Valley JE Simulation")
    st.sidebar.markdown(
        f"**Day:** {st.session_state.current_day} / 5\n\n"
        f"**Budget:** ${st.session_state.budget}\n"
        f"**Lab credits:** {st.session_state.lab_credits}"
    )

    # Progress indicator
    st.sidebar.markdown("### Progress")
    for day in range(1, 6):
        if day < st.session_state.current_day:
            status = "✅"
        elif day == st.session_state.current_day:
            status = "🟡"
        else:
            status = "⬜"
        st.sidebar.markdown(f"{status} Day {day}")

    st.sidebar.markdown("---")

    labels = ["Overview / Briefing", "Interviews", "Data & Study Design", "Lab & Environment", "Interventions & Outcome"]
    internal = ["overview", "interviews", "study", "lab", "outcome"]
    current_label = labels[internal.index(st.session_state.current_view)] if st.session_state.current_view in internal else labels[0]

    choice = st.sidebar.radio("Go to:", labels, index=labels.index(current_label))
    st.session_state.current_view = internal[labels.index(choice)]

    st.sidebar.markdown("---")
    # Advance day button
    if st.sidebar.button("Advance to next day"):
        can_advance, missing = check_day_prerequisites(st.session_state.current_day, st.session_state)
        if can_advance:
            if st.session_state.current_day < 5:
                st.session_state.current_day += 1
                st.session_state.advance_missing_tasks = []
            else:
                st.sidebar.success("Already at Day 5.")
        else:
            st.session_state.advance_missing_tasks = missing
            st.sidebar.warning("Cannot advance yet. See missing tasks on Overview.")


def day_briefing_text(day: int) -> str:
    if day == 1:
        return (
            "Day 1 focuses on detection and initial description of the outbreak. "
            "Your goals are to understand the basic pattern (time, place, person), "
            "draft a working case definition, and begin hypothesis-generating interviews."
        )
    if day == 2:
        return (
            "Day 2 focuses on hypothesis generation and study design. "
            "You will design an analytic study and develop a questionnaire to collect data."
        )
    if day == 3:
        return (
            "Day 3 is dedicated to data cleaning and analysis. "
            "You will work with your generated dataset to describe the outbreak and identify risk factors."
        )
    if day == 4:
        return (
            "Day 4 focuses on laboratory and environmental investigations. "
            "You will decide which human, animal, and environmental samples to collect and how to test them."
        )
    return (
        "Day 5 focuses on recommendations and communication. "
        "You will integrate all evidence and present your findings and interventions to leadership."
    )


def day_task_list(day: int):
    """Show tasks and whether they are completed."""
    st.markdown("### Key tasks for today")
    if day == 1:
        st.checkbox("Write a working case definition", value=st.session_state.case_definition_written, disabled=True)
        st.checkbox("Complete at least 2 interviews", value=len(st.session_state.interview_history) >= 2, disabled=True)
        st.checkbox("Review line list and epi curve", value=st.session_state.line_list_viewed, disabled=True)
    elif day == 2:
        st.checkbox("Choose a study design", value=st.session_state.decisions.get("study_design") is not None, disabled=True)
        st.checkbox("Submit questionnaire", value=st.session_state.questionnaire_submitted, disabled=True)
        st.checkbox("Generate study dataset", value=st.session_state.generated_dataset is not None, disabled=True)
    elif day == 3:
        st.checkbox("Complete descriptive analysis", value=st.session_state.descriptive_analysis_done, disabled=True)
    elif day == 4:
        st.checkbox("Submit at least one lab sample", value=len(st.session_state.lab_samples_submitted) > 0, disabled=True)
    else:
        st.checkbox("Enter final diagnosis", value=bool(st.session_state.decisions.get("final_diagnosis")), disabled=True)
        st.checkbox("Record main recommendations", value=bool(st.session_state.decisions.get("recommendations")), disabled=True)

    if st.session_state.advance_missing_tasks:
        st.warning("To advance to the next day, you still need to:\n- " + "\n- ".join(st.session_state.advance_missing_tasks))


def view_overview():
    truth = st.session_state.truth

    st.title("JE Outbreak Investigation – Sidero Valley")
    st.subheader(f"Day {st.session_state.current_day} briefing")

    st.markdown(day_briefing_text(st.session_state.current_day))

    day_task_list(st.session_state.current_day)

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


def view_interviews():
    truth = st.session_state.truth
    npc_truth = truth["npc_truth"]

    st.header("👥 Interviews")
    st.info("Interview community members and officials. Each interview costs budget; some unlock One Health perspectives.")

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
                    st.session_state.interview_history.setdefault(npc_key, [])
                else:
                    st.error("Insufficient budget for this interview.")

    npc_key = st.session_state.current_npc
    if npc_key:
        npc = npc_truth[npc_key]
        st.markdown("---")
        st.subheader(f"Talking to {npc['name']} ({npc['role']})")

        history = st.session_state.interview_history.get(npc_key, [])
        for msg in history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_q = st.chat_input("Ask your question...")
        if user_q:
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

    # Case definition
    st.markdown("### Step 1: Case Definition")
    text = st.text_area(
        "Write your working case definition:",
        value=st.session_state.decisions.get("case_definition_text", ""),
        height=120,
    )
    if st.button("Save Case Definition"):
        st.session_state.decisions["case_definition_text"] = text
        st.session_state.decisions["case_definition"] = {"clinical_AES": True}
        st.session_state.case_definition_written = True
        st.success("Case definition saved.")

    # Study design
    st.markdown("### Step 2: Study Design")
    sd_type = st.radio("Choose a study design:", ["Case-control", "Retrospective cohort"])
    if sd_type == "Case-control":
        st.session_state.decisions["study_design"] = {"type": "case_control"}
    else:
        st.session_state.decisions["study_design"] = {"type": "cohort"}

    # Questionnaire
    st.markdown("### Step 3: Questionnaire")
    st.caption("List the key questions or variables you plan to include (one per line).")
    q_text = st.text_area(
        "Questionnaire items:",
        value="\n".join(st.session_state.decisions.get("questionnaire_raw", [])),
        height=160,
    )
    if st.button("Save Questionnaire"):
        lines = [ln for ln in q_text.splitlines() if ln.strip()]
        st.session_state.decisions["questionnaire_raw"] = lines
        st.session_state.decisions["mapped_columns"] = lines  # je_logic maps keywords → real cols
        st.session_state.questionnaire_submitted = True
        st.success("Questionnaire saved.")

    # Dataset generation
    st.markdown("### Step 4: Generate Simulated Study Dataset")
    if st.button("Generate Dataset"):
        truth = st.session_state.truth
        df = generate_study_dataset(
            truth["individuals"], truth["households"], st.session_state.decisions
        )
        st.session_state.generated_dataset = df
        st.session_state.descriptive_analysis_done = True  # simple proxy
        st.success("Dataset generated. Preview below; export for analysis as needed.")
        st.dataframe(df.head())


def view_lab_and_environment():
    st.header("🧪 Lab & Environment")

    st.markdown(
        "Order lab tests and environmental investigations. "
        "This simple interface demonstrates how orders flow into `process_lab_order()`."
    )

    truth = st.session_state.truth
    villages = truth["villages"]

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
            "Test",
            ["JE_IgM_CSF", "JE_IgM_serum", "JE_PCR_mosquito", "JE_Ab_pig"],
        )

    source_description = st.text_input("Source description (e.g., 'Case from Nalu')", "")

    if st.button("Submit lab order"):
        order = {
            "sample_type": sample_type,
            "village_id": village_id,
            "test": test,
            "source_description": source_description or "Unspecified source",
        }
        result = process_lab_order(order, truth["lab_samples"])
        st.session_state.lab_results.append(result)
        st.session_state.lab_samples_submitted.append(order)
        st.session_state.lab_credits -= result.get("cost", 0)
        st.success(
            f"Lab order submitted. Result: {result['result']} "
            f"(turnaround {result['days_to_result']} days)."
        )

    if st.session_state.lab_results:
        st.markdown("### Lab results so far")
        st.dataframe(pd.DataFrame(st.session_state.lab_results))


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

    view = st.session_state.current_view
    if view == "overview":
        view_overview()
    elif view == "interviews":
        view_interviews()
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
