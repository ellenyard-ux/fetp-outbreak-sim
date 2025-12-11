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
FACILITATOR_PASSWORD = "fetp2025" 

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
    .clock-display {
        font-family: 'Courier New', monospace;
        font-weight: bold;
        color: #2c3e50;
        font-size: 1.2em;
        text-align: center;
        background: #e8f6f3;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border: 1px solid #a2d9ce;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    defaults = {
        'current_view': 'briefing',
        'investigation_day': 1,
        'exercise_started': False,
        
        # Resources
        'budget': 3000,
        'lab_credits': 15,
        'current_time': 8.0,  # 8.0 = 8:00 AM, 18.0 = 6:00 PM
        'day_ended': False,

        # Notebook
        'notebook_entries': "",
        
        # Inject Tracking
        'inject_shown': {},
        'current_inject': None,
        
        # Unlocks
        'npcs_unlocked': ['dr_chen', 'nurse_joy', 'mama_kofi', 'foreman_rex', 'teacher_grace'],
        'one_health_triggered': False,
        'vet_unlocked': False,
        'env_officer_unlocked': False,
        'healer_unlocked': False,
        
        # Data Access
        'interview_history': {},
        'current_character': None,
        'questions_asked_about': set(),
        
        # Progress
        'case_definition_written': False,
        'study_design_chosen': None,
        'questionnaire_submitted': False,
        
        # Data Cleaning & Analysis
        'raw_dataset_generated': False,
        'data_cleaned': False,
        'cleaning_steps_taken': set(),
        'dataset_received': False,
        
        # Lab & Field
        'lab_samples_submitted': [],
        'lab_results': [],
        'sites_inspected': [],
        'confirmed_cases': [],
        'manually_entered_cases': [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_time(float_time):
    hours = int(float_time)
    minutes = int((float_time - hours) * 60)
    return f"{hours:02d}:{minutes:02d}"

def spend_time(hours):
    """ Deducts time. Returns True if successful, False if out of time. """
    if st.session_state.day_ended:
        st.error("🌙 The day has ended. Go to the sidebar to start the next day.")
        return False
        
    if st.session_state.current_time + hours > 18.0:
        st.error("⛔ Not enough daylight remaining! You must end the day.")
        return False
        
    st.session_state.current_time += hours
    if st.session_state.current_time >= 18.0:
        st.session_state.day_ended = True
        st.warning("🌙 The sun has set. Work stops for today.")
    return True

def advance_day():
    current_day = st.session_state.investigation_day
    if current_day < 5:
        st.session_state.investigation_day += 1
        new_day = st.session_state.investigation_day
        
        # Reset Time for new day
        st.session_state.current_time = 8.0
        st.session_state.day_ended = False
        
        # Trigger Injects
        st.session_state.current_inject = DAY_STRUCTURE[new_day]['inject']
        st.session_state.inject_shown[new_day] = False
        
        # Unlock logic
        if new_day == 2 and st.session_state.one_health_triggered:
            if 'vet_amina' not in st.session_state.npcs_unlocked: st.session_state.npcs_unlocked.append('vet_amina')
        if new_day == 3: st.session_state.dataset_received = True
        return True
    return False

# ============================================================================
# DAY STRUCTURE
# ============================================================================
DAY_STRUCTURE = {
    1: {
        'title': 'Day 1: Detect, Confirm, Describe',
        'objectives': [
            '1. Review initial cases',
            '2. Develop a working case definition',
            '3. Begin interviews',
            '4. Document initial hypotheses'
        ],
        'inject': {'title': 'Initial Alert', 'source': 'DHO', 'message': 'Reports of seizures in children...'}
    },
    2: {
        'title': 'Day 2: Interviews & Design',
        'objectives': ['Conduct interviews', 'Develop hypotheses', 'Design study'],
        'inject': {
            'title': '🏥 New Cases',
            'source': 'Hospital',
            'message': 'Four new cases admitted overnight. Two are siblings.'
        }
    },
    3: {
        'title': 'Day 3: Data Analysis',
        'objectives': ['Clean dataset', 'Run descriptive epi', 'Calculate ORs'],
        'inject': {
            'title': '📊 Data Arrived',
            'source': 'Field Team',
            'message': 'Field data collection complete. The dataset looks messy.'
        }
    },
    4: {
        'title': 'Day 4: Lab & Environment',
        'objectives': ['Order lab tests', 'Environmental sampling', 'Triangulate data'],
        'inject': {
            'title': '🔬 Lab Unlocked',
            'source': 'Regional Lab',
            'message': 'You have authority to order diagnostic tests.'
        }
    },
    5: {
        'title': 'Day 5: Briefing',
        'objectives': ['Prepare briefing', 'Recommend interventions'],
        'inject': {'title': 'Briefing Time', 'source': 'MOH', 'message': 'The Director is ready for you.'}
    }
}

# ============================================================================
# DIRTY DATA GENERATOR (DAY 3)
# ============================================================================
def get_dirty_dataset():
    """Generates a dataset with deliberate errors for the cleaning module."""
    data = {
        'ID': [f'P{i:03d}' for i in range(1, 21)],
        'Age': [5, 6, 4, 32, 8, 45, 150, 6, 7, 5, '7 yrs', 65, 4, 9, 3, 5, 28, 6, 7, 5], # 150, '7 yrs'
        'Sex': ['M', 'F', 'M', 'M', 'Fem', 'Male', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'F', 'm', 'F', 'M', 'F', 'F', 'M'], # Fem, Male, m
        'Onset': ['2025-06-03', '2025-06-04', '03/06/2025', '2025-06-06', '2025-06-03', 
                 '2025-06-07', '2025-06-04', 'June 6', '2025-06-05', '2025-06-08',
                 '2025-06-03', '2025-06-04', '2025-06-05', '2025-06-06', '2025-06-07',
                 '2025-06-03', '2025-06-08', '2025-06-04', '2025-06-05', '2025-06-06'], # Mixed formats
        'Village': ['Nalu', 'Nalu', 'Kabwe', 'Kabwe', 'Nalu', 'Tamu', 'Nalu', 'Kabwe', 'Nalu', 'Nalu',
                   'Nalu', 'Tamu', 'Nalu', 'Kabwe', 'Nalu', 'Nalu', 'Kabwe', 'Nalu', 'Nalu', 'Nalu'],
        'Outcome': ['Recovered', 'Died', 'Died', 'Recovered', 'Recovered', 'Recovered', 'Recovered', 'Recovered', 'Recovered', 'Recovered',
                   'Recovered', 'Recovered', 'Recovered', 'Recovered', 'Died', 'Recovered', 'Recovered', 'Recovered', 'Recovered', 'Recovered']
    }
    return pd.DataFrame(data)

# ============================================================================
# CHARACTERS
# ============================================================================
CHARACTERS = {
    "dr_chen": {"name": "Dr. Chen", "role": "Hospital Director", "avatar": "👨‍⚕️", "cost": 50, "location": "Hospital", "knowledge": ["Full line list", "No adult cases except miners"], "personality": "Precise."},
    "nurse_joy": {"name": "Nurse Joy", "role": "Triage Nurse", "avatar": "🩺", "cost": 50, "location": "Hospital", "knowledge": ["Clusters of kids", "Mosquito bites"], "personality": "Exhausted."},
    "healer_marcus": {"name": "Healer Marcus", "role": "Private Clinic", "avatar": "🌿", "cost": 150, "location": "Nalu", "knowledge": ["Saw early cases", "Pigs restless"], "personality": "Suspicious."},
    "mama_kofi": {"name": "Mama Kofi", "role": "Mother", "avatar": "👵", "cost": 50, "location": "Nalu", "knowledge": ["House near pigs", "No nets"], "personality": "Emotional."},
    "vet_amina": {"name": "Vet Amina", "role": "Vet Officer", "avatar": "🐄", "cost": 150, "location": "District Office", "knowledge": ["Pigs amplifying JE"], "personality": "Technical."},
    "mr_osei": {"name": "Mr. Osei", "role": "Env Officer", "avatar": "🌊", "cost": 150, "location": "District Office", "knowledge": ["Rice paddy expansion"], "personality": "Practical."}
}

# ============================================================================
# AI
# ============================================================================
def get_ai_response(char_key, user_input, history):
    char = CHARACTERS[char_key]
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key: return "⚠️ API Key not configured."
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({"role": "user", "content": user_input})
        
        system_prompt = f"""You are {char['name']}, {char['role']}.
        PERSONALITY: {char['personality']}
        KNOWLEDGE: {char['knowledge']}
        CONTEXT: An outbreak of Encephalitis in Sidero Valley.
        Respond naturally. Keep answers brief."""
        
        response = client.messages.create(model="claude-3-haiku-20240307", max_tokens=250, system=system_prompt, messages=messages)
        return response.content[0].text
    except Exception as e: return f"[Error: {str(e)}]"

# ============================================================================
# UI LAYOUT
# ============================================================================
current_day = st.session_state.investigation_day
day_info = DAY_STRUCTURE.get(current_day, {})

st.markdown(f"""
<div class="main-header">
    <h1>🦟 Sidero Valley Outbreak Investigation</h1>
    <p style="margin:0; opacity:0.9;">FETP Intermediate 2.0 | {day_info.get('title', 'Day ' + str(current_day))}</p>
</div>
""", unsafe_allow_html=True)

# DAILY INJECT
if current_day > 1 and not st.session_state.inject_shown.get(current_day, False):
    inject = DAY_STRUCTURE[current_day]['inject']
    st.markdown(f"""<div class="alert-box"><h3>{inject['title']}</h3><p>{inject['message']}</p></div>""", unsafe_allow_html=True)
    if st.button("✅ Acknowledge"):
        st.session_state.inject_shown[current_day] = True
        st.rerun()

# SIDEBAR
with st.sidebar:
    # CLOCK
    st.markdown(f"""<div class="clock-display">🕒 TIME: {format_time(st.session_state.current_time)}</div>""", unsafe_allow_html=True)
    if st.session_state.day_ended:
        st.error("Day Ended. Sleep to continue.")
    else:
        st.caption("Day ends at 18:00")

    st.markdown(f"## Day {current_day}")
    if 'objectives' in day_info:
        with st.expander(f"🎯 Day {current_day} Objectives", expanded=True):
            for obj in day_info['objectives']:
                st.markdown(f"{obj}")
    
    st.markdown("### 💰 Resources")
    col1, col2 = st.columns(2)
    col1.metric("Budget", f"${st.session_state.budget:,}")
    col2.metric("Credits", st.session_state.lab_credits)
    
    st.markdown("---")
    # FIELD NOTEBOOK
    with st.expander("📓 Field Notebook", expanded=False):
        st.caption("Notes persist across days.")
        st.session_state.notebook_entries = st.text_area(
            "My Notes:", 
            value=st.session_state.notebook_entries, 
            height=200, 
            key="persistent_notes"
        )

    st.markdown("### 🧭 Navigation")
    if st.button("📞 Briefing", use_container_width=True): st.session_state.current_view = 'briefing'
    if st.button("👥 Interviews", use_container_width=True): st.session_state.current_view = 'interviews'
    if st.button("📋 Line List", use_container_width=True): st.session_state.current_view = 'linelist'
    if st.button("📍 Spot Map", use_container_width=True): st.session_state.current_view = 'spotmap'
    if st.button("🗺️ Field Sites", use_container_width=True): st.session_state.current_view = 'map'
    
    if current_day >= 2: st.button("🔬 Study Design", use_container_width=True)
    
    # Dirty Data Logic for Day 3 Analysis
    if current_day >= 3:
        if st.button("📊 Analysis", use_container_width=True):
            st.session_state.current_view = 'analysis'
            st.rerun()
            
    if current_day >= 4: st.button("🧪 Laboratory", use_container_width=True)
    if current_day >= 5: st.button("🏛️ MOH Briefing", use_container_width=True)
    
    st.markdown("---")
    if current_day < 5:
        if st.button(f"💤 End Day {current_day}", use_container_width=True, type="primary"):
            if advance_day(): st.rerun()

# ============================================================================
# VIEWS
# ============================================================================

if st.session_state.current_view == 'briefing':
    st.markdown("### 🚨 Incoming Alert")
    st.markdown("""<div class="transcript-box"><strong>From:</strong> DHO<br>"8 confirmed AES cases. 2 deaths. Nalu Village. Investigate immediately."</div>""", unsafe_allow_html=True)
    
    # Fictional Map
    st.markdown("### 📍 Sidero Valley Overview")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 400], y=[50, 100], mode='lines', line=dict(color='blue', width=4), name='River'))
    fig.add_shape(type="rect", x0=50, y0=250, x1=150, y1=350, fillcolor="rgba(144,238,144,0.4)", line_width=0)
    fig.add_trace(go.Scatter(x=[100, 250, 350], y=[300, 200, 100], mode='markers+text', marker=dict(size=20, color=['red','orange','yellow']), text=['Nalu', 'Kabwe', 'Tamu']))
    fig.update_layout(height=300, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    with st.form("case_def"):
        st.markdown("### 📝 Develop Case Definition")
        st.text_area("Clinical Criteria:")
        st.form_submit_button("Save")

elif st.session_state.current_view == 'interviews':
    st.markdown("### 👥 Key Informant Interviews")
    st.info("🕒 Cost: 1 Hour per interview | 💰 Cost: varies")
    
    if st.session_state.current_character:
        char = CHARACTERS[st.session_state.current_character]
        st.markdown(f"#### Interviewing {char['name']}")
        if st.button("🔙 End"):
            st.session_state.current_character = None
            st.rerun()
            
        for msg in st.session_state.interview_history.get(st.session_state.current_character, []):
            with st.chat_message(msg['role']): st.write(msg['content'])
            
        if prompt := st.chat_input("Ask a question..."):
            with st.chat_message("user"): st.write(prompt)
            # Add logic here to check unlock triggers
            resp = get_ai_response(st.session_state.current_character, prompt, st.session_state.interview_history.get(st.session_state.current_character, []))
            with st.chat_message("assistant"): st.write(resp)
            if st.session_state.current_character not in st.session_state.interview_history:
                 st.session_state.interview_history[st.session_state.current_character] = []
            st.session_state.interview_history[st.session_state.current_character].append({"role": "user", "content": prompt})
            st.session_state.interview_history[st.session_state.current_character].append({"role": "assistant", "content": resp})
    else:
        cols = st.columns(4)
        for k, v in CHARACTERS.items():
            is_unlocked = k in st.session_state.npcs_unlocked
            with cols[list(CHARACTERS.keys()).index(k)%4]:
                if is_unlocked:
                    st.markdown(f"**{v['avatar']} {v['name']}**")
                    st.caption(f"${v['cost']}")
                    if st.button("Talk (1hr)", key=k):
                        if spend_time(1.0):
                            if st.session_state.budget >= v['cost']:
                                st.session_state.budget -= v['cost']
                                st.session_state.current_character = k
                                st.rerun()
                            else: st.error("No Funds")
                else:
                    st.markdown(f"🔒 {v['name']}")

elif st.session_state.current_view == 'linelist':
    st.markdown("### 📋 Case Line List (Hospital Data)")
    st.dataframe(pd.DataFrame([
        {'ID': 'DH-01', 'Age': 4, 'Village': 'Nalu', 'Onset': 'Jun 03', 'Outcome': 'Recovered'},
        {'ID': 'DH-02', 'Age': 6, 'Village': 'Nalu', 'Onset': 'Jun 04', 'Outcome': 'Died'},
        {'ID': 'DH-03', 'Age': 32, 'Village': 'Kabwe', 'Onset': 'Jun 04', 'Outcome': 'Died'}
    ]), use_container_width=True)

elif st.session_state.current_view == 'spotmap':
    st.markdown("### 📍 Spot Map")
    # Fixed Spot Map logic from previous correct iteration
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 400], y=[50, 100], mode='lines', line=dict(color='lightblue', width=10), name='River'))
    fig.add_shape(type="rect", x0=50, y0=250, x1=150, y1=350, fillcolor="rgba(144,238,144,0.3)", line_width=0)
    fig.add_trace(go.Scatter(x=[100, 250, 350], y=[300, 200, 100], mode='text', text=['Nalu', 'Kabwe', 'Tamu']))
    # Cases
    fig.add_trace(go.Scatter(x=[90, 110, 105, 240, 260], y=[310, 290, 305, 210, 190], mode='markers', marker=dict(color='red', size=10), name='Case'))
    fig.update_layout(height=500, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

elif st.session_state.current_view == 'map':
    st.markdown("### 🗺️ Field Sites")
    st.info("🕒 Cost: 2 Hours + $100 per inspection")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Inspect Pig Farms"):
            if spend_time(2.0):
                if st.session_state.budget >= 100:
                    st.session_state.budget -= 100
                    st.image("https://images.unsplash.com/photo-1516467508483-a72120608ae7?w=400", caption="Pig Farms")
                else: st.error("No Funds")
    with col2:
        if st.button("Inspect Rice Paddies"):
            if spend_time(2.0):
                 if st.session_state.budget >= 100:
                    st.session_state.budget -= 100
                    st.image("https://images.unsplash.com/photo-1550989460-0adf9ea622e2?w=400", caption="Rice Paddies")
                 else: st.error("No Funds")

# ============================================================================
# ANALYSIS & DIRTY DATA MODULE
# ============================================================================
elif st.session_state.current_view == 'analysis':
    st.markdown("### 📊 Data Analysis Lab")
    
    if not st.session_state.data_cleaned:
        st.warning("⚠️ ALERT: The field team sent the raw dataset. It contains errors. You must clean it before analysis.")
        
        # Show Dirty Data
        df_dirty = get_dirty_dataset()
        st.dataframe(df_dirty.head(10))
        
        st.markdown("#### 🧹 Data Cleaning Workbench")
        
        col1, col2 = st.columns(2)
        with col1:
            clean_sex = st.checkbox("Standardize 'Sex' (M/F)")
            clean_age = st.checkbox("Remove impossible Ages (>100)")
        with col2:
            clean_date = st.checkbox("Fix Date Formats (YYYY-MM-DD)")
            clean_village = st.checkbox("Standardize Village Names")
            
        if st.button("Run Cleaning Script"):
            if clean_sex and clean_age and clean_date:
                st.session_state.data_cleaned = True
                st.success("✅ Data Cleaned Successfully! Analysis unlocked.")
                st.rerun()
            else:
                st.error("❌ You missed some errors. Look closely at Age, Sex, and Dates.")
    
    else:
        # CLEANED DATA ANALYSIS
        st.success("✅ Working with Cleaned Dataset (n=20)")
        
        # Dummy Clean Data for visualization
        df_clean = pd.DataFrame({
            'Sex': ['M', 'F', 'M', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
            'Outcome': ['Recovered', 'Died', 'Died', 'Recovered', 'Recovered', 'Recovered', 'Recovered', 'Recovered', 'Recovered', 'Recovered']
        })
        
        tab1, tab2 = st.tabs(["Descriptive Epi", "Measures of Association"])
        
        with tab1:
            st.markdown("**Sex Distribution**")
            st.bar_chart(df_clean['Sex'].value_counts())
            
        with tab2:
            st.markdown("**2x2 Tables**")
            st.caption("Calculate Odds Ratios here.")
            st.table(pd.DataFrame({'Cases': [10, 5], 'Controls': [2, 12]}, index=['Exposed', 'Unexposed']))
            st.metric("Odds Ratio", "12.0")