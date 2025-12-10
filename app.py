import streamlit as st
import anthropic
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(
    page_title="FETP Sim: Sidero Valley",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background-color: #2E4053;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .handwritten-note {
        font-family: 'Comic Sans MS', 'Chalkboard SE', 'Marker Felt', sans-serif;
        font-size: 16px;
        background-color: #fdf6e3;
        color: #2c3e50;
        padding: 15px;
        border: 1px solid #d6d6d6;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
        transform: rotate(-0.5deg);
    }
    .stChatInput {
        position: fixed;
        bottom: 0;
        padding-bottom: 20px;
        z-index: 1000;
    }
    .step-completed {
        color: green;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'current_view' not in st.session_state: st.session_state.current_view = 'home'
if 'interview_history' not in st.session_state: st.session_state.interview_history = {}
if 'interviewed_characters' not in st.session_state: st.session_state.interviewed_characters = set()
if 'private_clinic_unlocked' not in st.session_state: st.session_state.private_clinic_unlocked = False
if 'manually_entered_cases' not in st.session_state: st.session_state.manually_entered_cases = []
if 'notes' not in st.session_state: st.session_state.notes = []

# Tracker State (The 13 Steps)
if 'completed_steps' not in st.session_state: st.session_state.completed_steps = set()

# --- STORY & DATA ---
STORY_CONTEXT = "Situation: 'Shaking Sickness' in Sidero Valley. High fever, tremors, seizures. Setting: Mine (North), Farms (South)."

# Clean Hospital Data (6 Cases)
PUBLIC_CASES = [
    {"ID": "H-01", "Age": 45, "Sex": "M", "Occupation": "Miner", "Onset": "2025-12-08", "Symptoms": "Fever, Tremors", "Status": "Alive"},
    {"ID": "H-02", "Age": 28, "Sex": "M", "Occupation": "Miner", "Onset": "2025-12-09", "Symptoms": "Fever, Confusion", "Status": "Alive"},
    {"ID": "H-03", "Age": 52, "Sex": "M", "Occupation": "Miner", "Onset": "2025-12-09", "Symptoms": "Seizures, Coma", "Status": "Deceased"},
    {"ID": "H-04", "Age": 33, "Sex": "M", "Occupation": "Miner", "Onset": "2025-12-10", "Symptoms": "Fever, Tremors", "Status": "Alive"},
    {"ID": "H-05", "Age": 41, "Sex": "M", "Occupation": "Miner", "Onset": "2025-12-10", "Symptoms": "Fever, Ataxia", "Status": "Alive"},
    {"ID": "H-06", "Age": 29, "Sex": "M", "Occupation": "Miner", "Onset": "2025-12-10", "Symptoms": "Headache, Fever", "Status": "Alive"},
]

# Messy Clinic Notes (User must filter these)
CLINIC_NOTES_PILE = [
    "Dec 7. Sarah (6y F). Pig farm. High fever, shaking hands. Mom says she fell in mud.",
    "Dec 7. Old John (65y M). Farmer. Complains of back pain from planting rice. No fever.",
    "Dec 8. Twin boys (8y M). Farm B. Both vomiting and twitching. Fever very high.",
    "Dec 8. Mary (24y F). Pregnant. Routine checkup. Healthy.",
    "Dec 9. Mrs. Adama (40y F). Collapsed in field. Eyes rolling back. Seizing.",
    "Dec 9. Boy (12y). Cut leg on rusty fence. Tetanus shot given.",
    "Dec 9. Baby K (2y M). High fever, stiff neck, screaming. Fontanelle bulging.",
    "Dec 10. Miner Tom (30y M). Coughing blood. TB suspect. Refer to hospital.",
    "Dec 10. Girl (5y). Pig farm. Fever, confused, can't walk straight.",
    "Dec 10. Farmer Ben (50y). Broken arm from tractor accident.",
]

CHARACTERS = {
    "dr_chen": {
        "name": "Dr. Elena Chen",
        "role": "Director, St. Mary's Hospital",
        "avatar": "👩‍⚕️",
        "personality": "Professional, academic. Relies on data.",
        "truth_document": "You have 6 adult male patients (miners). You suspect toxic gas.",
        "data_access": PUBLIC_CASES
    },
    "healer_marcus": {
        "name": "Marcus the Healer",
        "role": "Private Clinic Practitioner",
        "avatar": "🌿",
        "personality": "Suspicious, traditional. Protective of farmers.",
        "truth_document": "You treat farmers and children. They have the shaking sickness too.",
        "data_access": CLINIC_NOTES_PILE
    }
}

# --- FUNCTIONS ---

def get_ai_response(char_key, user_input, history):
    """AI with Data Injection Fix"""
    char = CHARACTERS[char_key]
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key: return "⚠️ API Key Missing"
    
    # Inject Data so AI doesn't hallucinate
    data_context = f"DATA ACCESS: {str(char.get('data_access', 'None'))}"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msgs = [{"role": m["role"], "content": m["content"]} for m in history]
        msgs.append({"role": "user", "content": user_input})
        
        system_prompt = f"""
        Roleplay {char['name']}. Context: {STORY_CONTEXT}. 
        {data_context}
        INSTRUCTIONS: Refer strictly to your Data Access for patient counts/details. Do not invent patients.
        """
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=250,
            system=system_prompt,
            messages=msgs
        )
        return response.content[0].text
    except Exception as e:
        return f"System Error: {e}"

# --- MAIN LAYOUT ---

st.markdown('<div class="main-header"><h1>🏔️ Sidero Valley Investigation</h1></div>', unsafe_allow_html=True)

# --- SIDEBAR NAV & TRACKER ---
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    if st.button("🏠 Briefing", use_container_width=True): st.session_state.current_view = 'home'
    if st.button("🗺️ Valley Map", use_container_width=True): st.session_state.current_view = 'map'
    if st.button("👥 Interviews", use_container_width=True): st.session_state.current_view = 'contacts'
    if st.button("🏥 Hospital Records", use_container_width=True): st.session_state.current_view = 'hospital_data'
    if st.button("🏚️ Clinic Records (Triage)", use_container_width=True): st.session_state.current_view = 'clinic_data'
    st.markdown("---")
    if st.button("📊 **Data Analysis Lab**", use_container_width=True): st.session_state.current_view = 'analysis'

    # THE 13 STEPS CHECKLIST
    st.markdown("---")
    with st.expander("📋 **Investigation Checklist**", expanded=False):
        st.caption("Mark steps as you complete them:")
        steps = [
            "1. Confirm Outbreak", "2. Prepare for Field Work", "3. Verify Diagnosis",
            "4. Case Definition", "5. Find & Record Cases", "6. Descriptive Epi",
            "7. Develop Hypotheses", "8. Evaluate Hypotheses", "9. Refine/Execute Studies",
            "10. Triangulate Data", "11. Prevention Measures", "12. Surveillance",
            "13. Communicate Findings"
        ]
        for step in steps:
            is_done = step in st.session_state.completed_steps
            if st.checkbox(step, value=is_done, key=f"chk_{step}"):
                st.session_state.completed_steps.add(step)
            else:
                st.session_state.completed_steps.discard(step)

# --- VIEWS ---

if st.session_state.current_view == 'home':
    st.info("MISSION: Investigate cluster of 'Acute Encephalitis' in Sidero Valley.")
    st.markdown("""
    **Intelligence:**
    * **Dr. Chen (Hospital):** Reports 6 cases (Miners). Suspects Toxin.
    * **Rumors:** Farmers are also sick, but avoiding the hospital.
    
    **Immediate Tasks:**
    1.  Interview Dr. Chen for hospital records.
    2.  Find Healer Marcus to access community cases.
    3.  Build a Master Line List in the Analysis Lab.
    """)

elif st.session_state.current_view == 'map':
    st.markdown("### 🗺️ Sidero Valley")
    # Simple image placeholder or plotly map can go here
    st.image("https://placehold.co/800x400/2c3e50/FFF?text=North:+Mines+|+South:+Farms+|+Center:+Hospital", caption="Valley Layout")

elif st.session_state.current_view == 'contacts':
    st.markdown("### 👥 Interviews")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Talk to Dr. Chen (Hospital)"):
            st.session_state.current_character = 'dr_chen'
            st.session_state.current_view = 'interview'
            st.rerun()
    with col2:
        if st.button("Talk to Healer Marcus (Clinic)"):
            st.session_state.current_character = 'healer_marcus'
            st.session_state.current_view = 'interview'
            st.rerun()

elif st.session_state.current_view == 'interview':
    char = CHARACTERS[st.session_state.current_character]
    if st.session_state.current_character == 'healer_marcus': st.session_state.private_clinic_unlocked = True
    
    st.markdown(f"### 💬 Interview: {char['name']}")
    if st.button("🔙 Back"): st.session_state.current_view = 'contacts'; st.rerun()
    
    # Init history
    if st.session_state.current_character not in st.session_state.interview_history:
        st.session_state.interview_history[st.session_state.current_character] = []
    
    history = st.session_state.interview_history[st.session_state.current_character]
    for msg in history:
        with st.chat_message(msg['role']): st.write(msg['content'])
            
    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user"): st.write(prompt)
        history.append({"role": "user", "content": prompt})
        with st.chat_message("assistant", avatar=char['avatar']):
            resp = get_ai_response(st.session_state.current_character, prompt, history[:-1])
            st.write(resp)
        history.append({"role": "assistant", "content": resp})

elif st.session_state.current_view == 'hospital_data':
    st.markdown("### 🏥 Hospital Records (Clean)")
    if 'dr_chen' in st.session_state.interviewed_characters: # Check if interviewed
        st.dataframe(pd.DataFrame(PUBLIC_CASES))
    else:
        st.warning("🔒 Talk to Dr. Chen first.")

elif st.session_state.current_view == 'clinic_data':
    st.markdown("### 🏚️ Private Clinic Records (Abstraction Task)")
    
    if not st.session_state.private_clinic_unlocked:
        st.warning("🔒 Find Healer Marcus to unlock this location.")
    else:
        st.info("📝 **Task:** Read the notes. Digitize ONLY the cases fitting the symptoms (Fever + Neuro).")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 📥 The Notes")
            for note in CLINIC_NOTES_PILE:
                st.markdown(f'<div class="handwritten-note">{note}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 💻 Data Entry")
            with st.form("entry_form"):
                c_id = st.text_input("Patient ID (e.g. C-01)")
                c_age = st.number_input("Age", 0, 100)
                c_sex = st.selectbox("Sex", ["M", "F"])
                c_occ = st.selectbox("Occupation", ["Miner", "Farmer", "Child", "Other"])
                c_onset = st.date_input("Onset Date")
                c_sympt = st.multiselect("Symptoms", ["Fever", "Tremors", "Seizures", "Confusion", "Vomiting"])
                
                if st.form_submit_button("➕ Add Case"):
                    if c_id:
                        new_case = {
                            "ID": c_id, "Age": c_age, "Sex": c_sex, "Occupation": c_occ, 
                            "Onset": str(c_onset), "Symptoms": ", ".join(c_sympt), "Source": "Clinic"
                        }
                        st.session_state.manually_entered_cases.append(new_case)
                        st.success(f"Added {c_id}")

elif st.session_state.current_view == 'analysis':
    st.markdown("## 📊 Data Analysis Lab")
    st.info("Combine Hospital Data with your Manual Entries here. Then build your charts.")
    
    # 1. Merge Data
    df_h = pd.DataFrame(PUBLIC_CASES)
    df_h['Source'] = 'Hospital'
    
    if st.session_state.manually_entered_cases:
        df_c = pd.DataFrame(st.session_state.manually_entered_cases)
        df_master = pd.concat([df_h, df_c], ignore_index=True)
    else:
        df_master = df_h
        st.caption("Only Hospital data available. Go to 'Clinic Records' to add more.")

    with st.expander("📄 View Master Line List", expanded=True):
        st.dataframe(df_master, use_container_width=True)
        csv = df_master.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "linelist.csv")

    st.markdown("---")
    st.markdown("### 📈 Chart Builder (Step 6)")
    
    colA, colB, colC = st.columns(3)
    with colA:
        x_axis = st.selectbox("X-Axis Variable", options=df_master.columns)
    with colB:
        y_axis = st.selectbox("Y-Axis Variable (Optional)", options=["None"] + list(df_master.columns))
    with colC:
        chart_type = st.selectbox("Chart Type", ["Bar Chart", "Histogram", "Pie Chart", "Scatter Plot"])
        
    if st.button("Generate Chart"):
        try:
            if chart_type == "Bar Chart":
                if y_axis != "None":
                    fig = px.bar(df_master, x=x_axis, y=y_axis)
                else:
                    fig = px.bar(df_master, x=x_axis, title=f"Count by {x_axis}")
            elif chart_type == "Histogram":
                fig = px.histogram(df_master, x=x_axis)
            elif chart_type == "Pie Chart":
                fig = px.pie(df_master, names=x_axis)
            elif chart_type == "Scatter Plot":
                fig = px.scatter(df_master, x=x_axis, y=y_axis)
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate chart: {e}")