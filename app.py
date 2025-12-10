import streamlit as st
import anthropic
import pandas as pd
import plotly.graph_objects as go
import numpy as np
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
    .transcript-box {
        background-color: #f8f9fa;
        border-left: 4px solid #dc3545;
        padding: 15px;
        font-family: 'Courier New', monospace;
        margin-bottom: 15px;
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
    .mentor-feedback {
        background-color: #e8f4f8;
        border-left: 5px solid #00a0dc;
        padding: 15px;
        margin-top: 10px;
    }
    .stChatInput {
        position: fixed;
        bottom: 0;
        padding-bottom: 20px;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'current_view' not in st.session_state: st.session_state.current_view = 'briefing'
if 'interview_history' not in st.session_state: st.session_state.interview_history = {}
if 'interviewed_count' not in st.session_state: st.session_state.interviewed_count = 0
if 'interview_limit' not in st.session_state: st.session_state.interview_limit = 3
if 'private_clinic_unlocked' not in st.session_state: st.session_state.private_clinic_unlocked = False
if 'manually_entered_cases' not in st.session_state: st.session_state.manually_entered_cases = []
if 'completed_steps' not in st.session_state: st.session_state.completed_steps = set()

# Study Design State
if 'case_definition' not in st.session_state: st.session_state.case_definition = ""
if 'mentor_feedback' not in st.session_state: st.session_state.mentor_feedback = ""
if 'questionnaire_text' not in st.session_state: st.session_state.questionnaire_text = ""
if 'mapped_columns' not in st.session_state: st.session_state.mapped_columns = []

# --- TRUTH DATA (HIDDEN) ---
@st.cache_data
def generate_hidden_population():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'ID': range(1, n+1),
        'Age': np.random.randint(1, 80, n),
        'Sex': np.random.choice(['M', 'F'], n),
        'Occupation': np.random.choice(['Miner', 'Farmer', 'Other'], n, p=[0.3, 0.5, 0.2]),
        'Zone': np.random.choice(['North (Industrial)', 'South (Agri)', 'Central (Res)'], n),
        # True Risk Factors (Hidden)
        'Pigs_Near_Home': np.random.choice([True, False], n),
        'Mosquito_Net_Use': np.random.choice([True, False], n),
        'Drank_River_Water': np.random.choice([True, False], n),
        'Vaccinated_JE': np.random.choice([True, False], n),
        'Recent_Travel': np.random.choice([True, False], n),
    })
    
    # Disease Logic (Japanese Encephalitis Proxy)
    # Risk = Pigs + No Net + South Zone
    df['Risk_Score'] = 0
    df.loc[df['Pigs_Near_Home'], 'Risk_Score'] += 3
    df.loc[~df['Mosquito_Net_Use'], 'Risk_Score'] += 2
    df.loc[(df['Zone']=='South (Agri)'), 'Risk_Score'] += 1
    
    probs = df['Risk_Score'] / 12
    df['Is_Case'] = np.random.rand(n) < probs
    
    return df

HIDDEN_POP = generate_hidden_population()

# --- CHARACTERS (8 Options) ---
CHARACTERS = {
    "dr_chen": {
        "name": "Dr. Elena Chen", "role": "Hospital Director", "avatar": "👩‍⚕️",
        "bio": "Overwhelmed. Focused on the severe cases in miners.",
        "truth": "All my severe cases are miners. I think it's the mine ventilation."
    },
    "healer_marcus": {
        "name": "Marcus the Healer", "role": "Private Practitioner", "avatar": "🌿",
        "bio": "Treats the poor. Distrusts authority.",
        "truth": "It's not just miners. Children are dying too. I have notes."
    },
    "foreman_rex": {
        "name": "Foreman Rex", "role": "Mine Manager", "avatar": "👷",
        "bio": "Defensive. Worried about mine closure.",
        "truth": "The mine is safe! We pass all inspections. It's the water."
    },
    "mama_kofi": {
        "name": "Mama Kofi", "role": "Mother of Case", "avatar": "👵",
        "bio": "Her 6-year-old is in coma. Lives near pig farm.",
        "truth": "My boy was playing near the sty. Then the fever came."
    },
    "mayor_simon": {
        "name": "Mayor Simon", "role": "Politician", "avatar": "👔",
        "bio": "Worried about the election and the economy.",
        "truth": "Fix this quietly. Don't scare the investors."
    },
    "nurse_joy": {
        "name": "Nurse Joy", "role": "Triage Nurse", "avatar": "🩹",
        "bio": "Sees everyone who walks in.",
        "truth": "We are running out of beds. It's mostly fever and shakes."
    },
    "teacher_grace": {
        "name": "Teacher Grace", "role": "School Principal", "avatar": "📚",
        "bio": "Noticed absences.",
        "truth": "Many children from the South farms are missing school."
    },
    "patient_zero": {
        "name": "Patient Tom", "role": "Recovering Case (Miner)", "avatar": "🤕",
        "bio": "Confused but recovering.",
        "truth": "I don't know. I just woke up shaking. I work in the deep shaft."
    }
}

# --- FUNCTIONS ---

def get_ai_response(char_key, user_input, history):
    """Character AI"""
    char = CHARACTERS[char_key]
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key: return "⚠️ API Key Missing"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msgs = [{"role": m["role"], "content": m["content"]} for m in history]
        msgs.append({"role": "user", "content": user_input})
        system_prompt = f"Roleplay {char['name']}. Truth: {char['truth']}. Bio: {char['bio']}."
        response = client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=200, system=system_prompt, messages=msgs
        )
        return response.content[0].text
    except Exception as e: return f"Error: {e}"

def map_questions_to_columns(user_questions):
    """LLM Mapper: Converts free text to DataFrame columns"""
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key: return []
    
    # The columns available in our 'Truth' dataset
    available_cols = list(HIDDEN_POP.columns)
    
    system_prompt = f"""
    You are a Data Mapper. 
    User Questions: "{user_questions}"
    Available Database Columns: {available_cols}
    
    Task: Return a JSON list of columns that best match the user's questions. 
    Example: "Do you use nets?" -> ["Mosquito_Net_Use"]
    If no match, ignore. return ONLY the JSON list.
    """
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=100, system=system_prompt, 
            messages=[{"role": "user", "content": "Map these."}]
        )
        # Parse logic would go here (simplified for demo)
        # For now, simplistic string matching as fallback if AI fails or for robustness
        import json
        text = response.content[0].text
        # Clean up the AI response to ensure it's a list
        start = text.find('[')
        end = text.find(']') + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
        return []
    except:
        return []

def render_map():
    """Improved Plotly Map"""
    fig = go.Figure()

    # Zones (Rectangles)
    fig.add_shape(type="rect", x0=0, y0=200, x1=200, y1=400, fillcolor="rgba(169, 169, 169, 0.3)", line_width=0) # North (Industrial)
    fig.add_annotation(x=100, y=350, text="🏭 NORTH (Mines)", showarrow=False)

    fig.add_shape(type="rect", x0=0, y0=0, x1=400, y1=200, fillcolor="rgba(144, 238, 144, 0.3)", line_width=0) # South (Agri)
    fig.add_annotation(x=200, y=50, text="🌾 SOUTH (Farms)", showarrow=False)
    
    fig.add_shape(type="rect", x0=200, y0=200, x1=400, y1=400, fillcolor="rgba(255, 228, 196, 0.3)", line_width=0) # Central
    fig.add_annotation(x=300, y=350, text="🏘️ CENTRAL (Town)", showarrow=False)

    # River (Path)
    fig.add_trace(go.Scatter(x=[0, 100, 200, 300, 400], y=[50, 80, 150, 180, 100], 
                             mode='lines', line=dict(color='blue', width=4), name='River Sidero'))

    # Landmarks
    fig.add_trace(go.Scatter(x=[300], y=[300], mode='markers+text', marker=dict(size=15, color='red', symbol='cross'), text=["Hospital"], textposition="top center", name="Hospital"))
    fig.add_trace(go.Scatter(x=[50], y=[100], mode='markers+text', marker=dict(size=15, color='green', symbol='circle'), text=["Pigs"], textposition="top center", name="Pig Farms"))
    
    fig.update_layout(xaxis=dict(range=[0,400], showgrid=False, showticklabels=False), 
                      yaxis=dict(range=[0,400], showgrid=False, showticklabels=False),
                      height=400, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP ---
st.markdown('<div class="main-header"><h1>🏔️ Sidero Valley Investigation</h1></div>', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    if st.button("📞 Briefing", use_container_width=True): st.session_state.current_view = 'briefing'
    if st.button("👥 Interviews (Pick 3)", use_container_width=True): st.session_state.current_view = 'contacts'
    if st.button("🏥 Data Abstraction", use_container_width=True): st.session_state.current_view = 'clinic_data'
    st.markdown("---")
    if st.button("🔬 **Study Design Lab**", use_container_width=True): st.session_state.current_view = 'study_design'

    st.markdown("---")
    with st.expander("📋 **Checklist**"):
        for step in ["1. Confirm Outbreak", "4. Case Definition", "7. Hypotheses", "8. Study Design", "13. Report"]:
            st.checkbox(step, value=step in st.session_state.completed_steps, key=step)

# --- VIEWS ---

if st.session_state.current_view == 'briefing':
    st.markdown("### 🚨 Incoming Call: District Health Officer")
    st.markdown("""
    <div class="transcript-box">
    <strong>[AUDIO TRANSCRIPT - 08:42 AM]</strong><br><br>
    <strong>DHO:</strong> "Is this Team Alpha? Listen, we have a situation. Sidero Valley.<br>
    <strong>You:</strong> "Go ahead."<br>
    <strong>DHO:</strong> "St. Mary's Hospital is reporting six deaths in 48 hours. Neuro symptoms. Seizures, rigidity. The Hospital Director says it's poison gas from the mines, but the rumors... my phone is blowing up. Farmers say their kids are waking up screaming."<br>
    <strong>You:</strong> "Has anyone gone out?"<br>
    <strong>DHO:</strong> "No. The miners are threatening to strike if we close the shaft. The farmers are blaming the miners. It's a powder keg. I need you there NOW. Figure out what is killing these people."<br>
    <strong>[CALL ENDED]</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Objective:** You have limited time. Review the map, then choose your interview targets carefully.")
    render_map()
    if st.button("Deploy to Field"): st.session_state.current_view = 'contacts'; st.rerun()

elif st.session_state.current_view == 'contacts':
    st.markdown(f"### 👥 Key Contacts (Interviews Remaining: {3 - st.session_state.interviewed_count})")
    
    if st.session_state.interviewed_count >= 3:
        st.warning("⛔ You have reached your interview limit for the Initial Assessment phase.")
    
    cols = st.columns(4)
    for idx, (key, char) in enumerate(CHARACTERS.items()):
        with cols[idx % 4]:
            st.markdown(f"**{char['avatar']} {char['name']}**")
            st.caption(char['role'])
            if st.button(f"Talk", key=key, disabled=st.session_state.interviewed_count >= 3):
                st.session_state.current_character = key
                st.session_state.current_view = 'interview'
                st.rerun()

elif st.session_state.current_view == 'interview':
    char = CHARACTERS[st.session_state.current_character]
    if st.session_state.current_character == 'healer_marcus': st.session_state.private_clinic_unlocked = True
    
    st.markdown(f"### 💬 {char['name']}")
    if st.button("🔙 End Interview"): 
        if st.session_state.current_character not in st.session_state.interviewed_characters:
            st.session_state.interviewed_count += 1
            st.session_state.interviewed_characters.add(st.session_state.current_character)
        st.session_state.current_view = 'contacts'
        st.rerun()
    
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

elif st.session_state.current_view == 'clinic_data':
    st.markdown("### 🏥 Data Abstraction")
    st.info("If you found the Private Clinic, digitize the records here.")
    if st.session_state.private_clinic_unlocked:
        st.write("*(Notes available...)*")
        # Reuse previous logic for notes/abstraction
    else:
        st.warning("🔒 You need to find the Private Practitioner first.")

elif st.session_state.current_view == 'study_design':
    st.markdown("## 🔬 Phase 2: Analytic Study Protocol")
    
    # 1. Study Type
    st.markdown("### 1. Study Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Study Design", ["Case-Control", "Retrospective Cohort", "Cross-Sectional"])
    with col2:
        st.selectbox("Sampling Strategy", ["Random Sampling", "Cluster Sampling", "Systematic Sampling"])
    
    st.markdown("---")
    
    # 2. Questionnaire (Free Text)
    st.markdown("### 2. Questionnaire Design")
    st.write("Write the questions you want to ask the community. Be specific.")
    
    q_text = st.text_area("Enter your questions (one per line):", height=150, 
                          placeholder="e.g.\nDid you eat pork recently?\nDo you sleep under a mosquito net?")
    
    if st.button("Submit Protocol & Generate Data"):
        if not q_text:
            st.error("Please write your questionnaire.")
        else:
            with st.spinner("AI is analyzing your questions against the Truth Engine..."):
                # LLM Maps questions to columns
                mapped_cols = map_questions_to_columns(q_text)
                st.session_state.mapped_columns = mapped_cols
                
                # Show results
                st.success("Protocol Approved. Field Team deployed.")
                st.markdown(f"**Data Collected:** Based on your questions, we gathered data on: `{mapped_cols}`")
                
                # Preview the resulting dataframe (The 'Truth' filtered by their questions)
                if mapped_cols:
                    preview_df = HIDDEN_POP[['ID', 'Age', 'Sex', 'Zone', 'Is_Case'] + [c for c in mapped_cols if c in HIDDEN_POP.columns]].head(10)
                    st.dataframe(preview_df)
                else:
                    st.warning("Your questions didn't match any known risk factors in the database. Try asking about animals, water, or protection.")