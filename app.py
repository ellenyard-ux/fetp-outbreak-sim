import streamlit as st
import anthropic
import pandas as pd
import plotly.graph_objects as go
import numpy as np

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
    .resource-box {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #b8daff;
        text-align: center;
        margin-bottom: 10px;
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
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'current_view' not in st.session_state: st.session_state.current_view = 'briefing'
if 'interview_history' not in st.session_state: st.session_state.interview_history = {}
if 'private_clinic_unlocked' not in st.session_state: st.session_state.private_clinic_unlocked = False
if 'manually_entered_cases' not in st.session_state: st.session_state.manually_entered_cases = []
if 'completed_steps' not in st.session_state: st.session_state.completed_steps = set()

# Resources
if 'budget' not in st.session_state: st.session_state.budget = 2500
if 'lab_credits' not in st.session_state: st.session_state.lab_credits = 10

# Study Design
if 'case_definition' not in st.session_state: st.session_state.case_definition = ""
if 'mapped_columns' not in st.session_state: st.session_state.mapped_columns = []

# --- TRUTH DATA ---
@st.cache_data
def generate_hidden_population():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'ID': range(1, n+1),
        'Age': np.random.randint(1, 80, n),
        'Sex': np.random.choice(['M', 'F'], n),
        'Occupation': np.random.choice(['Miner', 'Farmer', 'Merchant', 'Student'], n, p=[0.25, 0.4, 0.15, 0.2]),
        'Zone': np.random.choice(['North (Mines)', 'South (Farms)', 'Central (Town)', 'East (Forest)'], n),
        # Risk Factors
        'Pigs_Near_Home': np.random.choice([True, False], n),
        'Mosquito_Net_Use': np.random.choice([True, False], n),
        'Drank_River_Water': np.random.choice([True, False], n),
        'Eat_Bushmeat': np.random.choice([True, False], n),
    })
    # Disease Logic (JE Proxy: Pigs + South + No Net)
    df['Risk_Score'] = 0
    df.loc[df['Pigs_Near_Home'], 'Risk_Score'] += 3
    df.loc[~df['Mosquito_Net_Use'], 'Risk_Score'] += 2
    df.loc[(df['Zone']=='South (Farms)'), 'Risk_Score'] += 1
    probs = df['Risk_Score'] / 12
    df['Is_Case'] = np.random.rand(n) < probs
    return df

HIDDEN_POP = generate_hidden_population()

# Hospital Data (Always Visible)
PUBLIC_CASES = [
    {"ID": "H-01", "Age": 45, "Sex": "M", "Occ": "Miner", "Onset": "Dec 8", "Symp": "Fever, Tremors"},
    {"ID": "H-02", "Age": 28, "Sex": "M", "Occ": "Miner", "Onset": "Dec 9", "Symp": "Confusion, Fever"},
    {"ID": "H-03", "Age": 52, "Sex": "M", "Occ": "Miner", "Onset": "Dec 9", "Symp": "Seizures (Died)"},
    {"ID": "H-04", "Age": 33, "Sex": "M", "Occ": "Miner", "Onset": "Dec 10", "Symp": "Rigidity"},
]

# Clinic Notes (Locked)
CLINIC_NOTES_PILE = [
    "Dec 7. Sarah (6y F). Pig farm. High fever, shaking hands.",
    "Dec 8. Twin boys (8y M). Farm B. Both vomiting and twitching.",
    "Dec 9. Mrs. Adama (40y F). Collapsed in field. Seizing.",
    "Dec 9. Baby K (2y M). High fever, stiff neck, screaming.",
    "Dec 10. Girl (5y). Pig farm. Fever, confused.",
]

# Characters
CHARACTERS = {
    "dr_chen": {"name": "Dr. Chen", "role": "Hospital Director", "avatar": "👩‍⚕️", "cost": 100, "bio": "Focuses on Miners."},
    "healer_marcus": {"name": "Healer Marcus", "role": "Private Clinic", "avatar": "🌿", "cost": 150, "bio": "Suspicious of govt."},
    "foreman_rex": {"name": "Foreman Rex", "role": "Mine Manager", "avatar": "👷", "cost": 200, "bio": "Defensive."},
    "mama_kofi": {"name": "Mama Kofi", "role": "Mother of Case", "avatar": "👵", "cost": 100, "bio": "Grieving."},
    "mayor_simon": {"name": "Mayor Simon", "role": "Politician", "avatar": "👔", "cost": 50, "bio": "Worried about money."},
    "nurse_joy": {"name": "Nurse Joy", "role": "Triage Nurse", "avatar": "🩹", "cost": 50, "bio": "Overworked."},
    "teacher_grace": {"name": "Teacher Grace", "role": "School Principal", "avatar": "📚", "cost": 50, "bio": "Observant."},
    "market_lady": {"name": "Auntie Ama", "role": "Market Vendor", "avatar": "🍎", "cost": 50, "bio": "Knows gossip."}
}

# --- FUNCTIONS ---
def get_ai_response(char_key, user_input, history):
    char = CHARACTERS[char_key]
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key: return "⚠️ API Key Missing"
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msgs = [{"role": m["role"], "content": m["content"]} for m in history]
        msgs.append({"role": "user", "content": user_input})
        system_prompt = f"Roleplay {char['name']}. Bio: {char['bio']}."
        response = client.messages.create(model="claude-3-haiku-20240307", max_tokens=150, system=system_prompt, messages=msgs)
        return response.content[0].text
    except: return "AI Error"

def map_questions(q_text):
    """Maps free text to columns"""
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key: return []
    avail = list(HIDDEN_POP.columns)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        sys = f"Map user questions to these columns: {avail}. Return ONLY JSON list e.g. ['Age']"
        resp = client.messages.create(model="claude-3-haiku-20240307", max_tokens=100, system=sys, messages=[{"role": "user", "content": q_text}])
        import json
        txt = resp.content[0].text
        return json.loads(txt[txt.find('['):txt.find(']')+1])
    except: return []

# --- MAIN APP ---
st.markdown('<div class="main-header"><h1>🏔️ Sidero Valley Investigation</h1></div>', unsafe_allow_html=True)

# SIDEBAR: RESOURCES
with st.sidebar:
    st.markdown("### 🎒 Resources")
    colA, colB = st.columns(2)
    colA.metric("Budget", f"${st.session_state.budget}")
    colB.metric("Lab Credits", st.session_state.lab_credits)
    
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    if st.button("📞 Briefing"): st.session_state.current_view = 'briefing'
    if st.button("👥 Interviews"): st.session_state.current_view = 'contacts'
    if st.button("📋 Line List"): st.session_state.current_view = 'linelist'
    if st.button("🔬 Study Design"): st.session_state.current_view = 'study_design'

# VIEWS
if st.session_state.current_view == 'briefing':
    st.markdown("### 🚨 Incoming Call")
    st.markdown('<div class="transcript-box"><strong>DHO:</strong> "We have 6 deaths. Miners and kids. Go to Sidero now."</div>', unsafe_allow_html=True)
    
    st.markdown("### 🗺️ Field Map")
    
    # Map with "Noise"
    fig = go.Figure()
    # Zones
    fig.add_shape(type="rect", x0=0, y0=200, x1=200, y1=400, fillcolor="rgba(169,169,169,0.3)", line_width=0)
    fig.add_annotation(x=50, y=380, text="MINES", showarrow=False)
    fig.add_shape(type="rect", x0=0, y0=0, x1=400, y1=200, fillcolor="rgba(144,238,144,0.3)", line_width=0)
    fig.add_annotation(x=200, y=20, text="FARMS", showarrow=False)
    # River
    fig.add_trace(go.Scatter(x=[0,400], y=[50,150], mode='lines', line=dict(color='blue', width=5), name='River'))
    # Landmarks (Signal + Noise)
    locs = [
        (300, 300, 'Hospital', 'red', 'cross'),
        (50, 100, 'Pig Farms', 'pink', 'circle'),
        (150, 300, 'Market', 'orange', 'square'),
        (350, 100, 'School', 'blue', 'diamond'),
        (250, 150, 'Temple', 'purple', 'triangle'),
        (50, 300, 'Mine Shaft', 'black', 'star')
    ]
    for x, y, txt, col, sym in locs:
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', marker=dict(size=12, color=col, symbol=sym), text=[txt], textposition="top center", name=txt))
    
    fig.update_layout(height=400, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 📷 Site Inspections ($50 each)")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Inspect Mine"):
            if st.session_state.budget >= 50:
                st.session_state.budget -= 50
                st.image("https://placehold.co/300x200?text=Dark+Mine+Shaft", caption="Ventilation looks poor.")
                st.rerun()
    with col2:
        if st.button("Inspect Pig Farms"):
            if st.session_state.budget >= 50:
                st.session_state.budget -= 50
                st.image("https://placehold.co/300x200?text=Muddy+Pig+Sty", caption="Stagnant water nearby.")
                st.rerun()
    with col3:
        if st.button("Inspect River"):
            if st.session_state.budget >= 50:
                st.session_state.budget -= 50
                st.image("https://placehold.co/300x200?text=River+Bank", caption="Children swimming.")
                st.rerun()

elif st.session_state.current_view == 'contacts':
    st.markdown("### 👥 Interviews")
    st.info(f"Budget: ${st.session_state.budget}. Each interview costs money.")
    
    cols = st.columns(4)
    for i, (k, v) in enumerate(CHARACTERS.items()):
        with cols[i%4]:
            st.markdown(f"**{v['name']}**")
            st.caption(f"Cost: ${v['cost']}")
            if st.button(f"Talk", key=k):
                if st.session_state.budget >= v['cost']:
                    st.session_state.budget -= v['cost']
                    st.session_state.current_character = k
                    st.session_state.current_view = 'interview'
                    st.rerun()
                else:
                    st.error("Insufficient Funds!")

elif st.session_state.current_view == 'interview':
    char = CHARACTERS[st.session_state.current_character]
    if st.session_state.current_character == 'healer_marcus': st.session_state.private_clinic_unlocked = True
    
    st.markdown(f"### 💬 {char['name']}")
    if st.button("🔙 End Call"): st.session_state.current_view = 'contacts'; st.rerun()
    
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

elif st.session_state.current_view == 'linelist':
    st.markdown("### 📋 Master Line List")
    
    # Public Data
    st.subheader("1. Hospital Records (Public)")
    st.dataframe(pd.DataFrame(PUBLIC_CASES))
    
    # Private Data
    st.subheader("2. Private Clinic Records")
    if st.session_state.private_clinic_unlocked:
        st.success("✅ Access Unlocked by Healer Marcus")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw Notes:**")
            for n in CLINIC_NOTES_PILE: st.markdown(f'<div class="handwritten-note">{n}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("**Abstraction Form:**")
            with st.form("add_case"):
                st.text_input("ID")
                st.text_input("Age")
                if st.form_submit_button("Add to List"):
                    st.success("Added (Simulation)")
    else:
        st.warning("🔒 Data Locked. Interview Healer Marcus to access.")

elif st.session_state.current_view == 'study_design':
    st.markdown("## 🔬 Study Design Lab")
    
    with st.form("protocol"):
        st.markdown("#### 1. Architecture")
        # No pre-population (Text Input)
        st.text_input("Study Design (e.g., Case-Control, Cohort)", placeholder="Type your design here...")
        st.text_input("Sampling Strategy", placeholder="How will you find people?")
        st.number_input("Sample Size", min_value=10, max_value=1000, value=100)
        
        st.markdown("#### 2. Questionnaire (Kobo/Word)")
        st.markdown("*Paste your survey questions below. Be specific about answer formats.*")
        q_text = st.text_area("Questionnaire Content", height=200, placeholder="1. Age (Integer)\n2. Sex (M/F)\n3. Did you eat pork? (Y/N)...")
        
        if st.form_submit_button("🚀 Deploy Field Team ($500)"):
            if st.session_state.budget >= 500:
                st.session_state.budget -= 500
                cols = map_questions(q_text)
                st.session_state.mapped_columns = cols
                st.success(f"Protocol Deployed! Collected data on: {cols}")
                if cols:
                    st.dataframe(HIDDEN_POP[cols + ['Is_Case']].sample(5))
            else:
                st.error("Insufficient Funds")