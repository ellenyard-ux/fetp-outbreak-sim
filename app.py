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
if 'current_inspection' not in st.session_state: st.session_state.current_inspection = None 

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
        'Pigs_Near_Home': np.random.choice([True, False], n),
        'Mosquito_Net_Use': np.random.choice([True, False], n),
        'Drank_River_Water': np.random.choice([True, False], n),
    })
    # Disease Logic
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
    {"ID": "H-05", "Age": 41, "Sex": "M", "Occ": "Miner", "Onset": "Dec 10", "Symp": "Tremors, Ataxia"},
    {"ID": "H-06", "Age": 29, "Sex": "M", "Occ": "Miner", "Onset": "Dec 10", "Symp": "Headache, Fever"},
]

# Clinic Notes
CLINIC_NOTES_PILE = [
    "Dec 7. Sarah (6y F). Pig farm. High fever, shaking hands.",
    "Dec 8. Twin boys (8y M). Farm B. Both vomiting and twitching.",
    "Dec 9. Mrs. Adama (40y F). Collapsed in field. Seizing.",
    "Dec 9. Baby K (2y M). High fever, stiff neck, screaming.",
    "Dec 10. Girl (5y). Pig farm. Fever, confused.",
]

# Characters
CHARACTERS = {
    "dr_chen": {"name": "Dr. Chen", "role": "Hospital Director", "avatar": "👩‍⚕️", "cost": 100, 
                "bio": "Focuses on Miners. Precise.", 
                "data_access": str(PUBLIC_CASES)},
    "healer_marcus": {"name": "Healer Marcus", "role": "Private Clinic", "avatar": "🌿", "cost": 150, 
                      "bio": "Suspicious of govt.", 
                      "data_access": str(CLINIC_NOTES_PILE)},
    "foreman_rex": {"name": "Foreman Rex", "role": "Mine Manager", "avatar": "👷", "cost": 200, "bio": "Defensive.", "data_access": "None"},
    "mama_kofi": {"name": "Mama Kofi", "role": "Mother of Case", "avatar": "👵", "cost": 100, "bio": "Grieving.", "data_access": "None"},
    "mayor_simon": {"name": "Mayor Simon", "role": "Politician", "avatar": "👔", "cost": 50, "bio": "Worried about money.", "data_access": "None"},
    "nurse_joy": {"name": "Nurse Joy", "role": "Triage Nurse", "avatar": "🩹", "cost": 50, "bio": "Overworked.", "data_access": "None"},
    "teacher_grace": {"name": "Teacher Grace", "role": "School Principal", "avatar": "📚", "cost": 50, "bio": "Observant.", "data_access": "None"},
    "market_lady": {"name": "Auntie Ama", "role": "Market Vendor", "avatar": "🍎", "cost": 50, "bio": "Knows gossip.", "data_access": "None"}
}

# --- FUNCTIONS ---
def get_ai_response(char_key, user_input, history):
    char = CHARACTERS[char_key]
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key: return "⚠️ API Key Missing"
    
    data_context = f"DATA YOU HAVE: {char.get('data_access', 'None')}"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msgs = [{"role": m["role"], "content": m["content"]} for m in history]
        msgs.append({"role": "user", "content": user_input})
        
        system_prompt = f"""
        Roleplay {char['name']}. Bio: {char['bio']}.
        {data_context}
        CRITICAL INSTRUCTIONS:
        1. If you have data (like a list of cases), READ FROM IT EXACTLY. 
        2. Do NOT invent new patient names or numbers.
        3. If the data says "H-01", use that ID. Do not make up "Lukas Petrov".
        4. If you have no data, speak from general knowledge.
        """
        
        response = client.messages.create(model="claude-3-haiku-20240307", max_tokens=250, system=system_prompt, messages=msgs)
        return response.content[0].text
    except: return "AI Error"

def map_questions(q_text):
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
    
    # FIXED MAP LOGIC
    fig = go.Figure()
    # Zones
    fig.add_shape(type="rect", x0=0, y0=200, x1=200, y1=400, fillcolor="rgba(169,169,169,0.3)", line_width=0)
    fig.add_annotation(x=50, y=380, text="MINES", showarrow=False)
    fig.add_shape(type="rect", x0=0, y0=0, x1=400, y1=200, fillcolor="rgba(144,238,144,0.3)", line_width=0)
    fig.add_annotation(x=200, y=20, text="FARMS", showarrow=False)
    # River
    fig.add_trace(go.Scatter(x=[0,400], y=[50,150], mode='lines', line=dict(color='blue', width=5), name='River'))
    
    # Valid Symbols
    fig.add_trace(go.Scatter(x=[300], y=[300], mode='markers+text', marker=dict(size=12, color='red', symbol='cross'), text=["Hospital"], textposition="top center", name="Hospital"))
    fig.add_trace(go.Scatter(x=[50], y=[100], mode='markers+text', marker=dict(size=12, color='pink', symbol='circle'), text=["Pig Farms"], textposition="top center", name="Pig Farms"))
    fig.add_trace(go.Scatter(x=[150], y=[300], mode='markers+text', marker=dict(size=12, color='orange', symbol='square'), text=["Market"], textposition="top center", name="Market"))
    fig.add_trace(go.Scatter(x=[350], y=[100], mode='markers+text', marker=dict(size=12, color='blue', symbol='diamond'), text=["School"], textposition="top center", name="School"))
    fig.add_trace(go.Scatter(x=[50], y=[300], mode='markers+text', marker=dict(size=12, color='black', symbol='x'), text=["Mine Shaft"], textposition="top center", name="Mine Shaft"))
    
    fig.update_layout(height=400, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 📷 Visual Inspection ($50)")
    st.caption("Click to send a team to take photos. Draw your own inferences.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Inspect Mine"):
            if st.session_state.budget >= 50:
                st.session_state.budget -= 50
                st.session_state.current_inspection = "mine"
                st.rerun()
            else: st.error("No Funds")
            
    with col2:
        if st.button("Inspect Pig Farms"):
            if st.session_state.budget >= 50:
                st.session_state.budget -= 50
                st.session_state.current_inspection = "pigs"
                st.rerun()
            else: st.error("No Funds")
            
    with col3:
        if st.button("Inspect River"):
            if st.session_state.budget >= 50:
                st.session_state.budget -= 50
                st.session_state.current_inspection = "river"
                st.rerun()
            else: st.error("No Funds")
    
    # RENDER THE PHOTO GALLERY
    if st.session_state.current_inspection:
        st.markdown("---")
        st.markdown(f"### 📸 Field Photos: {st.session_state.current_inspection.title()}")
        
        # Define image sets (Using Unsplash source URLs for realism)
        gallery = {
            "mine": [
                "https://images.unsplash.com/photo-1516937941344-00b4ec0c42cd?w=400&h=300&fit=crop", # Industrial
                "https://images.unsplash.com/photo-1574352067721-72d5913ef8e1?w=400&h=300&fit=crop", # Worker
                "https://images.unsplash.com/photo-1581092921461-eab62e97a78e?w=400&h=300&fit=crop"  # Factory
            ],
            "pigs": [
                "https://images.unsplash.com/photo-1516467508483-a72120608ae7?w=400&h=300&fit=crop", # Pigs
                "https://images.unsplash.com/photo-1567306301408-9b74779a11af?w=400&h=300&fit=crop", # Mud
                "https://images.unsplash.com/photo-1596464716127-f9a865d1d643?w=400&h=300&fit=crop"  # Rural water
            ],
            "river": [
                "https://images.unsplash.com/photo-1437482078695-73f5ca6c96e2?w=400&h=300&fit=crop", # River
                "https://images.unsplash.com/photo-1519817914152-c2d22195a6ae?w=400&h=300&fit=crop", # Water
                "https://images.unsplash.com/photo-1550989460-0adf9ea622e2?w=400&h=300&fit=crop"  # Cooking/Water
            ]
        }
        
        photos = gallery.get(st.session_state.current_inspection, [])
        p1, p2, p3 = st.columns(3)
        with p1: st.image(photos[0], caption="Site Photo A")
        with p2: st.image(photos[1], caption="Site Photo B")
        with p3: st.image(photos[2], caption="Site Photo C")
        
        st.markdown("**🕵️ Field Notebook:**")
        st.text_area("What do you see? (Look for: vectors, hygiene, environment)", placeholder="e.g., I see standing water near the animal pens...")

elif st.session_state.current_view == 'contacts':
    st.markdown("### 👥 Interviews")
    st.info(f"Budget: ${st.session_state.budget}. Each interview costs money.")
    
    cols = st.columns(4)
    for i, (k, v) in enumerate(CHARACTERS.items()):
        with cols[i%4]:
            st.markdown(f"**{v['avatar']} {v['name']}**")
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
                st.text_input("ID (e.g., C-01)")
                st.text_input("Age")
                st.text_input("Symptoms")
                if st.form_submit_button("Add to List"):
                    st.success("Case added to database (Simulation)")
    else:
        st.warning("🔒 Data Locked. Interview Healer Marcus to access.")

elif st.session_state.current_view == 'study_design':
    st.markdown("## 🔬 Study Design Lab")
    
    with st.form("protocol"):
        st.markdown("#### 1. Architecture")
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