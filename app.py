import streamlit as st
import anthropic
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="FETP Outbreak Sim: Sidero Valley",
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
    .messy-note {
        font-family: 'Courier New', Courier, monospace;
        background-color: #fdf6e3;
        padding: 15px;
        border: 1px solid #d6d6d6;
        border-radius: 2px;
        transform: rotate(-1deg);
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
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
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'home'
if 'interview_history' not in st.session_state:
    st.session_state.interview_history = {}
if 'interviewed_characters' not in st.session_state:
    st.session_state.interviewed_characters = set()
if 'private_clinic_unlocked' not in st.session_state:
    st.session_state.private_clinic_unlocked = False
if 'manually_entered_cases' not in st.session_state:
    st.session_state.manually_entered_cases = []
if 'notes' not in st.session_state:
    st.session_state.notes = []

# --- STORY & CHARACTERS ---
STORY_CONTEXT = """
**Situation:** Reports of "The Shaking Sickness" in Sidero Valley. 
Patients present with high fever, tremors, confusion, and in severe cases, seizures.
Two fatalities reported this morning.

**Setting:**
Sidero Valley is a mining region. 
- **North:** The Iron Mine (major employer).
- **South:** Rice paddies and pig farms.
- **Center:** The Market and Residential zone.
"""

CHARACTERS = {
    "dr_chen": {
        "name": "Dr. Elena Chen",
        "role": "Director, St. Mary's Hospital (Public)",
        "avatar": "👩‍⚕️",
        "emoji": "👩‍⚕️",
        "personality": "Professional, exhausted, academic. Relies on data.",
        "truth_document": """
        - You run the main public hospital.
        - You have admitted 6 patients with 'Acute Encephalitis Syndrome' (AES).
        - Symptoms: Fever >39C, altered mental status, tremors.
        - All your patients are adult men who work at the Iron Mine.
        - You suspect it's a toxin at the mine (Heavy metal? Gas?).
        - You haven't seen any children.
        - You have electronic records for these 6 cases.
        """,
        "initial_greeting": "I'm glad the FETP team is here. I'm convinced this is an occupational hazard at the mine. All my patients are miners."
    },
    "healer_marcus": {
        "name": "Marcus the Healer",
        "role": "Private Clinic Practitioner",
        "avatar": "🌿",
        "emoji": "🌿",
        "personality": "Defensive, traditional, observant. Distrusts the government.",
        "truth_document": """
        - You run a small private clinic near the rice paddies.
        - You have seen 5 patients with 'The Shaking Sickness'.
        - UNLIKE the hospital, your patients are children and farmers, not miners.
        - You think the 'Mine Doctors' don't care about the farmers.
        - You keep messy handwritten notes. You don't use computers.
        - You will only share your notes if the user treats you with respect.
        """,
        "initial_greeting": "Why does the government come now? My people have been sick for a week. You only care about the mine workers."
    },
    "foreman_rex": {
        "name": "Foreman Rex",
        "role": "Mine Supervisor",
        "avatar": "👷",
        "emoji": "👷",
        "personality": "Gruff, defensive, worried about the mine closing.",
        "truth_document": """
        - The mine is safe! We follow all protocols.
        - Yes, some men are sick, but they probably ate bad food at the market.
        - We had a ventilation check last month. It passed.
        - The sick men? They all hang out at the 'Blue Heron' bar after work.
        """,
        "initial_greeting": "Look, if you're here to shut us down, you better have proof. My men are safe."
    }
}

# --- DATASETS ---

# Dataset A: Clean Data from Public Hospital (6 Cases)
PUBLIC_CASES = [
    {"ID": "P01", "Age": 45, "Sex": "M", "Occupation": "Miner", "Onset": "Dec 8", "Location": "Mining Camp", "Status": "Alive"},
    {"ID": "P02", "Age": 28, "Sex": "M", "Occupation": "Miner", "Onset": "Dec 9", "Location": "Mining Camp", "Status": "Alive"},
    {"ID": "P03", "Age": 52, "Sex": "M", "Occupation": "Miner", "Onset": "Dec 9", "Location": "Town Center", "Status": "Deceased"},
    {"ID": "P04", "Age": 33, "Sex": "M", "Occupation": "Miner", "Onset": "Dec 10", "Location": "Mining Camp", "Status": "Alive"},
    {"ID": "P05", "Age": 41, "Sex": "M", "Occupation": "Miner", "Onset": "Dec 10", "Location": "Town Center", "Status": "Alive"},
    {"ID": "P06", "Age": 29, "Sex": "M", "Occupation": "Miner", "Onset": "Dec 10", "Location": "Mining Camp", "Status": "Alive"},
]

# Dataset B: Messy Notes from Private Clinic (User must extract these)
PRIVATE_NOTES = [
    "Dec 7: Little Sarah (6yo female). Lives near Pig Farm A. High fever, seizing. Mother says she played in the mud.",
    "Dec 8: Old Man John (65, Farmer). Rice paddy worker. Shaking hands, confusion. Says the birds stopped singing.",
    "Dec 9: Twin boys (8yo). Farm B. Both vomiting and twitching. Fever 40C.",
    "Dec 10: Mrs. Adama (40, Rice Farmer). Collapsed in the field. Eyes rolling back."
]

# --- FUNCTIONS ---

def get_ai_response(char_key, user_input, history):
    """Call Claude API"""
    char = CHARACTERS[char_key]
    system_prompt = f"""
    Roleplay as {char['name']}, {char['role']}.
    Context: {STORY_CONTEXT}
    Personality: {char['personality']}
    Truths: {char['truth_document']}
    
    Rules:
    1. Be concise.
    2. If you are Healer Marcus, be suspicious. Only agree to share notes if the user asks nicely or explains why it helps the community.
    3. If you are Dr. Chen, focus on the miners and the toxin theory.
    """
    
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key: return "⚠️ API Key Missing"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msgs = [{"role": m["role"], "content": m["content"]} for m in history]
        msgs.append({"role": "user", "content": user_input})
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=250,
            system=system_prompt,
            messages=msgs
        )
        return response.content[0].text
    except Exception as e:
        return f"Error: {e}"

def render_map():
    """Draws a complex Sidero Valley map using Plotly shapes"""
    fig = go.Figure()
    
    # 1. The River (Blue Path)
    fig.add_shape(type="path", path="M 0,0 Q 150,100 300,50 T 600,100", 
                  line=dict(color="lightblue", width=15), layer="below")
    
    # 2. The Road (Grey Path)
    fig.add_shape(type="path", path="M 300,400 L 300,200 L 100,100", 
                  line=dict(color="#d3d3d3", width=8, dash="solid"), layer="below")

    # 3. Zones
    # Mine (Top Left)
    fig.add_shape(type="rect", x0=20, y0=300, x1=150, y1=380, fillcolor="rgba(100,100,100,0.2)", line_width=0)
    fig.add_annotation(x=85, y=340, text="🏗️ IRON MINE", showarrow=False)
    
    # Farms (Bottom Right)
    fig.add_shape(type="rect", x0=350, y0=20, x1=550, y1=150, fillcolor="rgba(144, 238, 144, 0.2)", line_width=0)
    fig.add_annotation(x=450, y=85, text="🌾 RICE PADDIES & PIG FARMS", showarrow=False)

    # 4. Locations
    fig.add_trace(go.Scatter(x=[300], y=[200], mode='markers+text', marker=dict(size=25, color='red', symbol='cross'), text=["St. Mary's<br>(Public Hospital)"], textposition="top center", name="Public Hospital"))
    fig.add_trace(go.Scatter(x=[480], y=[60], mode='markers+text', marker=dict(size=20, color='green', symbol='circle'), text=["Marcus's Clinic<br>(Private)"], textposition="bottom center", name="Private Clinic"))
    
    # 5. Fog of War Cases
    # Public cases (Miners) - Only show if Dr. Chen interviewed
    if 'dr_chen' in st.session_state.interviewed_characters:
        fig.add_trace(go.Scatter(x=[50, 60, 40, 70, 280, 290], y=[320, 330, 310, 325, 210, 220], 
                                mode='markers', marker=dict(color='orange', size=10), name="Hospital Cases (Miners)"))
    
    # Private cases (Farmers) - Only show if data abstracted
    if len(st.session_state.manually_entered_cases) > 2:
        fig.add_trace(go.Scatter(x=[400, 420, 380, 450], y=[80, 90, 70, 100], 
                                mode='markers', marker=dict(color='purple', size=10), name="Clinic Cases (Farmers)"))

    fig.update_layout(
        xaxis=dict(range=[0, 600], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 400], showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='#F0F2F6',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN APP LAYOUT ---

st.markdown("""
<div class="main-header">
    <h1>🏔️ Mystery in Sidero Valley</h1>
    <p>Mission: Investigate a cluster of "Acute Encephalitis Syndrome" (AES)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/mosquito.png", width=50) # Placeholder icon
    st.markdown("### 🕵️ Investigation Hub")
    if st.button("🏠 Briefing", use_container_width=True): st.session_state.current_view = 'home'
    if st.button("🗺️ Valley Map", use_container_width=True): st.session_state.current_view = 'map'
    if st.button("👥 Interviews", use_container_width=True): st.session_state.current_view = 'contacts'
    if st.button("🏥 Hospital Records", use_container_width=True): st.session_state.current_view = 'hospital_data'
    if st.button("🏚️ Private Clinic Records", use_container_width=True): st.session_state.current_view = 'clinic_data'
    
    st.markdown("---")
    st.progress(len(st.session_state.manually_entered_cases)/5 if len(st.session_state.manually_entered_cases) < 5 else 1.0, text="Private Data Abstracted")

# 1. BRIEFING
if st.session_state.current_view == 'home':
    st.markdown("### 🚨 Urgent Deployment Order")
    st.markdown("""
    **To:** FETP Team Alpha  
    **From:** Ministry of Health  
    **Subject:** Unexplained Neurological Illness - Sidero Valley
    
    **Details:** Local media is reporting a "Shaking Sickness" in Sidero Valley. 
    St. Mary's Hospital reports 6 admissions with high fever, tremors, and altered mental status.
    One death occurred this morning.
    
    **Initial Hypothesis:** Dr. Chen (Hospital Director) believes it is **industrial poisoning** from the Iron Mine.
    However, rumors from the farming villages suggest children are also sick.
    
    **Your Tasks:**
    1.  Interview **Dr. Chen** at St. Mary's Hospital.
    2.  Locate the **Private Clinic** used by farmers (they often avoid the hospital).
    3.  **Unify the data:** The hospital has computers; the private clinic does not. You must abstract the data manually.
    4.  Determine if this is Poisoning (Miners only) or Infectious (Community-wide).
    """)
    if st.button("Begin Operation"):
        st.session_state.current_view = 'map'
        st.rerun()

# 2. MAP
elif st.session_state.current_view == 'map':
    st.markdown("### 🗺️ Geographical Overview")
    st.info("💡 **Hint:** Notice the distinct zones. The Hospital is central, but the Private Clinic is deep in the farming zone.")
    render_map()

# 3. INTERVIEWS
elif st.session_state.current_view == 'contacts':
    st.markdown("### 👥 Key Witnesses")
    cols = st.columns(3)
    for key, char in CHARACTERS.items():
        with cols[list(CHARACTERS.keys()).index(key)]:
            st.markdown(f"### {char['emoji']}")
            st.markdown(f"**{char['name']}**")
            st.caption(char['role'])
            if st.button(f"Interview", key=key):
                st.session_state.current_character = key
                st.session_state.current_view = 'interview'
                st.rerun()

# 4. CHAT INTERFACE
elif st.session_state.current_view == 'interview':
    char = CHARACTERS[st.session_state.current_character]
    st.markdown(f"### 💬 Interviewing: {char['name']}")
    if st.button("🔙 Back"): 
        st.session_state.current_view = 'contacts'
        st.rerun()
    
    # Unlock Private Clinic logic
    if st.session_state.current_character == 'healer_marcus':
        st.session_state.private_clinic_unlocked = True
    
    if st.session_state.current_character not in st.session_state.interview_history:
        st.session_state.interview_history[st.session_state.current_character] = [{"role": "assistant", "content": char['initial_greeting']}]
        st.session_state.interviewed_characters.add(st.session_state.current_character)
    
    history = st.session_state.interview_history[st.session_state.current_character]
    
    for msg in history:
        with st.chat_message(msg['role'], avatar=char['avatar'] if msg['role'] == "assistant" else None):
            st.write(msg['content'])
            
    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user"): st.write(prompt)
        history.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant", avatar=char['avatar']):
            with st.spinner("Thinking..."):
                resp = get_ai_response(st.session_state.current_character, prompt, history[:-1])
                st.write(resp)
        history.append({"role": "assistant", "content": resp})

# 5. HOSPITAL DATA (EASY)
elif st.session_state.current_view == 'hospital_data':
    st.markdown("### 🏥 St. Mary's Hospital Records (EMR)")
    if 'dr_chen' in st.session_state.interviewed_characters:
        st.success("✅ Access Granted by Dr. Chen")
        df_public = pd.DataFrame(PUBLIC_CASES)
        st.dataframe(df_public, use_container_width=True)
        st.caption("Observation: All patients here are Adult Males (Miners).")
    else:
        st.error("🔒 Access Denied. You must interview Dr. Chen first.")

# 6. PRIVATE CLINIC DATA (HARD - ABSTRACTION TASK)
elif st.session_state.current_view == 'clinic_data':
    st.markdown("### 🏚️ Healer Marcus's Private Clinic")
    
    if not st.session_state.private_clinic_unlocked:
        st.error("🔒 You don't know where this clinic is. Find 'Marcus' in the contacts list.")
    else:
        st.markdown("""
        **Task:** Marcus doesn't have a computer. He handed you a pile of greasy, handwritten notes. 
        You must read them and **manually enter the data** into the line list below.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📝 The Handwritten Notes")
            for note in PRIVATE_NOTES:
                st.markdown(f'<div class="messy-note">{note}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 💻 Data Abstraction Form")
            with st.form("abstraction_form"):
                c_age = st.number_input("Age", min_value=0, max_value=100)
                c_sex = st.selectbox("Sex", ["M", "F"])
                c_occ = st.selectbox("Occupation/Status", ["Miner", "Farmer", "Child", "Other"])
                c_onset = st.selectbox("Onset Date", ["Dec 7", "Dec 8", "Dec 9", "Dec 10"])
                
                submitted = st.form_submit_button("➕ Add Case to Line List")
                if submitted:
                    new_case = {"Age": c_age, "Sex": c_sex, "Occupation": c_occ, "Onset": c_onset, "Source": "Private Clinic"}
                    st.session_state.manually_entered_cases.append(new_case)
                    st.success("Case Abstracted!")
            
            if st.session_state.manually_entered_cases:
                st.markdown("#### Your Abstracted Data")
                st.dataframe(pd.DataFrame(st.session_state.manually_entered_cases))
                
                if len(st.session_state.manually_entered_cases) >= 4:
                    st.success("🎉 Excellent! You have digitized the private records.")
                    st.info("🤔 **Critical Thinking:** Look at your combined data now. The Hospital had only Miners. The Clinic has Children and Farmers. Does the 'Toxic Mine' hypothesis still hold up? Or is this something environmental affecting everyone (like mosquitoes or pigs)?")