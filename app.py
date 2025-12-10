import streamlit as st
import anthropic
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="FETP Outbreak Sim",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for FETP branding and layout
st.markdown("""
<style>
    .main-header {
        background-color: #003366;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #003366;
    }
    .stChatInput {
        position: fixed;
        bottom: 0;
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'home'
if 'interview_history' not in st.session_state:
    st.session_state.interview_history = {} # Format: {char_key: [{"role": "user", "content": ""}, ...]}
if 'current_character' not in st.session_state:
    st.session_state.current_character = None
if 'notes' not in st.session_state:
    st.session_state.notes = []
if 'interviewed_characters' not in st.session_state:
    st.session_state.interviewed_characters = set()
if 'case_definition' not in st.session_state:
    st.session_state.case_definition = ""
if 'mentor_feedback' not in st.session_state:
    st.session_state.mentor_feedback = None

# --- DATA: CHARACTERS ---
CHARACTERS = {
    "nurse_sarah": {
        "name": "Nurse Sarah Opoku",
        "role": "Clinic Nurse",
        "emoji": "👩‍⚕️",
        "avatar": "👩‍⚕️",
        "personality": "Overworked, organized, protective of patient data. Will not give the line list unless you explicitly ask for patient records or data.",
        "truth_document": """
        - You are the Head Nurse at Riverside Health Center.
        - You have a line list of 15 confirmed cases.
        - First case: March 3rd (4-year-old Sarah Mensah).
        - Symptoms: Acute watery diarrhea ('rice water'), vomiting, severe dehydration.
        - Demographics: Mostly children <10 and elderly >60.
        - Location: You noticed they all live in the Northern part of the village (Neighborhood A).
        - Deaths: 3 total (Mama Esi, Baby Ama, Blessing M).
        - Testing: Stool samples sent to National Lab 3 days ago, results pending.
        - Supplies: Running low on ORS.
        """,
        "initial_greeting": "I'm Nurse Opoku. We are extremely busy with patients right now. Please make this quick. How can I help?"
    },
    "dr_mensah": {
        "name": "Dr. Kwame Mensah",
        "role": "District Health Officer",
        "emoji": "👨‍⚕️",
        "avatar": "👨‍⚕️",
        "personality": "Formal, administrative, concerned with protocol. Knows the 'big picture' but lacks patient-level details.",
        "truth_document": """
        - You are the District Health Officer.
        - You called the FETP team because of a spike in diarrhea cases.
        - You know there are 15 cases and 3 deaths.
        - You are worried about the Market Day coming up in 2 days.
        - You have NOT visited the specific houses; you rely on the nurse for that.
        - You can authorize resources but need evidence first.
        """,
        "initial_greeting": "Welcome, colleagues. I am Dr. Mensah. I appreciate your prompt arrival. The situation is concerning, and we need to contain this before the weekly market."
    },
    "mrs_abena": {
        "name": "Mrs. Abena Osei",
        "role": "Community Health Worker",
        "emoji": "🧣",
        "avatar": "🧣",
        "personality": "Chatty, knows everyone's business, helpful but scattered. Needs specific questions to stay on track.",
        "truth_document": """
        - You live in Neighborhood A.
        - You know all the families who are sick.
        - You know they all use Well #1.
        - You know Well #1 is uncovered and kids play near it.
        - You know the Latrine is uphill from Well #1.
        - You use Well #1 yourself but you boil your water (so you aren't sick).
        - People don't use Well #2 because it costs money (10 shillings).
        """,
        "initial_greeting": "Oh, hello, hello! I am Abena. It is so sad, what is happening to our children. I have never seen them so weak."
    },
    "mohammed": {
        "name": "Mohammed Hassan",
        "role": "Water Vendor",
        "emoji": "🚰",
        "avatar": "🚰",
        "personality": "Defensive, business-minded, observant. Worried you will blame him.",
        "truth_document": """
        - You sell water from Well #2 (the safe well).
        - You treat your water with chlorine.
        - You have been telling people Well #1 is dirty for months.
        - You saw the latrine overflow during the rains last week.
        - You saw a dead rat in Well #1 last month.
        - None of your customers are sick.
        """,
        "initial_greeting": "I sell clean water. My water is the best. If people are sick, it is not from me."
    }
}

# --- DATA: CASES ---
CASES = [
    {"id": 1, "name": "Sarah M.", "age": 4, "sex": "F", "onset": "Mar 3", "neighborhood": "A", "well": 1},
    {"id": 2, "name": "David K.", "age": 6, "sex": "M", "onset": "Mar 4", "neighborhood": "A", "well": 1},
    {"id": 3, "name": "Grace O.", "age": 3, "sex": "F", "onset": "Mar 4", "neighborhood": "A", "well": 1},
    {"id": 4, "name": "Mama Esi", "age": 67, "sex": "F", "onset": "Mar 5", "neighborhood": "A", "well": 1},
    {"id": 5, "name": "Kofi A.", "age": 8, "sex": "M", "onset": "Mar 5", "neighborhood": "A", "well": 1},
    {"id": 6, "name": "Baby Ama", "age": 1, "sex": "F", "onset": "Mar 5", "neighborhood": "A", "well": 1},
    {"id": 7, "name": "Peter M.", "age": 5, "sex": "M", "onset": "Mar 6", "neighborhood": "A", "well": 1},
    {"id": 8, "name": "Ruth N.", "age": 7, "sex": "F", "onset": "Mar 6", "neighborhood": "A", "well": 1},
    {"id": 9, "name": "Samuel O.", "age": 45, "sex": "M", "onset": "Mar 6", "neighborhood": "A", "well": 1},
    {"id": 10, "name": "Mary K.", "age": 4, "sex": "F", "onset": "Mar 7", "neighborhood": "A", "well": 1},
    {"id": 11, "name": "James T.", "age": 6, "sex": "M", "onset": "Mar 7", "neighborhood": "A", "well": 1},
    {"id": 12, "name": "Fatima A.", "age": 35, "sex": "F", "onset": "Mar 8", "neighborhood": "A", "well": 1},
    {"id": 13, "name": "Ibrahim S.", "age": 9, "sex": "M", "onset": "Mar 8", "neighborhood": "A", "well": 1},
    {"id": 14, "name": "Blessing M.", "age": 2, "sex": "F", "onset": "Mar 9", "neighborhood": "A", "well": 1},
    {"id": 15, "name": "Joseph K.", "age": 72, "sex": "M", "onset": "Mar 9", "neighborhood": "A", "well": 1},
]

# --- HELPER FUNCTIONS ---

def get_ai_response(character_key, user_input, history):
    """Generates response from Claude with 'Harder to Crack' logic"""
    character = CHARACTERS[character_key]
    
    # System prompt enforcing information hiding
    system_prompt = f"""
    You are roleplaying as {character['name']}, a {character['role']} in a cholera outbreak.
    
    CORE TRAITS: {character['personality']}
    
    INFORMATION YOU KNOW (TRUTH DOCUMENT):
    {character['truth_document']}
    
    INSTRUCTIONS FOR INTERACTION:
    1. DO NOT volunteer information. Be reactive, not proactive.
    2. If the user asks a vague question (e.g., "What's happening?"), give a vague answer.
    3. Only reveal specific details (dates, numbers, specific locations) if specifically asked.
    4. Keep your answers relatively short and conversational. 
    5. If asked about something not in your Truth Document, say you don't know or refer them to someone else.
    6. Maintain your persona.
    """
    
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ Error: API Key missing."

    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Format history for Claude
        messages = []
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current input
        messages.append({"role": "user", "content": user_input})
        
        response = client.messages.create(
            model="claude-3-haiku-20240307", # Fast and efficient
            max_tokens=300,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

def get_mentor_critique(case_definition):
    """AI Mentor function to critique case definitions"""
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ Error: API Key missing."
        
    system_prompt = """
    You are a strict but helpful Senior Epidemiologist Mentor at the CDC.
    The trainee has submitted a Case Definition for a suspected Cholera outbreak.
    
    Critique their definition based on the standard 4 components:
    1. Clinical Criteria (Is it specific? e.g., acute watery diarrhea)
    2. Person (Who?)
    3. Place (Specific location?)
    4. Time (Date range?)
    
    If they are missing components, point it out clearly. 
    If it is good, give them a 'High Confidence' rating.
    Keep the feedback concise and actionable.
    """
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Here is my case definition: {case_definition}"}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error getting feedback: {str(e)}"

# --- MAIN UI HEADER ---
st.markdown("""
<div class="main-header">
    <h1 style='margin:0'>🦠 FETP Outbreak Simulator: Day 1</h1>
    <p style='margin:0; opacity: 0.8'>Investigation Phase: Initial Assessment | Location: Riverside Village</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    if st.button("🏠 HQ (Home)", use_container_width=True):
        st.session_state.current_view = 'home'
        st.rerun()
    
    if st.button("🗺️ Village Map", use_container_width=True):
        st.session_state.current_view = 'map'
        st.rerun()
    
    if st.button("👥 Interviews", use_container_width=True):
        st.session_state.current_view = 'contacts'
        st.rerun()
        
    if st.button("📋 Data & Line List", use_container_width=True):
        st.session_state.current_view = 'data'
        st.rerun()
        
    if st.button("🎯 Objectives & Tasks", use_container_width=True):
        st.session_state.current_view = 'objectives'
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 📓 Pocket Notebook")
    if st.session_state.notes:
        for i, note in enumerate(st.session_state.notes[-3:]): # Show last 3
            st.caption(f"• {note}")
    else:
        st.caption("No notes taken yet.")

# --- VIEWS ---

# 1. HOME VIEW
if st.session_state.current_view == 'home':
    st.markdown("## 🚨 Incident Report: Unknown GI Illness")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Date:** December 10, 2025  
        **Location:** Riverside Village, Northern Coastal Region  
        
        **Situation:** The District Health Officer (DHO) has requested immediate assistance. 
        Riverside Health Center is reporting a rapid increase in patients presenting with 
        severe acute watery diarrhea and dehydration.
        
        **Current Status:**
        - **15** Suspected Cases
        - **3** Reported Deaths
        - **Etiology:** Unknown (Cholera suspected)
        
        **Your Orders:**
        Deploy to Riverside immediately. Confirm the outbreak, establish a case definition, 
        and identify the source of infection to prevent further mortality.
        """)
        
        if st.button("🚀 Begin Investigation", type="primary"):
            st.session_state.current_view = 'contacts'
            st.rerun()
            
    with col2:
        st.info("💡 **FETP Tip:** Start by interviewing the District Health Officer to get authorization and the Clinic Nurse to get patient data.")

# 2. MAP VIEW (With Fog of War)
elif st.session_state.current_view == 'map':
    st.markdown("## 📍 Village Map")
    
    # Layout the map
    fig = go.Figure()
    
    # Neighborhood A Zone
    fig.add_shape(type="rect", x0=0, y0=0, x1=250, y1=300, line=dict(color="gray", width=1, dash="dash"), fillcolor="rgba(200,200,200,0.1)")
    fig.add_annotation(x=125, y=150, text="Neighborhood A", showarrow=False, font=dict(color="gray"))
    
    # Neighborhood B Zone
    fig.add_shape(type="rect", x0=300, y0=0, x1=550, y1=300, line=dict(color="gray", width=1, dash="dash"), fillcolor="rgba(200,200,200,0.1)")
    fig.add_annotation(x=425, y=150, text="Neighborhood B", showarrow=False, font=dict(color="gray"))

    # Clinic
    fig.add_trace(go.Scatter(x=[275], y=[350], mode='markers+text', marker=dict(size=20, color='blue', symbol='cross'), text=["🏥 Clinic"], textposition="top center", name="Clinic"))
    
    # Water Sources (Visible initially? Or usually visible on a map)
    fig.add_trace(go.Scatter(x=[140], y=[80], mode='markers+text', marker=dict(size=15, color='cyan'), text=["Well #1"], textposition="bottom center", name="Well #1"))
    fig.add_trace(go.Scatter(x=[445], y=[80], mode='markers+text', marker=dict(size=15, color='cyan'), text=["Well #2"], textposition="bottom center", name="Well #2"))
    
    # Latrine
    fig.add_trace(go.Scatter(x=[100], y=[50], mode='markers+text', marker=dict(size=12, color='brown', symbol='square'), text=["Latrine"], textposition="bottom center", name="Latrine"))

    # --- FOG OF WAR LOGIC ---
    if 'nurse_sarah' in st.session_state.interviewed_characters:
        # User has interviewed nurse, reveal cases
        case_x = [80, 110, 140, 170, 200, 80, 110, 140, 170, 200, 80, 110, 140, 170, 200]
        case_y = [180, 180, 180, 180, 180, 150, 150, 150, 150, 150, 120, 120, 120, 120, 120]
        
        fig.add_trace(go.Scatter(
            x=case_x, y=case_y, 
            mode='markers', 
            marker=dict(size=10, color='red', opacity=0.8),
            name='Confirmed Cases'
        ))
        st.success("✅ Case locations plotted based on Nurse Sarah's records.")
    else:
        # Fog of War active
        st.warning("⚠️ **Map Data Incomplete:** Case locations are unknown. Interview health staff to obtain patient addresses.")

    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, range=[-20, 580]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[-20, 400]),
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor='white',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 3. CONTACTS VIEW
elif st.session_state.current_view == 'contacts':
    st.markdown("## 👥 Key Contacts")
    
    cols = st.columns(4)
    for idx, (key, char) in enumerate(CHARACTERS.items()):
        with cols[idx]:
            st.markdown(f"""
            <div style="text-align: center; border: 1px solid #ddd; padding: 20px; border-radius: 10px;">
                <div style="font-size: 50px;">{char['avatar']}</div>
                <h4>{char['name']}</h4>
                <p style="color: gray; font-size: 12px;">{char['role']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Talk to {char['name'].split()[0]}", key=f"btn_{key}", use_container_width=True):
                st.session_state.current_character = key
                st.session_state.current_view = 'interview'
                st.rerun()

# 4. INTERVIEW VIEW (Modern Chat UI)
elif st.session_state.current_view == 'interview':
    char_key = st.session_state.current_character
    if not char_key:
        st.session_state.current_view = 'contacts'
        st.rerun()
        
    char = CHARACTERS[char_key]
    
    # Initialize history for this char if needed
    if char_key not in st.session_state.interview_history:
        st.session_state.interview_history[char_key] = []
        st.session_state.interviewed_characters.add(char_key)
        # Add initial greeting if empty
        st.session_state.interview_history[char_key].append({
            "role": "assistant", 
            "content": char['initial_greeting']
        })

    # Header
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back"):
            st.session_state.current_view = 'contacts'
            st.rerun()
    with col2:
        st.markdown(f"### Interviewing: {char['name']} ({char['role']})")
    
    st.markdown("---")

    # DISPLAY CHAT HISTORY
    for msg in st.session_state.interview_history[char_key]:
        with st.chat_message(msg["role"], avatar=char['avatar'] if msg["role"] == "assistant" else None):
            st.write(msg["content"])

    # CHAT INPUT
    if prompt := st.chat_input(f"Ask {char['name'].split()[0]} a question..."):
        # 1. Show User Message
        with st.chat_message("user"):
            st.write(prompt)
        
        # 2. Append to history
        st.session_state.interview_history[char_key].append({"role": "user", "content": prompt})
        
        # 3. Generate Response
        with st.chat_message("assistant", avatar=char['avatar']):
            with st.spinner(f"{char['name'].split()[0]} is thinking..."):
                response_text = get_ai_response(char_key, prompt, st.session_state.interview_history[char_key][:-1])
                st.write(response_text)
                
        # 4. Append Response to history
        st.session_state.interview_history[char_key].append({"role": "assistant", "content": response_text})
        
        # 5. Quick Save Note Button Logic (Optional: Auto-save logic could go here)
        st.session_state.notes.append(f"Observation from {char['name']}: {response_text[:50]}...")

# 5. DATA VIEW
elif st.session_state.current_view == 'data':
    st.markdown("## 📋 Epidemiological Data")
    
    if 'nurse_sarah' in st.session_state.interviewed_characters:
        st.success("✅ Line list accessed via Nurse Sarah.")
        df = pd.DataFrame(CASES)
        
        tab1, tab2 = st.tabs(["Line List", "Descriptive Stats"])
        
        with tab1:
            st.dataframe(df, use_container_width=True)
            
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Age Distribution**")
                fig_age = px.histogram(df, x="age", nbins=10, title="Cases by Age")
                st.plotly_chart(fig_age, use_container_width=True)
            with col2:
                st.markdown("**Epicurve**")
                # Simple count by date
                df_counts = df['onset'].value_counts().reset_index()
                df_counts.columns = ['date', 'count']
                fig_epi = px.bar(df_counts, x='date', y='count', title="Cases by Onset Date")
                st.plotly_chart(fig_epi, use_container_width=True)
    else:
        st.info("🔒 **Data Locked**")
        st.warning("You must interview the **Clinic Nurse** to obtain the patient line list.")
        if st.button("Go to Clinic"):
            st.session_state.current_view = 'contacts'
            st.rerun()

# 6. OBJECTIVES VIEW (With AI Mentor)
elif st.session_state.current_view == 'objectives':
    st.markdown("## 🎯 Investigation Tasks")
    
    st.markdown("### Task 1: Establish Case Definition")
    st.write("A case definition is standard criteria for deciding whether a person has a particular disease or other health-related condition.")
    
    current_def = st.text_area("Draft your Case Definition here:", value=st.session_state.case_definition, height=100)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("💾 Save Definition"):
            st.session_state.case_definition = current_def
            st.success("Definition saved.")
            
    with col2:
        if st.button("🤖 Ask AI Mentor to Critique"):
            if len(current_def) < 10:
                st.error("Please write a definition first.")
            else:
                with st.spinner("Analyzing your definition against standard criteria (Person, Place, Time, Clinical)..."):
                    feedback = get_mentor_critique(current_def)
                    st.session_state.mentor_feedback = feedback
    
    if st.session_state.mentor_feedback:
        st.markdown(f"""
        <div class="info-box">
            <strong>👮 AI Mentor Feedback:</strong><br>
            {st.session_state.mentor_feedback}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Task 2: Descriptive Epidemiology")
    st.checkbox("Generate Line List (Interview Nurse)", value='nurse_sarah' in st.session_state.interviewed_characters, disabled=True)
    st.checkbox("Create Epicurve (View Data Tab)", value=False) # Placeholder for future logic
    st.checkbox("Map Cases (View Map)", value=False) 

    st.markdown("---")
    st.markdown("### Task 3: Hypothesis Generation")
    st.text_input("Based on the interviews, what is the likely source?", placeholder="e.g., Well #1 contamination due to...")