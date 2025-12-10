import streamlit as st
import anthropic
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="FETP Outbreak Investigation Simulation",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for FETP branding
st.markdown("""
<style>
    .main-header {
        background-color: #FF6B35;
        color: white;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .case-marker {
        color: #D32F2F;
        font-size: 24px;
    }
    .character-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #FF6B35;
    }
    .interview-bubble {
        background-color: #E3F2FD;
        padding: 12px;
        border-radius: 10px;
        margin: 8px 0;
    }
    .user-bubble {
        background-color: #FFF3E0;
        padding: 12px;
        border-radius: 10px;
        margin: 8px 0;
    }
    .objective-complete {
        color: #4CAF50;
    }
    .objective-pending {
        color: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'home'  # Start on home screen
if 'interview_history' not in st.session_state:
    st.session_state.interview_history = {}
if 'current_character' not in st.session_state:
    st.session_state.current_character = None
if 'notes' not in st.session_state:
    st.session_state.notes = []
if 'day' not in st.session_state:
    st.session_state.day = 1
if 'interviewed_characters' not in st.session_state:
    st.session_state.interviewed_characters = set()
if 'visited_clinic' not in st.session_state:
    st.session_state.visited_clinic = False

# Character database with truth documents
CHARACTERS = {
    "nurse_sarah": {
        "name": "Nurse Sarah Opoku",
        "role": "Clinic Nurse, Riverside Health Center",
        "emoji": "👩‍⚕️",
        "personality": "Professional, organized, detail-oriented. Has maintained careful records of all patients. Concerned but methodical.",
        "truth_document": """
        - Head nurse at Riverside Health Center for 12 years
        - Has complete line list of all 15 cases with detailed information
        - First patient arrived March 3rd at 06:00 - 4-year-old Sarah Mensah, severe dehydration
        - By March 4th, had 3 more cases, all children from same neighborhood
        - Pattern continued - mostly children under 10 and elderly over 60
        - All patients present with profuse watery diarrhea ("rice water" stool), vomiting, rapid dehydration
        - 3 deaths so far - all were severely dehydrated on arrival, delayed treatment
        - Case details she can provide when asked:
          * Case 1: Sarah M., 4F, onset Mar 3, Neighborhood A, lives near Well #1
          * Case 2: David K., 6M, onset Mar 4, Neighborhood A, near Well #1
          * Case 3: Grace O., 3F, onset Mar 4, Neighborhood A, near Well #1
          * Case 4: Mama Esi, 67F, onset Mar 5, Neighborhood A, near Well #1 (deceased)
          * Case 5: Kofi A., 8M, onset Mar 5, Neighborhood A, near Well #1
          * Case 6: Baby Ama, 1F, onset Mar 5, Neighborhood A, near Well #1 (deceased)
          * Case 7: Peter M., 5M, onset Mar 6, Neighborhood A, near Well #1
          * Case 8: Ruth N., 7F, onset Mar 6, Neighborhood A, near Well #1
          * Case 9: Samuel O., 45M, onset Mar 6, Neighborhood A, near Well #1
          * Case 10: Mary K., 4F, onset Mar 7, Neighborhood A, near Well #1
          * Case 11: James T., 6M, onset Mar 7, Neighborhood A, near Well #1
          * Case 12: Fatima A., 35F, onset Mar 8, Neighborhood A, near Well #1
          * Case 13: Ibrahim S., 9M, onset Mar 8, Neighborhood A, near Well #1
          * Case 14: Blessing M., 2F, onset Mar 9, Neighborhood A, near Well #1 (deceased)
          * Case 15: Joseph K., 72M, onset Mar 9, Neighborhood A, near Well #1
        - Noticed all cases are from same area of village - northern section
        - When asked about families, she notes they all mention using Well #1 for drinking water
        - No cases from the southern part of village (Well #2 users)
        - Symptoms start 6-12 hours after reported exposure
        - Most severe cases are those who delayed coming to clinic
        - Clinic running low on ORS and IV fluids
        - Sent stool samples to National Lab 3 days ago, still waiting for results
        - Clinical diagnosis: strongly suspects cholera based on presentation
        - Has been asking patients about food and water sources in routine intake
        """,
        "initial_greeting": "Good morning, I'm Nurse Sarah Opoku, head nurse here at the clinic. We've been overwhelmed these past few days. I've been keeping detailed records of every patient. What would you like to know about the cases?"
    },
    "dr_mensah": {
        "name": "Dr. Kwame Mensah",
        "role": "District Health Officer",
        "emoji": "👨‍⚕️",
        "personality": "Professional, knowledgeable, somewhat formal. Provides overview but refers to nurse for detailed records.",
        "truth_document": """
        - District Health Officer for Riverside District for 8 years
        - Oversees 12 health facilities in the district
        - First notified of cluster on March 5th when Nurse Sarah called with concerns
        - Visited clinic on March 6th, confirmed outbreak investigation needed
        - Total of 15 confirmed cases, 3 deaths (case fatality rate ~20%)
        - Clinical presentation consistent with cholera
        - Requested FETP team support on March 8th
        - Arranged for stool samples to be sent to National Lab
        - Concerned about potential for wider spread
        - Village has two main water sources but doesn't know details (refers to community members)
        - Last cholera outbreak in district was 5 years ago in different village
        - Worried about upcoming market day (2 days away) - could spread to other villages
        - Has basic supplies at district level but not enough for large outbreak
        - Can authorize interventions and mobilize resources
        - Expects team to conduct thorough investigation before recommendations
        - Nurse Sarah has the detailed case information and line list
        """,
        "initial_greeting": "Welcome to Riverside Village. I'm Dr. Mensah, the District Health Officer. I'm very glad the FETP team is here. We've been dealing with a serious outbreak of severe diarrheal illness. Nurse Sarah at the clinic has been maintaining detailed records. What can I tell you about the situation?"
    },
    "mrs_abena": {
        "name": "Mrs. Abena Osei",
        "role": "Community Health Worker",
        "emoji": "👩‍⚕️",
        "personality": "Warm, talkative, very familiar with the community. Knows personal details about families.",
        "truth_document": """
        - Community health worker in Riverside Village for 15 years
        - Lives in Neighborhood A, uses Well #1 herself
        - Knows most of the affected families personally
        - First case was 4-year-old Sarah Mensah (no relation to Dr. Mensah) on March 3rd
        - Sarah's mother brought her to health center with severe diarrhea and vomiting
        - By March 5th, Sarah's two siblings also became ill
        - Most cases are children, but some adults affected too
        - Noticed that affected families all live within 200 meters of Well #1
        - Her own family drinks boiled water, hasn't gotten sick
        - Some families drink water directly from the well without treatment
        - Well #1 is an open well (no proper cover) about 15 meters deep
        - During last month's heavy rains, she noticed the well water looked cloudy for a few days
        - Saw children playing near the well and animals (goats) drinking from buckets nearby
        - One family reported their well water "tasted different" starting in late February
        - Well #2 is deeper (25 meters) and has a concrete cover with hand pump
        - Families using Well #2 have not reported any illness
        - Village women usually collect water early morning and evening
        - Some families store water in containers for several days
        """,
        "initial_greeting": "Hello! I'm Abena, the community health worker here. I've lived in this village my whole life. This outbreak has been terrible - I know many of these families personally. How can I help?"
    },
    "chief_okoye": {
        "name": "Chief Adebayo Okoye",
        "role": "Village Chief",
        "emoji": "👴",
        "personality": "Respected elder, speaks carefully, concerned about village reputation. Protective of community.",
        "truth_document": """
        - Village chief for 20 years
        - Very concerned about outbreak affecting village's reputation and economy
        - Market day is in 2 days - worried about canceling it (major income source for village)
        - Well #1 was dug 30 years ago by his father
        - Well #2 was built 10 years ago with NGO support (why it's better constructed)
        - Knows there have been problems with Well #1 for years
        - Last year during rainy season, noticed the area around Well #1 floods
        - Village latrine is located about 30 meters uphill from Well #1
        - The latrine is old (15 years) and in poor condition - cracks visible in structure
        - After heavy rains in February, the latrine pit overflowed slightly
        - He wanted to relocate the latrine but village couldn't afford it
        - Some community members complained about Well #1 water quality but continued using it (free, close to homes)
        - Well #2 requires small payment for maintenance (10 shillings per bucket)
        - Poorer families prefer free Well #1 despite quality concerns
        - Willing to close Well #1 if investigators recommend it, but worried about water access for poor families
        - Can help organize community meeting to discuss interventions
        - Village has about 100 households total
        """,
        "initial_greeting": "Welcome, welcome. I am Chief Okoye. This outbreak troubles me deeply. Our village has always been healthy. I hope you can help us quickly - we have important market day coming soon."
    },
    "mohammed": {
        "name": "Mohammed Hassan",
        "role": "Water Vendor",
        "emoji": "🚰",
        "personality": "Young, entrepreneurial, observant. Slightly defensive about his business.",
        "truth_document": """
        - 25-year-old water vendor, sells treated water from Well #2
        - Charges 10 shillings per 20-liter container
        - Business has been slow because most families use free Well #1
        - Since outbreak started, more families are buying his water (business is up)
        - He treats water from Well #2 with chlorine tablets before selling
        - Never gotten sick himself, none of his customers have gotten sick
        - Has been warning people for months that Well #1 water isn't safe
        - Noticed several concerning things about Well #1:
          * Water sometimes has brown tinge after rains
          * Frogs and insects often found in the well
          * No cover - leaves, dirt, and debris fall in
          * Children sometimes drop buckets in the water while playing
          * Saw a dead rat floating in the well once (about 3 months ago)
        - After February rains, he noticed the area around Well #1 was muddy and smelled bad
        - Suspects runoff from the latrine might be seeping into Well #1
        - The ground slopes from the latrine toward the well
        - Offered to sell chlorine tablets to Well #1 users but they said too expensive
        - Knows basic water treatment from NGO training 2 years ago
        - Can demonstrate proper chlorination if asked
        - Willing to help distribute safe water during outbreak if paid fairly
        - His water source (Well #2) has never had any problems
        """,
        "initial_greeting": "Good morning. I'm Mohammed, I sell water here. I've been saying for months that Well #1 isn't safe, but people don't want to pay for clean water. Now look what's happened."
    }
}

# Case data
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

def get_ai_response(character_key, user_question, conversation_history):
    """Get response from Claude API based on character's truth document"""
    
    character = CHARACTERS[character_key]
    
    # Build the system prompt
    system_prompt = f"""You are roleplaying as {character['name']}, a {character['role']} in Riverside Village during a cholera outbreak investigation.

PERSONALITY: {character['personality']}

TRUTH DOCUMENT (information you know):
{character['truth_document']}

INSTRUCTIONS:
1. Stay in character at all times
2. Only share information from your truth document
3. Answer based on what this character would realistically know
4. If asked something not in your truth document, respond naturally as the character would:
   - "I'm not sure about that"
   - "You'd need to ask [other character] about that"
   - Or make a reasonable inference that doesn't contradict the truth document
5. Be conversational and natural
6. Reveal information progressively - don't dump everything at once
7. Some information should only come out if specifically asked
8. Show appropriate emotion (concern for outbreak, etc.)

Remember: You are being interviewed by an FETP epidemiologist investigating this outbreak."""

    # Get API key from secrets
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    
    if not api_key:
        return "⚠️ API key not configured. Please add ANTHROPIC_API_KEY to Streamlit secrets."
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Build messages from conversation history
        messages = []
        for entry in conversation_history:
            messages.append({"role": "user", "content": entry["question"]})
            messages.append({"role": "assistant", "content": entry["response"]})
        
        # Add current question
        messages.append({"role": "user", "content": user_question})
        
        # Get response from Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=messages
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"⚠️ Error communicating with AI: {str(e)}"

# Header
st.markdown("""
<div class="main-header">
    <h1>🦠 FETP Outbreak Investigation Simulation</h1>
    <p style="margin:0;">Day 1: Initial Investigation | Riverside Village | December 10, 2025</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("### 📊 Navigation")
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.current_view = 'home'
        st.rerun()
    
    if st.button("📍 Map View", use_container_width=True):
        st.session_state.current_view = 'map'
        st.rerun()
    
    if st.button("👥 Contacts", use_container_width=True):
        st.session_state.current_view = 'contacts'
        st.rerun()
    
    if st.button("📋 Line List", use_container_width=True):
        st.session_state.current_view = 'cases'
        st.rerun()
    
    if st.button("✍️ Investigation Notes", use_container_width=True):
        st.session_state.current_view = 'notes'
        st.rerun()
    
    if st.button("🎯 Investigation Steps", use_container_width=True):
        st.session_state.current_view = 'objectives'
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("Use navigation above to explore different aspects of the investigation.")

# Main content area based on current view
if st.session_state.current_view == 'home':
    st.markdown("## 🚨 Outbreak Investigation Exercise: Northern Coastal Region")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin: 20px 0;">
        <h2 style="margin-top: 0; color: white;">⚠️ URGENT: Public Health Emergency</h2>
        <p style="font-size: 18px; margin-bottom: 0;">You are the newly deployed Outbreak Investigation Team</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📅 Current Date & Time")
        st.info("**December 10, 2025 | 08:00 Hours**")
        
        st.markdown("### 🌍 Location")
        st.write("**Riverside Village, Northern Coastal Region**")
        st.write("A rural community of approximately 100 households located near the coast.")
        
        st.markdown("### 🏥 Initial Report")
        st.warning("""
        Over the last **48 hours**, the local health clinic has reported an alarming spike in patients 
        presenting with **acute watery diarrhea and severe dehydration**.
        """)
        
        st.markdown("### 📊 Initial Data Snapshot")
        st.error("""
        **As of 08:00 this morning:**
        - **15 confirmed cases** admitted to the health center
        - **3 deaths** reported
        - **2 patients** requiring immediate aggressive intravenous fluid resuscitation
        - **Patient age range:** 1 to 72 years
        - **Clinical presentation:** Severe watery "rice-water" diarrhea, vomiting, rapid dehydration
        """)
        
        st.markdown("### 🔬 Preliminary Hypothesis")
        st.info("""
        Based on the clinical presentation and rapid onset, the **leading preliminary differential diagnosis is Cholera**.
        
        However, this remains unconfirmed pending laboratory results and epidemiological investigation.
        """)
    
    with col2:
        st.markdown("### 🎯 Your Mission")
        st.success("""
        Your team's objective is to **systematically manage this unfolding public health crisis**.
        
        Over the course of this exercise, you will be tasked with:
        
        1. **Defining the problem** and establishing a case definition
        
        2. **Conducting a field investigation** to find cases and generate hypotheses (person, place, time)
        
        3. **Analyzing epidemiological data** to identify the source and mode of transmission
        
        4. **Developing evidence-based control measures**
        
        5. **Communicating findings** and recommendations to stakeholders
        """)
        
        st.markdown("---")
        
        st.markdown("### 👥 Team Resources")
        st.write("""
        You have access to:
        - District Health Officer
        - Community Health Worker
        - Village leadership
        - Local community members
        - Basic laboratory services
        """)
    
    st.markdown("---")
    
    # Call to action
    st.markdown("### 🚀 Ready to Begin?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📍 View Village Map", use_container_width=True, type="primary"):
            st.session_state.current_view = 'map'
            st.rerun()
    
    with col2:
        if st.button("👥 Interview Key Contacts", use_container_width=True):
            st.session_state.current_view = 'contacts'
            st.rerun()
    
    with col3:
        if st.button("📋 Review Available Data", use_container_width=True):
            st.session_state.current_view = 'cases'
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #FFF3E0; padding: 15px; border-radius: 5px; border-left: 4px solid #FF6B35;">
        <p style="margin: 0;"><strong>💡 Investigation Tip:</strong> In a real outbreak, your first step would typically be 
        to speak with the District Health Officer to understand the situation, review available case information, 
        and visit the affected area. Use the navigation menu on the left to explore different aspects of the investigation.</p>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_view == 'map':
    st.markdown("## 📍 Riverside Village Map")
    
    st.markdown("""
    <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h3 style="margin-top: 0;">🗺️ Village Layout</h3>
    """, unsafe_allow_html=True)
    
    # Create a simple schematic map
    map_col1, map_col2 = st.columns([2, 1])
    
    with map_col1:
        import plotly.graph_objects as go
        
        # Create figure
        fig = go.Figure()
        
        # Add neighborhood boundaries
        # Neighborhood A (affected)
        fig.add_shape(type="rect",
            x0=0, y0=0, x1=250, y1=300,
            line=dict(color="red", width=2, dash="dash"),
            fillcolor="rgba(255,235,238,0.3)",
            layer="below"
        )
        
        # Neighborhood B (unaffected)  
        fig.add_shape(type="rect",
            x0=320, y0=0, x1=570, y1=300,
            line=dict(color="green", width=2, dash="dash"),
            fillcolor="rgba(232,245,233,0.3)",
            layer="below"
        )
        
        # Health Center at top
        fig.add_trace(go.Scatter(
            x=[285], y=[350],
            mode='markers+text',
            marker=dict(size=25, color='lightblue', symbol='square', line=dict(color='blue', width=2)),
            text=['🏥 Clinic'],
            textposition='bottom center',
            textfont=dict(size=12),
            showlegend=False,
            hoverinfo='text',
            hovertext='Health Center'
        ))
        
        # Cases in Neighborhood A (clustered around Well #1)
        case_x = [80, 110, 140, 170, 200,
                  80, 110, 140, 170, 200,
                  80, 110, 140, 170, 200]
        case_y = [180, 180, 180, 180, 180,
                  150, 150, 150, 150, 150,
                  120, 120, 120, 120, 120]
        
        fig.add_trace(go.Scatter(
            x=case_x[:15], y=case_y[:15],
            mode='markers',
            marker=dict(size=12, color='red', symbol='circle'),
            name='Cases',
            hoverinfo='text',
            hovertext=['Case ' + str(i+1) for i in range(15)]
        ))
        
        # Well #1 (contaminated)
        fig.add_trace(go.Scatter(
            x=[140], y=[80],
            mode='markers+text',
            marker=dict(size=30, color='lightblue', symbol='circle', line=dict(color='darkblue', width=3)),
            text=['💧 Well #1<br>(Open)'],
            textposition='bottom center',
            textfont=dict(size=10),
            showlegend=False,
            hoverinfo='text',
            hovertext='Well #1: Open well, no cover<br>Serves Neighborhood A'
        ))
        
        # Latrine (uphill from Well #1)
        fig.add_trace(go.Scatter(
            x=[100], y=[50],
            mode='markers+text',
            marker=dict(size=20, color='brown', symbol='square'),
            text=['🚽'],
            textposition='top center',
            textfont=dict(size=16),
            showlegend=False,
            hoverinfo='text',
            hovertext='Latrine (30m uphill from Well #1)'
        ))
        
        # Arrow from latrine to well
        fig.add_annotation(
            x=140, y=80,
            ax=100, ay=50,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="brown"
        )
        
        # Well #2 (safe)
        fig.add_trace(go.Scatter(
            x=[445], y=[80],
            mode='markers+text',
            marker=dict(size=30, color='lightgreen', symbol='circle', line=dict(color='darkgreen', width=3)),
            text=['💧 Well #2<br>(Covered)'],
            textposition='bottom center',
            textfont=dict(size=10),
            showlegend=False,
            hoverinfo='text',
            hovertext='Well #2: Covered with hand pump<br>Serves Neighborhood B<br>No cases'
        ))
        
        # Add annotations for neighborhoods
        fig.add_annotation(x=125, y=280, text="NEIGHBORHOOD A<br>(Well #1 users)<br>15 Cases",
                          showarrow=False, font=dict(size=12, color="darkred"), bgcolor="rgba(255,255,255,0.7)")
        
        fig.add_annotation(x=445, y=280, text="NEIGHBORHOOD B<br>(Well #2 users)<br>0 Cases",
                          showarrow=False, font=dict(size=12, color="darkgreen"), bgcolor="rgba(255,255,255,0.7)")
        
        # Update layout
        fig.update_layout(
            title="RIVERSIDE VILLAGE",
            title_x=0.5,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-20, 590]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-20, 380]),
            height=500,
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with map_col2:
        st.markdown("### 📍 Map Legend")
        st.markdown("""
        **Red dots (🔴):** Confirmed case locations
        
        **Blue circles (💧):** Water sources
        
        **Neighborhoods:**
        - Neighborhood A (left, shaded red)
        - Neighborhood B (right, shaded green)
        
        **Other features:**
        - 🏥 Health clinic
        - 🚽 Latrine
        """)
        
        st.markdown("---")
        
        st.info("""
        💡 **Using the map:**
        
        Hover over elements to see details. Look for spatial patterns in case distribution.
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add quick action buttons
    st.markdown("### 🎯 Next Steps")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗣️ Interview Community Members", use_container_width=True):
            st.session_state.current_view = 'contacts'
            st.rerun()
    with col2:
        if st.button("📊 Review Case Data", use_container_width=True):
            st.session_state.current_view = 'cases'
            st.rerun()

elif st.session_state.current_view == 'contacts':
    st.markdown("## 👥 People You Can Interview")
    
    st.info("💡 **Tip:** In outbreak investigations, start with health facility staff who have patient information, then expand to community members.")
    
    # Order characters: Nurse first, then Dr. Mensah, then community members
    character_order = ['nurse_sarah', 'dr_mensah', 'mrs_abena', 'chief_okoye', 'mohammed']
    
    for char_key in character_order:
        if char_key not in CHARACTERS:
            continue
            
        char_data = CHARACTERS[char_key]
        
        with st.container():
            st.markdown(f"""
            <div class="character-card">
                <h3>{char_data['emoji']} {char_data['name']}</h3>
                <p><strong>{char_data['role']}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            status = "✅ Interviewed" if char_key in st.session_state.interviewed_characters else "⭕ Not yet contacted"
            st.caption(f"Status: {status}")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button(f"Interview {char_data['name'].split()[0]}", key=f"interview_{char_key}"):
                    st.session_state.current_character = char_key
                    st.session_state.current_view = 'interview'
                    st.rerun()
            
            with col2:
                if char_key in st.session_state.interviewed_characters:
                    if st.button(f"View Notes from {char_data['name'].split()[0]}", key=f"notes_{char_key}"):
                        st.info("Feature coming in full version: View previous interview notes")

elif st.session_state.current_view == 'interview':
    character_key = st.session_state.current_character
    character = CHARACTERS[character_key]
    
    # Interview header
    st.markdown(f"## {character['emoji']} Interview with {character['name']}")
    st.caption(f"**Role:** {character['role']}")
    
    if st.button("← Back to Contacts"):
        st.session_state.current_view = 'contacts'
        st.rerun()
    
    st.markdown("---")
    
    # Initialize interview history for this character
    if character_key not in st.session_state.interview_history:
        st.session_state.interview_history[character_key] = []
        st.session_state.interviewed_characters.add(character_key)
    
    # Show conversation history
    conversation = st.session_state.interview_history[character_key]
    
    # Initial greeting if first interaction
    if len(conversation) == 0:
        st.markdown(f"""
        <div class="interview-bubble">
            <strong>{character['emoji']} {character['name']}:</strong><br>
            {character['initial_greeting']}
        </div>
        """, unsafe_allow_html=True)
    
    # Show previous conversation
    for entry in conversation:
        st.markdown(f"""
        <div class="user-bubble">
            <strong>You:</strong><br>
            {entry['question']}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="interview-bubble">
            <strong>{character['emoji']} {character['name']}:</strong><br>
            {entry['response']}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Question input area
    st.markdown("### 💬 Ask a Question")
    
    # Suggested questions (only show if no questions asked yet)
    if len(conversation) == 0:
        st.markdown("**💡 Suggested starter questions:**")
        suggestions = {
            "dr_mensah": [
                "Can you tell me about the outbreak so far?",
                "What are the main symptoms patients are showing?",
                "When did the first case occur?"
            ],
            "mrs_abena": [
                "What have you observed in the community?",
                "Do you know the affected families?",
                "Have you noticed any patterns?"
            ],
            "chief_okoye": [
                "Can you tell me about the village water sources?",
                "When was the last time you had a health problem like this?",
                "What are your concerns about this outbreak?"
            ],
            "mohammed": [
                "Tell me about your water business",
                "What have you noticed about the water sources?",
                "Why do you think some people got sick and others didn't?"
            ]
        }
        
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions.get(character_key, [])):
            with cols[i]:
                if st.button(suggestion, key=f"suggest_{i}"):
                    # Auto-fill the question
                    st.session_state['auto_question'] = suggestion

    # Question input
    user_question = st.text_area(
        "Type your question:",
        value=st.session_state.get('auto_question', ''),
        height=100,
        key="question_input",
        placeholder="What would you like to ask?"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🗣️ Ask Question", type="primary")
    with col2:
        if st.button("📝 Save Important Info to Notes"):
            if len(conversation) > 0:
                last_exchange = conversation[-1]
                note = f"**{character['name']}** ({datetime.now().strftime('%H:%M')}): {last_exchange['response'][:200]}..."
                st.session_state.notes.append(note)
                st.success("✅ Added to investigation notes!")
    
    if ask_button and user_question.strip():
        with st.spinner(f"💭 {character['name']} is responding..."):
            # Get AI response
            response = get_ai_response(character_key, user_question, conversation)
            
            # Add to conversation history
            st.session_state.interview_history[character_key].append({
                "question": user_question,
                "response": response,
                "timestamp": datetime.now().isoformat()
            })
            
            # Clear auto question if it exists
            if 'auto_question' in st.session_state:
                del st.session_state['auto_question']
            
            st.rerun()

elif st.session_state.current_view == 'cases':
    st.markdown("## 📋 Line List & Data Analysis")
    
    import pandas as pd
    
    # Check if they've interviewed the nurse for complete data
    if 'nurse_sarah' in st.session_state.interviewed_characters:
        st.success("✅ Complete case information obtained from Nurse Sarah")
        
        # Create full dataframe
        df = pd.DataFrame(CASES)
        df['status'] = ['Confirmed'] * 15
        df['outcome'] = ['Alive'] * 15
        df['outcome'][3] = 'Deceased'  # Mama Esi
        df['outcome'][5] = 'Deceased'  # Baby Ama
        df['outcome'][13] = 'Deceased'  # Blessing M.
        
        # Show line list in expandable section
        with st.expander("📋 View Complete Line List", expanded=False):
            st.dataframe(
                df[['id', 'name', 'age', 'sex', 'onset', 'neighborhood', 'well', 'status', 'outcome']],
                use_container_width=True,
                hide_index=True
            )
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Line List (CSV)",
                data=csv,
                file_name="riverside_linelist.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # Interactive Analytics Dashboard
        st.markdown("## 📊 Descriptive Epidemiology Dashboard")
        st.info("Use the tools below to analyze your line list data by Person, Place, and Time")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["👥 Person", "📍 Place", "📅 Time", "📈 Custom Analysis"])
        
        with tab1:
            st.markdown("### Analysis by Person")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Select variable to analyze:**")
                person_var = st.selectbox(
                    "Variable",
                    ["age", "sex", "outcome"],
                    key="person_var"
                )
                
                if st.button("Generate Frequency Table", key="person_table"):
                    if person_var == "age":
                        # Create age groups
                        df['age_group'] = pd.cut(df['age'], 
                                                  bins=[0, 5, 18, 60, 100], 
                                                  labels=['<5', '5-17', '18-59', '60+'])
                        freq_table = df['age_group'].value_counts().sort_index()
                        st.markdown("**Cases by Age Group:**")
                        st.dataframe(freq_table.reset_index().rename(columns={'index': 'Age Group', 'age_group': 'Count'}))
                        
                        # Calculate percentages
                        pct_table = (freq_table / freq_table.sum() * 100).round(1)
                        st.markdown("**Percentage Distribution:**")
                        st.dataframe(pct_table.reset_index().rename(columns={'index': 'Age Group', 'age_group': 'Percent'}))
                        
                    else:
                        freq_table = df[person_var].value_counts()
                        st.markdown(f"**Cases by {person_var.title()}:**")
                        st.dataframe(freq_table.reset_index().rename(columns={'index': person_var.title(), person_var: 'Count'}))
                        
                        pct_table = (freq_table / freq_table.sum() * 100).round(1)
                        st.markdown("**Percentage Distribution:**")
                        st.dataframe(pct_table.reset_index().rename(columns={'index': person_var.title(), person_var: 'Percent'}))
            
            with col2:
                st.markdown("**Generate visualization:**")
                chart_type = st.radio("Chart type", ["Bar Chart", "Pie Chart"], key="person_chart_type")
                
                if st.button("Create Chart", key="person_chart"):
                    import plotly.express as px
                    
                    if person_var == "age":
                        df['age_group'] = pd.cut(df['age'], 
                                                  bins=[0, 5, 18, 60, 100], 
                                                  labels=['<5', '5-17', '18-59', '60+'])
                        plot_data = df['age_group'].value_counts().sort_index()
                        plot_df = plot_data.reset_index()
                        plot_df.columns = ['Age Group', 'Count']
                        
                        if chart_type == "Bar Chart":
                            fig = px.bar(plot_df, x='Age Group', y='Count', 
                                       title=f'Cases by Age Group',
                                       color='Count',
                                       color_continuous_scale='Reds')
                        else:
                            fig = px.pie(plot_df, values='Count', names='Age Group',
                                       title=f'Cases by Age Group')
                    else:
                        plot_data = df[person_var].value_counts()
                        plot_df = plot_data.reset_index()
                        plot_df.columns = [person_var.title(), 'Count']
                        
                        if chart_type == "Bar Chart":
                            fig = px.bar(plot_df, x=person_var.title(), y='Count',
                                       title=f'Cases by {person_var.title()}',
                                       color='Count',
                                       color_continuous_scale='Reds')
                        else:
                            fig = px.pie(plot_df, values='Count', names=person_var.title(),
                                       title=f'Cases by {person_var.title()}')
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Analysis by Place")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Select geographic variable:**")
                place_var = st.selectbox(
                    "Variable",
                    ["neighborhood", "well"],
                    key="place_var",
                    format_func=lambda x: "Neighborhood" if x == "neighborhood" else "Water Source (Well)"
                )
                
                if st.button("Generate Frequency Table", key="place_table"):
                    freq_table = df[place_var].value_counts()
                    st.markdown(f"**Cases by {place_var.title()}:**")
                    
                    if place_var == "neighborhood":
                        st.dataframe(freq_table.reset_index().rename(columns={'index': 'Neighborhood', place_var: 'Count'}))
                    else:
                        freq_display = freq_table.reset_index()
                        freq_display.columns = ['Well Number', 'Count']
                        freq_display['Well Number'] = 'Well #' + freq_display['Well Number'].astype(str)
                        st.dataframe(freq_display)
                    
                    # Calculate attack rates if they want
                    st.markdown("**Calculate Attack Rate:**")
                    st.write("Attack Rate = (Cases / Population at Risk) × 100")
                    
                    if place_var == "well":
                        st.info("""
                        **Population data:**
                        - Well #1 serves ~60 households (≈300 people)
                        - Well #2 serves ~40 households (≈200 people)
                        """)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            well1_cases = len(df[df['well'] == 1])
                            if st.button("Calculate Attack Rate (Well #1)"):
                                ar = (well1_cases / 300) * 100
                                st.success(f"Well #1 Attack Rate: **{ar:.1f}%** ({well1_cases}/300)")
                        
                        with col_b:
                            well2_cases = len(df[df['well'] == 2])
                            if st.button("Calculate Attack Rate (Well #2)"):
                                ar = (well2_cases / 200) * 100
                                if well2_cases == 0:
                                    st.success(f"Well #2 Attack Rate: **{ar:.1f}%** ({well2_cases}/200)")
                                else:
                                    st.warning(f"Well #2 Attack Rate: **{ar:.1f}%** ({well2_cases}/200)")
            
            with col2:
                st.markdown("**Generate visualization:**")
                
                if st.button("Create Chart", key="place_chart"):
                    import plotly.express as px
                    
                    plot_data = df[place_var].value_counts()
                    plot_df = plot_data.reset_index()
                    
                    if place_var == "neighborhood":
                        plot_df.columns = ['Neighborhood', 'Count']
                        fig = px.bar(plot_df, x='Neighborhood', y='Count',
                                   title='Cases by Neighborhood',
                                   color='Count',
                                   color_continuous_scale='Reds')
                    else:
                        plot_df.columns = ['Well', 'Count']
                        plot_df['Well'] = 'Well #' + plot_df['Well'].astype(str)
                        fig = px.bar(plot_df, x='Well', y='Count',
                                   title='Cases by Water Source',
                                   color='Count',
                                   color_continuous_scale='Reds')
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Analysis by Time")
            
            st.info("📈 Create an epidemic curve to visualize the outbreak over time")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Epidemic Curve Settings:**")
                
                time_unit = st.radio(
                    "Time unit",
                    ["Day", "Week"],
                    key="time_unit"
                )
                
                show_labels = st.checkbox("Show case counts on bars", value=True)
                
                if st.button("Generate Epidemic Curve", type="primary", key="epi_curve"):
                    st.session_state['show_epi_curve'] = True
            
            with col2:
                if st.session_state.get('show_epi_curve', False):
                    import plotly.graph_objects as go
                    
                    # Convert onset to datetime for proper sorting
                    df['onset_date'] = pd.to_datetime('2025-' + df['onset'].str.replace('Mar ', 'March '))
                    
                    # Count cases by date
                    epi_data = df.groupby('onset_date').size().reset_index(name='cases')
                    epi_data = epi_data.sort_values('onset_date')
                    
                    # Create bar chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=epi_data['onset_date'],
                        y=epi_data['cases'],
                        marker_color='#FF6B35',
                        text=epi_data['cases'] if show_labels else None,
                        textposition='outside',
                        name='Cases'
                    ))
                    
                    fig.update_layout(
                        title='Epidemic Curve: Cases by Date of Onset',
                        xaxis_title='Date of Onset',
                        yaxis_title='Number of Cases',
                        showlegend=False,
                        height=400,
                        yaxis=dict(dtick=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("**Epidemic Curve Interpretation:**")
                    
                    first_case = epi_data['onset_date'].min().strftime('%B %d')
                    last_case = epi_data['onset_date'].max().strftime('%B %d')
                    peak_date = epi_data.loc[epi_data['cases'].idxmax(), 'onset_date'].strftime('%B %d')
                    peak_cases = epi_data['cases'].max()
                    duration = (epi_data['onset_date'].max() - epi_data['onset_date'].min()).days
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("First Case", first_case)
                        st.metric("Duration", f"{duration} days")
                    with col_b:
                        st.metric("Peak Date", peak_date)
                        st.metric("Peak Cases", peak_cases)
                    with col_c:
                        st.metric("Last Case", last_case)
                        st.metric("Total Cases", len(df))
                    
                    st.info("""
                    💡 **Pattern Analysis:**
                    - What does the shape suggest about the outbreak type?
                    - Point source? Continuous common source? Propagated?
                    - When was the likely exposure period?
                    """)
        
        with tab4:
            st.markdown("### Custom Analysis")
            
            st.markdown("**Cross-tabulation:**")
            st.write("Create a two-way table to explore relationships between variables")
            
            col1, col2 = st.columns(2)
            
            with col1:
                row_var = st.selectbox(
                    "Row variable",
                    ["sex", "neighborhood", "well", "outcome"],
                    key="row_var"
                )
            
            with col2:
                col_var = st.selectbox(
                    "Column variable",
                    ["sex", "neighborhood", "well", "outcome"],
                    key="col_var"
                )
            
            if st.button("Generate Cross-tabulation"):
                if row_var == col_var:
                    st.error("Please select different variables for rows and columns")
                else:
                    crosstab = pd.crosstab(df[row_var], df[col_var], margins=True)
                    st.markdown(f"**{row_var.title()} × {col_var.title()}**")
                    st.dataframe(crosstab)
                    
                    # Show percentages
                    if st.checkbox("Show row percentages"):
                        crosstab_pct = pd.crosstab(df[row_var], df[col_var], normalize='index') * 100
                        crosstab_pct = crosstab_pct.round(1)
                        st.markdown("**Row Percentages:**")
                        st.dataframe(crosstab_pct)
            
            st.markdown("---")
            
            st.markdown("**Calculate Case Fatality Rate:**")
            
            if st.button("Calculate CFR"):
                deaths = len(df[df['outcome'] == 'Deceased'])
                total_cases = len(df)
                cfr = (deaths / total_cases) * 100
                
                st.success(f"**Case Fatality Rate: {cfr:.1f}%**")
                st.write(f"Deaths: {deaths} / Total Cases: {total_cases}")
                
                st.info("""
                💡 **Interpretation:**
                - How does this CFR compare to expected CFR for cholera?
                - What factors might influence the CFR?
                - Typical cholera CFR with treatment: <1%
                - Typical cholera CFR without treatment: 25-50%
                """)
        
    else:
        st.warning("⚠️ Incomplete case information - Visit the clinic and speak with the nurse for detailed records")
        
        # Show limited initial data
        limited_data = {
            'id': [1, 2, 3, '...', 15],
            'name': ['Sarah M.', 'David K.', 'Grace O.', '...', 'Joseph K.'],
            'age': [4, 6, 3, '?', 72],
            'sex': ['F', 'M', 'F', '?', 'M'],
            'onset': ['Mar 3', 'Mar 4', 'Mar 4', '?', 'Mar 9'],
            'details': ['See nurse', 'See nurse', 'See nurse', 'See nurse', 'See nurse']
        }
        
        df_limited = pd.DataFrame(limited_data)
        st.dataframe(
            df_limited,
            use_container_width=True,
            hide_index=True
        )
        
        st.info("""
        **📝 What you know so far:**
        - Approximately 15 cases reported to the clinic
        - Cases started appearing on March 3rd
        - Range of ages affected
        - Symptoms: severe watery diarrhea, vomiting, dehydration
        - 3 deaths reported
        
        **👉 Next step:** Interview Nurse Sarah at the clinic to get complete case details, addresses, and exposure information.
        """)
        
        if st.button("🏥 Go to Clinic (Interview Nurse Sarah)", type="primary"):
            st.session_state.current_view = 'contacts'
            st.rerun()
    
    st.markdown("---")

elif st.session_state.current_view == 'notes':
    st.markdown("## ✍️ Investigation Notes")
    
    if len(st.session_state.notes) == 0:
        st.info("📝 No notes yet. During interviews, click 'Save Important Info to Notes' to capture key findings here.")
    else:
        st.success(f"You have {len(st.session_state.notes)} saved notes")
        
        for i, note in enumerate(st.session_state.notes):
            with st.expander(f"Note {i+1}", expanded=(i == len(st.session_state.notes)-1)):
                st.markdown(note)
                if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                    st.session_state.notes.pop(i)
                    st.rerun()
    
    st.markdown("---")
    st.markdown("### ➕ Add Manual Note")
    new_note = st.text_area("Type your observation or finding:")
    if st.button("💾 Save Note"):
        if new_note.strip():
            timestamp = datetime.now().strftime('%H:%M')
            st.session_state.notes.append(f"**Manual Note** ({timestamp}): {new_note}")
            st.success("✅ Note saved!")
            st.rerun()

elif st.session_state.current_view == 'objectives':
    st.markdown("## 🎯 Outbreak Investigation Steps")
    
    st.info("""
    These are the standard steps for conducting an outbreak investigation. Work through them systematically.
    Some steps may require multiple actions or interviews.
    """)
    
    # Initialize step completion tracking
    if 'steps_completed' not in st.session_state:
        st.session_state.steps_completed = {
            'step1': False,
            'step2': False,
            'step3': False,
            'step4': False,
            'step5': False,
            'step6': False,
            'step7': False,
            'step8': False,
            'step9': False,
            'step10': False
        }
    
    if 'case_definition' not in st.session_state:
        st.session_state.case_definition = ""
    
    if 'hypotheses' not in st.session_state:
        st.session_state.hypotheses = []
    
    if 'interventions' not in st.session_state:
        st.session_state.interventions = []
    
    # Step 1: Prepare for Field Work
    with st.expander("**Step 1: Prepare for Field Work**", expanded=True):
        st.markdown("### 📋 Objectives:")
        st.write("- Establish the existence of an outbreak")
        st.write("- Verify the diagnosis")
        st.write("- Define and identify cases")
        
        if 'nurse_sarah' in st.session_state.interviewed_characters or 'dr_mensah' in st.session_state.interviewed_characters:
            st.success("✅ You have interviewed health staff and verified the outbreak")
            st.session_state.steps_completed['step1'] = True
        else:
            st.warning("⚠️ Action needed: Interview health facility staff to verify the outbreak")
            if st.button("Go to Contacts", key="step1_contacts"):
                st.session_state.current_view = 'contacts'
                st.rerun()
    
    # Step 2: Verify the Diagnosis
    with st.expander("**Step 2: Verify the Diagnosis**"):
        st.markdown("### 🔬 Objectives:")
        st.write("- Confirm clinical diagnosis")
        st.write("- Review laboratory results (when available)")
        st.write("- Ensure cases meet diagnostic criteria")
        
        st.info("""
        **Current status:**
        - Clinical presentation consistent with cholera
        - Laboratory samples sent to National Lab (results pending)
        - Awaiting confirmatory testing
        """)
        
        st.session_state.steps_completed['step2'] = True
    
    # Step 3: Define a Case and Count Cases
    with st.expander("**Step 3: Establish Case Definition**"):
        st.markdown("### 📝 Objectives:")
        st.write("- Create a working case definition (clinical, time, place)")
        st.write("- Identify and count all cases meeting the definition")
        
        st.markdown("---")
        st.markdown("**Enter your case definition:**")
        
        case_def = st.text_area(
            "Case Definition",
            value=st.session_state.case_definition,
            height=150,
            placeholder="Example: Any person residing in Riverside Village with onset of acute watery diarrhea and vomiting between March 1-10, 2025...",
            help="A case definition should include clinical criteria, time period, and geographic area"
        )
        
        if st.button("Save Case Definition"):
            st.session_state.case_definition = case_def
            if case_def.strip():
                st.success("✅ Case definition saved")
                st.session_state.steps_completed['step3'] = True
            st.rerun()
        
        if st.session_state.case_definition:
            st.success(f"✅ Case definition recorded: {st.session_state.case_definition[:100]}...")
            st.session_state.steps_completed['step3'] = True
    
    # Step 4: Perform Descriptive Epidemiology
    with st.expander("**Step 4: Descriptive Epidemiology (Person, Place, Time)**"):
        st.markdown("### 📊 Objectives:")
        st.write("- Characterize cases by person (age, sex, occupation, etc.)")
        st.write("- Characterize cases by place (residence, workplace, exposures)")
        st.write("- Characterize cases by time (epidemic curve, onset dates)")
        
        if 'nurse_sarah' in st.session_state.interviewed_characters:
            st.success("✅ Complete line list obtained from Nurse Sarah")
            st.info("""
            **Actions you can take:**
            - Review the Line List for demographic patterns
            - Review the Map for geographic clustering
            - Examine the epidemic curve for temporal patterns
            - Calculate attack rates by subgroup
            """)
            st.session_state.steps_completed['step4'] = True
        else:
            st.warning("⚠️ Obtain complete case line list first (interview Nurse Sarah)")
    
    # Step 5: Develop Hypotheses
    with st.expander("**Step 5: Develop Hypotheses**"):
        st.markdown("### 💡 Objectives:")
        st.write("- Generate hypotheses about source and mode of transmission")
        st.write("- Consider all plausible explanations")
        st.write("- Prioritize hypotheses for testing")
        
        st.markdown("---")
        st.markdown("**What are your hypotheses about the source/transmission?**")
        
        new_hypothesis = st.text_input(
            "Add a hypothesis",
            placeholder="Example: Contaminated water from Well #1 is the source",
            key="new_hyp"
        )
        
        if st.button("Add Hypothesis") and new_hypothesis:
            st.session_state.hypotheses.append(new_hypothesis)
            st.rerun()
        
        if st.session_state.hypotheses:
            st.markdown("**Your hypotheses:**")
            for i, hyp in enumerate(st.session_state.hypotheses, 1):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.write(f"{i}. {hyp}")
                with col2:
                    if st.button("🗑️", key=f"del_hyp_{i}"):
                        st.session_state.hypotheses.pop(i-1)
                        st.rerun()
            st.session_state.steps_completed['step5'] = True
    
    # Step 6: Evaluate Hypotheses / Conduct Studies
    with st.expander("**Step 6: Evaluate Hypotheses / Conduct Studies**"):
        st.markdown("### 🔍 Objectives:")
        st.write("- Design and conduct analytical studies if needed")
        st.write("- Conduct environmental investigations")
        st.write("- Collect additional samples for testing")
        
        st.info("""
        **In the full simulation, you would:**
        - Design a case-control or cohort study
        - Create a questionnaire
        - Collect environmental samples
        - Request laboratory testing
        
        **For this demo:** Continue interviewing community members to gather evidence supporting or refuting your hypotheses.
        """)
        
        if len(st.session_state.interviewed_characters) >= 3:
            st.session_state.steps_completed['step6'] = True
    
    # Step 7: Implement Control Measures
    with st.expander("**Step 7: Implement Control and Prevention Measures**"):
        st.markdown("### 🛡️ Objectives:")
        st.write("- Recommend immediate control measures")
        st.write("- Implement interventions to stop transmission")
        st.write("- Prevent additional cases")
        
        st.markdown("---")
        st.markdown("**What control measures do you recommend?**")
        
        new_intervention = st.text_input(
            "Add a control measure",
            placeholder="Example: Chlorinate Well #1 and distribute safe water",
            key="new_int"
        )
        
        if st.button("Add Control Measure") and new_intervention:
            st.session_state.interventions.append(new_intervention)
            st.rerun()
        
        if st.session_state.interventions:
            st.markdown("**Your recommended interventions:**")
            for i, intervention in enumerate(st.session_state.interventions, 1):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.write(f"{i}. {intervention}")
                with col2:
                    if st.button("🗑️", key=f"del_int_{i}"):
                        st.session_state.interventions.pop(i-1)
                        st.rerun()
            st.session_state.steps_completed['step7'] = True
    
    # Step 8: Communicate Findings
    with st.expander("**Step 8: Communicate Findings**"):
        st.markdown("### 📢 Objectives:")
        st.write("- Prepare written report")
        st.write("- Present findings to stakeholders")
        st.write("- Provide recommendations")
        
        st.info("""
        **In a real investigation, you would:**
        - Write an outbreak investigation report
        - Present to Ministry of Health officials
        - Brief community leaders
        - Provide recommendations for prevention
        
        **For this demo:** This step would involve presenting your findings in the full simulation.
        """)
    
    # Step 9: Execute Additional Studies
    with st.expander("**Step 9: Execute Additional Studies (If Needed)**"):
        st.markdown("### 🔬 Objectives:")
        st.write("- Conduct follow-up investigations if questions remain")
        st.write("- Perform additional laboratory testing")
        st.write("- Long-term monitoring")
        
        st.info("This step may not be needed for all outbreaks. Depends on initial findings.")
    
    # Step 10: Maintain Surveillance
    with st.expander("**Step 10: Maintain Surveillance**"):
        st.markdown("### 👁️ Objectives:")
        st.write("- Continue monitoring for new cases")
        st.write("- Evaluate effectiveness of control measures")
        st.write("- Document lessons learned")
        
        st.info("""
        **Ongoing activities:**
        - Daily case count monitoring
        - Evaluate if interventions are working
        - Plan for sustained prevention
        """)
    
    # Progress Summary
    st.markdown("---")
    st.markdown("### 📈 Investigation Progress")
    
    completed = sum(1 for v in st.session_state.steps_completed.values() if v)
    total_steps = len(st.session_state.steps_completed)
    
    progress = completed / total_steps
    st.progress(progress)
    st.caption(f"{completed} of {total_steps} key steps completed or in progress")
    
    if completed >= 6:
        st.success("🎉 You're making excellent progress! Continue your investigation and finalize your recommendations.")
    elif completed >= 3:
        st.info("👍 Good start! Keep gathering information and developing your hypotheses.")
    else:
        st.warning("🔍 Early stages - Begin by interviewing key contacts and gathering case information.")

# Footer
st.markdown("---")
st.caption("FETP Intermediate 2.0 - Outbreak Investigation Simulation | Demo Version")