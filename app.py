import streamlit as st
import anthropic
import pandas as pd
import plotly.express as px
import numpy as np

# Page config
st.set_page_config(
    page_title="FETP Sim: Sidero Valley",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
<style>
    .step-box { padding: 10px; border-radius: 5px; margin-bottom: 5px; font-size: 0.9em; }
    .step-done { background-color: #d4edda; color: #155724; border-left: 4px solid #28a745; }
    .step-todo { background-color: #f8f9fa; color: #6c757d; border-left: 4px solid #dee2e6; }
    .main-header { background-color: #2c3e50; color: white; padding: 15px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'current_view' not in st.session_state: st.session_state.current_view = 'home'
if 'steps_status' not in st.session_state:
    # 0=Not Started, 1=In Progress, 2=Complete
    st.session_state.steps_status = {i: 0 for i in range(1, 14)}
if 'step_notes' not in st.session_state:
    st.session_state.step_notes = {i: "" for i in range(1, 14)}
if 'master_linelist' not in st.session_state:
    st.session_state.master_linelist = pd.DataFrame()
if 'budget' not in st.session_state: st.session_state.budget = 5000 # Resource constraint ($)
if 'time_days' not in st.session_state: st.session_state.time_days = 0 # Time elapsed

# --- DATA GENERATION (GOD VIEW) ---
# Simulating a large dataset that exists in the background
@st.cache_data
def load_god_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'ID': range(1, n+1),
        'Age': np.random.randint(1, 80, n),
        'Sex': np.random.choice(['M', 'F'], n),
        'Occupation': np.random.choice(['Miner', 'Farmer', 'Student', 'Merchant'], n, p=[0.2, 0.5, 0.2, 0.1]),
        'Zone': np.random.choice(['North (Mine)', 'South (Farms)', 'Central'], n),
        'Has_Pigs': np.random.choice([True, False], n),
        'Water_Source': np.random.choice(['River', 'Well', 'Piped'], n),
        'Outcome': 'Healthy' # Default
    })
    
    # Introduce the Outbreak (Japanese Encephalitis logic: Pigs + Farms + Kids)
    # Risk Factor: South Zone + Pigs + Age < 15
    risk_idx = df[
        (df['Zone'] == 'South (Farms)') & 
        (df['Has_Pigs'] == True) & 
        (df['Age'] < 15)
    ].index
    
    # Infect 40% of risk group
    infect_idx = np.random.choice(risk_idx, size=int(len(risk_idx)*0.4), replace=False)
    df.loc[infect_idx, 'Outcome'] = 'Case'
    
    # Add symptoms for cases
    df['Fever'] = False
    df['Seizures'] = False
    df.loc[df['Outcome'] == 'Case', 'Fever'] = True
    # Only 30% of cases have seizures
    seizure_idx = np.random.choice(infect_idx, size=int(len(infect_idx)*0.3), replace=False)
    df.loc[seizure_idx, 'Seizures'] = True
    
    return df

GOD_DATA = load_god_data()

# --- SIDEBAR: 13 STEPS TRACKER ---
with st.sidebar:
    st.markdown("### 📋 Investigation Steps")
    st.progress(sum([1 for v in st.session_state.steps_status.values() if v == 2])/13)
    
    STEPS = {
        1: "Confirm Outbreak",
        2: "Prepare for Field Work",
        3: "Verify Diagnosis",
        4: "Case Definition",
        5: "Find & Record Cases",
        6: "Descriptive Epi",
        7: "Develop Hypotheses",
        8: "Evaluate Hypotheses",
        9: "Refine/Execute Studies",
        10: "Triangulate Data",
        11: "Prevention Measures",
        12: "Surveillance",
        13: "Communicate Findings"
    }
    
    with st.expander("Tracker Details", expanded=True):
        selected_step = st.selectbox("Update Status for:", options=STEPS.keys(), format_func=lambda x: f"{x}. {STEPS[x]}")
        
        status = st.radio("Status:", ["Not Started", "In Progress", "Complete"], 
                          index=st.session_state.steps_status[selected_step],
                          key=f"status_{selected_step}")
        
        # Map status back to int
        status_map = {"Not Started": 0, "In Progress": 1, "Complete": 2}
        st.session_state.steps_status[selected_step] = status_map[status]
        
        # Note taking for the step
        note = st.text_area("Step Notes/Justification:", value=st.session_state.step_notes[selected_step])
        if st.button("Save Step Note"):
            st.session_state.step_notes[selected_step] = note
            st.success("Saved")

    st.markdown("---")
    st.metric("💰 Budget", f"${st.session_state.budget}")
    st.metric("⏱️ Days Elapsed", st.session_state.time_days)

# --- MAIN APP ---

st.markdown('<div class="main-header"><h1>🏔️ Sidero Valley Investigation</h1></div>', unsafe_allow_html=True)

# Navigation
tabs = st.tabs(["🏠 HQ", "🗺️ Field Map", "💻 Data & Analysis", "🔬 Lab Request"])

with tabs[0]:
    st.subheader("Situation Room")
    st.write("Welcome, FETP Team. Use the sidebar to track your progress through the 13 steps.")
    st.info(f"**Current Focus:** Step {selected_step} - {STEPS[selected_step]}")
    
    if st.session_state.step_notes[selected_step]:
        st.markdown(f"**Your Notes:** {st.session_state.step_notes[selected_step]}")
    else:
        st.caption("No notes for this step yet.")

with tabs[1]:
    st.subheader("Field Map & Interviews")
    st.write("Click on a zone to deploy a team for interviews.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://placehold.co/600x400/2c3e50/FFF?text=Sidero+Valley+Map", caption="Sidero Valley (North: Mines | South: Farms)")
    
    with col2:
        st.markdown("### 📋 Survey Deployment")
        zone = st.selectbox("Target Zone", ["North (Mine)", "South (Farms)", "Central"])
        n_samples = st.slider("Sample Size (Households)", 10, 50, 20)
        cost = n_samples * 10
        
        st.warning(f"**Cost:** ${cost} | **Time:** 1 Day")
        
        if st.button("Deploy Survey Team"):
            if st.session_state.budget >= cost:
                st.session_state.budget -= cost
                st.session_state.time_days += 1
                
                # THE TRUTH ENGINE LOGIC
                # Filter God Data based on selection
                subset = GOD_DATA[GOD_DATA['Zone'] == zone].sample(n_samples)
                
                # Reveal specific columns (Simulating a specific questionnaire)
                # In a full version, user would select WHICH columns to ask for
                observed_data = subset[['ID', 'Age', 'Sex', 'Occupation', 'Zone', 'Has_Pigs', 'Fever', 'Seizures']]
                
                # Append to Master Linelist
                st.session_state.master_linelist = pd.concat([st.session_state.master_linelist, observed_data]).drop_duplicates(subset=['ID'])
                
                st.success(f"Survey complete! Added {n_samples} records to your dataset.")
                st.rerun()
            else:
                st.error("Insufficient Budget!")

with tabs[2]:
    st.subheader("💻 Data Analysis Builder")
    
    if st.session_state.master_linelist.empty:
        st.info("No data collected yet. Go to 'Field Map' to conduct surveys.")
    else:
        df = st.session_state.master_linelist
        st.write(f"**Current Dataset:** {len(df)} records")
        with st.expander("View Raw Data"):
            st.dataframe(df)
            
        st.markdown("### 🛠️ Epi Info Tool (Descriptive)")
        st.caption("Construct your analysis manually.")
        
        colA, colB, colC = st.columns(3)
        with colA:
            var_x = st.selectbox("X-Axis Variable", df.columns)
        with colB:
            var_y = st.selectbox("Y-Axis Variable (Optional)", ["None"] + list(df.columns))
        with colC:
            chart_type = st.selectbox("Chart Type", ["Bar Chart", "Histogram", "Scatter", "Pie"])
            
        if st.button("Generate Graph"):
            try:
                if chart_type == "Bar Chart":
                    if var_y != "None":
                        fig = px.bar(df, x=var_x, y=var_y)
                    else:
                        fig = px.bar(df, x=var_x) # Counts
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=var_x)
                elif chart_type == "Pie":
                    fig = px.pie(df, names=var_x)
                elif chart_type == "Scatter":
                    fig = px.scatter(df, x=var_x, y=var_y)
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Analysis of {var_x}")
                
            except Exception as e:
                st.error(f"Could not create chart: {e}")

with tabs[3]:
    st.subheader("Laboratory")
    st.write("Send samples for confirmation.")
    # Placeholder for Step 3 (Verify Diagnosis) logic
    st.info("Lab module coming soon.")