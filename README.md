# FETP JE Outbreak Simulation

A comprehensive Japanese Encephalitis outbreak investigation simulation for Field Epidemiology Training Program (FETP) Intermediate 2.0.

## Project Structure

```
fetp_sim/
├── app.py                 # Main Streamlit application
├── je_logic.py            # Core simulation logic (separated from UI)
├── requirements.txt       # Python dependencies
├── data/
│   ├── villages.csv           # Village-level truth data
│   ├── households_seed.csv    # Seed household data
│   ├── individuals_seed.csv   # Seed individual data (fixed cases)
│   ├── lab_samples.csv        # Lab sample truth table
│   ├── environment_sites.csv  # Environmental site data
│   └── npc_truth.json         # NPC knowledge documents
```

## Key Features

### 5-Day Training Structure
- **Day 1:** Detect, Confirm, Describe (case definition, descriptive epi)
- **Day 2:** Interviews, Hypotheses, Study Design
- **Day 3:** Data Analysis (with realistic noise)
- **Day 4:** Laboratory & Environmental Sampling
- **Day 5:** MOH Briefing & Consequences

### 10 AI-Powered NPCs
Each NPC has:
- Base knowledge (always shared)
- Hidden clues (revealed only when asked specific questions)
- Red herrings (misconceptions that add realism)
- Unknown topics (prevents hallucination)

### One Health Integration
- Veterinary and Environmental officers unlock based on trainee questions
- Rewards trainees who think across human-animal-environment domains

### Consequence Engine
Final outbreak outcome depends on:
- Correct diagnosis
- One Health approach used
- Comprehensive sampling strategy
- Quality of questionnaire
- Evidence-based recommendations

## Deployment

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud
1. Push to GitHub
2. Connect repository to Streamlit Cloud
3. Add `ANTHROPIC_API_KEY` to secrets
4. Set facilitator password in app.py (`FACILITATOR_PASSWORD`)

## Configuration

### Facilitator Password
Change `FACILITATOR_PASSWORD` in app.py to protect ground truth access.

### Customizing the Scenario
To create a new outbreak scenario:
1. Modify CSV files in `data/` folder
2. Update `npc_truth.json` with new NPC knowledge
3. Adjust risk parameters in `je_logic.py`

## Learning Objectives

By completing this simulation, trainees practice:
1. Developing a case definition for AES
2. Conducting key informant interviews
3. Building and interpreting epidemic curves
4. Formulating and testing hypotheses
5. Integrating One Health data sources
6. Designing case-control studies
7. Interpreting laboratory results
8. Making evidence-based recommendations

## Credits

Developed for FETP Intermediate 2.0 training.
Japanese Encephalitis epidemiological model based on WHO guidelines.
