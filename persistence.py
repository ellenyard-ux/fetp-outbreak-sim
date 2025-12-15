"""
Session persistence module for FETP Outbreak Simulation.
Handles serialization and deserialization of session state for save/load functionality.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import streamlit as st

# Version for save file format - increment if format changes
SAVE_FILE_VERSION = "1.0.0"

# Keys that should NOT be persisted (regenerated from files or session-specific)
EXCLUDED_KEYS = {
    'truth',  # Regenerated from CSV files
    'facilitator_mode',  # Security-sensitive, session-specific
}

# Keys that represent the core session state to persist
PERSISTENT_KEYS = {
    # Core progression
    'current_day', 'alert_acknowledged', 'current_view',

    # Resources
    'budget', 'time_remaining', 'lab_credits', 'language',

    # Investigation data
    'decisions', 'generated_dataset', 'lab_results', 'lab_orders',
    'lab_samples_submitted', 'environment_findings',

    # Interview & NPC state
    'interview_history', 'revealed_clues', 'npc_state', 'current_npc',
    'npcs_unlocked', 'questions_asked_about',

    # Progress flags
    'case_definition_written', 'questionnaire_submitted', 'descriptive_analysis_done',
    'line_list_viewed', 'clinic_records_reviewed', 'found_cases_added',
    'hypotheses_documented', 'analysis_confirmed', 'etiology_revealed',
    'one_health_triggered', 'vet_unlocked', 'env_officer_unlocked',
    'descriptive_epi_viewed',

    # Investigation artifacts
    'selected_clinic_cases', 'case_finding_score', 'initial_hypotheses',
    'notebook_entries', 'unlock_flags', 'advance_missing_tasks',
}


def serialize_value(value: Any) -> Any:
    """
    Serialize a single value to a JSON-compatible format.

    Args:
        value: The value to serialize

    Returns:
        JSON-compatible representation of the value
    """
    # Handle None
    if value is None:
        return None

    # Handle pandas DataFrame
    if isinstance(value, pd.DataFrame):
        return {
            '__type__': 'DataFrame',
            'data': value.to_json(orient='split', date_format='iso')
        }

    # Handle sets
    if isinstance(value, set):
        return {
            '__type__': 'set',
            'data': list(value)
        }

    # Handle dictionaries recursively
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}

    # Handle lists recursively
    if isinstance(value, list):
        return [serialize_value(item) for item in value]

    # Handle basic types (int, float, str, bool)
    if isinstance(value, (int, float, str, bool)):
        return value

    # For other types, try to convert to string
    try:
        return str(value)
    except Exception:
        return None


def deserialize_value(value: Any) -> Any:
    """
    Deserialize a value from JSON format back to its original type.

    Args:
        value: The serialized value

    Returns:
        Deserialized value in its original type
    """
    # Handle None
    if value is None:
        return None

    # Handle special type markers
    if isinstance(value, dict) and '__type__' in value:
        type_marker = value['__type__']
        data = value['data']

        if type_marker == 'DataFrame':
            try:
                return pd.read_json(data, orient='split')
            except Exception as e:
                print(f"Error deserializing DataFrame: {e}")
                return None

        elif type_marker == 'set':
            return set(data)

    # Handle dictionaries recursively
    if isinstance(value, dict):
        return {k: deserialize_value(v) for k, v in value.items()}

    # Handle lists recursively
    if isinstance(value, list):
        return [deserialize_value(item) for item in value]

    # Return basic types as-is
    return value


def serialize_session_state(session_state) -> Dict[str, Any]:
    """
    Serialize session state to a JSON-compatible dictionary.

    Args:
        session_state: Streamlit session state object

    Returns:
        Dictionary containing serialized session state
    """
    serialized = {
        'version': SAVE_FILE_VERSION,
        'timestamp': datetime.now().isoformat(),
        'state': {}
    }

    # Serialize each persistent key
    for key in PERSISTENT_KEYS:
        if key in session_state:
            try:
                serialized['state'][key] = serialize_value(session_state[key])
            except Exception as e:
                print(f"Warning: Could not serialize key '{key}': {e}")
                serialized['state'][key] = None

    return serialized


def deserialize_session_state(data: Dict[str, Any], session_state) -> bool:
    """
    Deserialize session state from a dictionary and load it into session_state.

    Args:
        data: Dictionary containing serialized session state
        session_state: Streamlit session state object to load into

    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate version
        if 'version' not in data:
            print("Error: Save file missing version information")
            return False

        saved_version = data['version']
        if saved_version != SAVE_FILE_VERSION:
            print(f"Warning: Save file version ({saved_version}) differs from current version ({SAVE_FILE_VERSION})")
            # Continue anyway - we can add migration logic later if needed

        # Validate state exists
        if 'state' not in data:
            print("Error: Save file missing state data")
            return False

        # Load each state value
        state_data = data['state']
        for key, value in state_data.items():
            try:
                session_state[key] = deserialize_value(value)
            except Exception as e:
                print(f"Warning: Could not deserialize key '{key}': {e}")
                # Continue with other keys

        return True

    except Exception as e:
        print(f"Error deserializing session state: {e}")
        return False


def create_save_file(session_state) -> bytes:
    """
    Create a save file from current session state.

    Args:
        session_state: Streamlit session state object

    Returns:
        Bytes containing JSON save file
    """
    serialized = serialize_session_state(session_state)
    json_str = json.dumps(serialized, indent=2)
    return json_str.encode('utf-8')


def load_save_file(uploaded_file, session_state) -> tuple[bool, str]:
    """
    Load a save file and restore session state.

    Args:
        uploaded_file: Streamlit UploadedFile object
        session_state: Streamlit session state object to load into

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Read and parse JSON
        content = uploaded_file.read()
        data = json.loads(content)

        # Deserialize into session state
        success = deserialize_session_state(data, session_state)

        if success:
            timestamp = data.get('timestamp', 'unknown time')
            return True, f"Session loaded successfully from {timestamp}"
        else:
            return False, "Failed to load session state. The save file may be corrupted."

    except json.JSONDecodeError as e:
        return False, f"Invalid save file format: {e}"
    except Exception as e:
        return False, f"Error loading save file: {e}"


def get_save_filename(session_state) -> str:
    """
    Generate a descriptive filename for the save file.

    Args:
        session_state: Streamlit session state object

    Returns:
        Suggested filename for the save file
    """
    current_day = session_state.get('current_day', 1)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"fetp_save_day{current_day}_{timestamp}.json"
