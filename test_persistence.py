"""
Simple test to verify persistence functions work correctly.
"""

import persistence
import pandas as pd
from datetime import datetime


def test_basic_serialization():
    """Test serialization of basic types."""
    print("Testing basic serialization...")

    # Test data
    test_data = {
        'current_day': 3,
        'budget': 5000,
        'time_remaining': 6,
        'lab_credits': 10,
        'language': 'en',
        'alert_acknowledged': True,
        'decisions': {
            'case_definition': {'clinical_AES': True},
            'hypotheses': ['JE virus', 'Water contamination']
        },
        'lab_results': ['Result 1', 'Result 2'],
        'questions_asked_about': {'symptoms', 'location', 'timeline'}
    }

    # Serialize
    serialized = {}
    for key, value in test_data.items():
        serialized[key] = persistence.serialize_value(value)
        print(f"  ✓ Serialized {key}: {type(value).__name__}")

    # Deserialize
    deserialized = {}
    for key, value in serialized.items():
        deserialized[key] = persistence.deserialize_value(value)
        print(f"  ✓ Deserialized {key}: {type(deserialized[key]).__name__}")

    # Verify set is restored correctly
    assert isinstance(deserialized['questions_asked_about'], set)
    assert deserialized['questions_asked_about'] == test_data['questions_asked_about']

    print("✅ Basic serialization test passed!\n")


def test_dataframe_serialization():
    """Test DataFrame serialization."""
    print("Testing DataFrame serialization...")

    # Create sample DataFrame
    df = pd.DataFrame({
        'name': ['John', 'Jane', 'Bob'],
        'age': [30, 25, 35],
        'village': ['Nalu', 'Wako', 'Nalu']
    })

    # Serialize
    serialized = persistence.serialize_value(df)
    print(f"  ✓ Serialized DataFrame with shape {df.shape}")

    # Deserialize
    deserialized = persistence.deserialize_value(serialized)
    print(f"  ✓ Deserialized DataFrame with shape {deserialized.shape}")

    # Verify
    assert isinstance(deserialized, pd.DataFrame)
    assert deserialized.shape == df.shape
    assert list(deserialized.columns) == list(df.columns)

    print("✅ DataFrame serialization test passed!\n")


def test_full_save_load():
    """Test full save/load cycle."""
    print("Testing full save/load cycle...")

    # Mock session state
    class MockSessionState:
        def __init__(self):
            self.data = {}

        def __getitem__(self, key):
            return self.data[key]

        def __setitem__(self, key, value):
            self.data[key] = value

        def __contains__(self, key):
            return key in self.data

        def get(self, key, default=None):
            return self.data.get(key, default)

    # Create mock state
    mock_state = MockSessionState()
    mock_state['current_day'] = 2
    mock_state['budget'] = 7500
    mock_state['time_remaining'] = 8
    mock_state['lab_credits'] = 15
    mock_state['decisions'] = {'test': 'value'}
    mock_state['interview_history'] = {'npc1': ['message1', 'message2']}

    # Serialize
    serialized = persistence.serialize_session_state(mock_state)
    print(f"  ✓ Created save file with version {serialized['version']}")
    print(f"  ✓ Timestamp: {serialized['timestamp']}")
    print(f"  ✓ State keys: {len(serialized['state'])} keys")

    # Create new mock state and deserialize
    new_state = MockSessionState()
    success = persistence.deserialize_session_state(serialized, new_state)

    assert success
    assert new_state['current_day'] == 2
    assert new_state['budget'] == 7500
    assert new_state['decisions'] == {'test': 'value'}

    print("✅ Full save/load cycle test passed!\n")


if __name__ == '__main__':
    try:
        test_basic_serialization()
        test_dataframe_serialization()
        test_full_save_load()
        print("=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
