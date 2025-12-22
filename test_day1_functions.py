"""
Test script for Day 1 medical records functions
"""

# Mock streamlit to avoid import errors
import sys
from unittest.mock import MagicMock
sys.modules['streamlit'] = MagicMock()

# Now import je_logic
import je_logic

def test_get_medical_chart():
    """Test that medical charts exclude exposure data"""
    print("\n=== Testing get_medical_chart ===")

    # Test valid patient
    chart = je_logic.get_medical_chart("HOSP-01")

    if chart:
        print(f"✓ Retrieved chart for {chart['Name']}")
        print(f"  Patient ID: {chart['Patient ID']}")
        print(f"  Age: {chart['Age']}, Sex: {chart['Sex']}")
        print(f"  Village: {chart['Village']}")
        print(f"  Date of Onset: {chart['Date of Onset']}")
        print(f"  Temperature: {chart['Temperature']}")
        print(f"  Neuro Signs: {chart['Neuro Signs']}")
        print(f"  WBC Count: {chart['WBC Count']}")
        print(f"  Outcome: {chart['Outcome']}")

        # Check that exposure data is NOT present
        assert 'pig' not in str(chart).lower(), "ERROR: Chart contains pig exposure data!"
        assert 'mosquito' not in str(chart).lower(), "ERROR: Chart contains mosquito data!"
        print("✓ Chart does NOT contain exposure data (pigs, mosquitoes)")
    else:
        print("✗ Failed to retrieve chart")

    # Test invalid patient
    invalid_chart = je_logic.get_medical_chart("INVALID-99")
    assert invalid_chart is None, "Should return None for invalid patient"
    print("✓ Returns None for invalid patient ID")


def test_get_clinic_log():
    """Test that clinic logs have natural language complaints"""
    print("\n=== Testing get_clinic_log ===")

    for village_id in ['V1', 'V2', 'V3']:
        log = je_logic.get_clinic_log(village_id)
        print(f"\n✓ Retrieved {len(log)} entries for village {village_id}")

        # Show first 3 entries
        for i, entry in enumerate(log[:3]):
            print(f"  {i+1}. {entry['name']}, age {entry['age']}: '{entry['complaint']}'")

        # Verify natural language (should not be clinical codes)
        for entry in log:
            complaint = entry['complaint'].lower()
            # Natural language should have common words
            has_natural_language = any(word in complaint for word in
                ['hot', 'shaking', 'sleeping', 'hurts', 'pain', 'cough', 'won\'t'])
            if has_natural_language:
                break

        assert has_natural_language, f"Clinic log for {village_id} doesn't have natural language"

    print("\n✓ All clinic logs use natural language complaints")

    # Test village name input
    log_by_name = je_logic.get_clinic_log("Nalu")
    assert len(log_by_name) > 0, "Should accept village name"
    print("✓ Accepts village names (e.g., 'Nalu')")


def test_check_case_definition():
    """Test case definition validation"""
    print("\n=== Testing check_case_definition ===")

    # Valid criteria (clinical only)
    valid_criteria = {
        'age': '< 15 years',
        'symptoms': 'fever and seizures',
        'village': 'Nalu',
        'onset_date': 'June 2025'
    }

    result = je_logic.check_case_definition(valid_criteria)
    assert result['valid'] == True, "Should accept clinical criteria"
    print("✓ Valid: Clinical/Person/Place/Time criteria accepted")
    print(f"  Message: {result['message']}")

    # Invalid criteria (includes pigs)
    invalid_criteria_pigs = {
        'symptoms': 'fever',
        'exposure': 'contact with pigs'
    }

    result = je_logic.check_case_definition(invalid_criteria_pigs)
    assert result['valid'] == False, "Should reject pig exposure criteria"
    print("\n✓ Invalid: Rejected criteria with 'pigs'")
    print(f"  Message: {result['message']}")

    # Invalid criteria (includes mosquitoes)
    invalid_criteria_mosquitoes = {
        'symptoms': 'fever',
        'risk_factor': 'mosquito exposure'
    }

    result = je_logic.check_case_definition(invalid_criteria_mosquitoes)
    assert result['valid'] == False, "Should reject mosquito exposure criteria"
    print("\n✓ Invalid: Rejected criteria with 'mosquitoes'")
    print(f"  Message: {result['message']}")

    # Invalid criteria (includes water)
    invalid_criteria_water = {
        'symptoms': 'fever',
        'exposure': 'rice paddy water'
    }

    result = je_logic.check_case_definition(invalid_criteria_water)
    assert result['valid'] == False, "Should reject water exposure criteria"
    print("\n✓ Invalid: Rejected criteria with 'water/rice paddy'")


if __name__ == '__main__':
    print("=" * 60)
    print("Day 1 Medical Records & Clinical Symptoms - Function Tests")
    print("=" * 60)

    try:
        test_get_medical_chart()
        test_get_clinic_log()
        test_check_case_definition()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
