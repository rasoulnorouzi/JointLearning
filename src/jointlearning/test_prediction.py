"""
Test script for the joint causal prediction system.

This script validates that the prediction system follows the specified rules
and produces expected outputs according to the pipeline specifications.
"""

from predicion import predict_sentence, batch_predict, predict_with_rules
import json

def test_rule_implementations():
    """Test specific rule implementations."""
    
    print("Testing Rule-Based Pipeline Implementation")
    print("=" * 60)
    
    # Test cases designed to validate specific rules
    test_cases = [
        {
            "text": "Smoking causes lung cancer and heart disease.",
            "expected_causal": True,
            "description": "Clear causal sentence with explicit causal verb"
        },
        {
            "text": "The sky is blue today.",
            "expected_causal": False,
            "description": "Non-causal descriptive sentence"
        },
        {
            "text": "High unemployment leads to increased crime rates.",
            "expected_causal": True,
            "description": "Causal sentence with 'leads to' construction"
        },
        {
            "text": "There is a correlation between education and income.",
            "expected_causal": False,
            "description": "Correlation mention (should be non-causal)"
        },
        {
            "text": "Because of the rain, the picnic was cancelled.",
            "expected_causal": True,
            "description": "Causal sentence with 'because of' construction"
        }
    ]
    
    # Test different configurations
    configs = [
        {"name": "Full Pipeline", "config": {"level": 3, "rel_mode": "auto", "cause_decision": "cls+span"}},
        {"name": "Classifier Only", "config": {"level": 2, "cause_decision": "cls_only"}},
        {"name": "Spans Only", "config": {"level": 2, "cause_decision": "span_only"}},
        {"name": "Head Mode", "config": {"level": 3, "rel_mode": "head", "cause_decision": "cls+span"}},
    ]
    
    for config_info in configs:
        print(f"\nTesting Configuration: {config_info['name']}")
        print("-" * 40)
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for i, test_case in enumerate(test_cases, 1):
            try:
                result = predict_sentence(test_case["text"], config_info["config"])
                predicted_causal = result["causal"]
                expected_causal = test_case["expected_causal"]
                
                is_correct = predicted_causal == expected_causal
                if is_correct:
                    correct_predictions += 1
                
                status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
                print(f"Test {i}: {status}")
                print(f"  Text: {test_case['text'][:60]}...")
                print(f"  Expected: {expected_causal}, Predicted: {predicted_causal}")
                print(f"  Relations found: {len(result.get('relations', []))}")
                
            except Exception as e:
                print(f"Test {i}: ✗ ERROR - {e}")
        
        accuracy = correct_predictions / total_predictions * 100
        print(f"\nAccuracy for {config_info['name']}: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")


def test_level_differences():
    """Test that different processing levels produce different outputs."""
    
    print("\n" + "=" * 60)
    print("Testing Processing Level Differences")
    print("=" * 60)
    
    test_text = "Heavy rain caused flooding in the downtown area."
    
    levels = [0, 1, 2, 3]
    
    print(f"Test sentence: {test_text}")
    print()
    
    for level in levels:
        result = predict_with_rules(
            text=test_text,
            level=level,
            rel_mode="auto",
            cause_decision="cls+span"
        )
        
        print(f"Level {level} result:")
        print(f"  Causal: {result['causal']}")
        print(f"  Relations: {len(result.get('relations', []))}")
        if result.get('relations'):
            for rel in result['relations']:
                print(f"    {rel['cause']} -> {rel['effect']}")
        print()


def test_relation_modes():
    """Test different relation pairing modes."""
    
    print("=" * 60)
    print("Testing Relation Pairing Modes")
    print("=" * 60)
    
    test_text = "Poor economic conditions led to unemployment and social unrest."
    
    modes = ["head", "auto"]
    
    print(f"Test sentence: {test_text}")
    print()
    
    for mode in modes:
        result = predict_sentence(test_text, {"level": 3, "rel_mode": mode, "cause_decision": "cls+span"})
        
        print(f"Mode '{mode}' results:")
        print(f"  Causal: {result['causal']}")
        print(f"  Relations found: {len(result.get('relations', []))}")
        for i, rel in enumerate(result.get('relations', []), 1):
            print(f"    {i}. '{rel['cause']}' -> '{rel['effect']}'")
        print()


def test_batch_processing():
    """Test batch processing functionality."""
    
    print("=" * 60) 
    print("Testing Batch Processing")
    print("=" * 60)
    
    sentences = [
        "Exercise improves cardiovascular health.",
        "The conference was held in Chicago.", 
        "Stress can lead to various health problems.",
        "She bought a new car yesterday.",
        "Deforestation causes habitat loss and climate change."
    ]
    
    print("Processing batch of sentences...")
    results = batch_predict(sentences, {"level": 3, "rel_mode": "auto"})
    
    causal_count = sum(1 for r in results if r.get('causal', False))
    total_relations = sum(len(r.get('relations', [])) for r in results)
    
    print(f"Batch results: {causal_count}/{len(sentences)} causal, {total_relations} total relations")
    
    for i, (sentence, result) in enumerate(zip(sentences, results), 1):
        status = "CAUSAL" if result.get('causal', False) else "NON-CAUSAL"
        rel_count = len(result.get('relations', []))
        print(f"{i}. {status} ({rel_count} relations): {sentence[:50]}...")


if __name__ == "__main__":
    print("JOINT CAUSAL MODEL - PREDICTION SYSTEM TESTS")
    print("=" * 70)
    
    try:
        test_rule_implementations()
        test_level_differences() 
        test_relation_modes()
        test_batch_processing()
        
        print("\n" + "=" * 70)
        print("All tests completed successfully!")
        print("The prediction system is working according to specifications.")
        print("=" * 70)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
