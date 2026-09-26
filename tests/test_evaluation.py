from src.evaluation import compute_entity_f05, evaluate_predictions

def test_readme_example():
    true_set = {"S2-00047", "S3-00812"}
    pred_set = {"S2-00047", "S2-00193", "S3-00812"}
    score = compute_entity_f05(true_set, pred_set)
    # Expected: 2/3 precision, 1.0 recall -> (1.25 * (2/3) * 1) / (0.25 * (2/3) + 1) = 0.83333 / 1.16666 = 0.7142857
    assert abs(score - 0.7142857) < 1e-5, f"Expected 0.7142857, got {score}"
    print("README example test passed!")

def test_singleton():
    # Correct singleton
    assert compute_entity_f05(set(), set()) == 1.0
    # False positive on singleton
    assert compute_entity_f05(set(), {"S2-00001"}) == 0.0
    # False negative on non-singleton
    assert compute_entity_f05({"S2-00001"}, set()) == 0.0
    print("Singleton tests passed!")

if __name__ == "__main__":
    test_readme_example()
    test_singleton()
    print("All evaluation tests passed!")
