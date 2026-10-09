"""Pin the CORAL probability decoding as mathematically correct and unchanged.

These are golden/behavioural tests: if anyone "optimises" or rewrites
`ordinal_to_class`/`ordinal_to_probs` in losses.py, these break. This is
deliberate — the decoding is a validated part of the research record and must
not drift silently.
"""
import torch

from kneevision.training.losses import ordinal_to_class, ordinal_to_probs

# ── ordinal_to_class: sum of rounded binary sigmoids ───────────────────────────

def test_decodes_to_zero_when_all_thresholds_confident_false():
    logits = torch.tensor([[-5.0, -5.0, -5.0, -5.0]])
    assert ordinal_to_class(logits).item() == 0


def test_decodes_to_top_grade_when_all_thresholds_confident_true():
    logits = torch.tensor([[+5.0, +5.0, +5.0, +5.0]])
    assert ordinal_to_class(logits).item() == 4


def test_decode_equals_count_of_asserted_thresholds():
    """CORAL reference: class == number of \"is grade > k\" thresholds that hold.

    For perfect thresholds z_k = +10 if true grade > k else -10, decoding must
    recover the true grade exactly.
    """
    for true_grade in range(5):
        thresholds = [+10.0 if true_grade > k else -10.0 for k in range(4)]
        logits = torch.tensor([thresholds])
        assert ordinal_to_class(logits).item() == true_grade, (
            f"failed for true grade {true_grade}"
        )


def test_decode_never_exceeds_num_classes_minus_one():
    rng = torch.Generator().manual_seed(0)
    logits = torch.randn(64, 4, generator=rng) * 20
    preds = ordinal_to_class(logits)
    assert preds.min() >= 0
    assert preds.max() <= 4


def test_decode_is_argmax_free_summation():
    """Numerical check that the rule is round(sigmoid) then sum, not argmax."""
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0]])  # sigmoid(2)~0.88→1, rest ~0.5→? 
    preds = ordinal_to_class(logits)
    # torch.round uses round-half-to-even: sigmoid(0)=0.5 rounds to 0,
    # so exactly one threshold asserts → grade 1 (argmax over the 4 logits would
    # also predict 0-threshold index, but the decoded grade is the COUNT, 1).
    assert preds.item() == 1


def test_decode_non_decreasing_in_each_threshold():
    """Raising any single threshold logit must never lower the predicted grade."""
    rng = torch.Generator().manual_seed(3)
    for _ in range(200):
        logits = torch.randn(1, 4, generator=rng)
        for j in range(4):
            raised = logits.clone()
            raised[0, j] += 4.0
            assert ordinal_to_class(logits).item() <= ordinal_to_class(raised).item(), (
                f"raising threshold {j} decreased the grade for {logits.tolist()}"
            )


# ── ordinal_to_probs: proper distribution over grades 0..4 ────────────────────

def test_probs_sum_to_one_over_all_grades():
    rng = torch.Generator().manual_seed(1)
    logits = torch.randn(32, 4, generator=rng)
    probs = ordinal_to_probs(logits)
    assert probs.shape == (32, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(32), atol=1e-6)


def test_probs_are_nonnegative_never_nan():
    rng = torch.Generator().manual_seed(2)
    logits = torch.randn(32, 4, generator=rng) * 3
    probs = ordinal_to_probs(logits)
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()


def test_probs_peak_agrees_with_hard_decode_for_decisive_logits():
    """For confident monotone logits, the argmax of the soft distribution must
    equal the hard decoded grade — the two decoding views agree."""
    for true_grade in range(5):
        thresholds = [+8.0 if true_grade > k else -8.0 for k in range(4)]
        logits = torch.tensor([thresholds])
        hard = ordinal_to_class(logits).item()
        soft = ordinal_to_probs(logits).argmax(dim=1).item()
        assert hard == soft, f"mismatch for true grade {true_grade}"