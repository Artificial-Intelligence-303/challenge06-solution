"""Checks for Challenge 6.

Readable on purpose. If a check fails, read the test: it says in words what
your code was supposed to do.

Most checks use the car's sensor data. Some use the AI-writing detector and
the email folders from class, so that code written to fit the car alone does
not pass.
"""

import copy
import math

from naive_bayes import Model, accuracy, bayes_rule, classify, train
from sensors import load

# The detector from class: 10% of essays are AI-written; it flags 90% of
# those and 5% of essays a person wrote.
DETECTOR_PRIOR = {"ai": 0.1, "human": 0.9}
FLAGGED = {"ai": 0.9, "human": 0.05}
TWO_DETECTORS = Model(
    DETECTOR_PRIOR,
    {
        "detector1": {"flagged": FLAGGED, "clean": {"ai": 0.1, "human": 0.95}},
        "detector2": {"flagged": FLAGGED, "clean": {"ai": 0.1, "human": 0.95}},
    },
)

# The email folders from class.
FOLDERS = {"personal": 0.5, "promotions": 0.3, "spam": 0.2}
FREE = {"personal": 0.05, "promotions": 0.6, "spam": 0.8}
WINNER = {"personal": 0.1, "promotions": 0.5, "spam": 0.7}
EMAIL = Model(FOLDERS, {"free": {"yes": FREE}, "winner": {"yes": WINNER}})

# Four labelled examples, small enough to count by hand.
TINY = [
    ("present", {"camera": "yes"}),
    ("present", {"camera": "yes"}),
    ("present", {"camera": "no"}),
    ("absent", {"camera": "no"}),
]

WITHOUT_CAMERA2 = ("camera", "lidar", "radar")


def close(first: float, second: float, tol: float = 1e-3) -> bool:
    """Return whether two numbers agree to within `tol`."""
    return math.isclose(first, second, abs_tol=tol)


def test_bayes_rule_gives_the_detector_answer_from_class() -> None:
    """A flagged essay: 0.9 x 0.1 / (0.9 x 0.1 + 0.05 x 0.9) = 0.67."""
    posterior = bayes_rule(DETECTOR_PRIOR, FLAGGED)
    assert close(posterior["ai"], 0.6667)
    assert close(posterior["human"], 0.3333)


def test_bayes_rule_works_for_three_classes() -> None:
    """An email with "free": 0.068, 0.493, 0.438, as in class."""
    posterior = bayes_rule(FOLDERS, FREE)
    assert close(posterior["personal"], 0.0685)
    assert close(posterior["promotions"], 0.4932)
    assert close(posterior["spam"], 0.4384)


def test_bayes_rule_leaves_the_prior_unchanged() -> None:
    """The prior given to bayes_rule must not change."""
    prior = dict(DETECTOR_PRIOR)
    bayes_rule(prior, FLAGGED)
    assert prior == DETECTOR_PRIOR


def test_train_counts_the_prior() -> None:
    """Three of the four tiny examples are present: P(present) = 0.75."""
    model = train(TINY)
    assert close(model.prior["present"], 0.75)
    assert close(model.prior["absent"], 0.25)


def test_train_without_smoothing_gives_plain_shares() -> None:
    """Two of the three present examples have camera yes: 2/3."""
    model = train(TINY, smoothing=0)
    assert close(model.likelihood["camera"]["yes"]["present"], 2 / 3)
    assert close(model.likelihood["camera"]["no"]["absent"], 1.0)


def test_train_smoothing_adds_one_to_every_count() -> None:
    """(2 + 1) / (3 + 2) = 0.6 for present; (0 + 1) / (1 + 2) for absent."""
    model = train(TINY)
    assert close(model.likelihood["camera"]["yes"]["present"], 0.6)
    assert close(model.likelihood["camera"]["yes"]["absent"], 1 / 3)


def test_train_never_gives_an_unseen_value_probability_zero() -> None:
    """No absent example had camera yes, but smoothing keeps it above 0."""
    model = train(TINY)
    assert model.likelihood["camera"]["yes"]["absent"] > 0


def test_train_on_the_car_data() -> None:
    """144 of 400 moments had a pedestrian; camera yes in 122 of them."""
    model = train(load("train"))
    assert close(model.prior["present"], 0.36)
    assert close(model.likelihood["camera"]["yes"]["present"], 123 / 146)
    assert close(model.likelihood["camera"]["yes"]["absent"], 25 / 258)


def test_classify_with_one_feature_is_bayes_rule() -> None:
    """With one piece of evidence, naive Bayes and Bayes' rule agree."""
    model = train(load("train"))
    alone = classify(model, {"camera": "yes"})
    by_rule = bayes_rule(model.prior, model.likelihood["camera"]["yes"])
    assert close(alone["present"], by_rule["present"], 1e-9)


def test_classify_multiplies_the_evidence() -> None:
    """Two detectors flag the essay: 0.081 / 0.08325 = 0.97, as in class."""
    both = {"detector1": "flagged", "detector2": "flagged"}
    assert close(classify(TWO_DETECTORS, both)["ai"], 0.9730)


def test_classify_reads_the_value_that_was_observed() -> None:
    """One detector flags, the other does not: 0.17."""
    mixed = {"detector1": "flagged", "detector2": "clean"}
    assert close(classify(TWO_DETECTORS, mixed)["ai"], 0.1739)


def test_classify_works_for_three_classes() -> None:
    """Email with "free" and "winner": 0.012, 0.440, 0.548, as in class."""
    posterior = classify(EMAIL, {"free": "yes", "winner": "yes"})
    assert close(posterior["personal"], 0.0122)
    assert close(posterior["promotions"], 0.4401)
    assert close(posterior["spam"], 0.5477)


def test_classify_adds_up_to_one() -> None:
    """Every posterior is a distribution."""
    model = train(load("train"))
    posterior = classify(
        model, {"camera": "yes", "lidar": "no", "radar": "yes"}
    )
    assert close(sum(posterior.values()), 1.0, 1e-9)


def test_classify_skips_a_feature_it_has_no_table_for() -> None:
    """A sensor the model never saw is ignored, not an error."""
    model = train(load("train", WITHOUT_CAMERA2))
    with_extra = classify(model, {"camera": "yes", "thermal": "yes"})
    alone = classify(model, {"camera": "yes"})
    assert close(with_extra["present"], alone["present"], 1e-9)


def test_classify_leaves_the_evidence_unchanged() -> None:
    """The evidence given to classify must not change."""
    model = train(load("train"))
    evidence = {"camera": "yes", "lidar": "no"}
    before = copy.deepcopy(evidence)
    classify(model, evidence)
    assert evidence == before


def test_classify_on_the_car_data() -> None:
    """Camera and camera2 both yes: naive Bayes says 0.977 present."""
    model = train(load("train"))
    posterior = classify(model, {"camera": "yes", "camera2": "yes"})
    assert close(posterior["present"], 0.9768)


def test_accuracy_on_examples_counted_by_hand() -> None:
    """The tiny model gets three of the four tiny examples right."""
    assert close(accuracy(train(TINY), TINY), 0.75)


def test_accuracy_on_the_test_data() -> None:
    """0.900 with all four sensors, 0.935 without camera2."""
    model = train(load("train"))
    without = train(load("train", WITHOUT_CAMERA2))
    assert close(accuracy(model, load("test")), 0.900)
    assert close(accuracy(without, load("test", WITHOUT_CAMERA2)), 0.935)
