"""Checks for Challenge 6.

Readable on purpose. If a check fails, read the test: it says in words what
your code was supposed to do.

Most checks use the coding agent in `agent.py`. Some use a second problem,
whether to take an umbrella, and one uses three states, so that a function
written to fit the coding agent alone does not pass.
"""

import copy
import math

from agent import TOOLS, UTILITIES, belief_from_risk, next_step
from decisions import (
    Utilities,
    best_action,
    expected_utility,
    posterior,
    value_of_information,
)

# Take an umbrella or not. The forecast says "rainy" on 70% of rainy days
# and on 15% of dry ones.
UMBRELLA: Utilities = {
    "take": {"rain": 70, "dry": 20},
    "leave": {"rain": 0, "dry": 100},
}
CLOUDY_MORNING = {"rain": 0.3, "dry": 0.7}
FORECAST = {
    "rainy": {"rain": 0.7, "dry": 0.15},
    "sunny": {"rain": 0.3, "dry": 0.85},
}

# Three states and three actions, each action best in one state.
THREE: Utilities = {
    "a": {"x": 5, "y": 0, "z": -5},
    "b": {"x": 0, "y": 4, "z": 0},
    "c": {"x": -5, "y": 0, "z": 6},
}


def close(first: float, second: float) -> bool:
    """Return whether two numbers agree to six decimal places."""
    return math.isclose(first, second, abs_tol=1e-6)


def test_expected_utility_weights_each_state_by_its_probability() -> None:
    """0.3 x 70 + 0.7 x 20 = 35 to take it; 0.3 x 0 + 0.7 x 100 = 70 not."""
    assert close(expected_utility(CLOUDY_MORNING, UMBRELLA, "take"), 35)
    assert close(expected_utility(CLOUDY_MORNING, UMBRELLA, "leave"), 70)


def test_expected_utility_of_each_merge_action() -> None:
    """At P(buggy) = 0.15: merge -3.5, review 4.35, discard -16.25."""
    belief = belief_from_risk(0.15)
    assert close(expected_utility(belief, UTILITIES, "merge"), -3.5)
    assert close(expected_utility(belief, UTILITIES, "review"), 4.35)
    assert close(expected_utility(belief, UTILITIES, "discard"), -16.25)


def test_expected_utility_when_the_state_is_certain() -> None:
    """If the change is surely buggy, merging is worth exactly -80."""
    certain = {"correct": 0.0, "buggy": 1.0}
    assert close(expected_utility(certain, UTILITIES, "merge"), -80)


def test_best_action_returns_the_action_and_its_value() -> None:
    """On a cloudy morning, leave the umbrella: expected utility 70."""
    action, value = best_action(CLOUDY_MORNING, UMBRELLA)
    assert action == "leave"
    assert close(value, 70)


def test_best_action_changes_with_the_belief() -> None:
    """Merge a typo fix, review a small feature, discard a rewrite."""
    assert best_action(belief_from_risk(0.01), UTILITIES)[0] == "merge"
    assert best_action(belief_from_risk(0.15), UTILITIES)[0] == "review"
    assert best_action(belief_from_risk(0.8), UTILITIES)[0] == "discard"


def test_best_action_works_for_three_states() -> None:
    """Nothing in best_action may assume two states or the merge actions."""
    action, value = best_action({"x": 0.2, "y": 0.5, "z": 0.3}, THREE)
    assert action == "b" and close(value, 2.0)
    action, value = best_action({"x": 0.1, "y": 0.1, "z": 0.8}, THREE)
    assert action == "c" and close(value, 4.3)


def test_best_action_breaks_a_tie_by_order() -> None:
    """Two actions with the same expected utility: the first one listed."""
    tied = {"first": {"s": 1.0}, "second": {"s": 1.0}}
    assert best_action({"s": 1.0}, tied)[0] == "first"


def test_posterior_follows_bayes_rule() -> None:
    """At 0.15, failing tests: 0.15 x 0.6 / (0.85 x 0.05 + 0.15 x 0.6)."""
    after = posterior(belief_from_risk(0.15), TOOLS["tests"]["fail"])
    assert close(after["buggy"], 0.09 / 0.1325)
    assert close(after["correct"], 0.0425 / 0.1325)


def test_posterior_adds_up_to_one() -> None:
    """A lint warning at 0.4 gives 0.375 and 0.625, which add to 1."""
    after = posterior(belief_from_risk(0.4), TOOLS["lint"]["warn"])
    assert close(after["correct"], 0.375)
    assert close(after["buggy"], 0.625)
    assert close(sum(after.values()), 1.0)


def test_posterior_ignores_evidence_every_state_shares() -> None:
    """An observation equally likely in every state leaves the belief alone."""
    after = posterior(
        {"x": 0.2, "y": 0.5, "z": 0.3}, {"x": 0.4, "y": 0.4, "z": 0.4}
    )
    assert close(after["x"], 0.2) and close(after["y"], 0.5)
    assert close(after["z"], 0.3)


def test_posterior_leaves_the_belief_it_was_given_unchanged() -> None:
    """Return a new belief; the caller's belief must not change."""
    belief = belief_from_risk(0.4)
    before = copy.deepcopy(belief)
    posterior(belief, TOOLS["tests"]["fail"])
    assert belief == before


def test_value_of_information_for_the_forecast() -> None:
    """Rainy forecast: take it (53.33); sunny: leave it (86.86). VOI 6.3."""
    assert close(value_of_information(CLOUDY_MORNING, UMBRELLA, FORECAST), 6.3)


def test_value_of_information_of_the_tests() -> None:
    """The tests are worth 1.608 on a rename and 1.62 on a refactor."""
    tests = TOOLS["tests"]
    assert close(
        value_of_information(belief_from_risk(0.04), UTILITIES, tests), 1.608
    )
    assert close(
        value_of_information(belief_from_risk(0.4), UTILITIES, tests), 1.62
    )


def test_value_of_information_is_zero_when_no_result_changes_the_action() -> (
    None
):
    """At 0.15 the agent reviews whether the tests pass or fail: worth 0."""
    belief = belief_from_risk(0.15)
    for after in (posterior(belief, like) for like in TOOLS["tests"].values()):
        assert best_action(after, UTILITIES)[0] == "review"
    assert close(value_of_information(belief, UTILITIES, TOOLS["tests"]), 0.0)


def test_value_of_information_is_zero_for_a_useless_tool() -> None:
    """A tool that answers the same way in every state is worth nothing."""
    coin = {
        "heads": {"correct": 0.5, "buggy": 0.5},
        "tails": {"correct": 0.5, "buggy": 0.5},
    }
    belief = belief_from_risk(0.4)
    assert close(value_of_information(belief, UTILITIES, coin), 0.0)


def test_value_of_information_is_never_negative() -> None:
    """The agent can always ignore an answer, so looking never hurts."""
    for risk in (0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.72, 0.9, 1.0):
        for tool in TOOLS.values():
            gain = value_of_information(belief_from_risk(risk), UTILITIES, tool)
            assert gain > -1e-9


def test_value_of_a_perfect_tool() -> None:
    """A tool that is always right is worth 21 on the cloudy morning."""
    oracle = {
        "says rain": {"rain": 1.0, "dry": 0.0},
        "says dry": {"rain": 0.0, "dry": 1.0},
    }
    assert close(value_of_information(CLOUDY_MORNING, UMBRELLA, oracle), 21.0)


def test_value_of_information_leaves_the_belief_unchanged() -> None:
    """Looking ahead must not change the belief the agent holds now."""
    belief = belief_from_risk(0.4)
    before = copy.deepcopy(belief)
    value_of_information(belief, UTILITIES, TOOLS["tests"])
    assert belief == before


def test_the_agent_merges_a_typo_fix_without_testing() -> None:
    """At 0.01 the tests are worth 0.25, less than they cost."""
    assert next_step(belief_from_risk(0.01)) == ("act", "merge")


def test_the_agent_runs_the_tests_on_a_refactor() -> None:
    """At 0.4 the tests are worth 1.62, more than their cost of 1."""
    assert next_step(belief_from_risk(0.4)) == ("call", "tests")
