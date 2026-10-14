"""Run the coding agent on five kinds of change.

Usage:
    uv run python src/main.py              every change, and the next step
    uv run python src/main.py risk 0.25    one change with P(buggy) = 0.25
"""

import sys

from agent import CHANGES, COSTS, TOOLS, UTILITIES, belief_from_risk, next_step
from decisions import (
    best_action,
    expected_utility,
    posterior,
    value_of_information,
)


def explain(name: str, risk: float) -> None:
    """Print everything the agent weighs for one change."""
    belief = belief_from_risk(risk)
    print(f"\n{name}: P(buggy) = {risk}")
    for action in UTILITIES:
        value = expected_utility(belief, UTILITIES, action)
        print(f"  EU({action}) = {value:.3f}")
    action, value = best_action(belief, UTILITIES)
    print(f"  acting now: {action}, expected utility {value:.3f}")
    for tool_name, tool in TOOLS.items():
        gain = value_of_information(belief, UTILITIES, tool)
        cost = COSTS[tool_name]
        print(f"  {tool_name}: value of information {gain:.3f}, cost {cost}")
        for result, likelihood in tool.items():
            after = posterior(belief, likelihood)
            best_after, value_after = best_action(after, UTILITIES)
            print(
                f"    if {result}: P(buggy) becomes {after['buggy']:.3f},"
                f" then {best_after} ({value_after:.3f})"
            )
    step, what = next_step(belief)
    print(f"  the agent's next step: {step} {what}")


def table() -> None:
    """Print one line per kind of change."""
    print(
        f"{'change':<28}{'P(buggy)':>9}  {'act now':<9}"
        f"{'VOI tests':>10}{'VOI lint':>10}  next step"
    )
    for name, risk in CHANGES.items():
        belief = belief_from_risk(risk)
        action = best_action(belief, UTILITIES)[0]
        tests = value_of_information(belief, UTILITIES, TOOLS["tests"])
        lint = value_of_information(belief, UTILITIES, TOOLS["lint"])
        step, what = next_step(belief)
        print(
            f"{name:<28}{risk:>9.2f}  {action:<9}"
            f"{tests:>10.3f}{lint:>10.3f}  {step} {what}"
        )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "risk":
        explain("this change", float(sys.argv[2]))
    else:
        table()
        for change, p_buggy in CHANGES.items():
            explain(change, p_buggy)
