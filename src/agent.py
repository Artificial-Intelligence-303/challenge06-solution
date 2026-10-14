"""A coding agent deciding what to do with a change it just wrote.

Provided: you do not need to change this file to finish the challenge.
Reflection question 2 asks you to change one utility here and run again.

The agent has finished a code change. Before anything happens to it, the
agent estimates how likely the change is to be buggy, from how large it is
and how much of the codebase it touches. Then it can:

    merge      put the change into the main branch now
    review     ask a person on the team to review it first
    discard    throw the change away and start over

Before acting it may call a tool, once:

    tests      run the test suite, which costs time
    lint       run a static checker, which is quick and much less telling

Every number in this file is invented for the challenge.
"""

from decisions import (
    Belief,
    Tool,
    Utilities,
    best_action,
    value_of_information,
)

STATES = ("correct", "buggy")

# How much the team prefers each outcome. A buggy merge reaches the users;
# a review costs a teammate's time and catches most bugs; discarding a
# correct change throws away the work.
UTILITIES: Utilities = {
    "merge": {"correct": 10, "buggy": -80},
    "review": {"correct": 6, "buggy": -5},
    "discard": {"correct": -20, "buggy": 5},
}

# P(result | state) for each tool. The tests fail on 60% of buggy changes
# and, being a little flaky, on 5% of correct ones.
TOOLS: dict[str, Tool] = {
    "tests": {
        "fail": {"correct": 0.05, "buggy": 0.6},
        "pass": {"correct": 0.95, "buggy": 0.4},
    },
    "lint": {
        "warn": {"correct": 0.2, "buggy": 0.5},
        "clean": {"correct": 0.8, "buggy": 0.5},
    },
}

# What each tool costs, in the same units as the utilities.
COSTS = {"tests": 1.0, "lint": 0.2}

# The agent's estimate of P(buggy) for five kinds of change.
CHANGES = {
    "fix a typo in a comment": 0.01,
    "rename a variable": 0.04,
    "add a small feature": 0.15,
    "refactor a module": 0.4,
    "rewrite the database layer": 0.8,
}


def belief_from_risk(p_buggy: float) -> Belief:
    """Return the agent's belief for a change with this chance of a bug."""
    return {"correct": 1 - p_buggy, "buggy": p_buggy}


def next_step(
    belief: Belief,
    utilities: Utilities = UTILITIES,
    tools: dict[str, Tool] = TOOLS,
    costs: dict[str, float] = COSTS,
) -> tuple[str, str]:
    """Return what the agent does next: call a tool, or act.

    A tool is worth calling when its value of information is more than its
    cost. Of those, the agent calls the one that gains the most. If none is
    worth its cost, the agent takes the best action now.

    Args:
        belief: P(correct) and P(buggy).
        utilities: The utility of each action in each state.
        tools: Every tool the agent could call.
        costs: What each tool costs.

    Returns:
        ("call", tool name) or ("act", action name).
    """
    gains = {
        name: value_of_information(belief, utilities, tool) - costs[name]
        for name, tool in tools.items()
    }
    best_tool = max(gains, key=lambda name: gains[name])
    if gains[best_tool] > 0:
        return "call", best_tool
    return "act", best_action(belief, utilities)[0]
