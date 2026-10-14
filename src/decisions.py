"""Choosing an action under uncertainty, and deciding when to look first.

**You implement three functions in this file**: `expected_utility`,
`best_action` and `value_of_information`. Everything else is provided,
including two helpers at the bottom, `result_probability` and `posterior`:
the same Bayes' rule you wrote in Challenge 5, for one tool result.

Three kinds of table, all plain dictionaries:

    belief      state -> probability
                {"correct": 0.5, "buggy": 0.5}

    utilities   action -> state -> utility
                {"merge": {"correct": 10, "buggy": -80}, ...}

    tool        result -> state -> P(result | state)
                {"fail": {"correct": 0.05, "buggy": 0.6},
                 "pass": {"correct": 0.95, "buggy": 0.4}}

Reading a table: `utilities["merge"]["buggy"]` is U(merge, buggy), -80, and
`tool["fail"]["buggy"]` is P(fail | buggy), 0.6.

Every docstring below works one example by hand: the coding agent from
`agent.py` with a belief of P(correct) = 0.5 and P(buggy) = 0.5. Use it to
check each function as you write it, for example in `uv run python`:

    >>> import sys; sys.path.insert(0, "src")
    >>> from decisions import expected_utility
    >>> from agent import UTILITIES
    >>> expected_utility({"correct": 0.5, "buggy": 0.5}, UTILITIES, "merge")
    -35.0

None of these functions knows anything about code or merging. They work
for any states, any actions and any tool, which the tests check.
"""

Belief = dict[str, float]
Utilities = dict[str, dict[str, float]]
Tool = dict[str, dict[str, float]]


def expected_utility(
    belief: Belief, utilities: Utilities, action: str
) -> float:
    """Return the expected utility of one action under a belief.

    The formula, with a sum over every state s:

        EU(action) = sum of P(s) * U(action, s)

    In code: start a total at 0, loop over the states in `belief`, and add
    `belief[state] * utilities[action][state]` each time.

    Worked example, P(correct) = 0.5 and P(buggy) = 0.5:

        EU(merge)   = 0.5 * 10  + 0.5 * -80 = -35.0
        EU(review)  = 0.5 * 6   + 0.5 * -5  =   0.5
        EU(discard) = 0.5 * -20 + 0.5 * 5   =  -7.5

    Args:
        belief: The probability of each state.
        utilities: The utility of each action in each state.
        action: The action to score.

    Returns:
        The sum over states of P(state) times U(action, state).
    """
    total = 0.0
    for state in belief:
        total += belief[state] * utilities[action][state]
    return total


def best_action(belief: Belief, utilities: Utilities) -> tuple[str, float]:
    """Return the action with the highest expected utility, and that value.

    In code: loop over the actions in `utilities`, compute each one's
    expected utility with the function above, and keep the largest seen so
    far. If two actions tie, keep the one that comes first, so replace the
    best only when a value is strictly larger.

    Worked example, P(buggy) = 0.5: the three values are -35.0, 0.5 and
    -7.5, so the answer is ("review", 0.5).

    Args:
        belief: The probability of each state.
        utilities: The utility of each action in each state.

    Returns:
        A pair: the best action's name and its expected utility.
    """
    best = ""
    best_value = float("-inf")
    for action in utilities:
        value = expected_utility(belief, utilities, action)
        if value > best_value:
            best = action
            best_value = value
    return best, best_value


def value_of_information(
    belief: Belief, utilities: Utilities, tool: Tool
) -> float:
    """Return how much expected utility the agent gains by using the tool.

    The formula, with a sum over every result r the tool can give:

        VOI = sum of P(r) * EU(best action after seeing r)
              - EU(best action now)

    In code, for each result in `tool`:

        1. P(r) is `result_probability(belief, tool[result])`. If it is 0,
           skip this result.
        2. Update the belief with `posterior(belief, tool[result])`.
        3. Find the best action under that new belief; keep its value.
        4. Add P(r) times that value to a running total.

    After the loop, subtract the value of the best action under the current
    belief.

    Worked example, P(buggy) = 0.5, the tests as the tool:

        fail: P(fail) = 0.325, belief becomes buggy 0.923,
              best action discard, EU 3.077
        pass: P(pass) = 0.675, belief becomes buggy 0.296,
              best action review, EU 2.741
        after looking: 0.325 * 3.077 + 0.675 * 2.741 = 2.85
        acting now: review, EU 0.5
        VOI = 2.85 - 0.5 = 2.35

    Args:
        belief: The probability of each state now.
        utilities: The utility of each action in each state.
        tool: P(result | state) for every result the tool can give.

    Returns:
        Expected utility after looking, minus expected utility now.
    """
    after_looking = 0.0
    for result in tool:
        likelihood = tool[result]
        p_result = result_probability(belief, likelihood)
        if p_result == 0:
            continue
        new_belief = posterior(belief, likelihood)
        action, value = best_action(new_belief, utilities)
        after_looking += p_result * value
    action, value_now = best_action(belief, utilities)
    return after_looking - value_now


def result_probability(belief: Belief, likelihood: dict[str, float]) -> float:
    """Return how likely a tool result is under the current belief.

    Provided. The law of total probability: P(r) = sum of P(r | s) * P(s).
    With P(buggy) = 0.5, P(fail) = 0.5 * 0.05 + 0.5 * 0.6 = 0.325.

    Args:
        belief: The probability of each state.
        likelihood: P(result | state) for one result, such as
            `tool["fail"]`.

    Returns:
        P(result).
    """
    total = 0.0
    for state in belief:
        total += likelihood[state] * belief[state]
    return total


def posterior(belief: Belief, likelihood: dict[str, float]) -> Belief:
    """Return the belief after one tool result, by Bayes' rule.

    Provided. P(s | r) = P(r | s) * P(s) / P(r), for every state s. With
    P(buggy) = 0.5 and the tests failing: buggy 0.3 / 0.325 = 0.923,
    correct 0.025 / 0.325 = 0.077. Returns a new dictionary; `belief` is
    not changed.

    Args:
        belief: The probability of each state before the result.
        likelihood: P(result | state) for each state.

    Returns:
        A new belief, P(state | result) for each state.
    """
    total = result_probability(belief, likelihood)
    after = {}
    for state in belief:
        after[state] = likelihood[state] * belief[state] / total
    return after
