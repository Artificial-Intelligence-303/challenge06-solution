"""Choosing an action under uncertainty, and deciding when to look first.

**You implement four functions in this file**: `expected_utility`,
`best_action`, `posterior` and `value_of_information`. Everything else is
provided.

Three kinds of table, all plain dictionaries:

    belief      state -> probability
                {"correct": 0.8, "buggy": 0.2}

    utilities   action -> state -> utility
                {"merge": {"correct": 10, "buggy": -80}, ...}

    tool        result -> state -> P(result | state)
                {"fail": {"correct": 0.05, "buggy": 0.6},
                 "pass": {"correct": 0.95, "buggy": 0.4}}

None of the four functions knows anything about code or merging. They work
for any states, any actions and any tool, which the tests check.
"""

Belief = dict[str, float]
Utilities = dict[str, dict[str, float]]
Tool = dict[str, dict[str, float]]


def expected_utility(
    belief: Belief, utilities: Utilities, action: str
) -> float:
    """Return the expected utility of one action under a belief.

    Each state's utility for this action, weighted by how likely that state
    is, added up.

    Args:
        belief: The probability of each state.
        utilities: The utility of each action in each state.
        action: The action to score.

    Returns:
        The sum over states of P(state) times U(action, state).
    """
    return sum(p * utilities[action][state] for state, p in belief.items())


def best_action(belief: Belief, utilities: Utilities) -> tuple[str, float]:
    """Return the action with the highest expected utility, and that value.

    If two actions tie, return the one that comes first in `utilities`.

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
            best, best_value = action, value
    return best, best_value


def posterior(belief: Belief, likelihood: dict[str, float]) -> Belief:
    """Return the belief after one observation, by Bayes' rule.

    Multiply each state's prior by the likelihood of the observation in that
    state, then divide every product by their sum. Do not change `belief`.

    Args:
        belief: The probability of each state before the observation.
        likelihood: P(observation | state) for each state.

    Returns:
        A new belief, P(state | observation) for each state.
    """
    products = {state: p * likelihood[state] for state, p in belief.items()}
    total = sum(products.values())
    return {state: product / total for state, product in products.items()}


def value_of_information(
    belief: Belief, utilities: Utilities, tool: Tool
) -> float:
    """Return how much expected utility the agent gains by using the tool.

    For each result the tool could give: how likely that result is, and the
    expected utility of the best action once the agent has seen it. Add
    those up, weighted by the result's probability, then subtract the
    expected utility of the best action taken now, without the tool. Skip a
    result that has probability 0.

    Args:
        belief: The probability of each state now.
        utilities: The utility of each action in each state.
        tool: P(result | state) for every result the tool can give.

    Returns:
        Expected utility after looking, minus expected utility now.
    """
    after = 0.0
    for likelihood in tool.values():
        p_result = sum(p * likelihood[state] for state, p in belief.items())
        if p_result == 0:
            continue
        after += (
            p_result * best_action(posterior(belief, likelihood), utilities)[1]
        )
    return after - best_action(belief, utilities)[1]
