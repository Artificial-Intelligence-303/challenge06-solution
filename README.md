> **INSTRUCTOR COPY. DO NOT DISTRIBUTE.**
>
> This repository is the reference solution for Challenge 6. It carries the
> completed functions in `src/decisions.py`, a reference reflection in
> `docs/summary.md`, and the grading notes at the bottom of this file. The
> student repository is `challenge-06-expected-utility/`.

# Challenge 6: An Agent That Decides by Expected Utility

|                    |                                             |
| :----------------- | :------------------------------------------ |
| `Thursday 22 Oct`  | Released, in class                          |
| `Thursday 5 Nov`   | Due at 11:00am, the start of class          |
| `Tuesday 10 Nov`   | Verbal checks, in lab                       |
| Points             | 3                                           |

A coding agent has just written a change. It estimates how likely the
change is to be buggy, and then it has to do something: merge it, ask a
teammate to review it, or throw it away. Before acting it may run the
tests, which costs time, or a quick static checker. Whether running them
is worth it depends on whether their answer could change what the agent
does.

You write the four functions that make those decisions. The agent, its
utilities and its tools are provided.

## Course learning outcomes

1. **Outcome 1.** Correctly implement an intelligent agent and accurately
   describe its properties. You implement the decision rule of an agent
   that acts under uncertainty, and explain from its output when it looks
   before it acts.
2. **Outcome 5.** Evaluate intelligent systems while considering their
   social, political, and ethical implications. You change one of the
   agent's utilities and argue who that change protects and who it costs.

## The agent

| Piece        | In `src/agent.py`                                         |
| :----------- | :-------------------------------------------------------- |
| States       | the change is `correct` or `buggy`                        |
| Actions      | `merge`, `review`, `discard`                              |
| `UTILITIES`  | how much the team prefers each action in each state       |
| `TOOLS`      | `tests` and `lint`: P(result \| state) for each result    |
| `COSTS`      | what calling each tool costs, in the same units           |
| `CHANGES`    | five kinds of change and the agent's P(buggy) for each    |
| `next_step`  | calls the tool that gains the most over its cost, or acts |

Every number in `agent.py` is invented for this challenge.

## What you implement

Four functions in `src/decisions.py`. Everything else is provided. None of
them knows anything about code: each works for any states, any actions and
any tool, and the tests check that on a second problem.

### 1. `expected_utility(belief, utilities, action)`

The sum over states of P(state) times U(action, state).

### 2. `best_action(belief, utilities)`

The action with the highest expected utility, and that expected utility.
A tie goes to the action listed first.

### 3. `posterior(belief, likelihood)`

The belief after one observation, by Bayes' rule: multiply each state's
prior by the likelihood, then divide by the sum. Return a new dictionary.

### 4. `value_of_information(belief, utilities, tool)`

For each result the tool can give, its probability times the expected
utility of the best action after seeing it. Add those, then subtract the
expected utility of the best action now.

The docstring at the top of `decisions.py` shows the shape of every table.

### From the formula to code

Every formula here is a sum, and every sum becomes a loop: start a total at
0, loop over the states (or the results), and add one term each time.

| Formula | Python |
| :------ | :----- |
| P(s) | `belief[state]` |
| U(a, s) | `utilities[action][state]` |
| P(r \| s) | `tool[result][state]` |
| sum over states of ... | `for state in belief:` with `total += ...` |

Each function's docstring works one example by hand, with P(buggy) = 0.5.
Check each function against it as you go, before moving to the next: the
later functions call the earlier ones.

```
uv run python
>>> import sys; sys.path.insert(0, "src")
>>> from decisions import expected_utility
>>> from agent import UTILITIES
>>> expected_utility({"correct": 0.5, "buggy": 0.5}, UTILITIES, "merge")
-35.0
```

## Running it

```
uv run python src/main.py              every change, and the agent's next step
uv run python src/main.py risk 0.25    one change with P(buggy) = 0.25
uv run pytest                          run the tests
uv run ruff check src tests            check your style
uv run ruff format src tests           fix most style problems automatically
uv run gatorgrade                      run the checks the way they are graded
```

## Style

Same as last time: the [Google Python Style
Guide](https://google.github.io/styleguide/pyguide.html), checked by
`uv run ruff check src tests`.

## Evaluation

This challenge is worth **3 points**.

| Component                 | Value   |
| :------------------------ | :------ |
| Programming               | 1       |
| Written reflection        | 1       |
| Verbal check              | 1       |
| **Total**                 | **3**   |

### Programming, 1 point

Run by `gatorgrade`, awarded as the fraction of the **code** checks that
pass. These are:

- `expected_utility` and `best_action` give the right values and actions
  on the coding agent, on an umbrella problem, and with three states.
- `posterior` follows Bayes' rule, adds up to 1, and leaves the belief it
  was given unchanged.
- `value_of_information` gives the right values, is 0 when no answer can
  change the action, and is never negative.
- The agent merges a typo fix without testing and runs the tests on a
  refactor.
- Style passes `ruff` and type annotations check out under `mypy`.
- No `TODO` markers are left in `src/decisions.py`.

`gatorgrade` also runs two checks on `docs/summary.md`. Those belong to the
reflection below, not to this point.

Autograder results are preliminary. The final grade is determined by the
instructor.

### Written reflection, 1 point

Complete `docs/summary.md`. Two questions and the disclosure, **200 words
minimum across all of them**. You are graded on whether the reasoning is
correct and specific to what your run printed, not on length.

### Verbal check, 1 point

During lab you answer two or three questions about your own code, in the
context of this assignment. Nothing to prepare beyond understanding what
you wrote.

## AI use on this challenge

Full policy: **[AI in this course](https://areweagentsyet.com/ai/)**. AI
tools are allowed here, you disclose them, and you have to be able to
explain what you submit.

> ### The skill this time: ask what the tool is assuming about people
>
> Every utility in this agent is a judgment about people: how much a bug
> costs the users, how much a review costs a teammate. When a tool, or an
> agent, gives you an answer, ask whose costs it counted and whose it left
> out.

Concretely, for this assignment:

- **Write the four functions yourself.** They are this week's idea.
- **Check one value of information by hand.** For "rename a variable",
  work out P(buggy) after a failing test and after a passing one, pick the
  best action for each, and compare with what `main.py` prints.
- **A tool is useful** for the Python around the decisions: how to find
  the key with the largest value in a dictionary, or how to build a new
  dictionary from an old one.

## How this assignment was built

You are asked below to disclose your AI use, so here is mine.

I decided the shape: an agent that acts on a belief, chooses between
acting now and calling a tool by the value of information, and has its
utilities written where you can change them.

I used Claude to draft `agent.py`, the tests and `main.py`, and to compute
every number the tests expect. I ran the reference solution and the
unfinished starter against every check, and checked the checks against
wrong versions of all four functions, not only against empty ones.

## Disclosing your AI use

At the end of `docs/summary.md` there is a disclosure section. Answer three
things, a sentence or two each:

1. **Which tool.** Name it. If you used more than one, name each.
2. **What you used it for.** Which part of the assignment.
3. **What you did with what it gave you.** Accepted it, edited it, checked
   it against something, or threw it out.

**If you did not use any tool, write that.** That is a complete and
perfectly good answer.

Disclosing costs you nothing. There is no version of this where naming a
tool loses you points.

---

## Instructor notes

### What differs from the student repository

| File                 | Here                         | Student copy                    |
| :------------------- | :--------------------------- | :------------------------------ |
| `src/decisions.py`   | four functions implemented   | four `TODO` stubs               |
| `docs/summary.md`    | reference reflection         | questions with `TODO`           |
| `README.md`          | this banner and these notes  | no banner, no notes             |

Every other file is byte-identical. A fix to a test, to `agent.py` or to
`gatorgrade.yml` has to be copied to both.

### What a correct run prints

```
change                       P(buggy)  act now   VOI tests  VOI lint  next step
fix a typo in a comment          0.01  merge         0.252     0.000  act merge
rename a variable                0.04  merge         1.608     0.732  call tests
add a small feature              0.15  review        0.000     0.000  act review
refactor a module                0.40  review        1.620     0.000  call tests
rewrite the database layer       0.80  discard       1.740     0.160  call tests
```

Thresholds: merge below 4/79 ≈ 0.051, review up to 26/36 ≈ 0.722, discard
above. The small feature's tests are worth 0 because a failure (P(buggy)
0.679) and a pass (0.069) both leave the agent in the review range.

### Verification

- Solution: 25 of 25 `gatorgrade` checks.
- Starter: 20 of 20 tests fail; 2 of 25 checks pass (ruff and mypy, on
  purpose, so students start type-clean).
- Eight wrong versions, each caught by a check that names the mistake, none
  hanging (every run under a second):

| Wrong version                                   | Tests failed |
| :---------------------------------------------- | -----------: |
| Expected utility averages, ignoring the belief  | 11           |
| `best_action` keeps the last of a tie (`>=`)    | 1            |
| `best_action` picks the lowest                  | 10           |
| `posterior` does not normalize                  | 10           |
| `posterior` changes the belief in place         | 9            |
| VOI forgets to subtract acting now              | 6            |
| VOI keeps the action chosen before looking      | 4            |
| VOI does not weight by P(result)                | 5            |

### Verbal check bank

Hypotheticals, as in earlier challenges. No question asks the student to
walk through code. Every answer below was computed with the reference.

**Expected utility**

- *E1.* What is EU(merge) for a change that is certainly correct? *10, the
  utility itself; the sum has one term.*
- *E2.* As P(buggy) rises from 0 to 1, in what order are the actions best?
  *Merge, review, discard, switching near 0.05 and 0.72.*

**Posterior**

- *P1.* The tests fail on a rename (P(buggy) 0.04). Is P(buggy) now above
  or below one half? *Below: 0.333. A flaky failure is still more likely
  from a correct change, because correct changes are 24 times as common.*
- *P2.* If your `posterior` changed the belief it was given, which part of
  the agent would go wrong? *`value_of_information`: the second result
  would start from the first result's posterior.*

**Value of information**

- *V1.* Why are the tests worth exactly 0 on a small feature? *Pass or fail,
  the agent reviews; information that cannot change the action is worth
  nothing.*
- *V2.* If the tests cost 2 instead of 1, what does the agent do with a
  refactor? *Reviews without testing: the tests are worth 1.62.*
- *V3.* Could the value of information ever be negative? *No; the agent
  can always ignore the answer and do what it would have done.*

**Reading the agent**

- *A1.* What does the lint check add on a rename, and why does the agent
  still pick the tests? *0.732 minus 0.2 is 0.532; the tests give 1.608
  minus 1, which is 0.608.*
- *A2.* What did your changed utility in question 2 do, and to whom?
  *Open; listen for a specific number and a specific person.*

### Grading notes

- Reflection: half each question. Question 1 needs both values (1.608 and
  0.000) and the pass-or-fail argument; naming the threshold range is the
  full answer. Question 2 needs a changed number, its effect on the table,
  and a named person on each side.
- A reflection that says the small feature is "too risky to test" has the
  argument backwards: risk is not what makes information worthless.
