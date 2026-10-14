# Challenge 6 Reflection

Answer both questions. **200 words minimum across the document.**
You are graded on whether the reasoning is correct and specific to what
your own run printed. Write each answer in place of the placeholder under
its question.

Run `uv run python src/main.py` before you start writing. Everything these
questions ask about is in its output, or one `risk` away.

---

> **1. Riskier is not always more worth testing.**
>
> Your run's table shows the agent running the tests on "rename a
> variable" but not on "add a small feature", which is more likely to be
> buggy. Give the value of information of the tests for each.
>
> Then explain the difference using the lines your run printed under each
> change: what the agent does if the tests pass, and what it does if they
> fail.

For "rename a variable" (P(buggy) = 0.04) the tests are worth 1.608, more
than their cost of 1, so the agent runs them. For "add a small feature"
(0.15) they are worth 0.000.

Acting now, the rename gets merged (expected utility 6.4). If the tests
fail, P(buggy) becomes 0.333 and the agent asks for a review instead; if
they pass, it becomes 0.017 and it merges. So a failure changes the
action, and that change is what the tests are worth. The small feature
gets a review whatever happens: a failure takes P(buggy) to 0.679 and a
pass takes it to 0.069, and both are inside the range where review is the
best action (0.051 to 0.722). An answer that cannot change the action is
worth nothing, however risky the change is. The tests matter most for
changes near a threshold, where one result tips the decision.

---

> **2. Who sets the utilities.**
>
> Change one number in `UTILITIES` in `src/agent.py` to what you think a
> real team should use, and run again. Say which number you changed, from
> what to what, and what changed in the table.
>
> Then say who your change protects and who it costs: the users of the
> software, the teammate who reviews, or the person who wrote the change.
> Put the number back before you submit.

I changed U(merge, buggy) from -80 to -200, for software where a bug that
reaches users is serious. The rename is now reviewed rather than merged,
and the tests drop to 0.528 for it, below their cost, so the agent skips
them and goes straight to review. The typo fix is still merged without
testing, but the tests are now worth 0.972 there, almost their cost.

The change protects the users: fewer bugs reach them, because more
changes go to review. It costs the teammate who reviews, who now reviews
every rename, and it slows down the author. Who chooses -200 decides how
that time is spent, and the users who carry the cost of a bug are not in
the room when it is chosen.

---

> **AI tool disclosure**
>
> State what AI tools you used on this challenge, if any, and what they
> did. "None" is a complete answer if it is true. Be specific: "I used X to
> explain what expected utility means" and "I used X to write
> `value_of_information`" are very different disclosures.

Reference reflection written by the instructor with Claude; the numbers
come from running the reference solution.
