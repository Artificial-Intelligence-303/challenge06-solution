# Challenge 6 Reflection

Answer both questions. **200 words minimum across the document.**
You are graded on whether the reasoning is correct and specific to what
your own run printed. Write each answer in place of the placeholder under
its question.

Run `uv run python src/main.py` before you start writing. Everything these
questions ask about is in its output.

---

> **1. Where naive Bayes is Bayes' rule.**
>
> Your run prints P(present | camera yes) twice, once from `bayes_rule` and
> once from `classify`. Give the number, and explain why the two agree.
>
> Then give P(present | camera yes, lidar no) from your run. Name the
> assumption `classify` makes to combine the two sensors that `bayes_rule`
> never had to make.

Both print P(present | camera yes) = 0.830. They agree because with one
sensor naive Bayes does exactly what Bayes' rule does: multiply the prior
(0.36 present, 0.64 absent) by the camera's likelihoods (0.842 and 0.097),
then divide by the sum, which is P(camera yes) by the law of total
probability. There is nothing to combine, so nothing to assume.

With the camera saying yes and the lidar saying no, my run gives
P(present) = 0.375. To get it, `classify` multiplies P(camera yes | class)
by P(lidar no | class), which is only right if the two sensors are
independent once you know whether a pedestrian is there. That is the naive
Bayes assumption; Bayes' rule with one sensor never needed it.

---

> **2. A second camera.**
>
> With camera2 the detector is more confident a pedestrian is there when both
> cameras say yes. Give the two naive Bayes numbers and the two numbers your
> run counted directly in the training data. Explain the difference using
> what camera2 is.
>
> Then give the accuracy and both kinds of mistake, with and without camera2.
> Which version would you put in a car, and why? Name who carries the cost of
> each kind of mistake.

With both cameras saying yes, naive Bayes gives 0.977 present; with the
camera alone it gives 0.830. Counted directly in the training data, a
pedestrian was there in 121 of 144 moments where both cameras said yes
(0.840), and in 122 of 146 where the camera said yes (0.836). So the second
camera adds almost nothing, but naive Bayes counts it as a second, separate
witness. camera2 runs the same software as the first camera and repeats its
answer most of the time, so the two are not independent, and multiplying
their likelihoods counts the same evidence twice.

With all four sensors the detector is right 0.900 of the time, with 1
missed pedestrian and 19 false alarms. Without camera2 it is right 0.935 of
the time, with 5 missed pedestrians and 8 false alarms. I would still keep
camera2 for a car near a school, because a missed pedestrian can kill
someone outside the car who never agreed to the risk, while a false alarm
costs the passengers a hard stop and maybe a rear-end bump. Someone has to
decide that trade; accuracy alone hides it.

---

> **AI tool disclosure**
>
> State what AI tools you used on this challenge, if any, and what they
> did. "None" is a complete answer if it is true. Be specific: "I used X to
> explain what smoothing does" and "I used X to write `classify`" are very
> different disclosures.

Reference reflection written by the instructor with Claude; every number
comes from running the reference solution.
