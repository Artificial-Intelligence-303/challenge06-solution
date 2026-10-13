> **INSTRUCTOR COPY. DO NOT DISTRIBUTE.**
>
> This repository is the reference solution for Challenge 6. It carries the
> completed functions in `src/naive_bayes.py`, a reference reflection in
> `docs/summary.md`, the data generator in `scripts/make_data.py`, and the
> grading notes at the bottom of this file. The student repository is
> `challenge-06-naive-bayes/`.

# Challenge 6: Is There a Pedestrian Ahead?

|                    |                                             |
| :----------------- | :------------------------------------------ |
| `Thursday 22 Oct`  | Released, in class                          |
| `Thursday 5 Nov`   | Due at 11:00am, the start of class          |
| `Tuesday 10 Nov`   | Verbal checks, in lab                       |
| Points             | 3                                           |

An autonomous car has four sensors pointed at the road ahead: a camera, a
lidar, a radar, and a second camera from the same supplier. Each says yes,
someone is there, or no. The car has to decide how likely it is that a
pedestrian is really there.

You build that decision with Bayes' rule and naive Bayes. The car learns its
probabilities by counting a record of past moments, then combines whatever
its sensors report. The data, the sensors and the questions are provided.

## Course learning outcomes

1. **Outcome 3.** Design, implement, and assess an intelligent system. You
   implement a naive Bayes classifier, train it on data, and measure how
   often it is right and how it goes wrong.
2. **Outcome 5.** Evaluate intelligent systems while considering their
   social, political, and ethical implications. You compare two versions of
   the car's detector by who each kind of mistake puts at risk.

## The data

`data/train.csv` holds 400 past moments and `data/test.csv` 200 more. Each
row says whether a pedestrian was really there, `present` or `absent`, and
what each sensor reported.

| Column    | Values             | What it is                                   |
| :-------- | :----------------- | :------------------------------------------- |
| `label`   | `present`, `absent`| whether a pedestrian was really there        |
| `camera`  | `yes`, `no`        | the main camera                              |
| `lidar`   | `yes`, `no`        | the laser scanner                            |
| `radar`   | `yes`, `no`        | the radar                                    |
| `camera2` | `yes`, `no`        | a second camera, same supplier, same software|

Every row is invented for this challenge.

## What you implement

Four functions in `src/naive_bayes.py`. Everything else is provided. None of
them knows anything about cars: each works for any classes and any features,
and the tests check that on examples from class.

### 1. `bayes_rule(prior, likelihood)`

The posterior after one observation: multiply each class's prior by the
likelihood, then divide by the sum.

### 2. `train(examples, smoothing=1.0)`

Count the examples. The prior is the share of examples in each class. Each
likelihood is the share of a class's examples with that value, with
`smoothing` added to every count so no value ever gets probability 0.

### 3. `classify(model, evidence)`

Naive Bayes: multiply the prior by one likelihood for each observed sensor,
then divide by the sum.

### 4. `accuracy(model, examples)`

The share of examples whose most likely class is their label.

The docstring at the top of `naive_bayes.py` shows the shape of every table.

## Running it

```
uv run python src/main.py                      tables, moments, accuracy
uv run python src/main.py camera=yes lidar=no  one moment, by its evidence
uv run pytest                                  run the tests
uv run ruff check src tests                    check your style
uv run ruff format src tests                   fix most style problems automatically
uv run gatorgrade                              run the checks the way they are graded
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

- `bayes_rule` gives the answers from class and leaves the prior unchanged.
- `train` counts the prior and the likelihoods, with and without smoothing,
  and never gives a value probability 0.
- `classify` multiplies one likelihood per observed sensor, skips a sensor
  it has no table for, adds up to 1, and gives the answers from class.
- `accuracy` gives the right share on a small example and on the test data.
- Style passes `ruff` and type annotations check out under `mypy`.
- No `TODO` markers are left in `src/naive_bayes.py`.

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
> The car's numbers decide who it protects. When a tool, or a model you
> build, gives you an answer, ask what it assumed and whose risk that
> assumption moves.

Concretely, for this assignment:

- **Write the four functions yourself.** They are this week's idea.
- **Check one number by hand.** P(present | camera yes) needs the prior and
  one row of the camera table. Compare it with what `main.py` prints.
- **A tool is useful** for the Python around the counting: how to read a
  CSV file, or how to find the key with the largest value in a dictionary.

## How this assignment was built

You are asked below to disclose your AI use, so here is mine.

I decided the shape: the car from Challenge 5, now with several sensors to
combine, a second camera that is not independent of the first, and four
functions for you to write.

I used Claude to draft the data generator, `sensors.py`, the tests and
`main.py`, and to compute every number the tests expect. I ran the reference
solution and the unfinished starter against every check, and checked the
checks against wrong versions of all four functions, not only against empty
ones.

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

| File                    | Here                         | Student copy             |
| :---------------------- | :--------------------------- | :----------------------- |
| `src/naive_bayes.py`    | four functions implemented   | four `TODO` stubs        |
| `docs/summary.md`       | reference reflection         | questions with `TODO`    |
| `README.md`             | this banner and these notes  | no banner, no notes      |
| `scripts/make_data.py`  | the data generator           | not shipped              |

Every other file is byte-identical, including `data/`. A fix to a test, to
`sensors.py` or to `gatorgrade.yml` has to be copied to both.

### The data

`scripts/make_data.py`, seed 6: P(present) 0.3; camera yes 0.85 / 0.10, lidar
0.90 / 0.05, radar 0.70 / 0.20 (present / absent); camera2 repeats the camera's
answer 90% of the time, otherwise draws fresh with the camera's rates. Training
data: 144 present, 256 absent. Rerunning the script reproduces the files
exactly; changing anything in it moves every number below and in the tests.

### What a correct run prints

| Question                                   | Answer                         |
| :----------------------------------------- | :----------------------------- |
| P(present), smoothed P(camera yes \| present) | 0.360, 123/146 = 0.842      |
| P(present \| camera yes), both functions    | 0.830                          |
| P(present \| camera yes, lidar no)          | 0.375                          |
| P(present \| camera yes, camera2 yes)       | 0.977 (counted: 121/144 = 0.840) |
| Accuracy, all four sensors                  | 0.900: 1 missed, 19 false alarms |
| Accuracy, without camera2                   | 0.935: 5 missed, 8 false alarms  |

### Verification

- Solution: 23 of 23 `gatorgrade` checks.
- Starter: 18 of 18 tests fail; 2 of 23 checks pass (ruff and mypy, on purpose).
- Ten wrong versions, each caught by a check that names the mistake, none
  hanging (every run under a second):

| Wrong version                                            | Tests failed |
| :------------------------------------------------------- | -----------: |
| `bayes_rule` does not divide by the sum                  | 3            |
| `bayes_rule` changes the prior in place                  | 4            |
| `train` ignores smoothing                                | 4            |
| `train` smooths the denominator by 1                     | 2            |
| `train` divides by all examples, not the class           | 5            |
| `classify` uses only the first feature                   | 5            |
| `classify` adds instead of multiplying                   | 6            |
| `classify` forgets the prior                             | 6            |
| `classify` does not normalize                            | 6            |
| `accuracy` picks the least likely class                  | 2            |

### Verbal check bank

Hypotheticals. No question asks the student to walk through code.

**Bayes' rule**

- *B1.* What does `bayes_rule` divide by, and what is that number in
  probability terms? *The sum of the products: P(evidence), by the law of
  total probability.*
- *B2.* If every class had the same likelihood, what would `bayes_rule`
  return? *The prior, unchanged; the likelihood cancels when you divide.*

**Training**

- *T1.* Without smoothing, what would `classify` say for a sensor value never
  seen with one class? *Probability 0 for that class, whatever every other
  sensor says.*
- *T2.* Where does P(camera yes | present) = 0.842 come from? *123 of 146: the
  122 present moments with camera yes plus 1, over 144 plus 2.*

**Classify**

- *C1.* Why does `classify` with only the camera give the same answer as
  `bayes_rule`? *One factor, nothing to combine, nothing assumed.*
- *C2.* What assumption lets `classify` multiply the camera's and the lidar's
  likelihoods? *Independence once you know whether a pedestrian is there.*

**Reading the car**

- *R1.* Why is 0.977 too confident? *camera2 repeats the camera, so the
  product counts one piece of evidence twice; the data itself says 0.840.*
- *R2.* camera2 lowers accuracy. Why might a team keep it anyway? *It
  misses 1 pedestrian instead of 5; listen for who carries each mistake.*

### Grading notes

- Reflection: half each question. Question 1 needs 0.830 with the reason the
  two agree, and 0.375 with the independence assumption named. Question 2
  needs the four numbers, the "counts the same evidence twice" reason, and a
  choice argued from both kinds of mistake, not from accuracy alone.
- An answer that calls camera2 "more evidence, so more confidence is right"
  has missed the point of the question.
