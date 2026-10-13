"""Train the car's pedestrian detector, show its tables, and ask it questions.

Usage:
    uv run python src/main.py                     tables, moments, accuracy
    uv run python src/main.py camera=yes lidar=no one moment, by its evidence
"""

import sys

from naive_bayes import (
    Evidence,
    Example,
    Model,
    accuracy,
    bayes_rule,
    classify,
    train,
)
from sensors import NEW_MOMENTS, load

WITHOUT_CAMERA2 = ("camera", "lidar", "radar")


def show_tables(model: Model) -> None:
    """Print the prior and every likelihood table."""
    cells = "  ".join(f"P({c}) = {p:.3f}" for c, p in model.prior.items())
    print("prior:  " + cells)
    for sensor, table in model.likelihood.items():
        for value, row in table.items():
            cells = "  ".join(f"{c} {p:.3f}" for c, p in row.items())
            print(f"  P({sensor} = {value} | truth):  {cells}")


def show(name: str, posterior: dict[str, float]) -> None:
    """Print one posterior on a line."""
    cells = "  ".join(f"{c} {p:.3f}" for c, p in posterior.items())
    print(f"  {name:<28} {cells}")


def count_present(examples: list[Example], evidence: Evidence) -> str:
    """Count, in the data itself, how often a pedestrian was there."""
    matching = [
        label
        for label, seen in examples
        if all(seen[s] == v for s, v in evidence.items())
    ]
    present = matching.count("present")
    return f"{present} of {len(matching)} = {present / len(matching):.3f}"


def mistakes(model: Model, examples: list[Example]) -> str:
    """Count both kinds of mistake the detector makes on the examples."""
    missed = false_alarms = 0
    for label, evidence in examples:
        posterior = classify(model, evidence)
        guess = max(posterior, key=lambda c: posterior[c])
        if label == "present" and guess == "absent":
            missed += 1
        if label == "absent" and guess == "present":
            false_alarms += 1
    return f"{missed} missed pedestrians, {false_alarms} false alarms"


def main() -> None:
    """Run the detector."""
    model = train(load("train"))
    if len(sys.argv) > 1:
        evidence = dict(arg.split("=", 1) for arg in sys.argv[1:])
        show(" ".join(sys.argv[1:]), classify(model, evidence))
        return

    print("== The tables, learned from data/train.csv ==")
    show_tables(model)

    print("\n== One sensor: Bayes' rule and naive Bayes agree ==")
    camera_yes = model.likelihood["camera"]["yes"]
    show("bayes_rule, camera yes", bayes_rule(model.prior, camera_yes))
    show("classify, camera yes", classify(model, {"camera": "yes"}))

    print("\n== New moments ==")
    for name, evidence in NEW_MOMENTS.items():
        show(name, classify(model, evidence))

    print("\n== The camera says yes: with and without camera2 ==")
    without = train(load("train", WITHOUT_CAMERA2))
    both = NEW_MOMENTS["camera yes, camera2 yes"]
    show("naive Bayes, both cameras", classify(model, both))
    show("naive Bayes, camera alone", classify(without, {"camera": "yes"}))
    train_data = load("train")
    counted_both = count_present(train_data, both)
    counted_one = count_present(train_data, {"camera": "yes"})
    print(f"  counted in the data, both cameras yes: {counted_both}")
    print(f"  counted in the data, camera yes:       {counted_one}")

    print("\n== On data/test.csv ==")
    test, test_without = load("test"), load("test", WITHOUT_CAMERA2)
    score, score_without = (
        accuracy(model, test),
        accuracy(without, test_without),
    )
    print(f"  all four sensors:  accuracy {score:.3f}, {mistakes(model, test)}")
    print(
        f"  without camera2:   accuracy {score_without:.3f}, "
        f"{mistakes(without, test_without)}"
    )


if __name__ == "__main__":
    main()
