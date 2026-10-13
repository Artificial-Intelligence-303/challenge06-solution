"""An autonomous car deciding whether a pedestrian is ahead.

Provided: you do not need to change this file.

At each moment a pedestrian is present ahead of the car or absent. Four
sensors each report yes, they detect someone, or no:

    camera    the main camera
    lidar     the laser scanner
    radar     the radar
    camera2   a second camera from the same supplier, running the same software

`data/train.csv` is the record the car learns from: moments where someone
later checked whether a pedestrian was really there. `data/test.csv` is held
back to measure how often the car is right. Every moment in both files is
invented for this challenge.
"""

import csv
import pathlib

from naive_bayes import Example

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
SENSORS = ("camera", "lidar", "radar", "camera2")


def load(name: str, sensors: tuple[str, ...] = SENSORS) -> list[Example]:
    """Return the moments in data/<name>.csv as (label, evidence) pairs.

    Args:
        name: "train" or "test".
        sensors: Which sensors to keep in the evidence.

    Returns:
        One (label, evidence) pair per moment.
    """
    with open(DATA / f"{name}.csv", newline="") as handle:
        return [
            (row["label"], {s: row[s] for s in sensors})
            for row in csv.DictReader(handle)
        ]


# Moments the car is asked about in main.py.
NEW_MOMENTS = {
    "camera yes": {"camera": "yes"},
    "camera yes, lidar no": {"camera": "yes", "lidar": "no"},
    "radar yes only": {"camera": "no", "lidar": "no", "radar": "yes"},
    "camera yes, camera2 yes": {"camera": "yes", "camera2": "yes"},
    "every sensor yes": {
        "camera": "yes",
        "lidar": "yes",
        "radar": "yes",
        "camera2": "yes",
    },
}
