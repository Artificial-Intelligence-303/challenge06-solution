"""Generate the sensor history for Challenge 6. Instructor only.

Run once:  python3 scripts/make_data.py
Writes data/train.csv (400 moments) and data/test.csv (200 moments) from a
fixed seed, so every student gets the same data and the tests can check exact
numbers. Every rate below is invented.

Each moment, a pedestrian is present ahead of the car or absent. Four
sensors each report yes (they detect someone) or no:
  camera   the main camera
  lidar    the laser scanner
  radar    the radar
  camera2  a second camera from the same supplier, running the same software;
           it repeats the camera's answer 90% of the time, so it is NOT
           independent of the camera once the truth is known
"""

import csv
import pathlib
import random

RNG = random.Random(6)
P_PRESENT = 0.3
RATE = {  # P(yes | truth)
    "camera": {"present": 0.85, "absent": 0.10},
    "lidar": {"present": 0.90, "absent": 0.05},
    "radar": {"present": 0.70, "absent": 0.20},
}
REPEAT = 0.9


def moment():
    label = "present" if RNG.random() < P_PRESENT else "absent"
    row = {"label": label}
    for sensor, rate in RATE.items():
        row[sensor] = "yes" if RNG.random() < rate[label] else "no"
    if RNG.random() < REPEAT:
        row["camera2"] = row["camera"]
    else:
        row["camera2"] = "yes" if RNG.random() < RATE["camera"][label] else "no"
    return row


def write(path, n):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, ["label", "camera", "lidar", "radar", "camera2"])
        writer.writeheader()
        for _ in range(n):
            writer.writerow(moment())


here = pathlib.Path(__file__).resolve().parents[1] / "data"
write(here / "train.csv", 400)
write(here / "test.csv", 200)
print("wrote", here / "train.csv", "and", here / "test.csv")
