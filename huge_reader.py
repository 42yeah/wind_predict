import pandas as pd
import numpy as np
import random


def load_data():
    df = pd.read_csv("huge.csv", sep=";")
    data = []
    for index, entry in df.iterrows():
        if entry["power_normalized"] <= 0:
            entry["power_normalized"] = 0
        if entry["power"] == 0 or \
            entry["speed"] <= 0 or \
            entry["speed"] > 20 or \
            entry["direction"] <= 0 or \
            entry["temp"] <= 0 or \
            entry["humidity"] <= 0 or \
            entry["pressure"] <= 0: # or \
            # index == 0:
            continue

        tod = entry["nope"].split(" ")
        tod = [0, 0] if len(tod) == 1 else [int(x) for x in tod[1].split(":")]
        tod_norm = (tod[0] * 60 + tod[1]) / (24 * 60)

        inputs = np.array([[entry["speed_normalized"]],
                           [entry["temp_normalized"]],
                           [entry["humidity_normalized"]],
                           [entry["pressure_normalized"]],
                           [tod_norm],
                           # [df.iloc[index - 1]["power_normalized"]]])
                           ])
        output = np.array([[entry["power_normalized"]]])
        data.append((inputs, output))
    test_data = data[-7000:]
    training_data = data[:-7000]
    random.shuffle(training_data)

    return training_data, test_data


if __name__ == "__main__":
    training_data, test_data = load_data()
    print(f"Training: #{len(training_data)}, test: #{len(test_data)}")
