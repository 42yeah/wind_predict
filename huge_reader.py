import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv("huge.csv", sep=";")
    draw = []
    for index, entry in df.iterrows():
        draw.append(entry["power"])
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(draw)
    plt.show()
    data = []
    for index, entry in df.iterrows():
        if entry["power"] == 0 or \
            entry["speed"] <= 0 or \
            entry["speed"] > 20 or \
            entry["direction"] <= 0 or \
            entry["temp"] <= 0 or \
            entry["humidity"] <= 0 or \
            entry["pressure"] <= 0:
            continue
        if entry["power_normalized"] <= 0:
            entry["power_normalized"] *= -1
        inputs = np.array([[entry["speed_normalized"]],
                           [entry["temp_normalized"]],
                           [entry["humidity_normalized"]],
                           [entry["pressure_normalized"]],
                           [entry["time_normalized"]]])
        output = np.array([[entry["power_normalized"]]])
        data.append((inputs, output))
    random.shuffle(data)

    return data[:-7000], data[-7000:]


if __name__ == "__main__":
    training_data, test_data = load_data()
    print(f"Training: #{len(training_data)}, test: #{len(test_data)}")
