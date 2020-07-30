import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict
import re
import itertools
import os

model_path = Path("code/models/trained_models/minisV10")

results = []
for folder in model_path.glob("*"):
    if not os.path.isdir(folder):
        continue
    eval_folder = folder / "evaluation_results"
    with open(eval_folder / "metrics.json") as js:
        metric_results = json.load(js)
    with open(folder / "train_config.json") as js:
        config = json.load(js)
    pattern = re.search("V[1-9]", config["model"])
    label = "base"
    if "lstm" in config["model"]:
        label += " + LSTM"
    if "gru" in config["model"]:
        label += " + GRU"
    if pattern:
        label += pattern.group(0)

    metric_results["Label"] = label
    metric_results["Model"] = str(config["model"]) + "_" + str(config["track_ID"])
    metric_results["model_class"] = "mobile" if "mobile" in str(config["model"]) else "resnet"
    results.append(metric_results)

data = defaultdict(list)
for category in results[0].keys():
    for model in results:
        data[category].append(model[category])

# print(data)
df = pd.DataFrame.from_dict(data)
df["time_in_ms"] = df["time_taken"] * 1000

mobile_df = df[df["model_class"] == "mobile"]
resnet_df = df[df["model_class"] != "mobile"]

categorys = ["Jaccard", "Overall_acc", "per_class_acc", "dice"]  # , "time_in_ms"
plots = []

# f, ax = plt.subplots(1, 2, figsize=(30, 10))
# ax[0].bar(mobile_df["Label"], mobile_df["Jaccard"])
# ax[1].bar(mobile_df["Label"], mobile_df["time_in_ms"])
fontdict = {'fontsize': 30,
              'fontweight': 1,
              'verticalalignment': 'baseline',
              'horizontalalignment': "center"}
tmp = [(0, 0), (0, 1), (1, 0), (1, 1)]
for name, df in [("mobile", mobile_df), ("resnet", resnet_df)]:
    f, ax = plt.subplots(2, 2, figsize=(50, 20))
    for pos, category in zip(tmp, categorys):
        ax[pos].bar(df["Label"], df[category])
        ax[pos].axhline(y=float(df[category][df["Label"] == "base"]), xmin=-0, xmax=1, color="r")
        ax[pos].set_title(category, fontdict=fontdict)
        plt.savefig(model_path / str("Eval_summary_" + name + ".png"))
# plt.show()
