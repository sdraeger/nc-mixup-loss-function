import os
import os.path as osp
import sys
from pathlib import Path
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import fire
import polars as pl
import dill


def dict_mean(dicts):
    # Assume all dicts have the same keys
    keys = dicts[0].keys()
    mean_dict = {}

    for key in keys:
        mean_dict[key] = sum([d[key] for d in dicts]) / len(dicts)

    return mean_dict


def group_by_seed(logdir):
    if not isinstance(logdir, Path):
        logdir = Path(logdir)

    ddict = {}
    for file in logdir.iterdir():
        if not file.is_dir():
            continue

        idx = file.stem.find("_seed_")
        key = file.stem[:idx]

        group = ddict.get(key, [])
        group.append(file.name)

        ddict[key] = group

    return ddict


def save_dict_mean(logdir, nseed=3):
    if not isinstance(logdir, Path):
        logdir = Path(logdir)

    grouped = group_by_seed(logdir)
    for common_name, dirs in grouped.items():
        if len(dirs) != nseed:
            print(f"Skipping {common_name} because it does not have {nseed} seeds")
            continue

        pkl_fnames = [f.name for f in (logdir / dirs[0]).rglob("*.pkl")]

        try:
            os.makedirs(logdir / common_name)
        except os.error:
            print(f"Directory {logdir / common_name} already exists, skipping")
            continue

        for pkl_fname in pkl_fnames:
            epoch = re.findall(r"\d+", pkl_fname)[-1]
            dicts = []

            for d in dirs:
                print(osp.join(logdir, d, pkl_fname))
                try:
                    with open(osp.join(logdir, d, pkl_fname), "rb") as f:
                        data = dill.load(f)
                        dicts.append(data)
                except:
                    pass

            mean_dict = dict_mean(dicts)
            with open(
                osp.join(
                    logdir, common_name, f"H_W_colors_class_epoch_{epoch}_mean.pkl"
                ),
                "wb",
            ) as f:
                dill.dump(mean_dict, f)


if __name__ == "__main__":
    fire.Fire(save_dict_mean)
