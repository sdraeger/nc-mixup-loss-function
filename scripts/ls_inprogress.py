from pathlib import Path

import fire


def ls_inprogress(path):
    if not isinstance(path, Path):
        path = Path(path)

    for file in path.iterdir():
        if file.is_dir() and file.stem.split("_")[-1].isnumeric():
            contains_500 = bool(
                {_ for _ in file.rglob("H_W_colors_class_epoch_500.pkl")}
            )
            highest_pkl = max(
                file.rglob("*.pkl"), key=lambda x: int(x.stem.split("_")[-1])
            )
            if not contains_500:
                yield f"{file.name} - {highest_pkl.stem.split('_')[-1]}"


if __name__ == "__main__":
    fire.Fire(ls_inprogress)
