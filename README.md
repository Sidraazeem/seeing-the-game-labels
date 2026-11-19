
# Seeing the Game: Soccer Player Identification from Still Images

**CSC 481 final prototype** · Muzammil Ali Khan, Sidra Azeem

Given a still frame from a soccer broadcast, the system finds every person on the pitch, groups them into the two teams,
the goalkeepers and the referees, and reads each player's jersey number, or answers "unreadable" when it is not
confident. Everything is classical image processing (colour spaces, thresholding, morphology, connected components,
K-means, hand-made features). No pretrained model is used: every classifier is trained from scratch in the notebooks.

## Contents of the submission

| File / folder | What it is |
|---|---|
| `01_field_and_teams.ipynb` | Part 1: grass mask, field region, player boxes, team / referee / goalkeeper grouping, detection and team metrics (Muzammil) |
| `02_jersey_numbers.ipynb` | Part 2: jersey number recognition, all experiments and metrics, and both parts run together at the end (Sidra) |
| `labels/legible_labels.json` | Our hand labels: for 168 train images, "can a human read the number?" (used to evaluate per image) |
| `labels/legible_labels_b3.json` | Extra hand labels of dev images, used only to train the visibility classifier (B3) |
| `requirements.txt` | Python libraries |
| `README.md` | This file |

These folders are **created by the notebooks**. They are not part of the ZIP:

| Folder | Created by | Contents |
|---|---|---|
| `sampled_data/` | notebook 01 | 60 GSR frames from 6 matches + their `labels.json` |
| `jersey_data/` | notebook 02 | `zips/` (the two downloaded zips, 2.3 GB) and the 6 frames per player that are kept |
| `outputs/` | notebook 02 | synthetic and real training digits, trained classifiers and predictions (`cache/`), figures (`figures/`), result files (`*.json`) |

## Setup

Python 3.10 or newer. Install the libraries (inside a virtual environment if you like):

```bash
pip install -r requirements.txt
```

| Library | Used for |
|---|---|
| `opencv-python` | colour conversion, filtering, Otsu, morphology, connected components, contours, convex hull, watershed, K-means (notebook 01) |
| `numpy` | arrays, counting, FFT (Wiener filter) |
| `matplotlib` | all figures |
| `scikit-learn` | K-means in LAB, kNN, SVM, logistic regression, random forest, cross-validation, confusion matrix |
| `scikit-image` | HOG features |
| `pillow` | drawing the synthetic training players |
| `huggingface_hub` | downloading the SoccerNet data from HuggingFace |
| `torch` (optional) | the small CNN of notebook 02. If it is missing, the CNN is skipped and everything else runs |
| `ipywidgets` (optional) | the click tool for hand-labelling more images (`RUN_LABELLING = True`) |
| `jupyter` | running the notebooks |

Everything runs on a CPU. A GPU only speeds up the CNN. The notebooks also run in Google Colab (upload the files,
then **Run all**). No Google Drive or Colab-specific code is needed.

## Data

Both datasets are public parts of **SoccerNet**, downloaded automatically from HuggingFace, so there is nothing to
download by hand. (If HuggingFace ever asks for a login, create a free account and run `hf auth login` once.)

| Notebook | SoccerNet task | HuggingFace dataset | What is used |
|---|---|---|---|
| 01 | Game State Reconstruction 2024 | [`SoccerNet/SN-GSR-2024`](https://huggingface.co/datasets/SoccerNet/SN-GSR-2024), `valid.zip` | 6 random clips (seed 481), 10 evenly spaced frames each. Only those bytes are read from the 11 GB zip |
| 02 | Jersey Number Recognition 2023 | [`SoccerNet/SN-Jersey-2023`](https://huggingface.co/datasets/SoccerNet/SN-Jersey-2023), `train.zip` + `test.zip` | 6 evenly spaced frames per player: 1,427 train players (8,544 images), 1,211 test players (7,242 images) |

Our own data is the hand labels in `labels/` (included in the ZIP) and, to reproduce the exact numbers of the
presentation, the saved results folder (see below).

## How to run

Put both notebooks in the same folder (with `labels/` next to them) and run them in this order:

1. **`01_field_and_teams.ipynb`**: Run all. The first run reads its frames from HuggingFace and saves them in
   `sampled_data/`. Later runs use the saved frames and do not need the internet.
2. **`02_jersey_numbers.ipynb`**: Run all.
   * First run: it downloads `train.zip` and `test.zip` from HuggingFace (2.3 GB, a few minutes on a normal
     connection) and copies 6 frames per player into `jersey_data/` (seconds). Then it builds
     the training digits, trains the classifiers and scores every experiment. Everything slow is saved in `outputs/`.
     Computing everything from scratch can take an hour or more on a CPU.
   * Later runs load the saved results, so a full run takes a few minutes.
   * Its last section ("Both halves together") reads `sampled_data/` from notebook 01.
   * After the first full run, `jersey_data/zips/` (2.3 GB) can be deleted. It is only needed again if `jersey_data/`
     is deleted.

**Hand labels.** Notebook 02 needs `legible_labels.json` and `legible_labels_b3.json`. It looks for them in this
order: the `labels/` folder (included in this ZIP), then the notebook's own folder (for example files dragged into
the Colab "Files" panel), then `LABELS_URL` (the raw GitHub folder of our repository, e.g.
`https://raw.githubusercontent.com/<user>/<repo>/main/labels/`), which is downloaded automatically. If none of these
works, the notebook stops with a clear message instead of scoring 0 images.

Settings at the top of notebook 02: `SHOW_CHECKS` (show or hide the large check grids), `RUN_LABELLING` (click tool),
`N_SYNTH` (number of synthetic players).

## Evaluation

Both notebooks compute their metrics against the SoccerNet ground truth. The labels are used only to measure, never by
the pipeline.

* **Notebook 01:** detection recall and precision (IoU ≥ 0.3), recall by player height, team accuracy (K = 2), purity
  (K = 5), the effect of each hyperparameter, and the best and worst frames.
* **Notebook 02:**
  * player accuracy (6 frames vote; "unreadable" counts as correct for players without a number), precision when
    answering, visible accuracy, and abstain rate;
  * per-image precision and coverage on the hand-labelled readable images, and the τ trade-off curve;
  * accuracy by image size and sharpness, an ablation study (one component off at a time), the digit confusion
    matrix, and a pipeline walkthrough for successes and failures;
  * the improvements B1-B6, each chosen on validation players and scored once on test;
  * the end-to-end funnel (not detected, unreadable, wrong, correct) on full match frames.

  Thresholds and variants are always chosen on **val** and scored once on **test**. The cell
  "[PRESENTATION] Check against the slides" prints every number from the slides next to the value from the run.

## Reproducing the exact numbers of the presentation

The presentation numbers come from our saved results. To get exactly the same values, download our results folder
from **[LINK TO THE SHARED RESULTS FOLDER]**. It holds `cache/`, `synthetic_digits.npz` and `real_digits.npz`. Put
its contents in `outputs/`, so you have `outputs/cache/...`, `outputs/synthetic_digits.npz` and
`outputs/real_digits.npz`, then run notebook 02. The notebook loads the saved classifiers and predictions instead of
recomputing them, and the check cell shows `ok` for every row.

If you compute everything from scratch, the dataset statistics are the same, but the scores can differ slightly. The
synthetic training digits are drawn with the bold fonts installed on the computer (our run used Google Colab's
fonts), and CNN training is not bit-identical on every machine.

## Code structure (notebook 02)

The notebook follows the pipeline: Step 0 data → 1 usable images → 2 shirt → 3 number mask → 4 digits → 5 classifiers
→ 6 decision and evaluation, then the analyses (A1-A4), the improvements (B1-B6) and the joint run. Each step is
one or a few small functions with comments. Every design choice that an experiment turns on or off is a switch in the
`CFG` dictionary. Functions receive the settings as their `cfg` argument, and `settings(dict(...))` makes a copy with
some switches changed. `cached(name, function, ...)` saves slow results in `outputs/cache/` so they are computed only
once.

## What is ours and what is from libraries

The libraries provide the basic algorithms (colour conversion, Otsu, morphology, connected components, K-means, SVM,
HOG, watershed, the CNN layers). Our work is the design of both pipelines and how the steps are combined, all the
settings, the synthetic training players, the digit harvesting, the hand-made shape features, the confidence and
"unreadable" decision, the visibility classifier, the CNN architecture and its training, and all the experiments and
metrics.
