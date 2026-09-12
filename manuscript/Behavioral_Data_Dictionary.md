# Behavioral Data Dictionary (`behavioral_compilate.csv`)

This document provides a comprehensive, field-by-field breakdown of the raw behavioral dataset located at `data/behavioral_compilate.csv`. It contains all necessary information to compute reaction times, inter-trial intervals, handle missing data, and safely feed the dataset into cognitive models (e.g., reinforcement learning, drift-diffusion).

## Overview
- **Total Rows:** 16,029 trials
- **Total Participants:** 128 unique subjects
- **Missing Data:** The raw CSV has exactly **0 NAs** in all columns. Missingness only arises when deriving lagged variables (like ITI) or when filtering invalid responses.

---

## Raw Column Definitions

| Column | Type | Description |
| :--- | :--- | :--- |
| `participant_id` | String | Unique identifier for the subject (e.g., `ACMO_06011994`). |
| `nt` | Integer | Trial number. Starts at 1 and increments sequentially for each participant. Resets to 1 for a new participant. |
| `Bd1` | Integer | Identity of the first bandit/fractal presented (typically the "Left" or "Option 1" stimulus). Values range from 2 to 8. |
| `Bd2` | Integer | Identity of the second bandit/fractal presented. Values range from 2 to 8. |
| `Resp` | Integer | The participant's response. <br> `1` = Chose Bd1 <br> `2` = Chose Bd2 <br> `3` = Invalid/Timeout. |
| `F` | Integer | Feedback (Reward) received on the current trial. `1` = Win, `0` = Loss. |
| `prob` | Float | The generative reward probability associated with the state or choice (takes values `0.15`, `0.25`, `0.75`, `0.85`). |
| `buena` | Integer | Binary indicator of optimal choice. `1` = participant chose the bandit with the higher expected value, `0` = chose the sub-optimal bandit. |
| `ttp` | Float | Timestamp (in milliseconds) of when the stimulus (the two bandits) was **presented** on screen. |
| `ttr` | Float | Timestamp (in milliseconds) of the participant's **response**. |
| `ttF` | Float | Timestamp (in milliseconds) of when the **feedback** (reward/loss) was displayed. |

---

## Derived Computations

To properly use this data in a cognitive model, you must derive Reaction Time (RT) and Inter-Trial Interval (ITI) from the raw millisecond timestamps.

### 1. Reaction Time (RT)
RT is the time elapsed between stimulus presentation and response.
- **Formula:** `rt = (ttr - ttp) / 1000` (converts to seconds).
- **Distribution:** Median is ~0.816s. Most responses fall between 0.5s and 1.3s.

### 2. Inter-Trial Interval (ITI)
ITI is the time elapsed between the *previous* trial's feedback and the *current* trial's presentation. It is critical for modeling memory decay in reinforcement learning.
- **Formula:** `iti = (ttp - lag(ttF)) / 1000` (grouped by `participant_id`).
- **Distribution:** Median is ~3.58s. Most ITIs fall between 3.3s and 3.8s.

---

## Quirks, Pitfalls, and Best Practices

When writing preprocessing scripts (e.g., in R or Python) or Stan data blocks, pay close attention to the following quirks:

> [!WARNING]
> **The First-Trial ITI Trap**
> Because ITI relies on `lag(ttF)`, the first trial (`nt == 1`) for every participant will inherently evaluate to `NA`. If your Stan model multiplies decay by `iti[t]`, it will crash on trial 1. 
> **Fix:** Impute `NA` ITIs with the median (e.g., `3.5`) or safely catch them in Stan: `real clean_iti = (iti[t] < 0 || is_nan(iti[t])) ? 3.5 : iti[t];`

> [!WARNING]
> **Anticipatory and Invalid Responses**
> - `Resp == 3` occurs 20 times in the dataset. These are timeouts or invalid keypresses where the subject made no choice. 
> - The computed `rt` column contains negative values (min `-0.03`s) and impossibly fast times (e.g., `0.01`s), representing anticipatory button mashing.
> **Fix:** Always filter responses before passing them to a likelihood function. For example: `filter(Resp %in% c(1, 2) & rt > 0.1)`. In Stan, encode invalid trials as `-999` and wrap the likelihood in `if (ch > 0 && rt[t] > 0.1)`.

> [!TIP]
> **Zero-Indexing in Stan**
> If you intend to use `Bd1` and `Bd2` as array indices for Q-values (e.g., `Q[Bd1]`), note that their minimum value is `2`. You can safely use a `vector[8] Q` array. If you dynamically calculate the chosen option index, it would be `ch_idx = (Resp == 1) ? Bd1 : Bd2`.
