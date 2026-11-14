# TIMBER

**T**ree **I**nstability **M**apping **B**y **E**levation & **R**eflectance.

A science-fair project: a consumer drone and free software (WebODM + TIMBER)
find which trees in a forest could fall on trails, roads and buildings.
*Elevation* = the drone's surface and ground models (DSM/DTM), used for tree
heights, slope and water flow. *Reflectance* = the photo colors (VARI/ExG),
used for the tree health check.

**Engineering goal.** Design an automated system that uses only a consumer drone and free software (WebODM + this app) to find which trees in a forest pose the greatest fall risk to trails, roads and buildings, accurately enough to direct a field inspection and faster than a manual walk-through survey.

> **Screening tool for a student science project. Not a substitute for a
> professional arborist assessment.**

## Download

**[Download the TIMBER installer for Windows](https://github.com/colinmcampbell1414-design/timber/releases/latest)**
(`TIMBER-Setup-<version>.exe` under *Assets*). Run it: it installs just for you,
no administrator password needed, and adds TIMBER to the Start menu. Windows
may say "Windows protected your PC" because the installer isn't signed: click
**More info → Run anyway**. Then put each WebODM survey (DTM, DSM and
orthophoto GeoTIFFs) in its own folder inside `Documents\TIMBER\data`, start
TIMBER, and pick the survey in the sidebar. Needs 64-bit Windows 10 or 11.

TIMBER grew out of the *Drone Terrain Hazard App* (the folder
`..\terrain-hazard-app`, which is kept unchanged with all 9 hazard tools). This
folder is TIMBER only: tree fall hazard first, slope stability second, and the
ground-model checks that show whether the results can be trusted.

## Run it

| How | Command / action |
|---|---|
| Desktop window (from this folder) | `.\.venv\Scripts\python timber.py` |
| Portable app (any Windows 10/11 PC) | double-click `dist\TIMBER\TIMBER.exe` |
| Installed app (future) | Start menu → TIMBER (see "Installer" below) |
| Browser tab (for development) | `.\.venv\Scripts\python -m streamlit run app.py` |

First-time setup of this folder (Python 3.11 or newer):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-build.txt
.\.venv\Scripts\python -m pytest
```

## The TIMBER window

The sidebar has the page menu and the **survey** picker. Pages:

| Page | What it is for |
|---|---|
| **Home** | What TIMBER does, the current survey at a glance (area, cell size, DSM/photo, pit repair), how to add surveys |
| **Tree hazard** | The main tool: every tree's hazard score, likely fall direction, fall exposure, features at risk, downed logs, field check, **storm check** and **design criteria** |
| **Slope stability** | Steep, wet, water-collecting ground that could slump (the mudslide score) |
| **Ground check** | Is the ground model believable? Overview (hillshade, heights), Slope, Water flow (the old pages 1, 2 and 4) |
| **Accuracy checker** | Ground model vs hand-measured check points: the best-case vertical error (the old page 3) |

On **Tree hazard** and **Slope stability**:
- **Settings** are in one box with tabs. *Basic* holds the few settings most
  people change (minimum tree height, wind, how classes are chosen); the other
  tabs hold the advanced ones. Nothing re-runs until you press **Run analysis**,
  so you can change several sliders without waiting after each one.
- The **maps are interactive**: scroll or drag a box to zoom, double-click to
  see the whole map, point at a tree for a summary, **click a tree** (or a row
  of the table) for its detail card: class, score, height, fall direction,
  reasons, what it could reach, latitude/longitude and a bar chart of its eight
  hazard factors. Low-hazard trees are small faded dots so the risky ones stand out.
- Choose a part of the result with the buttons above the map (Trees & hazard,
  Where trees could fall, Features at risk, Downed logs, Field check). Only
  that part is drawn, which keeps the page quick.
- **Prepare map PNG / overlay GeoTIFF** make the files only when asked (they
  are big), then show a save button. The PNG is drawn at full detail, 200 dpi.

## Speed

| Step | Before | First time now | After that (even after a restart) |
|---|---|---|---|
| Open the 555 m forest (25 cm) | 19 s, every time | 10 s | 0.3 s |
| Open the 900 m forest (50 cm) | 31 s, every time | 12 s | 0.3 s |
| Tree hazard on the forest | 14 s, after every slider move | 24 s (includes wetness and slope layers) | about 2 s |

How:
- **Faster photo reading.** WebODM orthophotos contain smaller "overview"
  copies. TIMBER reads the smallest copy that is still twice as sharp as the
  analysis grid instead of the full 5 cm photo (17 s → 1.2 s on the large forest).
- **Prepared surveys are saved** (`terrain/survey.py`): the DTM, DSM and photo
  on one grid plus the water flow, as one file per survey and settings. A
  "fingerprint" of the files' names, sizes and change times and the settings
  decides whether the saved copy still fits.
- **Analysis results are saved** too (`Layers.remember` in
  `terrain/apps/base.py`), per survey, settings and marked targets.
- **Settings in a form** and **one result part at a time** (see above).
- Maps send a picture of at most 1600 pixels to the window instead of
  millions of cells.

Saved copies live in `.timber_cache\` (portable app: `dist\TIMBER\cache\`;
installed app: `%LOCALAPPDATA%\TIMBER\cache`). Deleting them is safe; TIMBER
works them out again. Only the 8 newest surveys and 12 newest results are kept.

## Desktop app (TIMBER.exe)

```powershell
.\build_exe.ps1            # the app only
.\build_exe.ps1 -WithData  # also copies every survey from data\ next to the exe
```

The result is the **portable** folder `dist\TIMBER\` (about 450 MB, plus the
surveys) with `TIMBER.exe` inside. Python and every library are packed in, so
it runs on any Windows 10/11 PC without installing anything. Copy the **whole
folder**. Surveys go in `dist\TIMBER\data\<survey name>\`. The file
`portable.txt` tells TIMBER to keep surveys, saved results and `timber.log`
in that folder.

Double-click `TIMBER.exe`: a window shows "Starting TIMBER..." for a few
seconds, then the app. Downloads go to your Downloads folder. Closing the
window stops TIMBER. If it doesn't start, the reason is in `timber.log`.

How it works: `timber.py` starts the app's engine (Streamlit) as a hidden
background process that only this computer can reach (address 127.0.0.1), and
shows it in a desktop window using Microsoft Edge WebView2, which is built into
Windows 10/11. The build recipe is `TIMBER.spec` (PyInstaller). After changing
the code, run `.\build_exe.ps1` again.

## Installer (ready for later)

`installer\TIMBER.iss` is an installer recipe for **Inno Setup 6** (free). To
make `dist\installer\TIMBER-Setup-1.1.0.exe`:

```powershell
winget install JRSoftware.InnoSetup   # once
.\build_installer.ps1
```

The result is `dist\installer\TIMBER-Setup-1.1.0.exe` (about 125 MB; no
surveys inside). By default it installs **just for the current user**, so no
administrator password is needed (handy on school laptops), into
`%LOCALAPPDATA%\Programs\TIMBER`; an administrator can choose "all users"
(Program Files) instead. It adds TIMBER to the Start menu (and optionally the
desktop) with an uninstaller in Settings → Apps.

An installed TIMBER keeps each user's surveys in `Documents\TIMBER\data` (made
the first time TIMBER starts; the Home page has an "Open the data folder"
button) and saved results and its log in `%LOCALAPPDATA%\TIMBER`
(`timber.py`, `user_folders`). Uninstalling never deletes surveys. To release
a new version, run `.\build_exe.ps1`, change `AppVersion` in
`installer\TIMBER.iss` (never `AppId`), then run `.\build_installer.ps1`.

## Validation (Phase 6)

Tools that test whether the hazard score means anything. Math:
`terrain/validation.py` and `terrain/fusion.py`. Screens:
`timber_ui/validation_panel.py`. Every tool saves a results file in `results/`
holding its measured design criteria and a plain sentence for the summary.

**Baselines.** Next to the hazard score, every tree gets three simple
rankings: `baseline_height` (tallest first), `baseline_height_target` (tallest
first among trees that can reach a trail, road or building) and
`baseline_random` (a random order with a seed, default 42). A judge will ask
"is the 8-factor score better than something simple?"; these answer it.

**Weight sensitivity** (Tree hazard → Trees & hazard → *Sensitivity*). Do the
results depend on the weights I chose?
- One at a time: each weight × 0.5 and × 1.5 (the rest rescaled to add up to
  1); the Spearman rank correlation with the original scores and how many of
  the original top 20 stay in the top 20.
- 500 random weight sets from a Dirichlet distribution centered on the current
  weights (concentration 20). For each tree: how often it lands in the top 20.
  `weight_stability_top20` = the mean share of the original top 20 kept.
- All 500 sets are scored with one matrix product (trees × factors times
  factors × sets), which takes a fraction of a second on the forest.

**Season** (sidebar → Flight notes → Leaves; saved in the survey's
`survey.json`). Autumn color makes healthy trees look stressed, so with
*autumn color* the health weight starts at 0.05, a banner says so, and
"possibly dead" uses only the snag rule (among the least green 5% of crowns
and a small crown for its height). With *leaf-off* the health factor and
"possibly dead" are switched off and the other weights are rescaled.

**Leaf-on / leaf-off fusion** (sidebar → *Ground model (DTM) source*, and
Ground check → *DTM comparison*). Photogrammetry can't see the ground under
leaves. A ground model from a leaf-off survey of the same place (another
folder in `data/`) or from public lidar (a GeoTIFF in `data/outside_dtm/`) can
replace the survey's own: it is reprojected onto the survey's grid (bilinear),
feet are converted to meters (read from the file, or chosen by hand; a US
survey foot is 1200/3937 m), and it is lined up vertically by fitting a plane
on stable open ground (canopy under 0.5 m with both models, smooth, slope under
10°). The comparison shows every ground model against a reference, split into
open ground and under canopy, the canopy heights each produces, hillshades side
by side, and (after a field check) the tree-height RMSE for every ground model.
That table is the before/after result of the leaf-off flight.

### Field workflow

1. **Export** (Tree hazard → Field check → *Export the blind field sheet*).
   N highest-scoring trees and N random trees from the lowest third of scores
   (default 30 + 30), all taller than 10 m and optionally within 30 m of a
   trail, shuffled, each with a random 4-digit code. Saved in
   `results/field_check/`:
   - `field_sheet.csv` (take along): code, latitude, longitude, distance and
     bearing from the nearest trail point, and blank columns. **No score, group,
     height, reasons or tree number**, so you can't be influenced.
   - `field_booklet.html` (print): a 40 × 40 m photo of each tree with its crown
     outlined, a north arrow and a 10 m scale bar.
   - `field_key.csv` (stays home): which code is which tree.
2. **In the woods**, for each code fill in: `status` (found / not_found /
   inaccessible), `observed_height_m` (clinometer), and y or n for
   `dead_or_dying_crown`, `broken_or_dead_top`, `trunk_crack_or_cavity`,
   `root_damage_or_heaving`, `lean_over_15deg`, `hanging_dead_branches`,
   `undercut_bank`; plus `notes`.
3. **Check plots.** Click 3-5 plot centers (or upload plot_id, latitude,
   longitude, radius_m), save `plot_sheet.csv` and its cards, and count the
   trees taller than 10 m inside each circle (`field_count_over_10m`).
4. **Import** both filled sheets on the same page. Reported and saved in
   `results/field_check_results.json` / `.csv` and `results/detection_plots.json`:
   - defect rate in each group, their ratio, and Fisher's exact test with its 2×2 table;
   - AUC (the chance a defective tree outranks a sound one) for the full score,
     each baseline and each single factor;
   - tree height bias, MAE and RMSE with a 1:1 scatter plot, split into
     open/edge and interior trees;
   - the "possibly dead" flag against what was seen (2×2 table);
   - detection rate per plot and overall (sum of app counts / sum of field
     counts; count-based, so a missed tree and an extra tree cancel).
5. **Downed logs** (Tree hazard → Downed logs). Logs within 5 m of trails and
   roads are left out (trail crews cut them). Slope, wetness, edge exposure and
   canopy height around each log are compared with 1000 random forest points
   (Mann-Whitney U). Log directions are tested against the downhill direction
   with **axial statistics**: the angle between log and slope is doubled (so 30°
   and 210° count as the same axis), the mean resultant length is worked out,
   and a Rayleigh test gives the chance of that much agreement by luck. Logs
   you measured yourself can be uploaded (latitude, longitude, axis_azimuth_deg,
   length_m) and get the same tests.
6. **Timing** (Design criteria → *Survey speed*): type in your flying,
   processing and field hours. Speed = hectares / (your hours + computer time).

### Design criteria

`config/design_criteria.yaml` (created on first use; edit the targets there).
The **Design criteria** part of the Tree hazard page shows each with its
target, measured value, PASS / FAIL / NO DATA YET and the file it came from.

| Criterion | Target | Measured from |
|---|---|---|
| `detection_rate` | ≥ 0.80 | check plots, trees over 10 m |
| `height_rmse_m` | ≤ 1.5 | field sheet heights |
| `defect_rate_ratio` | ≥ 2.0 | field check: defect rate high / low |
| `fisher_p` | ≤ 0.05 | field check |
| `beats_baseline` | true | field check: full score AUC above every baseline |
| `log_factor_test_p` | ≤ 0.05 | downed-log factor test (smallest p × number of factors) |
| `hectares_per_hour` | ≥ 10 | your hours + compute time |
| `storm_auc_severe` | ≥ 0.70 | storm check, pooled |
| `storm_beats_baseline` | true | storm AUC: full score above "tallest first" |
| `fall_direction_mae_deg` | ≤ 45 | storm check, actual-wind model |
| `calm_false_change_rate` | ≤ 0.01 | calm pairs |
| `weight_stability_top20` | ≥ 0.75 | sensitivity |

Also reported, as background only: the best-case open-ground vertical error
from the Accuracy checker (yard survey).

**Export results for board** writes `results/summary.md` (the goal, the
criteria table and a plain-sentence result from every tool) and every chart as
a 300 dpi PNG in `results/figures/`.

### Audit

```powershell
.\.venv\Scripts\python scripts\audit.py
```

re-checks the whole project: every module imports, the required files exist,
the yaml has the 12 criteria, the seven parts of the tree page exist, pytest
passes, the full chain runs on a small made-up forest and every step writes its
file, and (if `data/forest` exists) the tree analysis and the sensitivity run
are fast enough. It prints one table and `AUDIT PASSED` or the failing IDs.
The requirement list with evidence is `docs/requirements_checklist.md`.

## Storm check (Phase 7)

**Why it matters.** A field check asks whether high-hazard trees *look*
defective. The storm check asks whether they actually *fail*. The same forest
is flown before and after a real wind storm, and because the scores are
**frozen before the storm**, this is a true prediction, not a fit made after
the damage was known. It is the strongest evidence in the project.

It is the **Storm check** part of the Tree hazard page (next to Field check),
with a **Design criteria** part beside it. Math: `terrain/storm.py` and
`terrain/validation.py`. Screens: `timber_ui/storm_panel.py`.

### Workflow for one storm

1. **Fly the before survey** and put it in its own folder,
   `data/forest_YYYY-MM-DD`. Fill in *Flight notes* in the sidebar (date,
   time, leaves, altitude, overlap); they are saved as `survey.json` in the folder.
2. **Add the storm** (Storm check → *Add or edit a storm*): a name and the
   before survey. Storms are kept in `config/storms.yaml`.
3. **Freeze predictions** with the before survey chosen in the sidebar. This
   writes, in `results/storms/<storm>/`:
   - `predictions_frozen.csv`: every tree's position, height, crown, all
     factors, hazard score, priority, baseline ranks, predicted fall direction,
     chance of landing on each feature and the chance of falling in each of
     36 directions;
   - `settings.json`: every slider value and the code version;
   - `frozen_grids.npz`: crown outlines, exposure map and feature masks;
   - `freeze_log.jsonl`: the time and SHA-256 fingerprints of those files.
   All later analysis only **reads** these files. If one is edited, the
   fingerprint no longer matches and the analysis refuses to run.
   *Tip: commit and push `results/storms/<storm>/` to GitHub right after
   freezing. GitHub's timestamp is independent proof of when the predictions existed.*
4. **After the storm**, fly again as soon as it is safe, put the survey in
   `data/forest_YYYY-MM-DD`, and set it as the storm's after survey.
5. **Wind data.** Download hourly readings for the nearest airport station
   (ASOS; for example from the Iowa Environmental Mesonet's ASOS download page)
   as CSV and upload it in the storm editor with the storm's start and end.
   The app works out the peak gust, the mean wind and the **gust-weighted wind
   direction**: each reading is an arrow pointing in its direction with length
   gust², the arrows are added, and the direction of the sum is the answer
   (so 350° and 10° average to 0°, not 180°).
6. **Make the actual-wind copy** (in the freeze box). It recomputes only the
   fall directions with the storm's real wind, from the frozen file, and is
   labeled "actual-wind prediction". The first copy ("prevailing-wind
   prediction") is the true forecast; both are reported.
7. **Run storm analysis.**

### What the analysis does

**Alignment.** Two drone models never line up exactly, so before anything is compared:
- *Sideways:* phase correlation on the small-scale shape of the surface models
  (heights, not colors, so lighting doesn't matter) finds the shift to 1/20 of
  a cell. On big maps the shift is then measured in 100 m blocks; if it changes
  across the map (the models differ in scale or rotation), a stretch-and-rotate
  (affine) fit corrects it.
- *Height:* on stable open ground (low **and smooth** in both surveys, slope
  under 10°, not a building) a plane is fitted to the difference and removed
  (offset + tilt).
- *Bend:* models made without ground control points also bend by different
  amounts. The typical (median) height difference within 40 m is removed. A
  median ignores the minority of cells that really changed.
- Both canopy heights use the **before** ground model: the ground doesn't move
  in a storm, and each flight's ground model has its own errors under trees.
- A warning appears if the leftover noise on stable ground is above 0.15 m or
  the leftover shift above 0.5 m, and every tree is then "uncertain" unless you
  choose to trust the alignment.

**Change per tree**, inside each frozen crown: `top_drop_m` (old top height −
highest after canopy within 1.5 m), `crown_loss_frac` (share of crown cells
that dropped more than the change threshold, default 3 × the stable-ground
noise, at least 0.5 m) and `volume_loss_frac`. Classes, all with sliders:

| Class | Rule |
|---|---|
| fallen | top dropped > 60% of height **and** crown lost > 70% |
| snapped top | top dropped > max(2 m, 20% of height) |
| crown damage | crown lost > 25% |
| no change | otherwise |
| uncertain | near the map edge, missing after data, or poor alignment (left out of every score) |

**New logs.** The downed-log detector runs on both surveys; a log with no
match before (center within 2 m, direction within 20°) is new. A new log with
one end within 1.5 × the crown radius of a fallen tree's trunk belongs to that
tree, and that end is the **root end**, so the fall direction is fully known
(root → top), not just an axis.

**Leaf-drop guard.** If the two surveys have different leaf conditions, or the
typical tree lost over 10% of its crown, a banner says the change looks like
leaf drop. *Leaf-drop mode* then classifies with the top drop and new logs only.

**Calm control (null test).** Add a pair of flights with no storm between them
as a storm with *Calm pair* ticked. Every change it finds is a false alarm, so
it measures the method's noise. Storm results show their change rate next to
the calm rate with Fisher's exact test.

**Did we predict it?** Using only frozen scores, with uncertain trees left out:
- **AUC** (the chance a damaged tree had a higher frozen score than an
  undamaged one; 0.5 = guessing) for the full score, three baselines (tallest
  first; tallest that can reach a target; random order) and each factor alone,
  with 95% bootstrap ranges (2000 resamples).
- Damage rate by hazard class, the high/low ratio and Fisher's exact test.
- How many damaged trees were in the top 10% and 20% of the frozen ranking.
- Logistic regression (Newton-Raphson, written out in `validation.py`): the
  odds of severe damage per 10 hazard points, with log(peak gust) when storms
  with different gusts are pooled.
- With fewer than 10 damaged trees a warning says to read counts, not p-values.

**Fall direction** (fallen trees with a linked log): the angle between the
observed direction and each model's most likely direction, for both frozen
copies and for "downhill only", "lean only", "wind only" and a uniform guess
(90° on average); the log-likelihood of the observed directions under the
36-direction forecast vs a uniform one (sum of ln(chance given to the true
direction); higher is better); and a rose diagram with the storm wind marked.

**Landing zone and exposure:** whether each log ended within the tree's reach
and inside the most likely landing strip; the frozen fall exposure under new
logs vs at random forest points (Mann-Whitney U); and whether a new log landed
on a detected trail, road, stream or building that the tree had been flagged for.

**Field confirmation** (blind): every fallen or snapped tree (up to 30) mixed
with as many random "no change" trees, 4-digit codes, and a printable booklet
of *before* photo crops. The sheet has only: status, storm_damage (none /
branches / top broken / fallen / uprooted), fall_azimuth_deg, notes. Uploading
it reports the change detection's precision and recall per class.

Results: `results/storms/<storm>/storm_results.json`, `storm_trees.csv`,
`new_logs.csv`, `fall_directions.csv`, and all storms pooled in
`results/storms/pooled_results.json`.

The four storm criteria (`storm_auc_severe`, `storm_beats_baseline`,
`fall_direction_mae_deg`, `calm_false_change_rate`) are in the Design criteria
table above and show NO DATA YET until a storm or calm pair has been analyzed.

### Flight checklist

- Same flight plan, altitude, overlap, camera angle and exposure settings for
  every survey of a site; save the plan file in the survey folder.
- Fly before a storm as close to it as is safely possible (ideally 1-3 days),
  and after as soon as winds are within the drone's limits and it is safe.
  **Never fly in or near storm conditions.**
- Similar time of day (shadows change the photo and the surface model).
- At least one calm-pair repeat flight per season as the null test.
- Note the leaf condition (on / turning / off) in the flight notes every flight.
- Check that the after survey fully covers the before area; damage at the map
  edge is "uncertain".

**Why the same flight plan matters (measured):** the two existing surveys
`forest` and `forest-large` were flown with different plans. Lined up by this
code they differ by a 2.8 m shift, a 1.5% scale difference, about 1° of tilt
and several meters of bend, and even after correction 2 m of noise is left on
open ground, which would falsely flag about 11% of trees. Open ground agrees to
0.18 m at small scale, so the surveys themselves are fine; they just weren't
flown as a matched pair.

**Tests** (`tests/test_storm.py`, 23 tests): a surface shifted 1.3 m east,
0.4 m north, 2.1 m up and tilted 0.3° is recovered to 0.1 m and 3 cm; a shift
that grows across the map is fitted and a bad block is dropped; a removed tree
with a log pointing 120° is "fallen" with direction 120° (not 300°); a tree cut
by 40% is "snapped top"; a crown missing one side is "crown damage"; a calm
pair with 0.1 m noise gives no false changes; lowering every crown 30% trips
the leaf-drop guard; frozen predictions survive a weight change and an edited
file is detected; damaged = top-scoring trees gives AUC 1.0 and random damage
about 0.5; perfect directions give error 0 and beat the uniform likelihood;
350° and 10° average to 0°; and with no storms every criterion says "no data yet".

## Input data

From WebODM, use **Download Assets** and put each survey's GeoTIFFs in its own
subfolder of the data folder, so files from different flights are never mixed:

```
data/
  yard/          ...-dtm.tif  ...-dsm.tif  ...-orthophoto.tif   (+ check point CSVs)
  forest/        ...-dtm.tif  ...-dsm.tif  ...-orthophoto.tif
  forest-large/  ...
```

Pick the survey in the sidebar. Large maps (over 100 m wide) start at a coarser
cell size (10, 25 or 50 cm) so they load quickly; change it under **Survey
settings** in the sidebar, together with the pit repair and the tilt removal.

In WebODM, make sure the task options include **dsm** and **dtm** (and
**pc-classify** for forests), otherwise no elevation models are produced.

| File | What it is | Needed? |
|---|---|---|
| `*dtm.tif` | Digital Terrain Model: bare-ground heights | Yes |
| `*dsm.tif` | Digital Surface Model: ground plus trees, buildings | Yes, for trees |
| `*orthophoto.tif` | Map photo | Strongly recommended: health check, buildings, roads, map background |

Notes on this data:
- Rasters must be in a projected coordinate system measured in **meters** (WebODM
  uses UTM). Cell size is always read from the file.
- Empty areas (nodata) are ignored, never treated as height 0.
- **Absolute elevation is not trustworthy** (no ground control points; drone GPS
  error is about 1 m). The app only shows heights *relative to the lowest point*.

## Project layout

```
timber.py             desktop launcher: own window, picks the data/cache/log folders
app.py                sets up the pages (the page code is in timber_ui/)
timber_ui/            the user interface
  home.py analysis_page.py ground.py accuracy_page.py   the pages
  survey_panel.py     sidebar survey picker + survey settings
  maps.py             interactive maps (Plotly) and full-detail PNG maps
  widgets.py          tables, summary cards, tree detail card, click-to-mark editor
  style.py            TIMBER's look (with .streamlit/config.toml)
  paths.py            where surveys and saved results are
terrain/              all the math (beginner-readable, with tests)
  survey.py           prepare a survey and save/reuse it on disk; flight notes (survey.json)
  storm.py            storm check: freeze, align, detect change, score the predictions
  fusion.py           another ground model (leaf-off survey, lidar): reproject, feet to m, align, compare
  selfcheck.py        end-to-end run on a made-up forest
  validation.py       baselines, AUC, bootstrap, Fisher, logistic regression, design criteria
  io.py slope.py hydrology.py vegetation.py accuracy.py treefall.py
  apps/               trees.py (tree fall hazard), mudslide.py (slope stability), base.py
tests/                pytest tests using synthetic (made-up) elevation models
assets/               TIMBER icon
TIMBER.spec build_exe.ps1 requirements-build.txt    build TIMBER.exe
installer/ build_installer.ps1                      build the installer (Inno Setup)
data/                 your WebODM surveys (not committed to git)
scripts/audit.py      re-check the whole project (prints AUDIT PASSED or the failing IDs)
docs/                 the original specs (docs/prompts/) and docs/requirements_checklist.md
config/               storms.yaml and design_criteria.yaml (made by the app)
results/              frozen predictions and storm results (made by the app)
```

The sections below describe each analysis in detail. Page names from the
original app: "1. Load and view", "2. Slope" and "4. Water flow" are now tabs
of **Ground check**, "3. Accuracy checker" is **Accuracy checker**, and
"5. Hazards & Applications" is now the **Tree hazard** and **Slope stability**
pages.

## Phase 1: Load and view

**What it does:** loads a DTM, hides nodata cells, and optionally *resamples* it
to a coarser grid (default 5 cm). Resampling averages the original cells inside
each new cell, which smooths out grass and noise and speeds things up. It never
makes cells smaller than the original.

It shows:
- native and working cell size, grid size, map width/height, area with data
- the relief (difference between the highest and lowest point)
- a **hillshade**: a shaded-relief picture, as if the ground were lit by a low
  sun. You can change the sun direction, sun height and vertical exaggeration.
- a **relative height** map (0 m = lowest point)

**How to use it:** start the app, choose **1. Load and view** in the sidebar,
pick your DTM, and set the cell size.

**Repair ground-model pits** (sidebar, on by default). Under thick canopy
WebODM sometimes can't see the ground, and its DTM drops into deep "pits"
tens of meters below the real ground. On the large forest these pits made
15 m trees look 55-67 m tall and turned pit walls into fake cliffs. A cell
counts as a pit if it is more than 10 m (slider) below the *typical ground*:
the median height within 30 m, which one deep hole can't pull down. Pit cells
and a 1 m rim around them are replaced by that typical height. Real valleys
are wider than 30 m, so the typical ground follows them and they are left
alone. The sidebar says how much of the map was repaired (about 0.9% of
`forest`, 1.2% of `forest-large`). Every page uses the repaired DTM except
the accuracy checker, which always reads the original file.

**Getting a better ground model under trees.** The repair only fixes obvious
pits. Under dense leaves the DTM also follows the treetops in many places,
which shows up as bumpy, blocky "hills" in the hillshade and makes slopes
read far too steep. Better input fixes this better than any code: fly when
the leaves are off (late fall to early spring), fly lower with more image
overlap, and in WebODM use *pc-classify* plus a higher *pc-quality*.

## Phase 2: Slope

**What it does:** calculates the slope (steepness) of every cell in degrees.

How slope is calculated (`terrain/slope.py`):
1. For each cell, compare the heights of the neighbors on the east and west
   sides, and on the north and south sides ("central differences"). Divide by
   the distance between them to get the rise per meter in each direction.
2. Combine both directions: `rise = sqrt(east_rise² + north_rise²)`.
3. Convert to an angle: `slope = arctan(rise)`.

Slope is grouped into five classes. Defaults: Flat (0–5°), Gentle (5–15°),
Moderate (15–25°), Steep (25–35°), Very steep (above 35°). These are common
screening ranges, not an official standard. **All boundaries are adjustable
with sliders.**

The page shows a slope class map over the hillshade, a continuous slope map,
a table of area and percent per class, and a histogram.

**Tip:** slope depends on cell size. Small cells pick up bumps from grass and
noise; larger cells show the overall shape of the land. Try 5, 10 and 25 cm.

**Tests** (`tests/test_slope.py`): a tilted plane built at a known angle (1°,
10°, 30°, 45°, 60°) must give exactly that slope everywhere, including with
non-square cells; a flat plane must give 0° slope. There are also tests for
aspect, hillshade, nodata handling, slope classes, and loading/resampling
(`tests/test_io.py`).

## Phase 3: Accuracy checker

**What it does:** compares the drone model's heights with heights measured by
hand at check points, to see how accurate the model is.

**Entering check points** (two ways):
1. **Load a CSV** with columns `id, x, y, measured_z`. Put it in `data/` or upload it.
   `x, y` are map coordinates of each marker center (same coordinate system as
   the DTM, e.g. UTM meters). `measured_z` is the hand-measured height relative
   to your reference point, in meters.
2. **Grid helper** (no need to look up every marker's coordinates):
   - **A.** In WebODM, read the coordinates of just **two** markers: the first
     marker (column 1, row 1) and the last marker in that row. Enter them as map
     x/y (meters) or latitude/longitude; lat/lon is converted automatically.
     Enter columns × rows (e.g. 17 × 6 = 102), row spacing, which side the
     other rows are on, and how the markers are numbered (row by row or zig-zag).
     A zoomed preview shows a red circle where the app thinks every marker is.
     If `*orthophoto*.tif` is in `data/`, the photo is shown, so you can check
     each circle sits on a golf ball or rock.
   - **B.** Type or paste your hand-measured heights **in marker order**, in the
     units you wrote them in (meters, cm, mm, inches or feet). Write `x` for a
     marker you couldn't measure. If your numbers are distances measured *down*
     from a level line (bigger number = lower ground), tick the box and the app
     flips them so higher ground has a bigger height. The number of values must
     match the number of markers.
   - **C.** Save the result to `data/` as a CSV (it keeps your original numbers
     and units too), then use **Load a CSV** next time.

**How it works** (`terrain/accuracy.py`):
1. Read the model height at each point with **bilinear interpolation**: a
   weighted average of the 4 nearest cells. It uses the original full-detail DTM.
2. `difference = model height − measured height` for each point.
3. The model's absolute height is shifted by an unknown amount (no ground
   control points; GPS error ~1 m), and the measured heights are relative to a
   reference point. So the **mean offset** (average difference) is subtracted:
   `error = difference − mean offset`. The offset is reported on its own.
4. Statistics: mean error (0 by design), standard deviation, RMSE, maximum
   absolute error, LE90 (90% of errors are smaller than this), number of points.
   Shown in meters and inches.

If a DSM is in `data/`, it is checked too, so you can compare DTM and DSM.

**Charts:** a map of errors at each point (red = model too high, blue = too
low), a histogram of errors, and error vs local slope with a trend line and R².

**Subset comparison:** picks an evenly spread subset of points (default 30)
and computes the same statistics, to see whether fewer check points would
give the same answer. Points are picked by "farthest point sampling": start
with one point, then keep adding the point farthest from those already chosen.

**Export:** download every point's heights, differences and errors as a CSV.

**Tests** (`tests/test_accuracy.py`): bilinear interpolation is exact on a
tilted plane; a model shifted by a constant 1.234 m gives zero error after
mean alignment; points with known errors (±3 cm, ±4 cm) give exactly the
expected RMSE; plus tests for the grid helper, CSV reading and the subset.

## Phase 4: Water flow

**What it does:** predicts where rain water flows and where it collects in
puddles (ponding). It uses the working grid (cell size from the sidebar,
default 5 cm).

How it works (`terrain/hydrology.py`, written with numpy instead of pysheds so
every step can be explained; pysheds also crashes with current numpy):
1. **Fill depressions** ("priority-flood"): imagine water rising from the
   edges of the map. The lowest unvisited cell is always handled next, and
   any neighbor lower than it is raised to its level. Every dip is filled to
   the height where it would overflow.
2. **Ponding depth** = filled surface − original ground. Cells deeper than the
   **minimum ponding depth** (slider, default 1 cm) are ponding. Touching
   ponding cells form one pond. The table lists each pond's area, maximum and
   average depth, water held (liters) and location. Ponds smaller than the
   minimum area (default 0.05 m²) are hidden, because tiny ones are usually
   noise from grass.
3. **Flow direction (D8)**: each cell sends its water to the one of its 8
   neighbors that is steepest downhill (drop ÷ distance; diagonal neighbors
   are farther away). On the flat top of a filled pond, water heads toward
   the spill point.
4. **Flow accumulation**: every cell gets one cell's worth of rain and passes
   everything it collects downhill. The result is the area (m²) of ground
   draining through each cell. Cells above the **flow path threshold**
   (slider) are drawn as lines; thicker lines carry more water. The slider
   starts at the smallest value that keeps flow paths to about 2% of the map
   (2 m² on the 5 cm yard, 100 m² on the 25 cm forest). A fixed 1 m² would
   mark a third of the forest as "flow paths" and draw over a million lines.

**Big maps:** the step-by-step version above takes minutes on a forest map
with millions of cells, so above 1 million cells the app switches to a fast
version that does the same three steps with whole-array operations (filling
by "morphological reconstruction", routing flats ring by ring, and passing
water downhill in batches). A test checks that both give the same answer.

Water reaching the map edge or a no-data area is assumed to flow away. Soil
soaking up water is ignored, so ponds are worst-case puddles in heavy rain.

**Remove best-fit tilt** (off by default): fits the flat plane closest to the
whole surface and subtracts its tilt before the analysis. Use it to compare:
without ground control points a whole model can be slightly tilted, but this
also removes real slope. On the yard data the fitted tilt is
about 14° down toward the north, which matches the real slope of the yard,
so with tilt removed the yard turns into one big fake "pond".

The background is the orthophoto if one is in `data/`, otherwise the hillshade.

**Tests** (`tests/test_hydrology.py`): a cone-shaped bowl fills to exactly its
rim depth at the center and forms one pond of the expected area; in a
V-shaped valley water collects along the valley line, increases downstream,
and all of it leaves at the bottom; a flat plane runs without errors and no
water is lost; no-data areas act as outlets; tilt removal recovers a known
5° tilt; and a tilted bowl that spills out holds its full depth again once
the tilt is removed.

## Phase 5: Hazards & Applications

**What it does (in TIMBER):** two pages, **Tree hazard** (the focus) and
**Slope stability**. (The other seven tools of the original app, erosion,
drainage, irrigation, rain garden, winter ice, trip hazards and volume, are
not part of TIMBER; they are still in the original `terrain-hazard-app`
folder.) Each page shows: a short description, the settings (all numbers are
*starting values — adjustable*), summary cards, interactive maps with a
legend, tables, downloads (overlay as GeoTIFF, map as PNG, table as CSV), and
a "How to test this at home" box.

> Screening tool for a student science project. Not a substitute for a
> professional assessment.

Survey settings (sidebar): analysis cell size, pit repair and **remove
best-fit tilt** (off by default). Above each map: photo or shaded-relief
background, and overlay strength.

The tree tool can optionally use extra targets you mark on the map (a shed, a
play area...). Open the "Mark on map" box and click on the map, or upload a CSV
with columns `x, y` (map coordinates) and optional `shape` (a number grouping
rows into one outline) and `label`.

| Tool | Needs | What it does |
|---|---|---|
| Mudslide / slope stability | DTM (+ photo) | Weighted score of slope, wetness (TWI), concave curvature and bare soil (low ExG) → low / moderate / high. Each factor only counts past a start point (slope 15° → full at 35°, wettest and most concave 25% of the map), so ordinary ground scores 0: a dry 35° slope is moderate, steep *and* wet or water-collecting ground is high. (Before, wetness and curvature were plain ranks, so 78% of the forest came out moderate or high; now 36%.) |
| Tree fall hazard | DTM + DSM (+ photo) | Works automatically on any dataset: hazard score for every tree, most likely fall direction, fall exposure map, auto-detected trails/roads/buildings/streams/clearings, downed logs, and a field-check sheet (see below) |

**Code structure:** each tool is a module in `terrain/apps/` with the same
parts (`NAME`, `DESCRIPTION`, `REQUIRED_INPUTS`, `SHAPES`, `TEST_AT_HOME`,
`params(st)`, `run(layers, params) -> Result`). The list in
`terrain/apps/__init__.py` builds the dropdown. Shared math is reused from
`terrain/slope.py` (slope, aspect, plan curvature, hillshade),
`terrain/hydrology.py` (fill, flow, ponding, TWI) and
`terrain/vegetation.py` (VARI, ExG). The tree math is in `terrain/treefall.py`.
Do not add new hazard tools to TIMBER; the original app is the place for those.

**Tests** (`tests/test_apps.py`, `tests/test_shared_helpers.py`): every tool
has the shared interface and missing inputs give a message instead of a crash;
a steep slope is moderate or high while flat ground is low; and two cone
"trees" (5 m and 8 m) are measured within 5% and only the 8 m tree reaches a
target 6 m away.

### Tree fall hazard (works automatically, e.g. on `data/forest`)

No clicking is needed; marking extra targets on the map is optional. The math
is in `terrain/treefall.py`; `terrain/apps/trees.py` only draws the sliders
and the tabs. On the 555 m forest at 25 cm (about 5 million cells, ~1,900
trees) the analysis takes about 10 seconds after the first load.

**Sliders** are grouped in five expanders: *Tree detection*, *Hazard weights*,
*When each factor starts counting*, *Fall direction*, *Features*. Maps wider than 100 m start with forest values
(minimum tree height 5 m, canopy smoothing 0.5 m, crown edge 2 m).

**1. Finding trees.** Canopy height = DSM − DTM. Tree tops are local maxima
within a search window that grows with height (window = 1.5 m + 0.1 × height,
"variable window"), because big trees have wide crowns. Crowns are outlined by
watershed segmentation. Detected building roofs are never counted as trees.

**2. Hazard score (0-100) for every tree** = 100 × weighted average of eight
factors, each 0-1 and shown in the table. Every factor has a **start point**
and a **full point** (sliders): it is 0 until the start point and rises in a
straight line to 1 at the full point. An ordinary tree is below every start
point and scores about 0, so only trees with something concerning are flagged.
(An earlier version gave every tree partial credit on every factor, so a
normal 20 m tree already scored about 45 and 93% of the forest came out
moderate or high.)

| Factor | 0 until (start) | 1 at (full) | Why |
|---|---|---|---|
| height | 15 m | 30 m | reaches farther, hits harder |
| slope | 15° | 35° (mean under the crown, ground smoothed over 2 m first) | uproots more easily |
| wetness | 75th percentile | the wettest tree in the map (TWI rank) | wet soil holds roots poorly |
| exposure | sheltered | crown touches a **real opening**, or 3 → 10 m taller than neighbors within 15 m | catches more wind |
| slenderness | height/crown width 3 | 7 | tall thin trees are less stable |
| health | 1 "typical spread" less green | 3 spreads less green than other crowns (VARI + ExG) | stressed or dead |
| lean | crown center 0.5 radius off the top | a full radius off | heavy side / lean (low confidence) |
| stream bank | - | within one tree height of a stream draining 5,000 m² AND slope > 15° | undercut bank |

A **real opening** is smooth, open ground (the same test as clearings) at
least 5 m wide and 100 m² big, and the exposure falls from 1 at the opening to
0 at 10 m inside the forest. The narrow spaces between neighboring crowns
don't count: they all join into one big network, so an area limit alone
would make almost every tree an "edge tree". The width test works like this:
shrink the gaps by 2.5 m (narrow slivers vanish), then grow what is left back
by 2.5 m.

Default weights: height 0.20, slope 0.15, wetness 0.10, exposure 0.15,
slenderness 0.10, health 0.20, lean 0.05, stream bank 0.05 (rescaled to add
up to 1). **Classes**, two choices:
- *by hazard score* (default): low < 35, moderate 35-60, high > 60 (sliders).
  A safe forest gets few flagged trees.
- *by percent of trees*: the riskiest 5% are high and the next 15% moderate
  (sliders). Always flags the same share, so it ranks trees within the map.

On the forest (25 cm) the default flags 9 high and 146 moderate of 1,874 trees
(was 96 and 1,639); by percent it flags 94 and 281. A table under the map shows
how many trees fall in each 10-point score band, to help set the limits. The
**reasons** column names up to 3 factors past their start points, biggest
first, e.g. "Tall (36 m); on steep 67° slope; 11 m taller than neighbors". "Possibly dead" = over
5 m tall, much less green than other crowns, and a small crown for its height.
This is a relative screening score inside one map, not an arborist assessment.

**3. Fall direction.** Chance of falling in each of 36 directions =
any direction (0.4) + downhill (0.3 × slope factor) + heavy side of the crown
(0.15 × lean factor) + with the wind (0.15; wind from 270° = west by default,
can be turned off). Each preference is a smooth "von Mises" bump (kappa 2).
A tree falling one way covers a strip from its trunk out to height × safety
factor, as wide as its crown. Arrows show the most likely direction of
moderate and high hazard trees.

**4. Fall exposure map** = for every spot, the sum over trees of
(hazard / 100) × (chance the tree's strip covers the spot). A second map
counts how many trees could reach each spot. Both download as GeoTIFF.

**5. Features at risk** (found automatically, each can be turned off):
- *Clearings*: open ground (canopy < 0.5 m **and** a smooth surface) bigger
  than 20 m². Only spots the photo actually saw count. In dense forest the
  drone often can't see the ground, so WebODM's "ground" follows the treetops
  and canopy height looks like 0; the bumpy crowns give those spots away.
- *Trails / roads*: open ground thinned to a center line (skeletonize); pieces
  1-8 m wide, longer than 15 m, at least 5× longer than wide and fairly
  straight. Gray, smooth ones are roads.
- *Buildings*: taller than 2 m, gray **and** bright in the photo (plants are
  colorful in every season, including red and yellow fall leaves), smooth,
  roof slope at most 45°, bigger than 15 m².
- *Streams*: flow accumulation above 500 m².
- *Other trees*: how many other tree tops each tree could hit (domino effect).
Priority = hazard × (1 + consequence), consequence = 1.0 building, 0.7 road,
0.5 trail, 0.3 stream, 0.1 per other tree (up to 0.5); all adjustable. The
table is sorted by priority, top 20 first.

**6. Validation.** *Downed logs*: long (> 4 m), thin (< 1 m), low (0.2-1.5 m)
objects, with their axis direction; a table compares slope, wetness, fall
exposure and nearby tree hazard at logs vs random spots. (Fences and hedges
have the same shape.) *Field check*: download a CSV of the top N priority
trees plus N random low-hazard trees with latitude/longitude, fill in
observed height, lean, dead/damaged and root damage on a phone, upload it,
and the app reports height error (mean, RMSE), how often "possibly dead"
matched, and whether high-hazard trees had more problems than low-hazard ones.

**What to try first on the forest:** pick the *forest* survey, page 5,
*Tree fall hazard*. Look at the *Features at risk* tab first (are buildings on
houses and clearings on real clearings?), then *Trees & hazard* (top 20 and
their reasons), then *Where trees could fall*.

**Tests** (`tests/test_treefall.py`): 10 m and 20 m cone trees are found with
heights within 5% and the 20 m tree has the higher height factor; a tree on a
30° slope has a higher slope factor and most likely falls within 20° of
downhill; a lopsided crown leans the right way; with wind off a single tree's
exposure is a disk of radius = height; a 2 m × 40 m strip is a trail but a
round clearing is not; in a planted forest of identical 15 m trees the middle
tree scores under 10 ("low") because the holes between crowns are too narrow
to be openings, while the trees next to a 20 × 20 m clearing become edge
trees; the percent classes give exactly 5 high and 15 moderate of 100 trees;
a gray 10 × 8 × 4 m box is a building but a cone tree
of the same height is not; a 6 m log's axis angle is within 10°; of two
identical trees, the one that can reach a building ranks first; and a
2000 × 2000 map with 1000 trees runs in well under 60 seconds.
