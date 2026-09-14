# GREENROOT — build prompt

Paste the section below into a fresh AI coding session working in this repo.

Everything in it is grounded in what already exists and what has already gone
wrong here. The bugs listed in §7 are real ones that were shipped and then
found — an agent that reads them will not repeat them.

---

## The prompt

> You are working on **GREENROOT**, a crop recommendation system for
> smallholder farmers in Karnataka, India. It already exists and works. Your
> job is to make it genuinely better for the people who use it — not to
> rebuild it, and not to add features to prove effort.
>
> ### 1. Hard constraints — read first
>
> **Never** delete, retrain, or overwrite:
> - `models/stacking_model.pkl`, `models/scaler.pkl`, `models/class_names.pkl`
> - anything in `data/`
>
> These are the evaluated artifacts. `train.py` overwrites them — do not run
> it. All inference must load these exact files. If you need to validate the
> model, use `verify_cv.py`, which refits in memory and asserts the pickles
> are byte-identical before exiting.
>
> ### 2. Who this is for
>
> **A farmer, primarily.** Often on a mid-range Android phone, outdoors, one
> hand, patchy signal, reading in a second language. They want three things:
> which crop, what to buy, when to do it. They do not want a model.
>
> **An examiner, secondarily.** Behind a single toggle
> (`Examiner / AI Mode`). Wants the evidence: Z-scores, SHAP/LIME agreement,
> sensitivity curves, the raw audit ledger with schema columns intact.
>
> One switch (`is_simple()`) drives both. Do not fork the codebase into two
> apps, and do not water down the examiner view to make it friendlier.
>
> ### 3. What exists — do not rebuild these
>
> ```
> src/models/inference.py      CropRecommender: validate → scale → rank → sweep
> src/models/xai_engine.py     TreeSHAP + LIME over a RandomForest surrogate,
>                              Jaccard at k=3 (threshold 0.50)
> src/models/batch.py          Bulk advisory over a whole soil survey
> src/services/soil_service.py District baselines, aliases, climate normals
> src/services/weather_service.py  OpenWeatherMap, 5s timeout, offline fallback
> src/database/db_manager.py   Thread-safe SQLite, WAL, migrations
> src/database/db.py           Dual-read façade: farmer_view / examiner_view
> src/utils/agronomy.py        22-crop metadata, 50kg bag converter, spray advice
> src/utils/agronomy_advisory.py   Nutrient/pH/water/thermal heuristics
> src/utils/seasons.py         Kharif / Rabi / Summer fit
> src/utils/plain_language.py  Farmer register, bags and acres
> src/core/config.py           Paths, feature contract, bounds, thresholds
> ```
>
> Read a module before changing it. Most of them carry a docstring explaining
> *why* they are the way they are; that reasoning is usually load-bearing.
>
> ### 4. The design rules that matter here
>
> **Remove before you add.** This app has repeatedly been made worse by
> addition. Five farmer tabs became two and it improved. If you are about to
> add a panel, first ask what it replaces.
>
> **The answer leads.** Controls collapse behind a one-line summary of what
> they currently say. On a phone the run button must be reachable without
> scrolling. Anything set once and rarely changed does not belong in the main
> flow.
>
> **Uncertainty must be visible in the design, not just readable in a
> caption.** A 43/100 match rendered in the same confident green capitals as a
> 95/100 match is the app lying with typography. Below 60, degrade the colour
> and say so above the crop name.
>
> **Never show a farmer a schema column name.** No `primary_shap_driver`, no
> `jaccard_index`, no ISO timestamps, no `90.855`. `db.farmer_view()` is the
> seam; keep everything going through it. Examiner mode keeps the raw row.
>
> **Plain language is not the same as farmer content.** Translating
> "explainer consensus" into simple words does not make it something a farmer
> needs. Cut it from their view instead.
>
> ### 5. Honesty rules — these are not negotiable
>
> - **Do not invent market data.** Mandi rates, yields and cultivation costs
>   in `agronomy.py` are indicative benchmarks, stamped with their basis and
>   editable in the UI. Never present them as live prices. `economics.py`
>   deliberately ships no defaults at all — preserve that.
> - **Do not claim corroboration you have not checked.** Sixteen of 22 Kannada
>   names are verified against the shipped survey's bilingual labels, and a
>   test re-derives that set from the CSV. The other six are flagged. If you
>   add translations, mark them as unreviewed.
> - **Measure, don't assume.** `train.py` scales before splitting, which is
>   leakage. The correct response was to quantify it (+0.000 pp), not to
>   assert it away.
> - When you flag a concern, say it once, then do the work.
>
> ### 6. Verification standard
>
> A change is not done until:
>
> 1. `pytest -q` is green (523 tests at time of writing, 1 env-conditional skip).
> 2. You have driven the real app in a browser at **390×844** and **1440×900**,
>    taken screenshots, and **looked at them**. Not "captured" — looked.
> 3. You have exercised **both** the weak-confidence and strong-confidence
>    paths. Udupi returns a ~43/100 match; Dharwad returns ~60/100. Several
>    bugs here appeared on only one of the two.
> 4. You have checked both personas after any layout change.
> 5. `md5sum` confirms the model artifacts are unchanged.
>
> Playwright is available; Chromium is at
> `/opt/pw-browsers/chromium-1194/chrome-linux/chrome`. Streamlit needs ~20s
> to first render (model load plus a 16k-row CSV parse) — short waits produce
> blank screenshots that look like failures.
>
> ### 7. Traps that have already caught someone here
>
> - **Blank line in an HTML template.** An empty interpolated value leaves a
>   whitespace-only line; Markdown then reads the indented lines after it as a
>   **code block** and prints raw `<div>` markup. Emit conditional HTML as one
>   unindented string.
> - **`st.tabs` renders every tab in one pass** and switches client-side. No
>   rerun happens when the user changes tab, so reading `session_state` at
>   render time leaves you a run behind. Use `st.empty()` placeholders filled
>   after the value is settled.
> - **`st.metric(delta=...)` always draws a direction arrow**, even at
>   `delta_color="off"`. Use `caption` or `help` for text that is not a delta.
> - **`:has()` in CSS matches more than you think.** A rule meant for metric
>   rows also matched the main two-column layout, because a metric sits
>   somewhere inside it. There is no safe selector for that; don't try.
> - **`advisory.nutrient_gaps` is signed** `observed − required`. A deficit is
>   **negative**. Passing it straight to the bag converter prescribes nothing
>   for every nutrient actually short.
> - **The collapsed Streamlit sidebar overlays ~20px of content** on mobile.
> - **Chart legends must sit outside the axes.** matplotlib's `loc="best"`
>   puts them over the data whenever curves span the full range.
> - **Python floor is 3.12.** `shap` has no cp313 wheel. Do not claim 3.13 works.
>
> ### 8. Where the real headroom is
>
> Ranked by actual benefit to a farmer, not by how impressive it sounds:
>
> 1. **Kannada interface.** Only crop names are localised; every label,
>    button and heading is English. This is the largest remaining barrier for
>    the intended user. Ship it opt-in with English default, and mark the
>    translations unreviewed until a native speaker has read them.
> 2. **Works with no signal.** Fields do not have coverage. The Android shell
>    (`android/`) stores saved advice on-device; the web app cannot. Consider
>    exporting the model to ONNX so a recommendation can be produced offline.
> 3. **Lower the input burden.** Seven soil parameters is a lot to ask of
>    someone without a soil test. The district baseline helps; consider
>    whether the app can be useful with two or three inputs.
> 4. **Voice or pictures** for low-literacy users.
>
> Do not start these without saying which one you are doing and why.
>
> ### 9. How to work
>
> Read before you write. Make one coherent change at a time, verify it, then
> commit with a message that says what was wrong and what the fix was — not a
> list of files touched. If you find a real problem with what you were asked
> to do, say so in a sentence or two and then deliver the work anyway under
> stated assumptions. Report honestly: if something is untested, say it is
> untested.

---

## Notes on using this

**If you want a rewrite rather than an improvement**, add this to §3:

> The Streamlit presentation layer may be replaced (React, Flutter, native
> Android) provided `src/` remains the single source of inference, advisory
> and persistence logic, and the model artifacts are loaded unchanged. Port
> the design rules in §4 and the traps in §7 — they are framework-independent
> and were learned expensively.

**Streamlit's ceiling, honestly.** It got this app built and deployed fast,
and it is genuinely serviceable on a phone. But it fights you on layout, it
cannot do responsive containers, it re-runs the whole script on every
interaction, and it is not offline-capable. If the priority becomes farmers
in fields with no signal, that is the point at which the framework, not the
code, is the limit.
