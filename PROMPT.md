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
> **Simple means few decisions, not few features.** The app may grow an
> account system, a settings screen and plot management and still be simple,
> provided the farmer's path from opening it to reading advice stays: open →
> (district already remembered) → one button → the answer. Judge simplicity
> by the number of choices on that path, not by the size of the codebase.
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
> **Nothing blocks the first answer.** No sign-up wall, no onboarding
> carousel, no permission prompt before a farmer can see a crop
> recommendation. Accounts are for keeping your history across devices, and
> they are earned by the app being useful first — offered *after* a useful
> answer, never before it.
>
> **Plain language is not the same as farmer content.** Translating
> "explainer consensus" into simple words does not make it something a farmer
> needs. Cut it from their view instead.
>
> ### 4a. Accounts, and the ordinary app furniture
>
> The app currently has no accounts and no settings screen. It needs the
> things people expect any app to have — but built for this audience, not
> copied from a SaaS dashboard.
>
> **Sign-in — phone number, not email.** Most users here do not use email.
> Phone + OTP, or phone + a 4-digit PIN if you have no SMS gateway. Never
> email/password as the only route.
>
> Required screens and states:
>
> | Surface | Requirement |
> |---|---|
> | **Guest mode** | Full recommendation flow with no account. This is the default and must never regress. |
> | **Sign in / Sign up** | One flow, not two. Entering a number either signs in or creates the account. |
> | **Logout** | Reachable in two taps. Warns that unsynced local advice stays on the device. |
> | **Account recovery** | A lost PIN must be recoverable by OTP. Do not strand people. |
> | **Delete my account** | Required by Google Play, and non-negotiable ethically. Must actually delete, and say what it deletes. |
> | **Profile** | Name, village, default district, plot size. Pre-fills the form so a returning user presses one button. |
> | **My plots** | A farmer with three fields needs three saved plots with their own readings — not one global form. This is the single most useful account feature. |
> | **Settings** | Language, units, notifications, clear local data. |
> | **Language switch** | English / ಕನ್ನಡ. See §8 item 1 — the translations are not yet native-reviewed and must be marked so. |
> | **Notifications** | Opt-in only. Sowing-window reminders, spray-weather warnings, top-dressing dates from the crop roadmap. Never marketing. |
> | **Help & support** | A short FAQ, and a real contact route. |
> | **About** | Version, what the model is, the disclaimer, links to privacy and terms. |
> | **Offline indicator** | Say plainly when advice is from cached data rather than live. |
>
> **Security requirements — an agent left alone gets these wrong:**
>
> - Hash with **bcrypt or argon2**. Never plaintext, never MD5/SHA-1, never
>   a hand-rolled scheme.
> - No secrets in the repo. Keys come from environment or
>   `.streamlit/secrets.toml`, which stays gitignored.
> - Rate-limit OTP and PIN attempts, or you have built a free SMS cannon and
>   a brute-force target.
> - Session tokens must be random, expiring and revocable on logout.
>   Streamlit's `session_state` alone is not authentication — it is per-browser
>   -session memory and does not survive a refresh.
> - Scope every read to the signed-in user. A farmer must never see another
>   farmer's ledger. Add `user_id` to queries, not just to the UI.
>
> **Database work this implies.** `audit_logs` has no user column and
> `SCHEMA_VERSION` is 1. Adding accounts means a `users` table, a
> `user_id` column on `audit_logs`, and bumping `SCHEMA_VERSION` to 2 with a
> forward migration in `_migrate()` — it already has an additive-column
> pattern to follow. **Existing rows must survive**, attributed to a guest
> or legacy user. Do not drop and recreate the table.
>
> **Privacy consequences.** A phone number is personal data, which the app
> does not currently collect. `PRIVACY.md` and the Play data-safety
> declaration in `android/PLAYSTORE.md` both say so today and would both
> become wrong. Update them in the same change, not afterwards.
>
> **What "simple" means alongside all this.** Every one of the above lives
> behind a profile icon or in Settings. None of it appears in the
> recommendation flow. If a farmer who never signs in notices any of this
> work except a small "Sign in" affordance, it has been built wrong.
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
> 4a. If you touched accounts: guest mode still reaches an answer with no
>    sign-in; logout actually clears the session; one user cannot read
>    another's ledger; and the schema migration was run against a database
>    containing pre-existing rows, which survived.
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
