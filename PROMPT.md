# GREENROOT — build brief

One self-contained specification. Copy everything inside the fence below and
paste it into a fresh AI coding session opened on this repository.

It is not a generic "build me an app" brief. Every constraint, rule and trap
in it comes from something that actually happened in this codebase — features
that were added and then removed for making the app worse, and bugs that were
shipped, found and fixed. An agent that reads §7 will not repeat them.

---

```text
================================================================================
GREENROOT — CROP RECOMMENDATION SYSTEM FOR KARNATAKA SMALLHOLDERS
Build order for an existing, working repository.
================================================================================

MISSION
--------------------------------------------------------------------------------
GREENROOT recommends one of 22 crops from 7 soil and weather parameters, shows
its reasoning, and tells a farmer what to buy and when to do it. The system
already exists and works. Your job is to make it genuinely better for the
people who use it — not to rebuild it, and not to add features to prove effort.

================================================================================
1. CRITICAL CONSTRAINTS — READ BEFORE ANYTHING ELSE
================================================================================
1. NEVER delete, retrain or overwrite:
     models/stacking_model.pkl
     models/scaler.pkl
     models/class_names.pkl
   and NEVER modify anything in data/.
   These are the evaluated artifacts. All inference must load these exact
   files.

2. DO NOT RUN train.py. It overwrites all three pickles. To validate the
   model, run verify_cv.py — it refits the architecture in memory and asserts
   the pickles are byte-identical before exiting.

3. Preserve these verified behaviours. Each was a defect that was fixed:
     - What-If chart legends sit ABOVE the axes, never over the curves.
     - Farmer views show localised dates ("14 Sep 2026"), integer scores
       ("91 / 100") and plain column names — never primary_shap_driver,
       jaccard_index, ISO timestamps or raw floats like 90.855.
     - Examiner mode keeps the stored row exactly as it is.
     - A low-confidence match is styled differently from a high-confidence one.

================================================================================
2. WHO THIS IS FOR
================================================================================
FARMER (primary). On a mid-range Android phone, outdoors, one hand, patchy
signal, reading in a second language. Wants three things: which crop, what to
buy, when to do it. Does not want a model.

EXAMINER (secondary). Behind one toggle, "Examiner / AI Mode". Wants the
evidence: Z-scores, SHAP/LIME agreement, sensitivity curves, the raw audit
ledger with schema columns intact.

A single predicate, is_simple(), drives both. Do not fork the codebase into
two apps. Do not water down the examiner view to make it friendlier.

================================================================================
3. WHAT ALREADY EXISTS — DO NOT REBUILD THESE
================================================================================
  src/models/inference.py        CropRecommender: validate -> scale -> rank
                                 -> sweep
  src/models/xai_engine.py       TreeSHAP + LIME over a RandomForest
                                 surrogate; Jaccard at k=3, threshold 0.50
  src/models/batch.py            Bulk advisory over a whole soil survey
  src/services/soil_service.py   District baselines, aliases, climate normals
  src/services/weather_service.py  OpenWeatherMap, 5s timeout, offline mock
  src/database/db_manager.py     Thread-safe SQLite, WAL, migrations
  src/database/db.py             Dual-read facade: farmer_view/examiner_view
  src/utils/agronomy.py          22-crop metadata, 50kg bag converter,
                                 spray advisory, crop roadmap
  src/utils/agronomy_advisory.py Nutrient / pH / water / thermal heuristics
  src/utils/seasons.py           Kharif / Rabi / Summer fit
  src/utils/plain_language.py    Farmer register, bags and acres
  src/core/config.py             Paths, feature contract, bounds, thresholds
  verify_cv.py                   Independent CV; never touches models/

Read a module before changing it. Most carry a docstring explaining WHY they
are the way they are. That reasoning is usually load-bearing.

================================================================================
4. DESIGN RULES
================================================================================
4.1 SIMPLE MEANS FEW DECISIONS, NOT FEW FEATURES.
    The app may grow accounts, settings and plot management and still be
    simple, provided the farmer's path stays:
        open -> (district already remembered) -> one button -> the answer.
    Judge simplicity by the number of choices on that path, not by the size
    of the codebase.

4.2 NOTHING BLOCKS THE FIRST ANSWER.
    No sign-up wall. No onboarding carousel. No permission prompt before a
    farmer can see a recommendation. Accounts keep history across devices and
    are offered AFTER a useful answer, never before one.

4.3 REMOVE BEFORE YOU ADD.
    This app has repeatedly been made worse by addition. Five farmer tabs
    became two and it improved. Before adding a panel, say what it replaces.

4.4 THE ANSWER LEADS.
    Controls collapse behind a one-line summary of what they currently say
    ("Udupi - 1 acre - Kharif"). On a phone the run button must be reachable
    without scrolling. Anything set once and rarely changed does not belong
    in the main flow.

4.5 UNCERTAINTY MUST BE VISIBLE IN THE DESIGN, NOT JUST READABLE IN A
    CAPTION. A 43/100 match rendered in the same confident green capitals as
    a 95/100 match is the app lying with typography. Below 60, degrade the
    colour and say so above the crop name. The banner and the card must agree
    — they contradicted each other once already.

4.6 NEVER SHOW A FARMER A SCHEMA COLUMN NAME.
    db.farmer_view() is the seam. Keep everything going through it.

4.7 PLAIN LANGUAGE IS NOT THE SAME AS FARMER CONTENT.
    Translating "explainer consensus" into simple words does not make it
    something a farmer needs. Cut it from their view instead.

================================================================================
5. ACCOUNTS, LOGIN, AND THE ORDINARY APP SURFACE
================================================================================
The app has no accounts and no settings screen. It needs what people expect
any app to have — built for this audience, not copied from a SaaS dashboard.

5.1 SIGN-IN IS BY PHONE NUMBER, NOT EMAIL.
    Phone + OTP, or phone + a 4-digit PIN where no SMS gateway is available.
    Never email/password as the only route: most users here do not use email.

5.2 REQUIRED SURFACES
    Guest mode        Full recommendation flow with NO account. This is the
                      default and must never regress.
    Sign in / Sign up One flow, not two. A number either signs in or creates
                      the account.
    Logout            Reachable in two taps. Warns that unsynced local advice
                      stays on the device.
    Account recovery  A lost PIN recoverable by OTP. Do not strand people.
    Delete my account Required by Google Play and non-negotiable ethically.
                      Must actually delete, and say what it deletes.
    Profile           Name, village, default district, plot size. Pre-fills
                      the form so a returning user presses one button.
    My plots          A farmer with three fields needs three saved plots with
                      their own readings, not one global form. This is the
                      most useful thing an account unlocks.
    Settings          Language, units, notifications, clear local data.
    Language switch   English / Kannada. See 9.1 — translations are not yet
                      native-reviewed and must be marked as such.
    Notifications     Opt-in only. Sowing windows, spray-weather warnings,
                      top-dressing dates from the crop roadmap. Never
                      marketing.
    Help & support    A short FAQ and a real contact route.
    About             Version, what the model is, the disclaimer, links to
                      privacy and terms.
    Offline indicator Say plainly when advice came from cached data.

5.3 SECURITY — AN AGENT LEFT ALONE GETS THESE WRONG
    - Hash with bcrypt or argon2. Never plaintext, never MD5/SHA-1, never a
      hand-rolled scheme.
    - No secrets in the repo. Read from environment or
      .streamlit/secrets.toml (already gitignored, along with .env, *.pem,
      *.key).
    - Rate-limit OTP and PIN attempts, or you have built a free SMS cannon
      and a brute-force target.
    - Session tokens must be random, expiring and revocable on logout.
      Streamlit's session_state is NOT authentication: it is per-browser-
      session memory and does not survive a refresh.
    - Scope every READ to the signed-in user. Add user_id to the queries,
      not just to the UI. One farmer must never see another's ledger.

5.4 DATABASE WORK THIS IMPLIES
    audit_logs has no user column and SCHEMA_VERSION is 1. Accounts need a
    users table, a user_id column on audit_logs, and SCHEMA_VERSION bumped to
    2 with a forward migration in _migrate() — which already has an additive-
    column pattern to follow. EXISTING ROWS MUST SURVIVE, attributed to a
    guest or legacy user. Do not drop and recreate the table.

5.5 PRIVACY CONSEQUENCES
    A phone number is personal data, which this app does not currently
    collect. PRIVACY.md and the Play data-safety declaration in
    android/PLAYSTORE.md both state that today, and both become wrong the
    moment login ships. Update them in the SAME change, not afterwards.

5.6 WHAT "SIMPLE" MEANS ALONGSIDE ALL THIS
    Every item above lives behind a profile icon or in Settings. None of it
    appears in the recommendation flow. If a farmer who never signs in
    notices any of this work beyond a small "Sign in" affordance, it has been
    built wrong.

================================================================================
6. HONESTY RULES — NOT NEGOTIABLE
================================================================================
1. DO NOT INVENT MARKET DATA. Mandi rates, yields and cultivation costs in
   agronomy.py are indicative benchmarks, stamped with their basis and
   editable in the UI. Never present them as live prices.
   economics.py deliberately ships no defaults at all — preserve that.

2. DO NOT CLAIM CORROBORATION YOU HAVE NOT CHECKED. 16 of the 22 Kannada
   names are verified against the shipped survey's bilingual labels, and a
   test re-derives that set from the CSV so the claim cannot rot. The other
   six are flagged. Mark any translation you add as unreviewed.

3. MEASURE, DO NOT ASSUME. train.py scales before splitting, which is
   leakage. The correct response was to quantify it (+0.000 pp), not to
   assert it away.

4. State a concern once, then do the work. If something is untested, say it
   is untested.

================================================================================
7. TRAPS THAT HAVE ALREADY COST A DEBUGGING CYCLE HERE
================================================================================
1. BLANK LINE IN AN HTML TEMPLATE. An empty interpolated value leaves a
   whitespace-only line; Markdown then reads the indented lines after it as a
   CODE BLOCK and prints raw <div> markup. Emit conditional HTML as one
   unindented string.

2. st.tabs RENDERS EVERY TAB IN ONE PASS and switches client-side. No rerun
   happens when the user changes tab, so reading session_state at render time
   leaves you a run behind. Use st.empty() placeholders filled after the
   value is settled.

3. st.metric(delta=...) ALWAYS DRAWS A DIRECTION ARROW, even at
   delta_color="off". Use caption or help for text that is not a delta.

4. :has() IN CSS MATCHES MORE THAN YOU THINK. A rule meant for metric rows
   also matched the main two-column layout, because a metric sits somewhere
   inside it, and the crop card stopped stacking on mobile. There is no safe
   selector for that case.

5. advisory.nutrient_gaps IS SIGNED (observed - required). A deficit is
   NEGATIVE. Passing it straight to the bag converter prescribes nothing for
   every nutrient actually short.

6. THE COLLAPSED STREAMLIT SIDEBAR OVERLAYS ~20px OF CONTENT on mobile.

7. CHART LEGENDS MUST SIT OUTSIDE THE AXES. matplotlib's loc="best" puts them
   over the data whenever curves span the full range.

8. PYTHON FLOOR IS 3.12. shap has no cp313 wheel. Do not claim 3.13 works.

9. A KEYED st.selectbox STORES THE LABEL ITS format_func PRODUCED, and restores
   the widget by looking that label back up among the formatted options. Change
   what format_func returns -- translating the options, say -- and the stored
   label no longer matches anything, so Streamlit hands back the raw label
   string instead of the option. st.session_state["season"] then holds
   "Rabi (winter)" where every reader expects "rabi", and the next
   SEASONS[...] takes the whole page down. If a format_func can ever change,
   map the key's value back onto a valid option before the widget is built --
   after it, assigning to a widget key is itself an error. See
   seasons.canonical().

10. A TRANSLATED STRING NOTHING READS is worse than a missing one: it reads as
   covered, it costs a reviewer time, and the screen it was written for is
   still in English. Eight keys had drifted into that state. tests/test_i18n.py
   parses app.py for the keys it actually asks for and fails in both
   directions.

================================================================================
8. VERIFICATION STANDARD — A CHANGE IS NOT DONE UNTIL ALL OF THESE PASS
================================================================================
1. pytest -q is green. (958 passed, 1 environment-conditional skip at time of
   writing.)

2. You have driven the real app in a browser at 390x844 AND 1440x900, taken
   screenshots, and LOOKED at them. Not "captured" — looked.

3. You have exercised BOTH the weak- and strong-confidence paths. Udupi
   returns roughly 43/100; Dharwad roughly 60/100. Several bugs here appeared
   on only one of the two.

4. You have checked both personas after any layout change.

5. If you touched accounts: guest mode still reaches an answer with no
   sign-in; logout actually clears the session; one user cannot read
   another's ledger; and the migration was run against a database containing
   pre-existing rows, which survived.

6. md5sum confirms the model artifacts are unchanged.

Notes: Playwright is available; Chromium is at
/opt/pw-browsers/chromium-1194/chrome-linux/chrome. Streamlit needs about 20
seconds to first render (model load plus a 16k-row CSV parse) — short waits
produce blank screenshots that look like failures.

================================================================================
9. WHERE THE REAL HEADROOM IS
================================================================================
Ranked by benefit to a farmer, not by how impressive it sounds. Say which one
you are doing and why before starting it.

9.1 KANNADA INTERFACE. Only crop names are localised; every label, button and
    heading is English. This is the largest remaining barrier for the
    intended user. Ship it opt-in with English as the default, and mark the
    translations unreviewed until a native speaker has read them.

9.2 WORKS WITH NO SIGNAL. Fields do not have coverage. The Android shell in
    android/ stores saved advice on-device; the web app cannot. Consider
    exporting the model to ONNX so a recommendation can be produced offline.

9.3 LOWER THE INPUT BURDEN. Seven soil parameters is a lot to ask of someone
    with no soil test. The district baseline helps; consider whether the app
    can be useful with two or three inputs.

9.4 VOICE OR PICTURES for low-literacy users.

================================================================================
10. HOW TO WORK
================================================================================
Read before you write. Make one coherent change at a time and verify it
before starting the next. Commit with a message that says what was wrong and
what the fix was — not a list of files touched.

If you find a real problem with what you were asked to do, say so in a
sentence or two, then deliver the work anyway under stated assumptions.
Finish the whole task; if part of it is blocked, complete everything else and
say explicitly what you left out and why.
================================================================================
```

---

## If you want a rewrite rather than an improvement

Add this to §3 of the brief:

> The Streamlit presentation layer may be replaced (React, Flutter, native
> Android) provided `src/` remains the single source of inference, advisory
> and persistence logic, and the model artifacts are loaded unchanged. Port
> §4 and §7 — they are framework-independent and were learned expensively.

**Streamlit's ceiling, honestly.** It got this built and deployed quickly and
it is serviceable on a phone. But it fights you on layout, it cannot do
responsive containers, it re-runs the whole script on every interaction, and
it is not offline-capable. If farmers in fields with no signal become the
priority, the framework — not the code — is the limit.
