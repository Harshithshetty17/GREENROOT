# Publishing GREENROOT to Google Play

Everything in this repo is ready to build. Read the first section before you
pay for anything — it is the part that decides whether the submission is
accepted.

---

## 1. The policy risk, stated plainly

Google Play's **Spam and Minimum Functionality** policy rejects apps whose
purpose is to display a website:

> We don't allow apps that … provide a webview of a website not owned by the
> developer, or that have no functionality beyond what the website provides.

GREENROOT is a WebView of the GREENROOT dashboard, so this policy is the
single most likely reason for a rejection. Two things work in your favour:

1. **You own the website.** The clause above bites hardest on wrappers of
   *someone else's* site. Say so in the review notes.
2. **The app does things the website cannot.** This is the part that matters,
   and it is why the native layer exists:

   | Native capability | Where | Why the browser cannot do it |
   |---|---|---|
   | Saved advice readable with **no connection** | `CardStore.kt` | The dashboard is server-rendered; with no signal a browser shows nothing |
   | Android **share sheet** for a card | `NativeBridge.shareCard` | Sends advice to WhatsApp/SMS, which farmers actually use |
   | Downloads to the phone's **Downloads** folder | `MainActivity` download listener | Soil Health Card PDFs land where the file manager expects |
   | Survives backup and **device transfer** | `xml/data_extraction_rules.xml` | Saved cards follow the farmer to a new phone |
   | Offline library UI | `MainActivity.showOfflineLibrary` | Rendered from on-device storage, not fetched |

**Put this in "App access / Notes for review" in Play Console:**

> GREENROOT is a crop recommendation tool for Indian farmers. The prediction
> model runs on our own server (we own and operate the site the app displays).
> The app adds functionality the website cannot provide: recommendations the
> farmer saves are written to the device and remain readable with no network
> connection, which is the normal condition in a field. The app also
> integrates the Android share sheet and the system download manager. No
> sign-in is required — open the app and press "Show me the best crop".

**Be realistic:** this is a genuine risk, not a formality. If you are rejected,
the appeal route is to point at the offline library. If you have time before
submitting, the strongest possible answer is to make the *prediction itself*
run on-device (export the ensemble to ONNX and bundle it), at which point the
app is unambiguously more than a wrapper. That is a substantial piece of work
and is not done here.

---

## 2. What you need before you start

| Thing | Cost | Note |
|---|---|---|
| Google Play Console account | **US$25**, one-off | Personal or organisation |
| A hosted privacy policy URL | free | See §5 |
| Android Studio | free | The build cannot be done in this repo's CI — see §3 |
| An upload keystore | free | **Back it up.** Lose it and you can never update the listing |

### The 12-tester rule

If your Play Console developer account is **personal** (not an organisation)
and was created after November 2023, Google requires **closed testing with at
least 12 testers who stay opted in for 14 continuous days** before you may
apply for production access.

Plan for this. It means your app cannot go public the day you upload it, and
you need twelve real Google accounts. Start the closed test early and use your
classmates.

---

## 3. Building the release bundle

This must be done on your own machine. The Android SDK is served from
`dl.google.com`, which is not reachable from the environment this project was
developed in — none of the Kotlin here has been compiled. Expect to fix small
things on the first build.

```bash
# One-time: create the upload key. Keep the .jks and the passwords safe.
cd android
keytool -genkey -v -keystore greenroot-upload.jks \
        -keyalg RSA -keysize 2048 -validity 10000 -alias greenroot

cp keystore.properties.example keystore.properties
#   ... edit keystore.properties with your passwords ...

# Build the Android App Bundle Play requires (an .apk is not accepted)
./gradlew bundleRelease
#   → app/build/outputs/bundle/release/app-release.aab
```

Check it installs before you upload:

```bash
./gradlew installRelease        # onto a connected phone
```

### Version bumps

Every upload needs a higher `versionCode` than the last, in
`app/build.gradle.kts`. Play rejects a repeat.

---

## 4. Store listing — copy to paste

**App name** (30 char limit) — 24 used:

```
GreenRoot: Crop Advisor
```

**Short description** (80 char limit) — 71 used:

```
Which crop suits your soil, and why. Clear advice you can read offline.
```

**Full description** (4000 char limit):

```
GreenRoot tells you which crop suits your land — and, unlike most farming
apps, it shows you why.

Enter seven readings from your soil test: nitrogen, phosphorus, potassium,
pH, temperature, humidity and rainfall. GreenRoot weighs them against 22
crops and names the one that fits best, with a match score out of 100.

WHY, NOT JUST WHAT

Every recommendation comes with its reasoning. GreenRoot shows which of your
readings decided the answer, and it cross-checks that explanation with a
second, independent method — so you can see when the two agree and when they
do not. Advice you cannot question is advice you cannot trust.

WHAT TO DO ABOUT IT

• Exactly which fertiliser to add, in bags per acre — not abstract kg/ha
• Whether the season suits the crop, for Kharif, Rabi and Summer sowing
• What would change if you followed the advice
• Whether the cost is worth it, using prices you enter yourself

READS IN THE FIELD, WITH NO SIGNAL

Every recommendation you save is written to your phone. Open GreenRoot with
no network at all and your saved advice is still there — the crop, the
readings, the fertiliser plan. Share any card to WhatsApp or SMS in one tap.

IN PLAIN WORDS, AND IN KANNADA

GreenRoot is written for farmers, not agronomists. Simple words, quantities
in bags and acres, and a Soil Health Card you can print or save as a PDF,
with crop names in Kannada alongside English.

FOR MANY FARMS AT ONCE

Extension workers can upload a spreadsheet of plots and get a recommendation
for every row, with the ones that need a second look flagged.

HOW IT WORKS

GreenRoot uses a stacking ensemble — several machine-learning models whose
answers are combined by a further model trained to weigh them. It reaches
99.4% accuracy under stratified 5-fold cross-validation across 22 crops, and
it flags readings that fall outside the range it was trained on rather than
guessing confidently about land it has never seen.

IMPORTANT

GreenRoot gives advice to help you decide. It is not a promise about yield or
income. Please check with your local agriculture officer before sowing.

No account. No advertising. No tracking.
```

**Category:** Tools
**Tags:** agriculture, farming, soil
**Contact email:** parantimedia@gmail.com

### Graphics (already generated, in `../store/`)

| Asset | Required size | File |
|---|---|---|
| App icon | 512 × 512 PNG, no alpha | `store/play-icon-512.png` |
| Feature graphic | 1024 × 500 PNG | `store/play-feature-graphic-1024x500.png` |

**Phone screenshots — you must supply these.** Play requires **at least 2**,
between 320 px and 3840 px on each side. Take them on a real phone or an
emulator. Suggested set of four:

1. The empty state, showing the soil readings in the green hero
2. A result — the crop card and the "What to do in your field" advice
3. The explanation tab, showing the two explainers being compared
4. The offline library (turn on aeroplane mode after saving a card)

---

## 5. Privacy policy URL

Play will not accept the listing without one. `PRIVACY.md` at the repo root is
written for this. Publish it:

1. GitHub → your repo → **Settings → Pages**
2. Source: **Deploy from a branch**, branch `main`, folder `/ (root)`
3. The URL becomes
   `https://harshithshetty17.github.io/GREENROOT/PRIVACY` — check it loads
   before pasting it into Play Console.

---

## 6. Data safety form — the answers

Play Console → **App content → Data safety**. These follow from what the app
actually does; re-check them if you change the code.

**Does your app collect or share any of the required user data types?** → Yes

| Data type | Collected | Shared | Required? | Purpose |
|---|---|---|---|---|
| Location → Approximate location | Yes | No | Optional | App functionality |
| App activity → Other user-generated content (soil readings) | Yes | No | Required | App functionality |
| Personal info → Phone number | Yes | No | **Optional** | Account management |
| Personal info → Name | Yes | No | Optional | Account management |
| Personal info → Other info (village) | Yes | No | Optional | App functionality |

**Mark the three account rows "Optional".** They are collected only if the
user chooses to create an account; the whole recommendation flow works as a
guest. Declaring them Required would be inaccurate and is the kind of thing
that gets a data-safety form rejected.

**You must also answer these, and the answers changed when login shipped:**

- **Is a user account required to use the app?** → **No.** Guest mode is the
  default path.
- **Can users request that their data be deleted?** → **Yes**, and the app has
  an in-app route: Settings → Delete my account. Give that as the method.
- **Do you collect passwords or PINs?** → The PIN is collected and stored only
  as a bcrypt hash. Declare it under Account management; never describe it as
  not collected.

Why *Approximate location* is declared: the district name you type is a rough
area. It is not read from GPS, but Play's definition covers user-provided
areas, and over-declaring is safe while under-declaring is a policy violation.

Then:

- **Is all data encrypted in transit?** → Yes (HTTPS; cleartext is disabled in
  the manifest)
- **Can users request data deletion?** → Yes, provide the contact email
- **Does your app collect a device or advertising ID?** → No
- **Does your app use third-party analytics or ads?** → No

## 7. The rest of App content

| Question | Answer |
|---|---|
| Content rating questionnaire | Category *Utility*; answer **No** to every content question → rated Everyone / 3+ |
| Target audience | **18 and over.** Declaring under-18 pulls you into the Families policy programme and a much longer review |
| Ads | No |
| Government app | No |
| Financial features | No |
| Health apps | No |
| Data deletion URL | Not needed if you give the contact email instead |

---

## 8. Before you hit submit

- [ ] `versionCode` is higher than the last upload
- [ ] The `.aab` is signed — `keystore.properties` exists and is **not** in git
- [ ] `greenroot-upload.jks` is backed up somewhere off this laptop
- [ ] `APP_URL` in `MainActivity.kt` points at your live deployment
- [ ] The live deployment actually loads on a phone, in mobile data, right now
- [ ] Privacy policy URL loads in a browser
- [ ] At least 2 phone screenshots uploaded
- [ ] Review notes explain the native functionality (§1)

### One thing to fix first

The dashboard is hosted on Streamlit Community Cloud, which **puts an app to
sleep after a period with no traffic**. A reviewer opening a sleeping app sees
a wake-up spinner for 30–60 seconds and may reasonably conclude the app does
not work.

Before you submit, either:

- open the dashboard yourself shortly before the review, and keep it warm with
  a scheduled ping (an UptimeRobot check every 5 minutes is free and enough), or
- move the dashboard to a host that does not sleep.

This is worth doing. It is the most likely cause of a rejection that has
nothing to do with your code.
