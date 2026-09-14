# Running GREENROOT on your phone

Two ways, depending on what you need.

| | Same Wi-Fi | Streamlit Cloud |
|---|---|---|
| Works away from your computer | ✗ | ✓ |
| Needs your computer switched on | ✓ | ✗ |
| Needs internet | ✗ (works offline) | ✓ |
| Setup time | 1 minute | ~10 minutes |
| Cost | free | free |
| Good for | demo in the lab, viva, field test with no signal | sharing a link with your guide or examiner |

---

## Option 1 — Same Wi-Fi (fastest)

Your phone and your computer must be on the **same Wi-Fi network**.

**Windows**

```bat
run_phone.bat
```

**macOS / Linux**

```bash
./run_phone.sh
```

The window prints an address like:

```
    http://192.168.1.42:8501
```

Type that into your phone's browser. That's it.

### If the phone cannot connect

1. **Windows firewall.** The first run pops up a firewall prompt. Tick
   **Private networks** and click **Allow**. If you clicked Cancel, re-run the
   script and allow it, or add an inbound rule for TCP port 8501.
2. **Same network?** Phone on mobile data instead of Wi-Fi is the usual cause.
   Also check the two are not on separate guest and main networks.
3. **Client isolation.** Many college and café networks block device-to-device
   traffic entirely. Nothing on your computer can fix that — use a phone
   hotspot instead: connect the computer to the phone's hotspot, then re-run
   the script and use the address it prints.
4. **Wrong address.** If the script cannot detect your IP, run `ipconfig`
   (Windows) or `ifconfig` (macOS/Linux), find your IPv4 address, and use
   `http://THAT-ADDRESS:8501`.

### A note on security

`run_phone.bat` / `run_phone.sh` start the server with
`--server.enableCORS false --server.enableXsrfProtection false`. Both are
required: without them Streamlit refuses the browser's WebSocket connection
from any address other than `localhost`, and the phone gets a permanently blank
page. This was verified rather than assumed — with the flags the app renders
all six tabs over the network address; without them it renders nothing.

The cost is that a browser-origin check is switched off and the app has no
login, so **anyone on the same network can open it** while the window is
running. That is fine on home or college Wi-Fi for a demo. Do not run it on
open public Wi-Fi, and close the window when you are finished.

`run.bat` (the normal desktop launcher) keeps both protections on.

---

## Option 2 — Streamlit Community Cloud (a link that works anywhere)

Free, and gives you a public URL you can send to your guide or open on any
phone without your computer running.

1. Push this repository to GitHub (it already is).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. **Create app** → pick your repository → branch → main file `app.py`.
4. Deploy. The first build takes a few minutes while it installs `shap`,
   `lime` and `scikit-learn`.
5. You get a URL like `https://greenroot.streamlit.app`. Open it on the phone
   and add it to your home screen.

### What to expect

- **`requirements.txt` is already correct** — every third-party import in the
  project is declared, verified by an import audit.
- **Set the Python version to 3.12 or 3.13.** In the deploy dialog open
  **Advanced settings** and choose it there. This matters: `shap` declares
  `requires_python >=3.12` and `scikit-learn` 1.9 declares `>=3.11`, so a
  build on 3.11 or older fails while installing. Checked against PyPI —
  scikit-learn 1.9.0 publishes wheels for cp311 through cp314.
- **`scikit-learn` is pinned to 1.9.0** on purpose: the artefacts in `models/`
  were serialised under that version. If a build fails, raise the Python
  version rather than relaxing the pin — unpickling an estimator across a
  minor version can change behaviour silently.
- **The audit ledger resets on restart.** Streamlit Cloud gives each app an
  ephemeral filesystem, so `crop_recommendations.db` is wiped when the app
  sleeps or redeploys. Saved recommendations are not lost during a session,
  but do not treat the cloud copy as permanent storage. The local runs keep
  their database normally.
- **Cold start.** A free app sleeps after inactivity and takes ~30 seconds to
  wake. Open it a minute before your demo.
- **Repository visibility.** A public repo deploys with no extra steps. For a
  private one, grant Streamlit access to it during sign-in.

---

## Option 3 — Install it to the home screen (looks like an app)

The dashboard ships a web-app manifest, so a phone will install it properly
rather than leaving a browser bookmark:

- **Android (Chrome):** ⋮ menu → **Install app** (or *Add to Home screen*)
- **iPhone (Safari):** Share → **Add to Home Screen**

You get the GreenRoot seedling icon in your launcher, and it opens full-screen
with no address bar, under its own name. For a demo this is indistinguishable
from an installed app, and it costs nothing.

It still needs a connection — the model runs on the server, so there is no
useful offline mode.

---

## Option 4 — A real APK, and the Play Store

An Android project is included in **[`android/`](android/)**. Point it at your
deployed URL, open it in Android Studio, and **Build → Build APK(s)** produces
an installable `.apk` you can put on any Android phone.

> That project has **never been compiled** — no Android SDK was available where
> it was written. It is standard boilerplate and should build, but budget time
> to fix a small issue or two.

**About the Play Store specifically.** It is possible but not straightforward:

- A Google Play Developer account costs **US$25** (one time)
- Review takes days
- **Google routinely rejects WebView wrappers** under the minimum-functionality
  policy — an app whose only content is a website. Passing usually requires
  real native value: offline use, push notifications, camera or GPS

For a capstone, Option 3 gives you the look of an installed app in thirty
seconds, and the APK in `android/` gives you something you can physically
install on a panel member's phone. Neither needs a store listing.

---

## How the phone layout works

The dashboard was tested at 390 × 844 (iPhone-class) and fixed for it:

- The sidebar is a **drawer**, closed on open. Tap **☰** at the top to enter
  your soil readings, then close it.
- Your current readings and the green **Show me the best crop** button sit in
  the main area, so the primary action is always visible without opening the
  drawer.
- Columns stack instead of squeezing, buttons are at least 46 px tall, and
  inputs use 16 px text so iOS Safari does not zoom the page when you tap a
  field.
- Charts use larger type than a desktop-only design would, so they stay
  readable once scaled down to phone width.
