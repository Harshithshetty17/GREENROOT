# GREENROOT for Android

A native shell around the deployed GREENROOT dashboard.

## What is native and what is not

The stacking ensemble runs on the server, so the recommendation screen is the
dashboard in a WebView: one implementation of the model, and the phone picks up
fixes the moment the server is redeployed.

What the shell adds is the part a browser cannot do — **the advice a farmer
saves is written to the phone and stays readable with no connection**. Fields
do not have signal. Advice you can only read when the network is up is advice
you cannot act on while standing in the crop.

| File | Role |
|---|---|
| `MainActivity.kt` | WebView host, splash, pull-to-refresh, downloads, the offline library |
| `CardStore.kt` | Saved cards, in app-private storage, atomic writes, capped at 100 |
| `NativeBridge.kt` | `window.GreenRootNative` — what the dashboard calls to save and share |

The web half of that bridge is `src/core/native.py`, covered by
`tests/test_native_bridge.py`.

## Before you build

Point `MainActivity.APP_URL` at your own deployment. It must be `https`:
cleartext is disabled in the manifest.

```bash
cp keystore.properties.example keystore.properties   # then fill it in
./gradlew bundleRelease
```

## Publishing

See **[PLAYSTORE.md](PLAYSTORE.md)** — store listing copy, the data safety
answers, the graphics, and an honest account of the Minimum Functionality
policy risk that applies to any WebView app.

## Status

**This Kotlin has never been compiled.** The Android SDK is served from
`dl.google.com`, which was not reachable from the environment this was written
in. The XML parses and the structure is sound, but open it in Android Studio
and expect to fix small things on the first build.
