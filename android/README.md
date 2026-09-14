# GreenRoot — Android app

A thin native shell around the deployed dashboard. The model runs on the
server, so this is a WebView container rather than a reimplementation: one
codebase, one set of behaviour, and the phone picks up server fixes without a
new release.

> **Not built or tested in this repository.** The project is written against
> standard Android APIs but no Android SDK was available where it was authored,
> so it has never been compiled or run on a device. Expect to fix a small issue
> or two on first build — treat it as a starting point, not a finished binary.

## Before you build

Open `app/src/main/java/in/greenroot/app/MainActivity.kt` and set your address:

```kotlin
private const val APP_URL = "https://greenroot.streamlit.app"
```

It must be **https**. The manifest sets `usesCleartextTraffic="false"`, so a
plain `http://192.168.x.x` LAN address will be blocked. If you deliberately
want to point at a LAN server for a demo, flip that flag — and understand you
are turning off a protection.

## Build an installable APK

1. Install **Android Studio** (free, ~1 GB) from
   <https://developer.android.com/studio>
2. **File → Open** → select this `android/` folder
3. Wait for Gradle to sync; it downloads the SDK and build tools on first run
4. **Build → Build Bundle(s) / APK(s) → Build APK(s)**
5. The APK lands in `app/build/outputs/apk/debug/app-debug.apk`

Copy that file to a phone and open it. Android will warn about installing from
an unknown source — allow it for your file manager. This is how you demo on any
Android phone without a store.

## What it does

- Full-screen, portrait, own icon and name in the launcher
- Back button walks web history before exiting
- Card exports (PDF, HTML, CSV) go to the phone's Downloads folder
- A readable "no internet" screen instead of a blank page
- External links open in the real browser, not inside the app

## Publishing to the Play Store

Possible, but read this first:

- A Google Play Developer account costs **US$25**, one time.
- Review takes days, sometimes longer for a first submission.
- **Google frequently rejects WebView wrappers** under the "minimum
  functionality" policy — an app whose only content is a website. To pass you
  generally need genuine native value: offline capability, push notifications,
  camera/GPS integration, home-screen widgets. A shell around a web page on its
  own is the classic rejection case.
- You also need a signed release build, a privacy policy URL, store artwork and
  a content rating questionnaire.

**For a capstone demo, the APK above is enough** — it installs and runs on any
Android phone, which is what a panel actually wants to see. The "Add to Home
screen" route (see `../PHONE.md`) gets you the same icon and full-screen feel
with no build step at all.
