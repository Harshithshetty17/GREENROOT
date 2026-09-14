# GREENROOT — Privacy Policy

**Last updated: 14 September 2026**

GREENROOT is a crop recommendation tool. This policy covers the GREENROOT
Android app (`in.greenroot.app`) and the GREENROOT web dashboard it displays.

There is no account, no sign-in, and no advertising in GREENROOT. Nothing here
is sold, and nothing is shared with a data broker.

---

## What the app collects

**Farm readings you type in.** Seven soil and weather values — nitrogen,
phosphorus, potassium, pH, temperature, humidity and rainfall — plus a
district name. You enter these yourself; the app does not read them from any
sensor on your phone.

**The recommendation that comes back.** When you press *Save this
recommendation*, the crop, the confidence score, the readings and the date are
written to two places: the GREENROOT server, and your own phone.

**Only if you choose to create an account:** your mobile number, a PIN, and
whatever you put in your profile (a name, a village, your usual district and
plot size). **You do not need an account.** Every part of the crop
recommendation works without one, and if you never sign in none of this is
collected.

**Nothing else, ever.** GREENROOT does not collect your email address,
contacts, photos, files, device identifiers, or advertising ID. It does not
use the phone's GPS or any other location sensor. It contains no analytics,
advertising, or crash-reporting SDK.

## About your PIN

Your PIN is never stored. What is stored is a bcrypt hash of it, which cannot
be reversed — nobody, including us, can read your PIN back.

**If you forget your PIN, use your recovery code.** You are shown an
eight-character code once, when you create the account. Write it down and keep
it somewhere safe: it is the only way back in, we cannot show it to you again,
and we cannot look it up. Using it sets a new PIN and gives you a fresh code.

Recovery codes are stored the same way as PINs — hashed, never in the clear.
Wrong codes count towards the same lockout as wrong PINs, so the recovery
route cannot be used to get around it.

After five wrong PINs an account locks for fifteen minutes. This is
deliberate: a 4-digit PIN is short, and the lock is what stops somebody
guessing their way in.

Be aware that a 4-digit PIN is not strong protection. We chose it because a
long password typed on a phone in a field is a barrier that stops people
using the app at all — and because an account here holds crop advice, not
money or identity documents. Do not reuse a PIN you use for banking.

## About the district name

The district you choose is a rough area, typically hundreds of square
kilometres. It is used to look up a regional soil baseline and local weather.
It is not a street address and it is not derived from your device. You may type
anything you like in this field, including nothing meaningful.

## Where the data goes

| Where | What | Why |
|---|---|---|
| GREENROOT server | Readings, recommended crop, confidence, timestamp, district | The recommendation runs on the server; the record makes past advice auditable |
| GREENROOT server | Your mobile number and PIN hash — **only if you create an account** | So your saved advice follows you to another phone |
| Your phone | The recommendations you chose to save | So you can read your advice in a field with no signal |
| OpenWeatherMap | The district name only, and only if live weather is enabled | To fetch current weather for that area |

OpenWeatherMap receives a place name and nothing else — no identifier, no
readings, no recommendation. Their handling of that request is governed by
their own privacy policy at https://openweather.co.uk/privacy-policy

The connection to the GREENROOT server is HTTPS. The app refuses unencrypted
traffic.

## What stays only on your phone

Saved recommendation cards are written to the app's private storage. No other
app can read them. They are included in Android's backup, so they follow you to
a new phone if you use Google backup or a device-to-device transfer.

**Deleting them:** uninstall the app, or go to Settings → Apps → GreenRoot →
Storage → Clear storage. Either removes every saved card from the device
immediately.

## Deleting your account

In the app: **Settings → Delete my account**. You will be asked to type DELETE
to confirm.

This removes your account, your mobile number, your PIN hash, your profile and
your saved plots. It happens immediately and cannot be undone.

Recommendations you saved are **kept but detached** — they remain in the
records with no link to you or your number, because they are the agronomic
audit trail the system exists to hold. If you want those deleted as well,
email the address below and say so.

## How long the server keeps records

Server-side records are kept for as long as the service runs, as an audit
trail of what the system advised and why. They are not linked to you as a
person — there is no account to link them to.

To request deletion of server records associated with a district, email the
address below with the district and approximate dates.

## Children

GREENROOT is intended for adult farmers and agricultural extension workers. It
is not directed at children and does not knowingly collect data from them.

## Security, stated honestly

GREENROOT is a final-year engineering project, not a commercial service. The
transport is encrypted and the server stores no personal identifiers, but it
does not carry a formal security certification. Do not enter anything into the
district field that you would not want stored.

## Changes

Material changes to this policy will be published at this URL and the "last
updated" date above will change.

## Contact

**parantimedia@gmail.com**

---

## Not a warranty

GREENROOT gives advice to help you decide. It is not a promise about yield,
income, or crop outcome. Please check with your local agriculture officer
before sowing. Neither the author nor this app is liable for decisions taken
on the basis of its output.
