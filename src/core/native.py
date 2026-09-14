"""Hand a saved recommendation to the Android shell, when there is one.

The dashboard is the same code in a browser and inside the GREENROOT Android
app. Inside the app a Java bridge is bound to ``window.GreenRootNative``; in a
browser that name is simply absent. Everything here is therefore best-effort
and must never change what the browser user sees.

Why bother: a saved card written through this bridge lives in the phone's own
storage and stays readable with no connection. Fields do not have signal, and
advice a farmer can only read when the network is up is advice they cannot act
on. See ``android/app/src/main/java/in/greenroot/app/NativeBridge.kt``.

The injection route is the same one ``pwa`` uses -- a zero-height component
iframe served same-origin, whose script reaches ``window.parent``. If a future
Streamlit tightens that boundary the call is a no-op rather than an error.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence

#: Must match ``MainActivity.BRIDGE_NAME``.
BRIDGE = "GreenRootNative"


def build_card(
    *,
    card_id: str,
    crop: str,
    confidence: float,
    district: str,
    readings: str,
    advice: Sequence[str],
    max_advice: int = 6,
) -> dict:
    """The payload ``NativeBridge.saveCard`` parses.

    Advice is capped because the phone store holds 100 cards and the whole
    file is read on every open; a card is a summary, not a transcript.
    """
    return {
        "id": str(card_id),
        "crop": str(crop),
        "confidence": round(float(confidence), 2),
        "district": str(district),
        "readings": str(readings),
        "advice": [str(line) for line in advice][:max_advice],
    }


def save_script(card: dict) -> str:
    """JavaScript that pushes ``card`` to the app, or does nothing in a browser."""
    # json.dumps twice: once for the object, once so it arrives at the bridge
    # as a single JS string argument with its quotes intact.
    payload = json.dumps(json.dumps(card, ensure_ascii=False))
    # District is a free-text field. json.dumps escapes quotes but not the
    # solidus, so a district of "</script>" would close this block early and
    # spill the rest as markup. Break the sequence the HTML parser looks for;
    # "<\/" is the same string to JavaScript.
    payload = payload.replace("</", "<\\/")
    return f"""
<script>
(function () {{
  try {{
    var host = window.parent || window;
    var bridge = host.{BRIDGE};
    if (!bridge || typeof bridge.saveCard !== 'function') return;  // a browser
    bridge.saveCard({payload});
  }} catch (e) {{
    /* Never let the handoff break the page the farmer is looking at. */
  }}
}})();
</script>
"""


def push_card(st_module, card: dict) -> bool:
    """Emit the handoff. Returns whether the script was rendered at all.

    True does not mean the phone stored it -- only the bridge knows that, and
    it is not worth a round trip to find out. The Android side raises its own
    confirmation toast.
    """
    try:
        import streamlit.components.v1 as components

        components.html(save_script(card), height=0, width=0)
        return True
    except Exception:  # noqa: BLE001 - a missing shell is the normal case.
        return False


def readings_summary(features: dict) -> str:
    """One line of soil numbers, in the order the hero shows them."""
    return (
        f"N {features['N']:.0f} · P {features['P']:.0f} · K {features['K']:.0f}"
        f" · pH {features['ph']:.1f} · {features['temperature']:.0f}°C"
        f" · {features['rainfall']:.0f} mm"
    )


def advice_lines(advisory, simple: bool = True) -> List[str]:
    """Advisory items as plain sentences, worst first.

    The phone shows these with no styling, so the severity has to survive in
    the words themselves.
    """
    order = {"critical": 0, "warning": 1, "info": 2}
    items = sorted(advisory.items, key=lambda i: order.get(i.severity, 3))
    return [item.say(simple) for item in items]
