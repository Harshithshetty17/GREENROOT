"""Make the dashboard installable on a phone's home screen.

Streamlit renders a plain web page. A phone will happily bookmark it, but
without a web-app manifest the result is a browser shortcut: it opens with an
address bar, carries a generic icon, and does not feel like an application.

This module supplies the manifest and icons, so "Add to Home screen" produces
something that launches full-screen under its own name and icon — the
appearance of an installed app, without an app store.

Injection mechanism
-------------------
The tags must live in ``<head>``, and ``st.markdown`` writes into ``<body>``.
Streamlit also strips ``<script>`` from markdown, so the usual workaround is a
zero-height component iframe whose script reaches into the parent document.
That iframe is served same-origin with ``allow-same-origin``, so
``window.parent.document`` is reachable. If a future Streamlit tightens that
sandbox the injection simply does nothing — the app still works, it just stops
being installable, so this is a safe enhancement rather than a dependency.

What this is not
----------------
This does **not** produce a Play Store listing, and it does not make the app
work offline — Streamlit needs its server for every interaction, so there is
nothing useful to cache. It makes the app *look and launch* like an installed
app. A store listing needs a signed Android package; see ``PHONE.md``.
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from src.core.config import BASE_DIR

#: Where the generated icons live.
ASSETS_DIR: Path = BASE_DIR / "assets"

APP_NAME = "GREENROOT"
APP_SHORT_NAME = "GreenRoot"
APP_DESCRIPTION = "Tells you which crop suits your land, and why"
THEME_COLOUR = "#1f7a4d"
BACKGROUND_COLOUR = "#ffffff"


@lru_cache(maxsize=8)
def _icon_data_uri(filename: str) -> Optional[str]:
    """Return a PNG asset as a ``data:`` URI, or ``None`` if absent."""
    path = ASSETS_DIR / filename
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_manifest() -> Optional[Dict[str, object]]:
    """Assemble the web-app manifest, or ``None`` if the icons are missing."""
    icons = []
    for filename, size, purpose in (
        ("icon-192.png", "192x192", "any"),
        ("icon-512.png", "512x512", "any"),
        ("icon-maskable-512.png", "512x512", "maskable"),
    ):
        uri = _icon_data_uri(filename)
        if uri:
            icons.append(
                {"src": uri, "sizes": size, "type": "image/png", "purpose": purpose}
            )
    if not icons:
        return None

    return {
        "name": APP_NAME,
        "short_name": APP_SHORT_NAME,
        "description": APP_DESCRIPTION,
        "start_url": ".",
        "scope": ".",
        "display": "standalone",
        "orientation": "portrait-primary",
        "theme_color": THEME_COLOUR,
        "background_color": BACKGROUND_COLOUR,
        "categories": ["productivity", "utilities"],
        "icons": icons,
    }


def head_injection_script() -> Optional[str]:
    """Return the HTML component that installs the manifest into ``<head>``."""
    manifest = build_manifest()
    if manifest is None:
        return None

    apple_icon = _icon_data_uri("icon-192.png") or ""
    manifest_uri = (
        "data:application/manifest+json;base64,"
        + base64.b64encode(
            json.dumps(manifest, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
    )

    tags = [
        {"tag": "link", "rel": "manifest", "href": manifest_uri},
        {"tag": "link", "rel": "apple-touch-icon", "href": apple_icon},
        {"tag": "link", "rel": "icon", "type": "image/png", "href": apple_icon},
        {"tag": "meta", "name": "theme-color", "content": THEME_COLOUR},
        {"tag": "meta", "name": "apple-mobile-web-app-capable", "content": "yes"},
        {"tag": "meta", "name": "mobile-web-app-capable", "content": "yes"},
        {"tag": "meta", "name": "apple-mobile-web-app-title", "content": APP_SHORT_NAME},
        {
            "tag": "meta",
            "name": "apple-mobile-web-app-status-bar-style",
            "content": "black-translucent",
        },
    ]

    return f"""
<script>
(function () {{
  try {{
    var doc = window.parent && window.parent.document;
    if (!doc || doc.getElementById('greenroot-pwa')) return;
    var marker = doc.createElement('meta');
    marker.id = 'greenroot-pwa';
    doc.head.appendChild(marker);

    var specs = {json.dumps(tags)};
    specs.forEach(function (spec) {{
      var el = doc.createElement(spec.tag);
      Object.keys(spec).forEach(function (key) {{
        if (key !== 'tag') el.setAttribute(key, spec[key]);
      }});
      doc.head.appendChild(el);
    }});

    var title = doc.querySelector('title');
    if (title) title.textContent = {json.dumps(APP_NAME)};
  }} catch (err) {{
    /* Sandbox tightened, or head unavailable: the app is simply not
       installable. Never let this break the page. */
  }}
}})();
</script>
"""


def install(st_module) -> bool:
    """Inject the manifest into the running Streamlit page.

    Parameters
    ----------
    st_module:
        The imported ``streamlit`` module, passed in so this stays importable
        (and testable) without Streamlit present.

    Returns
    -------
    bool
        ``True`` when the injection component was rendered.
    """
    script = head_injection_script()
    if script is None:
        return False
    try:
        import streamlit.components.v1 as components

        components.html(script, height=0, width=0)
        return True
    except Exception:  # noqa: BLE001 - cosmetic; must never break the app.
        return False


__all__ = [
    "APP_NAME",
    "APP_SHORT_NAME",
    "THEME_COLOUR",
    "ASSETS_DIR",
    "build_manifest",
    "head_injection_script",
    "install",
]
