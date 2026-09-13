#!/usr/bin/env bash
# Serve GREENROOT to a phone on the same Wi-Fi network (macOS / Linux).
# The phone and this computer must be on the same network.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if [ ! -f models/stacking_model.pkl ]; then
  echo "[ERROR] models/stacking_model.pkl is missing." >&2
  exit 1
fi

if ! "$PYTHON" -c "import streamlit, sklearn, shap, lime, fpdf" >/dev/null 2>&1; then
  echo "Installing dependencies…"
  "$PYTHON" -m pip install -r requirements.txt
fi

# Find a private-range IPv4 for this machine.
lan_ip() {
  # Try each source in turn; `ip addr` reports nothing on some hosts.
  if command -v ipconfig >/dev/null 2>&1; then        # macOS
    ipconfig getifaddr en0 2>/dev/null && return
    ipconfig getifaddr en1 2>/dev/null && return
  fi
  if command -v ip >/dev/null 2>&1; then
    ip -4 addr show scope global 2>/dev/null \
      | awk '/inet /{sub(/\/.*/, "", $2); print $2; exit}' | grep . && return
  fi
  if command -v hostname >/dev/null 2>&1; then
    hostname -I 2>/dev/null | awk '{print $1}' | grep . && return
  fi
  return 1
}

IP="$(lan_ip || true)"
echo
echo "=========================================================="
echo "  GREENROOT — Phone Access"
echo "=========================================================="
if [ -n "${IP:-}" ]; then
  echo
  echo "  On your phone's browser, open:"
  echo
  echo "      http://${IP}:8501"
  echo
  echo "  Both devices must be on the same Wi-Fi."
else
  echo "  Could not detect this machine's Wi-Fi address."
  echo "  Find it with 'ifconfig' or 'ip addr', then open"
  echo "  http://YOUR-IP:8501 on the phone."
fi
echo
echo "  Keep this window open. Press Ctrl+C to stop."
echo

# enableCORS / enableXsrfProtection must both be off, or Streamlit refuses
# the browser's WebSocket from any address other than localhost and the phone
# sees a permanently blank page. This drops a browser-origin check, so use it
# only on a network you trust — anyone on the same Wi-Fi can open the app.
exec "$PYTHON" -m streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
