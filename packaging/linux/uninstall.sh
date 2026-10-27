#!/bin/sh
# Remove Rhino ERP for the current user.
set -eu
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
rm -f "$HOME/.local/bin/rhino-erp" \
    "$DATA/applications/rhino-erp.desktop" \
    "$DATA/icons/hicolor/scalable/apps/rhino-erp.svg" \
    "$DATA/icons/hicolor/256x256/apps/rhino-erp.png"
rm -rf "$DATA/RhinoERP"
update-desktop-database "$DATA/applications" >/dev/null 2>&1 || true
echo "Rhino ERP is removed."
