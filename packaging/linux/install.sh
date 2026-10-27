#!/bin/sh
# Install Rhino ERP for the current user: copies this folder to
# ~/.local/share/RhinoERP/app, adds a menu entry with its icon and a
# rhino-erp command in ~/.local/bin, then sets it up (downloads g.Pype
# and the libraries from PyPI). No administrator rights needed.
set -eu

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
APP="$DATA/RhinoERP/app"
BIN="$HOME/.local/bin"

case "$(uname -m)" in
    x86_64|amd64) ;;
    *) echo "Rhino ERP needs an x86_64 PC (g.Pype has no build for $(uname -m))." >&2
       exit 1 ;;
esac
glibc="$(ldd --version 2>/dev/null | head -1 | grep -o '[0-9]\+\.[0-9]\+$' || true)"
if [ -n "$glibc" ] && [ "$(printf '%s\n2.34\n' "$glibc" | sort -V | head -1)" != "2.34" ]; then
    echo "Rhino ERP needs glibc 2.34 or newer (this system has $glibc)," >&2
    echo "e.g. Ubuntu 22.04, Debian 12, Fedora 35 or later." >&2
    exit 1
fi

if [ "$HERE" != "$APP" ]; then
    echo "Copying Rhino ERP to $APP"
    rm -rf "$APP"
    mkdir -p "$APP"
    cp -a "$HERE/." "$APP/"
fi

mkdir -p "$BIN" "$DATA/applications" \
    "$DATA/icons/hicolor/scalable/apps" "$DATA/icons/hicolor/256x256/apps"
ln -sf "$APP/rhino-erp" "$BIN/rhino-erp"
cp "$APP/rhino-erp.svg" "$DATA/icons/hicolor/scalable/apps/rhino-erp.svg"
cp "$APP/rhino-erp.png" "$DATA/icons/hicolor/256x256/apps/rhino-erp.png"
cat > "$DATA/applications/rhino-erp.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Rhino ERP
GenericName=ERP Analysis
Comment=Offline ERP analysis of EEG recordings
Exec="$APP/rhino-erp" %f
Icon=rhino-erp
Terminal=false
Categories=Science;Education;MedicalSoftware;
StartupWMClass=rhino-erp
EOF
update-desktop-database "$DATA/applications" >/dev/null 2>&1 || true
gtk-update-icon-cache -q "$DATA/icons/hicolor" >/dev/null 2>&1 || true

echo "Setting up (downloads g.Pype and the libraries; a few minutes)..."
build="$(cat "$APP/build-id")"
"$APP/python/bin/python3" -I "$APP/bootstrap.py" "$APP" "$DATA/RhinoERP/runtime/$build"

echo
echo "Rhino ERP is installed: open it from your applications menu, or run"
echo "rhino-erp (add ~/.local/bin to PATH if the shell cannot find it)."
