#!/bin/bash
# Build the Linux x86_64 package: dist/RhinoERP-<version>-linux-x86_64.tar.gz
#
#     packaging/linux/build_linux.sh
#
# Runs on macOS or Linux; it only downloads and arranges files. Needs uv,
# git and the project's .venv (pip install -e ".[test]").
#
# The package carries its own Python (a relocatable python-build-standalone
# build, which uv downloads) and the Rhino ERP wheel, but not g.Pype (see
# ../common.sh): the first launch installs g.Pype and the other pinned
# libraries from PyPI (../bootstrap.py). PySide6's Linux wheels need
# glibc 2.34, hence manylinux_2_34.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PLATFORM="x86_64-manylinux_2_34"
SUFFIX="linux-x86_64"
. "$HERE/../common.sh"

NAME="RhinoERP-$VERSION-linux-x86_64"
DIR="$WORK/$NAME"

uv python install --quiet --install-dir "$WORK/py" cpython-3.12-linux-x86_64-gnu
PYTHON_HOME="$(find "$WORK/py" -maxdepth 1 -type d -name 'cpython-3.12.*-linux-x86_64-gnu' | head -1)"
[ -n "$PYTHON_HOME" ] || { echo "uv did not install a Linux CPython 3.12" >&2; exit 1; }

mkdir -p "$DIR/wheels"
cp -R "$PYTHON_HOME" "$DIR/python"
rm -rf "$DIR/python/include" "$DIR/python/share" \
    "$DIR/python/lib/python3.12/idlelib" "$DIR/python/lib/python3.12/turtledemo"
cp "$WORK"/wheels/*.whl "$DIR/wheels/"
cp "$WORK/requirements.txt" "$DIR/requirements.txt"
cp "$HERE/../bootstrap.py" "$DIR/bootstrap.py"
cp "$ROOT/LICENSE" "$HERE/README.txt" "$DIR/"
echo "$BUILD_ID" > "$DIR/build-id"
install -m 755 "$HERE/rhino-erp" "$HERE/install.sh" "$HERE/uninstall.sh" "$DIR/"
cp "$ROOT/rhino_erp/resources/rhinoerp.svg" "$DIR/rhino-erp.svg"
render_icon 256 "$DIR/rhino-erp.png"

mkdir -p "$OUT"
TARBALL="$OUT/$NAME.tar.gz"
COPYFILE_DISABLE=1 tar -C "$WORK" -czf "$TARBALL" "$NAME"
echo "Built:"
du -sh "$TARBALL"
