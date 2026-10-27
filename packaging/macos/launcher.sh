#!/bin/bash
# Rhino ERP.app's executable.
#
# The app carries its own Python and Rhino ERP, but not g.Pype: the g.tec
# Non-Commercial License does not allow g.Pype to be redistributed in a
# bundle. On first launch, and after an update, bootstrap.py installs
# g.Pype and the other libraries from PyPI into
# ~/Library/Application Support/RhinoERP.
set -u

CONTENTS="$(cd "$(dirname "$0")/.." && pwd)"
RES="$CONTENTS/Resources"
PYTHON="$RES/python/bin/python3"
BUILD="$(cat "$RES/build-id")"
LIB="$HOME/Library/Application Support/RhinoERP/runtime/$BUILD"

# --version answers without setting anything up.
for arg in "$@"; do
    if [ "$arg" = "--version" ]; then
        echo "Rhino ERP ${BUILD%%-*}"
        exit 0
    fi
done

if [ ! -f "$LIB/.complete" ]; then
    "$PYTHON" -I -B "$RES/bootstrap.py" "$RES" "$LIB" || exit 1
fi

# -I: ignore the environment and the user's site-packages; -B: write no
# bytecode into the signed bundle. The libraries come from $LIB alone.
exec "$PYTHON" -I -B -c '
import sys
sys.path.insert(0, sys.argv.pop(1))
sys.argv[0] = "rhino-erp"
from rhino_erp.app import main
sys.exit(main())
' "$LIB" "$@"
