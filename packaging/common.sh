# Shared by the three build scripts; sourced, not run.
#
#     PLATFORM=<uv python platform> SUFFIX=<build id suffix> . packaging/common.sh
#
# Sets ROOT, OUT, DEV_PY, VERSION, BUILD_ID and WORK (a temporary folder,
# removed on exit), and leaves in WORK:
#   wheels/            the Rhino ERP wheel (pure Python, any platform)
#   requirements.txt   its dependencies, pinned to the versions in the
#                      project's .venv, which the tests ran with
#
# Packages never carry g.Pype: the g.tec Non-Commercial License forbids
# redistributing it as a standalone executable or other bundled artefact.
# requirements.txt lists it, and bootstrap.py installs it from PyPI on
# first launch.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$ROOT/dist}"
DEV_PY="$ROOT/.venv/bin/python"

[ -x "$DEV_PY" ] || { echo "Run pip install -e '.[test]' into .venv first." >&2; exit 1; }

VERSION="$("$DEV_PY" -c 'import rhino_erp; print(rhino_erp.__version__)')"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
git -C "$ROOT" diff --quiet HEAD -- rhino_erp pyproject.toml packaging || COMMIT="$COMMIT-dirty"
BUILD_ID="$VERSION-$COMMIT-$SUFFIX"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo "Building Rhino ERP $BUILD_ID"

mkdir -p "$WORK/wheels"
uv build --quiet --wheel --out-dir "$WORK/wheels" "$ROOT"

"$DEV_PY" - "$ROOT/pyproject.toml" > "$WORK/requirements.in" <<'PY'
import sys, tomllib
for dep in tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]:
    print(dep)
PY
uv pip freeze --python "$DEV_PY" --exclude-editable > "$WORK/constraints.txt"
uv pip compile --quiet --python-version 3.12 --python-platform "$PLATFORM" \
    --constraint "$WORK/constraints.txt" --no-header --no-annotate \
    "$WORK/requirements.in" -o "$WORK/requirements.txt"

# render_icon <size> <out> [format]: the app icon as a PNG or ICO.
render_icon() {
    QT_QPA_PLATFORM=offscreen "$DEV_PY" -c '
import sys
from PySide6.QtGui import QGuiApplication, QIcon
app = QGuiApplication([])
size = int(sys.argv[2])
QIcon(sys.argv[1]).pixmap(size, size).save(sys.argv[3], *sys.argv[4:])
' "$ROOT/rhino_erp/resources/rhinoerp.svg" "$@"
}
