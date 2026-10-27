"""Start Rhino ERP from its unpacked (or installed) folder, Windows.

    pythonw.exe -I launcher.py [rhino-erp options]

The folder carries its own Python and Rhino ERP, but not g.Pype: the
g.tec Non-Commercial License does not allow g.Pype to be redistributed in
a bundle. On first launch, and after an update, bootstrap.py installs
g.Pype and the other libraries from PyPI into
%LOCALAPPDATA%\\RhinoERP\\runtime.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = (HERE / "build-id").read_text().strip()
LIB = (Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
       / "RhinoERP" / "runtime" / BUILD)

if "--version" in sys.argv[1:]:  # answers without setting anything up
    if sys.stdout:
        print(f"Rhino ERP {BUILD.split('-')[0]}")
    sys.exit(0)

if not (LIB / ".complete").exists():
    setup = subprocess.run([sys.executable, "-I", str(HERE / "bootstrap.py"),
                            str(HERE), str(LIB)])
    if setup.returncode != 0:
        sys.exit(1)

# -I (on the command line): ignore the environment and the user's
# site-packages; the libraries come from LIB alone.
sys.path.insert(0, str(LIB))
sys.argv[0] = "rhino-erp"
from rhino_erp.app import main  # noqa: E402

sys.exit(main())
