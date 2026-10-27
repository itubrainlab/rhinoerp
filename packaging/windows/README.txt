Rhino ERP for Windows
=====================

Offline ERP analysis of EEG recordings (BDF/EDF), e.g. from Rhino
Recorder and the g.tec Unicorn Hybrid Black.

Needs
-----
- 64-bit Windows 10 or 11.
- An internet connection the first time: Rhino ERP then downloads
  g.Pype, MNE and the libraries it needs from PyPI (several hundred MB).

Install
-------
Unpack the zip, then double-click install.cmd.

This copies Rhino ERP to %LOCALAPPDATA%\RhinoERP, adds it to the Start
menu with its icon, and sets it up. No administrator rights are needed.
To run it without installing, double-click "Rhino ERP.cmd" in this
folder.

Remove it with uninstall.cmd.

Licence
-------
Rhino ERP is MIT licensed (LICENSE). g.Pype, which the first launch
installs from PyPI, is under the g.tec Non-Commercial License: free for
private and educational use with a g.tec amplifier; commercial use needs
a licence from g.tec. This package does not contain g.Pype.
