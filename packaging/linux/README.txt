Rhino ERP for Linux
===================

Offline ERP analysis of EEG recordings (BDF/EDF), e.g. from Rhino
Recorder and the g.tec Unicorn Hybrid Black.

Needs
-----
- An x86_64 PC with glibc 2.34 or newer (Ubuntu 22.04, Debian 12,
  Fedora 35 or later). g.Pype has no build for ARM Linux.
- An internet connection the first time: Rhino ERP then downloads
  g.Pype, MNE and the libraries it needs from PyPI (several hundred MB).
- Qt's usual desktop libraries. On Debian or Ubuntu, if the window does
  not open:  sudo apt install libxcb-cursor0 libegl1

Install
-------
    ./install.sh

This copies Rhino ERP to ~/.local/share/RhinoERP, adds it to the
applications menu with its icon, adds a rhino-erp command to
~/.local/bin, and sets it up. No administrator rights are needed. To run
it without installing, start ./rhino-erp from this folder; pass a
recording to open it directly: ./rhino-erp recording.bdf

Remove it with ./uninstall.sh.

Licence
-------
Rhino ERP is MIT licensed (LICENSE). g.Pype, which the first launch
installs from PyPI, is under the g.tec Non-Commercial License: free for
private and educational use with a g.tec amplifier; commercial use needs
a licence from g.tec. This package does not contain g.Pype.
