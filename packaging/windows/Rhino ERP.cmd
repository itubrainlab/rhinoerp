@echo off
rem Start Rhino ERP from this folder without installing it.
start "" "%~dp0python\pythonw.exe" -I "%~dp0launcher.py" %*
