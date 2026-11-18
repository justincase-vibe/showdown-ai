@echo off
rem P4.3 — launch the Showdown AI training console (http://127.0.0.1:8700).
rem The trainer itself runs as a DETACHED subprocess: closing this window
rem never stops training (stop it from the dashboard, or via the run files).
cd /d "%~dp0.."
python -m gui
pause
