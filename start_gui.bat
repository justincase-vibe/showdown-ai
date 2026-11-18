@echo off
rem P4.3 — launch the Showdown AI training console (http://127.0.0.1:8700).
rem The trainer itself runs as a DETACHED subprocess: closing this window
rem never stops training (stop it from the dashboard, or via the run files).
rem D80: run from the folder that contains THIS .bat (= the project root),
rem matching the convention of the other root launchers (%~dp0, not %~dp0..).
cd /d "%~dp0"
python -m gui
pause
