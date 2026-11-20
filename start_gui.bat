@echo off
rem Launch the Showdown AI **Mission Control** GUI (native desktop, tkinter).
rem The trainer itself runs as a DETACHED subprocess: closing this window
rem never stops training (stop it from the GUI, or via the run files).
rem Run from the folder that contains THIS .bat (= the project root).
cd /d "%~dp0"
python -m gui.desktop
pause
