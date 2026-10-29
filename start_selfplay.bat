@echo off
rem Launch the 1k-battle GPU self-play with error pop-ups. Works from ANY folder (double-click it).
rem Extra args are passed through, e.g.:  start_selfplay.bat --no-popup
cd /d "C:\Users\JustinCase\Documents\Projects\showdown AI"
"C:\Users\JustinCase\AppData\Local\Programs\Python\Python312\python.exe" "bot\run_gpu_watch.py" %*
echo.
echo Launcher finished. Full run log: bot\run_gpu.log
pause
