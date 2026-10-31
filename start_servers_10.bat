@echo off
rem Start 10 Showdown servers on ports 8000-8009 (minimized windows).
cd /d "C:\Users\JustinCase\Documents\Projects\showdown AI\pokemon-showdown"

echo Starting 10 Showdown servers (8000-8009)...

for /L %%i in (8000,1,8009) do (
    start "PS-%%i" /min cmd /c "set SHOWDOWN_PORT=%%i && node pokemon-showdown"
)

echo.
echo Waiting 15s for servers to come up...
timeout /t 15 /nobreak >nul

echo.
echo === Port check ===
set /a UP=0
for /L %%i in (8000,1,8009) do (
    netstat -an | findstr "0.0.0.0:%%i " | findstr "LISTENING" >nul
    if not errorlevel 1 (
        echo   Port %%i: OK
        set /a UP+=1
    ) else (
        echo   Port %%i: MISSING
    )
)

echo.
if %UP% geq 10 (
    echo All 10 servers UP. Run your training command now.
) else (
    echo WARNING: only %UP%/10 servers detected. Check the minimized PS-800x windows.
)
echo.
pause
