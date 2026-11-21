@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Showdown AI - Install Dependencies

rem === Run from the folder that contains THIS .bat ===
cd /d "%~dp0"

if not exist "bot\requirements.txt" (
    echo [ERROR] bot\requirements.txt not found in: %CD%
    echo         Make sure the whole project folder is here.
    pause
    exit /b 1
)

rem --- Pick a Python (same logic as Launcher.bat) ---
set "PYEXE="
set "PYARGS="
call :pick "python" ""
call :pick "py" "-3"
call :pick "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" ""
call :pick "%LOCALAPPDATA%\Programs\Python\Python315\python.exe" ""

if not defined PYEXE (
    echo [ERROR] No working Python found. Install from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH", then run this again.
    pause
    exit /b 1
)

echo [i] Using Python : %PYEXE% %PYARGS%
echo.

rem --- torch: must come from the official index on Python 3.15 ---
"%PYEXE%" -c "import torch" >nul 2>nul
if errorlevel 1 (
    echo [i] torch is not installed yet.
    echo.
    echo     Which torch build do you want?
    echo       1  = NVIDIA GPU   pick this if you have an NVIDIA graphics card
    echo       2  = CPU only     works on any machine, just slower
    set /p C="     Choice 2 if you are not sure [2]: "
    set "IDX="
    if "!C!"=="1" set "IDX=https://download.pytorch.org/whl/cu126"
    if not defined IDX set "IDX=https://download.pytorch.org/whl/cpu"
    echo.
    echo [i] Installing torch from: !IDX!
    echo     This is a large download, please wait...
    "%PYEXE%" -m pip install torch --index-url "!IDX!"
    if errorlevel 1 (
        echo.
        echo [ERROR] torch install failed.
        echo         If you do NOT have an NVIDIA GPU, rerun and pick 2 for CPU only.
        pause
        exit /b 1
    )
) else (
    echo [i] torch already installed - skipping.
)

echo.
echo [i] Installing the remaining dependencies from bot\requirements.txt ...
"%PYEXE%" -m pip install -r "bot\requirements.txt"
if errorlevel 1 (
    echo.
    echo [WARN] Some dependencies may not have installed. Run CheckEnv.bat to see which are missing.
    pause
    exit /b 1
)

echo.
echo [i] Verifying imports ...
call :report

echo.
echo [OK] Done. You can now run Launcher.bat
echo.
pause
endlocal & exit /b 0
goto :eof

:report
for %%m in (tkinter torch numpy gymnasium poke_env) do (
    "%PYEXE%" -c "import %%m" >nul 2>nul
    if not errorlevel 1 ( echo   [OK]       %%m ) else ( echo   [MISSING]  %%m )
)
goto :eof

:pick
if not defined PYEXE (
    set "_ex=%~1"
    if exist "%_ex%" (
        set "_ok=1"
    ) else (
        where %_ex% >nul 2>nul && set "_ok=1"
    )
    if defined _ok (
        "%_ex%" %~2 -c "import tkinter" >nul 2>nul
        if not errorlevel 1 (
            set "PYEXE=%~1"
            set "PYARGS=%~2"
        )
    )
)
goto :eof
