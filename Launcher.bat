@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Showdown AI - Launcher

rem === Always run from the folder that contains THIS .bat (location independent) ===
cd /d "%~dp0"

if not exist "gui\desktop\app.py" (
    echo [ERROR] gui\desktop\app.py (the Mission Control GUI) was not found in this folder:
    echo         %CD%
    echo         Make sure the whole project folder is here, including the gui\ subfolder.
    echo.
    pause
    exit /b 1
)

rem === Pick the first Python that can actually launch the GUI (has tkinter) ===
rem Order: python on PATH, then the py launcher, then the known-good per-user 3.12.
set "PYEXE="
set "PYARGS="
call :pick "python" ""
call :pick "py" "-3"
call :pick "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" ""

if not defined PYEXE (
    echo [ERROR] No working Python was found on this PC.
    echo.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" during install, then run this again.
    echo.
    pause
    exit /b 1
)

echo [i] Python used : !PYEXE! !PYARGS!
echo [i] Project dir : %CD%
echo.

"!PYEXE!" !PYARGS! -m gui.desktop
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
    echo [ERROR] Launcher exited with code %RC%. Read the message above for the cause.
    echo         Common cause: the training libraries are missing. Run  python -m pip install -r bot\requirements.txt  with the same Python, then install torch.
    pause
)
endlocal & exit /b %RC%
goto :eof

:pick
rem %~1 = executable name or full path, %~2 = extra args (e.g. -3 for the py launcher)
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
