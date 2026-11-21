@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Showdown AI - Environment Check

cd /d "%~dp0"

echo ===================== Showdown AI - Environment Check =====================
echo Project folder : %CD%
echo.

rem --- Python (same resolution as Launcher.bat) ---
set "PYEXE="
set "PYARGS="
call :pick "python" ""
call :pick "py" "-3"
call :pick "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" ""

if defined PYEXE (
    echo [OK]  Python       : !PYEXE! !PYARGS!
    for /f "delims=" %%v in ('"!PYEXE!" !PYARGS! --version 2^>^&1') do echo        version      : %%v
    echo.
    echo --- Python modules needed for training ---
    for %%m in (tkinter torch numpy gymnasium poke_env) do (
        "!PYEXE!" !PYARGS! -c "import %%m" >nul 2>nul
        if not errorlevel 1 ( echo   [OK]       %%m ) else ( echo   [MISSING]  %%m )
    )
) else (
    echo [MISSING] No working Python found. Install from https://www.python.org/downloads/
    echo            and tick "Add python.exe to PATH" during install.
)

echo.
echo --- Showdown server ---
where node >nul 2>nul
if not errorlevel 1 (
    for /f "delims=" %%v in ('node --version 2^>^&1') do echo [OK]  node %%v
) else (
    echo [MISSING] node not found on PATH - needed to run the Showdown servers
)
if exist "pokemon-showdown\pokemon-showdown" (
    echo [OK]  pokemon-showdown present
) else (
    echo [MISSING] pokemon-showdown\pokemon-showdown not found
)

echo.
echo ============================================================================
echo Done. Copy this whole block and send it back so we can pinpoint the issue.
echo ============================================================================
echo.
pause
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
