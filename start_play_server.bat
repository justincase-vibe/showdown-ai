@echo off
rem One local Showdown server (port 8000) for playing against the NN bot.
cd /d "%~dp0pokemon-showdown"
set SHOWDOWN_PORT=8000
node pokemon-showdown
