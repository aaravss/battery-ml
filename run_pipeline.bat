@echo off
rem Double-click or run from cmd. Passes options through, e.g.  run_pipeline.bat -Background
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_pipeline.ps1" %*
