@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_studio.ps1" -Qwen %*
if errorlevel 1 pause
