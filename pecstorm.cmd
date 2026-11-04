@echo off
rem pecstorm <command> [args] -- single entry point for the project.
rem Keep this file ASCII-only: chcp 65001 must not break batch parsing.
setlocal
chcp 65001 >nul
set "ROOT=%~dp0"
if exist "%ROOT%.venv-rocm\Scripts\python.exe" (
  "%ROOT%.venv-rocm\Scripts\python.exe" "%ROOT%pecstorm.py" %*
) else (
  py "%ROOT%pecstorm.py" %*
)
exit /b %ERRORLEVEL%
