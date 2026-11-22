@echo off
setlocal
pushd "%~dp0"
set "LAB_ENV=%TEMP%\songdo-sumo128-py312"
if exist "%LAB_ENV%\Scripts\python.exe" goto packages
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -m venv "%LAB_ENV%"
    if exist "%LAB_ENV%\Scripts\python.exe" goto packages
)
if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
    "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv "%LAB_ENV%"
) else (
    python -m venv "%LAB_ENV%"
)
if errorlevel 1 goto failed
:packages
"%LAB_ENV%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
"%LAB_ENV%\Scripts\python.exe" run.py --session AM3
if errorlevel 1 goto failed
popd
exit /b 0
:failed
echo Check the error above. Python 3.12 x64 and network access are needed for first-time setup.
popd
pause
exit /b 1

