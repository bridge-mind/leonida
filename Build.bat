@echo off
title Leonida - Game Builder
rem Double-click to build. Finds your copy of the game, checks what's installed, and builds into
rem Documents\Leonida. Extra options pass through, e.g.  Build.bat --out D:\Games\Leonida
setlocal
cd /d "%~dp0"
echo.
echo  ==========================================
echo       L E O N I D A  -  Game Builder
echo  ==========================================
echo.
set "PY="
for %%V in (3.14 3.13 3.12 3) do (
  if not defined PY (
    py -%%V -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  where python >nul 2>nul && (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && set "PY=python"
  )
)
rem No Python on PATH: look for a per-user install (python.org or winget, "Add to PATH" unticked), else offer to install one.
if not defined PY call :find_user_python
if not defined PY call :offer_python
if not defined PY (
  echo Python 3.12 or newer is needed: https://www.python.org/downloads/
  echo Tick "Add python.exe to PATH" during setup, then run Build.bat again.
  pause
  exit /b 1
)
%PY% -I "%~dp0make_game.py" %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (echo Done.) else (echo The build stopped. Read the message above.)
pause
exit /b %RC%

rem ---- Helpers (only reached by "call"; the script exits above) ----

rem Each per-user install under %LOCALAPPDATA%\Programs\Python, highest folder name first; each one is
rem version-checked, so the name order does not matter. Needs no PATH refresh.
:find_user_python
for /f "delims=" %%D in ('dir /b /ad /o-n "%LOCALAPPDATA%\Programs\Python\Python3*" 2^>nul') do call :try_python "%LOCALAPPDATA%\Programs\Python\%%D\python.exe"
exit /b 0

:try_python
if defined PY exit /b 0
if not exist %1 exit /b 0
%1 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>nul && set PY="%~1"
exit /b 0

rem Ask first, then install Python 3.12 for this user only (no admin): winget if present, else the pinned
rem python.org installer checked against its SHA-256 (also used if winget runs but installs nothing). LEONIDA_PY_INSTALL_DRYRUN=1 prints the command and skips it.
rem On "No" or any failure PY stays unset and the caller prints the manual instructions.
:offer_python
echo Python 3.12 or newer was not found, and the build needs it.
choice /c YN /n /t 60 /d N /m "Download and install Python 3.12 for your user account now? (No admin needed) [Y/N] "
if errorlevel 2 exit /b 0
if not errorlevel 1 exit /b 0
where winget >nul 2>nul && goto :py_winget
:py_download
set "PYURL=https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
set "PYSHA=67B5635E80EA51072B87941312D00EC8927C4DB9BA18938F7AD2D27B328B95FB"
set "PYEXE=%TEMP%\leonida-python-3.12.10-amd64.exe"
if defined LEONIDA_PY_INSTALL_DRYRUN goto :py_dry_download
echo Downloading Python 3.12.10 from python.org ...
powershell -NoProfile -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -UseBasicParsing -Uri $env:PYURL -OutFile $env:PYEXE; if ((Get-FileHash -Algorithm SHA256 -LiteralPath $env:PYEXE).Hash -ine $env:PYSHA) { Remove-Item -LiteralPath $env:PYEXE -Force; exit 3 }"
if errorlevel 1 goto :py_download_failed
echo Installing Python 3.12.10 ...
start "" /wait "%PYEXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1
del /q "%PYEXE%" >nul 2>nul
goto :py_installed

:py_download_failed
echo The download failed or did not match its checksum, so it was not run.
del /q "%PYEXE%" >nul 2>nul
exit /b 0

:py_dry_download
echo [dry run] would download %PYURL%
echo [dry run] would check SHA-256 %PYSHA%
echo [dry run] would run "%PYEXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1
exit /b 0

:py_winget
set "PYWG=winget install --id Python.Python.3.12 -e --scope user --silent --accept-package-agreements --accept-source-agreements"
if defined LEONIDA_PY_INSTALL_DRYRUN goto :py_dry_winget
echo Installing Python 3.12 with winget ...
call %PYWG%
call :find_user_python
if defined PY exit /b 0
echo winget did not install Python; trying the python.org installer instead.
goto :py_download

:py_dry_winget
echo [dry run] would run %PYWG%
goto :py_download

:py_installed
call :find_user_python
exit /b 0
