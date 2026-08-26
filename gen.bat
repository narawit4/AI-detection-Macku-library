@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

set "JITTER_ARG1=%~1"
set "JITTER_ARG2=%~2"
setlocal EnableDelayedExpansion
if not "!JITTER_ARG2!"=="" goto :usage
if "!JITTER_ARG1!"=="" goto :build
if /i "!JITTER_ARG1!"=="--help" goto :help
if /i "!JITTER_ARG1!"=="--review-json" goto :review
goto :usage

:build
python distribution_metadata.py --build
exit /b %errorlevel%

:help
python distribution_metadata.py --describe-build
exit /b %errorlevel%

:review
python distribution_metadata.py --review-json
exit /b %errorlevel%

:usage
echo Invalid arguments. Use exactly --help or --review-json, or no arguments to build. 1>&2
exit /b 2
