@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "JITTER_GEN_COMMAND_LINE=!cmdcmdline!"
set "JITTER_GEN_SCRIPT=%~f0"
setlocal DisableDelayedExpansion
cd /d "%~dp0"

set "JITTER_PACKAGE_COMPILE_TARGETS=main.py ui.py motion.py ai_targeting.py ai_detection.py ai_capture.py ai_service.py makcu_service.py hotkeys.py settings.py liquid_widgets.py distribution_metadata.py"
set "JITTER_PACKAGE_RUNTIME_IMPORTS=makcu serial onnxruntime dxcam comtypes numpy"
set "JITTER_PACKAGE_NUITKA_DATA_OPTIONS=--include-data-dir=models=models --include-data-dir=licenses=licenses"
set "JITTER_PACKAGE_RELEASE_MATERIALS=LICENSE THIRD_PARTY_NOTICES.md licenses"

python distribution_metadata.py --classify-gen-invocation
set "JITTER_GEN_ACTION=%errorlevel%"
if "%JITTER_GEN_ACTION%"=="0" goto :build
if "%JITTER_GEN_ACTION%"=="10" goto :help
if "%JITTER_GEN_ACTION%"=="11" goto :review
goto :usage

:build
if not exist build-output mkdir build-output
python -m pip install -r requirements.txt Nuitka ordered-set zstandard
if errorlevel 1 exit /b %errorlevel%
python distribution_metadata.py --review-json >nul
if errorlevel 1 exit /b %errorlevel%
python -m py_compile %JITTER_PACKAGE_COMPILE_TARGETS%
if errorlevel 1 exit /b %errorlevel%
python -m unittest discover -s tests -v
if errorlevel 1 exit /b %errorlevel%
python -c "import %JITTER_PACKAGE_RUNTIME_IMPORTS: =, %"
if errorlevel 1 exit /b %errorlevel%
if exist sound_service.py if exist sound python -c "import pygame"
if errorlevel 1 exit /b %errorlevel%
python -m nuitka --onefile --mingw64 --assume-yes-for-downloads --progress-bar=none --windows-console-mode=disable --enable-plugin=tk-inter %JITTER_PACKAGE_NUITKA_DATA_OPTIONS% --output-filename=Jitter.exe --output-dir=build-output main.py > build-output\build.log 2>&1
if errorlevel 1 (
    type build-output\build.log
    exit /b %errorlevel%
)
python distribution_metadata.py --copy-release-materials build-output
if errorlevel 1 exit /b %errorlevel%
echo Build complete: build-output\Jitter.exe
exit /b 0

:help
echo Builds build-output\Jitter.exe on explicit request.
echo Normal development runs python main.py; this script installs packaging tools and runs Nuitka.
echo The validated packaging configuration follows; no build is started.
python distribution_metadata.py --review-json
exit /b %errorlevel%

:review
python distribution_metadata.py --review-json
exit /b %errorlevel%

:usage
echo Invalid arguments. Use exactly --help or --review-json, or no arguments to build. 1>&2
exit /b 2
