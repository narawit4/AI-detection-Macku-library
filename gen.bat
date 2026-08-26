@echo off
setlocal
cd /d "%~dp0"

set "JITTER_PACKAGE_COMPILE_TARGETS=main.py ui.py motion.py ai_targeting.py ai_detection.py ai_capture.py ai_service.py makcu_service.py hotkeys.py settings.py liquid_widgets.py distribution_metadata.py"
set "JITTER_PACKAGE_RUNTIME_IMPORTS=makcu onnxruntime dxcam numpy"
set "JITTER_PACKAGE_NUITKA_DATA_OPTIONS=--include-data-dir=models=models --include-data-dir=licenses=licenses"
set "JITTER_PACKAGE_RELEASE_MATERIALS=LICENSE THIRD_PARTY_NOTICES.md licenses"

if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help
if /i "%~1"=="--review-json" goto :review
if not "%~1"=="" goto :usage

if not exist build-output mkdir build-output
python -m pip install -r requirements.txt Nuitka ordered-set zstandard
if errorlevel 1 exit /b %errorlevel%
python -m py_compile %JITTER_PACKAGE_COMPILE_TARGETS%
if errorlevel 1 exit /b %errorlevel%
python -m unittest discover -s tests -v
if errorlevel 1 exit /b %errorlevel%
python -c "import %JITTER_PACKAGE_RUNTIME_IMPORTS: =, %"
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
echo Unknown option: %~1 1>&2
echo Use --help or --review-json. 1>&2
exit /b 2
