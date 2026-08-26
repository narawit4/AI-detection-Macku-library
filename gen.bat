@echo off
setlocal
cd /d "%~dp0"

if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help

if not exist build-output mkdir build-output
python -m pip install -r requirements.txt Nuitka ordered-set zstandard
if errorlevel 1 exit /b %errorlevel%
python -m py_compile main.py ui.py motion.py ai_targeting.py ai_detection.py ai_capture.py ai_service.py makcu_service.py hotkeys.py settings.py liquid_widgets.py
if errorlevel 1 exit /b %errorlevel%
python -m unittest discover -s tests -v
if errorlevel 1 exit /b %errorlevel%
python -c "import makcu, onnxruntime, dxcam, numpy"
if errorlevel 1 exit /b %errorlevel%
python -m nuitka --onefile --mingw64 --assume-yes-for-downloads --progress-bar=none --windows-console-mode=disable --enable-plugin=tk-inter --include-data-dir=models=models --output-filename=Jitter.exe --output-dir=build-output main.py > build-output\build.log 2>&1
if errorlevel 1 (
    type build-output\build.log
    exit /b %errorlevel%
)
echo Build complete: build-output\Jitter.exe
exit /b 0

:help
echo Builds build-output\Jitter.exe on explicit request.
echo Normal development runs python main.py; this script installs packaging tools and runs Nuitka.
echo Packaging review: compiles ai_capture.py, ai_detection.py, ai_targeting.py, and ai_service.py.
echo Bundled AI resource: models\all_games_320.onnx.
exit /b 0
