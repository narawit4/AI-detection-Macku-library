@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

python distribution_metadata.py --confirm-build
