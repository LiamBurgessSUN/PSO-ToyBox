@echo off
REM Script to clear the Figures directory before running benchmarks

echo 🧹 Clearing Figures directory...

REM Get the directory path from Python config
for /f "delims=" %%i in ('python -c "from SAPSO_AGENT.CONFIG import CHECKPOINT_BASE_DIR; print(CHECKPOINT_BASE_DIR)"') do set FIGURES_DIR=%%i

if exist "%FIGURES_DIR%" (
    echo 🗑️  Removing existing Figures directory: %FIGURES_DIR%
    rmdir /s /q "%FIGURES_DIR%"
    if %errorlevel% equ 0 (
        echo ✅ Successfully removed existing Figures directory
    ) else (
        echo ❌ Failed to remove Figures directory
        exit /b 1
    )
) else (
    echo ℹ️  Figures directory does not exist: %FIGURES_DIR%
)

REM Create the directory
echo 📁 Creating Figures directory: %FIGURES_DIR%
mkdir "%FIGURES_DIR%"
if %errorlevel% equ 0 (
    echo ✅ Successfully created Figures directory
) else (
    echo ❌ Failed to create Figures directory
    exit /b 1
)

echo ✅ Figures directory cleared successfully! 