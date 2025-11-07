@echo off
setlocal EnableDelayedExpansion
goto :main

:log_info
echo [INFO] %*
exit /b 0
:log_success
echo [SUCCESS] %*
exit /b 0
:log_warn
echo [WARN] %*
exit /b 0
:log_error
echo [ERROR] %*
exit /b 1

:main
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." || ( call :log_error "Cannot change to project directory." & exit /b 1 )
set "PROJECT_DIR=%CD%"

echo === QUPATH INFERENCE SETUP ===

where curl >nul 2>&1
if %ERRORLEVEL% NEQ 0 ( call :log_error "curl not found." & exit /b 1 )

call :log_info "Step 1: Checking for 'uv'..."
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    call :log_warn "'uv' not found. Installing..."
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if !ERRORLEVEL! NEQ 0 ( call :log_error "Failed to install uv." & exit /b 1 )
    set "PATH=%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\bin;%APPDATA%\bin;!PATH!"
)

where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 ( call :log_error "'uv' not on PATH after install." & exit /b 1 )

for /f "delims=" %%v in ('uv --version 2^>nul') do set "UV_VERSION=%%v"
call :log_success "uv verification successful. Version: !UV_VERSION!"

call :log_info "Step 2: Setting up Python project..."
if not exist ".venv" (
    call :log_info "Creating virtual environment (.venv)..."
    uv venv >nul
    if !ERRORLEVEL! NEQ 0 ( call :log_error "Failed to create venv." & exit /b 1 )
    call :log_success "Virtual environment created."
) else (
    call :log_info "Existing .venv found."
)

set "VIRTUAL_ENV=%PROJECT_DIR%\.venv"
set "PATH=!VIRTUAL_ENV!\Scripts;!PATH!"

call :log_info "Looking for dependency files..."
if exist "pyproject.toml" (
    call :log_info "Found pyproject.toml. Installing with 'uv sync'..."
    uv sync
    if !ERRORLEVEL! NEQ 0 ( call :log_error "uv sync failed." & exit /b 1 )
    call :log_success "Dependencies installed via pyproject."
) else if exist "requirements.txt" (
    call :log_info "Found requirements.txt. Installing into venv..."
    uv pip install -r requirements.txt -q
    if !ERRORLEVEL! NEQ 0 ( call :log_error "uv pip install failed." & exit /b 1 )
    call :log_success "Dependencies from requirements.txt installed."
) else (
    call :log_error "No pyproject.toml or requirements.txt found." & exit /b 1
)

call :log_success "Project setup complete."
echo.

popd
endlocal
exit /b 0