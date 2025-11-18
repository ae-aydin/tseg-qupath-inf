@echo off
setlocal EnableDelayedExpansion

if "%~1" == "" (
    echo Usage: %~nx0 ^<log_file_path^> 1>&2
    exit /b 1
)

set "LOG_FILE=%~1"
type nul > "%LOG_FILE%"
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."
set "PROJECT_DIR=%CD%"

call :log_info "Inference setup started."

where curl >nul 2>nul
if %ERRORLEVEL% neq 0 (
    call :log_error "curl not found."
    exit /b 1
)

call :log_info "Checking for 'uv'..."
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    call :log_warn "'uv' not found. Installing..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\uv\bin;%PATH%"
)

where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    call :log_error "'uv' not on PATH after install."
    exit /b 1
)

set "UV_VERSION="
for /f "delims=" %%v in ('uv --version 2^>nul') do set "UV_VERSION=%%v"
if not defined UV_VERSION set "UV_VERSION=unknown"
call :log_success "uv verification successful. Version: !UV_VERSION!"

call :log_info "Setting up Python project..."
if not exist ".venv" (
    call :log_info "Creating virtual environment (.venv)..."
    uv venv -q >nul
    if !ERRORLEVEL! neq 0 (
        call :log_error "Failed to create venv."
        exit /b 1
    )
    call :log_success "Virtual environment created."
) else (
    call :log_info "Existing .venv found."
)

set "VIRTUAL_ENV=%PROJECT_DIR%\.venv"
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"

call :log_info "Looking for dependency files..."
if exist "pyproject.toml" (
    call :log_info "Found pyproject.toml. Installing with 'uv sync'..."
    uv sync -q
    if !ERRORLEVEL! neq 0 (
        call :log_error "Failed to sync dependencies."
        exit /b 1
    )
    call :log_success "Dependencies installed via pyproject."
) else if exist "requirements.txt" (
    call :log_info "Found requirements.txt. Installing into venv..."
    uv add -r requirements.txt -q
    if !ERRORLEVEL! neq 0 (
        call :log_error "Failed to add requirements."
        exit /b 1
    )
    call :log_success "Dependencies from requirements.txt installed."
) else (
    call :log_error "No pyproject.toml or requirements.txt found."
    exit /b 1
)

call :log_success "Project setup complete."
echo.

exit /b 0

:log_output
set "msg=%~1"
echo !msg!
echo !msg! >> "%LOG_FILE%"
exit /b

:log_info
call :log_output "[INFO] %~1"
exit /b

:log_success
call :log_output "[SUCCESS] %~1"
exit /b

:log_warn
call :log_output "[WARN] %~1"
exit /b

:log_error
call :log_output "[ERROR] %~1" 1>&2
exit /b