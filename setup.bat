@echo off
setlocal

set VENV_DIR=.venv
set PYTHON_EXE=python
set PYTHON_VERSION_PREF=python3.9.exe

echo --- Starting Windows Project Setup (Targeting Python 3.9) ---

:: 1. Try to find preferred Python version, then fall back to generic 'python'
where %PYTHON_VERSION_PREF% >NUL 2>&1

:: If errorlevel is 0, the specific python version was found
if %ERRORLEVEL% equ 0 (
    set PYTHON_EXE=%PYTHON_VERSION_PREF%
    echo Found specific Python: %PYTHON_EXE%
) else (
    :: Fallback - the check in bootstrap.py will enforce 3.9 if this fails
    echo Warning: Specific version "%PYTHON_VERSION_PREF%" not found. Falling back to generic 'python'.
    echo This will rely on the global 'python' being 3.9 or the user running 'python3.9 setup.bat'.
)

:: 2. Check if the found Python is available
%PYTHON_EXE% --version >NUL 2>&1
if errorlevel 1 (
    
    echo ERROR: Python executable "%PYTHON_EXE%" is not found in your system's PATH.
    pause
    exit /b 1
)

:: 3. Create Virtual Environment
if exist %VENV_DIR% (
    
    echo Directory "%VENV_DIR%" already exists. Skipping venv creation.
) else (
    
    echo 1. Creating virtual environment in "%VENV_DIR%"
    :: Use the determined executable to create the venv
    %PYTHON_EXE% -m venv %VENV_DIR%
)

:: Define the path to the VENV Python executable
set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe

:: 4. Install the project in editable mode

echo 2. Installing/Upgrading dependencies in venv
:: Use CALL to ensure the batch script resumes after executing the Python/pip command
call %VENV_PYTHON% -m pip install -U pip
call %VENV_PYTHON% -m pip install -e .

:: 5. Download Datasets (Example) - Assumes a download_data.py file
if exist download_data.py (
    
    echo 3. Running data download script 
    call %VENV_PYTHON% download_data.py
)


echo --- Setup Complete! ---
echo To activate the environment and start working, run:
echo call .\%VENV_DIR%\Scripts\activate

pause

endlocal