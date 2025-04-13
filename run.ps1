# Script to run Streamlit app: another_text2python_beta_v2.py

# Define paths
$venvPath = "$env:USERPROFILE\venv"  # Replace with your virtual environment path, e.g., C:\Users\YourUser\venv
$appPath = ".\another_text2python_beta_v2.py"  # Path to your Python script
# $appPath = ".\another_text2python.py"  # Path to your Python script

# Check if virtual environment exists
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment at $venvPath..."
    & "$venvPath\Scripts\Activate.ps1"
}
else {
    Write-Host "Virtual environment not found at $venvPath. Using system Python."
}

# Check if Python is installed
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed. Please install Python."
    exit 1
}

# Check if the Python script exists
if (-not (Test-Path $appPath)) {
    Write-Host "Error: Python script $appPath not found."
    exit 1
}

# Run the Streamlit app
Write-Host "Starting Streamlit app..."
python -m streamlit run $appPath

# Check if the command was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "Streamlit app started successfully."
}
else {
    Write-Host "Error: Failed to start Streamlit app."
    exit 1
}