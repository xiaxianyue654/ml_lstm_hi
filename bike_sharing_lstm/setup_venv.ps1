param(
    [string]$PythonVersion = "3.10",
    [string]$VenvName = ".venv",
    [switch]$RunTraining
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"
$VenvPath = Join-Path $ProjectRoot $VenvName
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

if (-not (Test-Path $RequirementsPath)) {
    throw "requirements.txt not found at: $RequirementsPath"
}

function Get-PythonCommand {
    param(
        [string]$PreferredVersion
    )

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        $versionCheck = & py "-$PreferredVersion" -c "import sys; print(sys.version)"
        if ($LASTEXITCODE -eq 0) {
            return @("py", "-$PreferredVersion")
        }
    }

    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return @("python")
    }

    throw "No usable Python interpreter found. Install Python $PreferredVersion or update PATH."
}

Write-Host "Project root: $ProjectRoot"
Write-Host "Requirements: $RequirementsPath"

$PythonCommand = Get-PythonCommand -PreferredVersion $PythonVersion

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment at $VenvPath ..."
    if ($PythonCommand.Length -gt 1) {
        & $PythonCommand[0] @($PythonCommand[1..($PythonCommand.Length - 1)]) -m venv $VenvPath
    } else {
        & $PythonCommand[0] -m venv $VenvPath
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment."
    }
} else {
    Write-Host "Virtual environment already exists: $VenvPath"
}

Write-Host "Upgrading pip/setuptools/wheel ..."
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip tools."
}

Write-Host "Installing dependencies from requirements.txt ..."
& $VenvPython -m pip install -r $RequirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install project dependencies."
}

Write-Host ""
Write-Host "Virtual environment is ready."
Write-Host "Activate with:"
Write-Host "  $ActivateScript"
Write-Host ""
Write-Host "Train with:"
Write-Host "  $VenvPython main.py --mode all"

if ($RunTraining) {
    Write-Host ""
    Write-Host "Running training pipeline ..."
    Push-Location $ProjectRoot
    try {
        & $VenvPython "main.py" "--mode" "all"
        if ($LASTEXITCODE -ne 0) {
            throw "Training pipeline failed."
        }
    } finally {
        Pop-Location
    }
}
