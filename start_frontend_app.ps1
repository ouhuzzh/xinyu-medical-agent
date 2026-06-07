param(
    [int]$ApiPort = 8000,
    [int]$FrontendPort = 5173,
    [int]$MockHospitalPort = 8001,
    [switch]$SkipInstall,
    [switch]$SkipMockHospital,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimeDir = Join-Path $Root "runtime"
$FrontendDir = Join-Path $Root "frontend"
$ScriptsDir = Join-Path $Root "scripts"
$PythonExe = Join-Path $Root "venv\Scripts\python.exe"

New-Item -ItemType Directory -Force $RuntimeDir | Out-Null

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment not found: $PythonExe. Please create venv and install requirements first."
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm is not available. Please install Node.js/npm first."
}

if (-not $SkipInstall -and -not (Test-Path (Join-Path $FrontendDir "node_modules"))) {
    Write-Host "Installing frontend dependencies..."
    Push-Location $FrontendDir
    try {
        npm install
    } finally {
        Pop-Location
    }
}

$apiLog = Join-Path $RuntimeDir "api_server.log"
$apiErr = Join-Path $RuntimeDir "api_server.err.log"
$frontendLog = Join-Path $RuntimeDir "frontend_server.log"
$frontendErr = Join-Path $RuntimeDir "frontend_server.err.log"
$mockHospitalLog = Join-Path $RuntimeDir "mock_hospital.log"
$mockHospitalErr = Join-Path $RuntimeDir "mock_hospital.err.log"

function Get-ListeningProcessId {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) { return [int]$conn.OwningProcess }
    return $null
}

function Stop-PortProcess {
    param([int]$Port, [string]$Name)
    $processId = Get-ListeningProcessId -Port $Port
    if ($processId) {
        Write-Host "Stopping $Name on port $Port (PID $processId) ..."
        Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

if ($Restart) {
    Stop-PortProcess -Port $ApiPort -Name "FastAPI backend"
    Stop-PortProcess -Port $FrontendPort -Name "React frontend"
    if (-not $SkipMockHospital) {
        Stop-PortProcess -Port $MockHospitalPort -Name "Mock hospital MCP"
    }
}

# --- Mock Hospital MCP Server (optional) ---
if (-not $SkipMockHospital) {
    $existingMock = Get-ListeningProcessId -Port $MockHospitalPort
    if (-not $existingMock) {
        $mockScript = Join-Path $ScriptsDir "mock_hospital_mcp_server.py"
        if (Test-Path $mockScript) {
            Write-Host "Starting Mock Hospital MCP on http://127.0.0.1:$MockHospitalPort/mcp ..."
            Start-Process `
                -FilePath $PythonExe `
                -ArgumentList "scripts\mock_hospital_mcp_server.py", "--port", $MockHospitalPort `
                -WorkingDirectory $Root `
                -RedirectStandardOutput $mockHospitalLog `
                -RedirectStandardError $mockHospitalErr `
                -WindowStyle Hidden
            Start-Sleep -Seconds 2

            # Seed mock hospital into DB (idempotent — safe to re-run)
            $seedScript = Join-Path $ScriptsDir "seed_mock_hospital.py"
            if (Test-Path $seedScript) {
                Write-Host "Seeding mock hospital registry ..."
                & $PythonExe $seedScript 2>&1 | ForEach-Object { Write-Host "  $_" }
            }
        } else {
            Write-Host "Mock hospital script not found at $mockScript, skipping." -ForegroundColor Yellow
        }
    } else {
        Write-Host "Mock hospital already running on port $MockHospitalPort."
    }
}

# --- FastAPI Backend ---
$existingApi = Get-ListeningProcessId -Port $ApiPort
if (-not $existingApi) {
    Write-Host "Starting FastAPI backend on http://127.0.0.1:$ApiPort ..."
    Start-Process `
        -FilePath $PythonExe `
        -ArgumentList "project\api_app.py" `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $apiLog `
        -RedirectStandardError $apiErr `
        -WindowStyle Hidden
} else {
    Write-Host "FastAPI backend already appears to be running on port $ApiPort."
}

# --- React Frontend ---
$existingFrontend = Get-ListeningProcessId -Port $FrontendPort
if (-not $existingFrontend) {
    Write-Host "Starting React frontend on http://127.0.0.1:$FrontendPort ..."
    Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList "run dev -- --host 127.0.0.1 --port $FrontendPort" `
        -WorkingDirectory $FrontendDir `
        -RedirectStandardOutput $frontendLog `
        -RedirectStandardError $frontendErr `
        -WindowStyle Hidden
} else {
    Write-Host "React frontend already appears to be running on port $FrontendPort."
}

Start-Sleep -Seconds 3

Write-Host ""
Write-Host "User frontend:   http://127.0.0.1:$FrontendPort"
Write-Host "API docs:        http://127.0.0.1:$ApiPort/docs"
Write-Host "API health:      http://127.0.0.1:$ApiPort/api/health"
if (-not $SkipMockHospital) {
    Write-Host "Mock hospital:   http://127.0.0.1:$MockHospitalPort/mcp"
}
Write-Host ""
Write-Host "Logs:"
Write-Host "  $apiLog"
Write-Host "  $apiErr"
Write-Host "  $frontendLog"
Write-Host "  $frontendErr"
if (-not $SkipMockHospital) {
    Write-Host "  $mockHospitalLog"
}
