# Build the C VecDot reference binary
$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $scriptDir
try {
    Write-Host "Building vecdot_ref.exe..."
    cl.exe /nologo /O2 /fp:precise /W4 /Fe:vecdot_ref.exe vecdot_ref.c
    if ($LASTEXITCODE -eq 0) {
        Write-Host "vecdot_ref.exe built successfully"
    } else {
        Write-Error "Build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
