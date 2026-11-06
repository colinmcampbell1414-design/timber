# Build the TIMBER installer (dist\installer\TIMBER-Setup-<version>.exe).
#
#   .\build_installer.ps1
#
# Needs Inno Setup 6 (free). Install it once with:
#   winget install JRSoftware.InnoSetup
# This script builds TIMBER.exe first if dist\TIMBER doesn't exist yet.

$ErrorActionPreference = "Stop"
if (-not (Test-Path (Join-Path $PSScriptRoot "dist\TIMBER\TIMBER.exe"))) {
    & (Join-Path $PSScriptRoot "build_exe.ps1")
}
$iscc = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    throw "Inno Setup 6 is not installed. Install it with:  winget install JRSoftware.InnoSetup"
}
& $iscc (Join-Path $PSScriptRoot "installer\TIMBER.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed." }
Write-Host "Done: $(Join-Path $PSScriptRoot 'dist\installer')"
