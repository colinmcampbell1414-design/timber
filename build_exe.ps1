# Build TIMBER.exe into dist\TIMBER\  (run from the project folder in PowerShell):
#
#   .\build_exe.ps1             build the app (surveys are NOT copied)
#   .\build_exe.ps1 -WithData   also copy every survey folder from data\ next to the exe
#
# The result is a PORTABLE folder: copy the whole dist\TIMBER folder (e.g. to a USB
# stick) and double-click TIMBER.exe. Surveys go in dist\TIMBER\data\<survey name>\
# and saved results in dist\TIMBER\cache\. (The file portable.txt tells TIMBER to keep
# everything next to the exe; an installed copy uses Documents\TIMBER instead, see
# build_installer.ps1.)

param([switch]$WithData)
$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

& $python -m pip install --quiet -r (Join-Path $PSScriptRoot "requirements-build.txt")
# Build inside this folder (dist\ and build\ here), wherever the script was started from.
Push-Location $PSScriptRoot
try {
    & $python -m PyInstaller --noconfirm --clean --distpath (Join-Path $PSScriptRoot "dist") `
        --workpath (Join-Path $PSScriptRoot "build") (Join-Path $PSScriptRoot "TIMBER.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }
} finally {
    Pop-Location
}

# build\ is only PyInstaller's scratch space. Its TIMBER.exe cannot run (no _internal
# folder), so remove it: the only TIMBER.exe left is the working one in dist\TIMBER.
Remove-Item (Join-Path $PSScriptRoot "build") -Recurse -Force -ErrorAction SilentlyContinue

$dist = Join-Path $PSScriptRoot "dist\TIMBER"
Set-Content -Path (Join-Path $dist "portable.txt") -Encoding utf8 -Value @(
    "This file makes TIMBER portable: surveys (data\), saved results (cache\) and timber.log",
    "are kept in this folder. Delete it to use Documents\TIMBER\data like an installed copy."
)
$target = Join-Path $dist "data"
New-Item -ItemType Directory -Force $target | Out-Null
if ($WithData) {
    Get-ChildItem (Join-Path $PSScriptRoot "data") -Directory | ForEach-Object {
        Write-Host "Copying survey $($_.Name)..."
        Copy-Item $_.FullName $target -Recurse -Force
    }
}
Write-Host "Done: $(Join-Path $dist 'TIMBER.exe')"
