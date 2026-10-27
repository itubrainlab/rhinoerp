# Install Rhino ERP for the current user: copies this folder to
# %LOCALAPPDATA%\RhinoERP\app, adds a Start menu shortcut with the icon,
# then sets it up (downloads g.Pype and the libraries from PyPI). No
# administrator rights needed. Started by install.cmd.
$ErrorActionPreference = 'Stop'

if (-not [Environment]::Is64BitOperatingSystem) {
    throw 'Rhino ERP needs 64-bit Windows.'
}

$here = $PSScriptRoot
$data = Join-Path $env:LOCALAPPDATA 'RhinoERP'
$app = Join-Path $data 'app'

if ($here.TrimEnd('\') -ne $app) {
    Write-Host "Copying Rhino ERP to $app"
    if (Test-Path $app) { Remove-Item -Recurse -Force $app }
    New-Item -ItemType Directory -Force -Path $app | Out-Null
    Copy-Item -Recurse -Force (Join-Path $here '*') $app
}
# Files unpacked from a downloaded zip carry its mark of the web, which
# makes Windows ask before running each program.
Get-ChildItem -Recurse -File $app | Unblock-File

$programs = [Environment]::GetFolderPath('Programs')
$link = (New-Object -ComObject WScript.Shell).CreateShortcut(
    (Join-Path $programs 'Rhino ERP.lnk'))
$link.TargetPath = Join-Path $app 'python\pythonw.exe'
$link.Arguments = '-I "' + (Join-Path $app 'launcher.py') + '"'
$link.WorkingDirectory = $app
$link.IconLocation = Join-Path $app 'rhino-erp.ico'
$link.Description = 'Offline ERP analysis of EEG recordings'
$link.Save()

Write-Host 'Setting up (downloads g.Pype and the libraries; a few minutes)...'
$build = (Get-Content (Join-Path $app 'build-id') -Raw).Trim()
& (Join-Path $app 'python\python.exe') -I (Join-Path $app 'bootstrap.py') `
    $app (Join-Path $data "runtime\$build")
if ($LASTEXITCODE -ne 0) {
    throw "Setup failed. See $(Join-Path $data 'runtime\setup.log')"
}

Write-Host ''
Write-Host 'Rhino ERP is installed: open it from the Start menu.'
