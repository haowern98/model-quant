$root = Join-Path $env:LOCALAPPDATA "MI\g"

if (Test-Path -LiteralPath $root) {
  Remove-Item -LiteralPath $root -Recurse -Force
  Write-Host "Deleted $root"
} else {
  Write-Host "Missing $root"
}
