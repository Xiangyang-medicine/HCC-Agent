$files = @(
    'src/prognostic_engine/src/prognostic_engine/training.py',
    'src/prognostic_engine/src/prognostic_engine/bootstrap.py',
    'src/prognostic_engine/src/prognostic_engine/metrics.py',
    'src/prognostic_engine/src/prognostic_engine/models.py',
    'src/prognostic_engine/scripts/run_formal_training.py'
)
$expected = @{
    'src/prognostic_engine/src/prognostic_engine/training.py' = 'abbd57cfebbd3cd9b6db39a7b2e1f60b9bf08773347f2df2fe5ffed96a8b3c45'
    'src/prognostic_engine/src/prognostic_engine/bootstrap.py' = '57894850bd7e9a87b70694ddd8266e1d1b92a5973f30d0fd8899c14d34732c05'
    'src/prognostic_engine/src/prognostic_engine/metrics.py' = 'ee4422bd1751279ba9a15e92a1c3d9489dc3e48c2643151bc62ed9ddbfdb72db'
    'src/prognostic_engine/src/prognostic_engine/models.py' = 'abee22d998bf9e955519bdb4b62d1e2ed86a3869ae5bbb2c5247357b8e86f3b1'
    'src/prognostic_engine/scripts/run_formal_training.py' = '0132a44bdb14784d22466f7b3ed5faefee59dd35cdc035187bee153d1a36291d'
}
$allMatch = $true
foreach ($f in $files) {
    $hash = (Get-FileHash -Path $f -Algorithm SHA256).Hash.ToLower()
    $expectedHash = $expected[$f]
    $match = $hash -eq $expectedHash
    if (-not $match) { $allMatch = $false }
    $status = if ($match) { 'OK' } else { 'MISMATCH' }
    Write-Host "$($f | Split-Path -Leaf): $status"
    Write-Host "  Expected: $expectedHash"
    Write-Host "  Got:      $hash"
}
if ($allMatch) {
    Write-Host "`nAll hashes match - pipeline is frozen"
} else {
    Write-Host "`nHASH MISMATCH DETECTED - STOPPING"
    exit 1
}
