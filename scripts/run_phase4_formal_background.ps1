$ErrorActionPreference = "Stop"

$root = "F:\ACM"
$python = Join-Path $root ".venv312\Scripts\python.exe"
$runner = Join-Path $root "scripts\run_phase4_formal_benchmark.py"
$secretFile = Join-Path $root ".secrets\phase4_api_key.dpapi"
$logDirectory = Join-Path $root "experiments\phase4\background_logs"
$stdoutLog = Join-Path $logDirectory "formal_resume_stdout.log"
$statusFile = Join-Path $logDirectory "formal_resume_status.json"

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

$existing = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -match "^python" -and
        $_.CommandLine -like "*run_phase4_formal_benchmark.py*"
    }
if ($existing) {
    @{
        status = "NOT_STARTED_DUPLICATE_PROCESS_PRESENT"
        process_ids = @($existing.ProcessId)
        generated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8
    exit 2
}

if (-not (Test-Path -LiteralPath $secretFile)) {
    throw "Encrypted Phase 4 credential file is missing."
}

$encryptedKey = (Get-Content -LiteralPath $secretFile -Raw).Trim()
$secureKey = $encryptedKey | ConvertTo-SecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    $plainKey = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    $env:PHASE4_LLM_API_KEY = $plainKey
    $env:PHASE4_LLM_MODEL = "claude-opus-4-8"
    $env:PHASE4_LLM_BASE_URL = "https://rsxermu666.cn/v1"
    $env:PYTHONPATH = $root

    @{
        status = "BACKGROUND_RESUME_STARTED"
        workers = 1
        generated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8

    Set-Location -LiteralPath $root
    & $python $runner `
        --authorization I_AUTHORIZE_FORMAL_LLM_COST `
        --mode all `
        --workers 1 `
        --resume *>> $stdoutLog
    $exitCode = $LASTEXITCODE

    @{
        status = if ($exitCode -eq 0) {
            "BACKGROUND_RESUME_PROCESS_COMPLETED"
        } else {
            "BACKGROUND_RESUME_PROCESS_FAILED"
        }
        exit_code = $exitCode
        generated_at = (Get-Date).ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $statusFile -Encoding UTF8
    exit $exitCode
}
finally {
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    $plainKey = $null
    Remove-Item Env:\PHASE4_LLM_API_KEY -ErrorAction SilentlyContinue
}
