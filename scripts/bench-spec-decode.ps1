#requires -Version 5.1
<#
.SYNOPSIS
  Speculative-decoding bench harness for shannon-prime-llama (v2.14.0-sp2+).

.DESCRIPTION
  Runs llama-cli through a battery of SP configurations on a target+draft
  pair and reports tok/sec, acceptance rate, and (when -CompareOutputs is
  set) Levenshtein-style output drift across configs. Output is a CSV ready
  to drop into CHANGELOG.md, SPECULATIVE-DECODING.md, or a PR description.

  Five configurations exercised:
    1. vanilla            - no SP at all (baseline)
    2. shared             - same SP for target+draft (sp1-style)
    3. per-model-same     - SHANNON_PRIME_SPEC=1 with same defaults (sp2 baseline)
    4. differential-agg   - SPEC=1 + DRAFT_PRESET=aggressive (K=2,1 V=1)
    5. differential-tern  - SPEC=1 + DRAFT_K_TERNARY_BANDS=3

  Configs 3-5 require the v2.14.0-sp2 patch surgery to be effective.
  Running them on an sp1 binary will silently route everything through
  the global g_sp context - the comparison is misleading on sp1.

.PARAMETER LlamaCli
  Path to llama-cli executable. Default: build/bin/Release/llama-cli.exe.

.PARAMETER Target
  Path to target GGUF (the larger model - typically 7B-70B).

.PARAMETER Draft
  Path to draft GGUF (the smaller model - typically 0.5B-2B). Must use
  the same tokeniser as the target.

.PARAMETER PromptFile
  Text file containing the prompt. Default: a built-in 256-token prompt.

.PARAMETER NPredict
  Number of tokens to generate per run. Default 256.

.PARAMETER NRuns
  Repeat each configuration this many times for tok/sec averaging.
  Default 3 (warmup-discarded; result is mean of last 2).

.PARAMETER OutputCsv
  Where to write the results. Default: bench-spec-results-<timestamp>.csv.

.PARAMETER CompareOutputs
  Capture generated text and compute pairwise edit-distance vs vanilla.

.PARAMETER DryRun
  Print the commands that would be invoked without running them.

.EXAMPLE
  .\scripts\bench-spec-decode.ps1 `
    -Target  "D:\models\qwen2.5-7b-instruct-q4_k_m.gguf" `
    -Draft   "D:\models\qwen2.5-0.5b-instruct-q8_0.gguf" `
    -OutputCsv "qwen25-bench.csv" -CompareOutputs

.EXAMPLE
  .\scripts\bench-spec-decode.ps1 -Target fake.gguf -Draft fake.gguf -DryRun

.NOTES
  Requires PowerShell 5.1+ and llama-cli built with -DLLAMA_SHANNON_PRIME=ON
  against shannon-prime-llama v2.14.0-sp2 or later.

  See lib/shannon-prime/docs/SPECULATIVE-DECODING.md for the architecture.
#>

[CmdletBinding()]
param(
    [string]$LlamaCli   = "build/bin/Release/llama-cli.exe",
    [Parameter(Mandatory=$true)] [string]$Target,
    [Parameter(Mandatory=$true)] [string]$Draft,
    [string]$PromptFile = "",
    [int]   $NPredict   = 256,
    [int]   $NRuns      = 3,
    [int]   $DraftMax   = 8,
    [string]$OutputCsv  = "",
    [switch]$CompareOutputs,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ---- Helpers (defined first so the main loop can call them) ----
function Measure-EditDistance {
    # Iterative Levenshtein. Quadratic in length; NPredict=256 is ~1k chars
    # worst case, ~1MB matrix - fine for bench post-processing.
    param([string]$a, [string]$b)
    if ([string]::IsNullOrEmpty($a)) { return $b.Length }
    if ([string]::IsNullOrEmpty($b)) { return $a.Length }
    $n = $a.Length; $m = $b.Length
    $d = New-Object 'int[,]' ($n + 1), ($m + 1)
    for ($i = 0; $i -le $n; $i++) { $d[$i, 0] = $i }
    for ($j = 0; $j -le $m; $j++) { $d[0, $j] = $j }
    for ($i = 1; $i -le $n; $i++) {
        for ($j = 1; $j -le $m; $j++) {
            $cost = if ($a[$i - 1] -eq $b[$j - 1]) { 0 } else { 1 }
            $im1 = $i - 1
            $jm1 = $j - 1
            $a1 = $d[$im1, $j] + 1
            $a2 = $d[$i, $jm1] + 1
            $a3 = $d[$im1, $jm1] + $cost
            $best = if ($a1 -lt $a2) { $a1 } else { $a2 }
            if ($a3 -lt $best) { $best = $a3 }
            $d[$i, $j] = $best
        }
    }
    return $d[$n, $m]
}

# ---- Default prompt (used when -PromptFile not given) ----
$defaultPrompt = @"
Write a clear, technically-correct paragraph explaining the Sieve of
Eratosthenes algorithm. Cover: its time complexity, why it works, the
key trick of marking multiples starting from p^2, and one small worked
example for primes up to 30. Be precise but accessible.
"@

if ($PromptFile -and (Test-Path $PromptFile)) {
    $prompt = Get-Content $PromptFile -Raw
} else {
    $prompt = $defaultPrompt
}

if (-not $OutputCsv) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputCsv = "bench-spec-results-$stamp.csv"
}

# ---- Configurations ----
$configs = @(
    @{
        Name = "vanilla"
        Env  = @{ "SHANNON_PRIME_ENABLED" = "0" }
        Note = "no SP, baseline"
    },
    @{
        Name = "shared"
        Env  = @{
            "SHANNON_PRIME_ENABLED" = "1"
            "SHANNON_PRIME_K_BITS"  = "5,5,4,3"
            "SHANNON_PRIME_V_BITS"  = "3"
        }
        Note = "ship preset, both ctx (sp1-style)"
    },
    @{
        Name = "per-model-same"
        Env  = @{
            "SHANNON_PRIME_ENABLED" = "1"
            "SHANNON_PRIME_SPEC"    = "1"
            "SHANNON_PRIME_K_BITS"  = "5,5,4,3"
            "SHANNON_PRIME_V_BITS"  = "3"
        }
        Note = "sp2 per-model, same defaults"
    },
    @{
        Name = "differential-agg"
        Env  = @{
            "SHANNON_PRIME_ENABLED"      = "1"
            "SHANNON_PRIME_SPEC"         = "1"
            "SHANNON_PRIME_K_BITS"       = "5,5,4,3"
            "SHANNON_PRIME_V_BITS"       = "3"
            "SHANNON_PRIME_DRAFT_PRESET" = "aggressive"
        }
        Note = "sp2 differential - draft K=2,1 V=1"
    },
    @{
        Name = "differential-tern"
        Env  = @{
            "SHANNON_PRIME_ENABLED"               = "1"
            "SHANNON_PRIME_SPEC"                  = "1"
            "SHANNON_PRIME_K_BITS"                = "5,5,4,3"
            "SHANNON_PRIME_V_BITS"                = "3"
            "SHANNON_PRIME_DRAFT_K_TERNARY_BANDS" = "3"
        }
        Note = "sp2 differential - draft band 3 ternary"
    }
)

# ---- Pre-flight ----
if (-not $DryRun) {
    if (-not (Test-Path $LlamaCli)) {
        Write-Error "llama-cli not found at $LlamaCli - build it or pass -LlamaCli"
        exit 1
    }
    if (-not (Test-Path $Target)) {
        Write-Error "Target GGUF not found: $Target"
        exit 1
    }
    if (-not (Test-Path $Draft)) {
        Write-Error "Draft GGUF not found: $Draft"
        exit 1
    }
}

$tmpPrompt = [System.IO.Path]::GetTempFileName()
Set-Content -Path $tmpPrompt -Value $prompt -NoNewline -Encoding utf8

# ---- Run loop ----
$results = @()
$vanillaOutput = $null

foreach ($cfg in $configs) {
    Write-Host ""
    Write-Host "==============================================================="
    Write-Host ("Config: {0} - {1}" -f $cfg.Name, $cfg.Note)
    Write-Host "==============================================================="

    $tokSecRuns = @()
    $acceptRuns = @()
    $lastOutput = ""

    for ($run = 1; $run -le $NRuns; $run++) {
        $isWarmup = ($run -eq 1) -and ($NRuns -gt 1)
        $tag = if ($isWarmup) { " [warmup, discarded]" } else { "" }
        Write-Host ("  Run {0}/{1}{2}" -f $run, $NRuns, $tag)

        $cliArgs = @(
            "-m",  $Target,
            "-md", $Draft,
            "--draft-max", "$DraftMax",
            "-f",  $tmpPrompt,
            "-n",  "$NPredict",
            "--temp", "0",
            "--no-warmup"
        )

        if ($DryRun) {
            $envEcho = ($cfg.Env.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join " "
            Write-Host "    DRY-RUN: $envEcho $LlamaCli $($cliArgs -join ' ')"
            $tokSecRuns += [double]0.0
            $acceptRuns += [double]0.0
            continue
        }

        $savedEnv = @{}
        foreach ($k in $cfg.Env.Keys) {
            $savedEnv[$k] = [System.Environment]::GetEnvironmentVariable($k, "Process")
            [System.Environment]::SetEnvironmentVariable($k, $cfg.Env[$k], "Process")
        }

        try {
            $stdout = & $LlamaCli @cliArgs 2>&1 | Out-String
        } finally {
            foreach ($k in $cfg.Env.Keys) {
                [System.Environment]::SetEnvironmentVariable($k, $savedEnv[$k], "Process")
            }
        }

        $tokSec = 0.0
        $accept = 0.0
        if ($stdout -match "(?s)eval time.*?\(([\d.]+)\s*tokens per second\)") {
            $tokSec = [double]$matches[1]
        }
        if ($stdout -match "draft.*?accepted[:\s]+(\d+)\s*/\s*(\d+)") {
            $a = [double]$matches[1]; $b = [double]$matches[2]
            if ($b -gt 0) { $accept = $a / $b }
        }

        Write-Host ("    tok/sec = {0:N2}, accept = {1:P1}" -f $tokSec, $accept)

        if (-not $isWarmup) {
            $tokSecRuns += $tokSec
            $acceptRuns += $accept
        }
        $lastOutput = $stdout
    }

    if ($DryRun) {
        $meanTokSec = 0.0; $meanAccept = 0.0; $editDist = 0
    } else {
        if ($tokSecRuns.Count -gt 0) {
            $meanTokSec = ($tokSecRuns | Measure-Object -Average).Average
            $meanAccept = ($acceptRuns | Measure-Object -Average).Average
        } else {
            $meanTokSec = 0.0; $meanAccept = 0.0
        }

        $editDist = 0
        if ($CompareOutputs) {
            $cleaned = ($lastOutput -split "`n" | Where-Object {
                $_ -notmatch "^\[shannon-prime|^llama_|^eval time|^load time|^sampling time|^draft.*accepted|^total time|^\s*$"
            }) -join "`n"
            if ($cfg.Name -eq "vanilla") {
                $vanillaOutput = $cleaned
            } elseif ($vanillaOutput) {
                $editDist = (Measure-EditDistance $vanillaOutput $cleaned)
            }
        }
    }

    $results += [PSCustomObject]@{
        Config        = $cfg.Name
        Note          = $cfg.Note
        TokSec        = [Math]::Round($meanTokSec, 2)
        Acceptance    = [Math]::Round($meanAccept, 4)
        EditDistVsVan = $editDist
    }
}

Remove-Item $tmpPrompt -Force -ErrorAction SilentlyContinue

# ---- Summary + CSV write ----
Write-Host ""
Write-Host "==============================================================="
Write-Host "Results"
Write-Host "==============================================================="

$results | Format-Table -AutoSize

$vanillaTokSec = ($results | Where-Object Config -eq "vanilla").TokSec
if ($vanillaTokSec -gt 0) {
    Write-Host ""
    Write-Host "Speedup vs vanilla:"
    foreach ($r in $results) {
        $ratio = if ($vanillaTokSec -gt 0) { $r.TokSec / $vanillaTokSec } else { 0 }
        Write-Host ("  {0,-22}  {1:N2}x" -f $r.Config, $ratio)
    }
}

$results | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding utf8
Write-Host ""
Write-Host "CSV written: $OutputCsv"
