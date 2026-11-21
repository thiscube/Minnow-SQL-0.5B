# The merge gate for Minnow-SQL. Every change passes this on the dev PC before it ships; remote
# CI is only a backup. On success it stamps the exact commit it checked, and scripts/ship.ps1
# refuses anything else.
#   powershell -File scripts/verify.ps1          the fast gate
#   ... -Full     also proves a fresh clone can install and pass the tests from nothing
param([switch]$Full)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$times = @()
$started = Get-Date
# the stamp at the end only goes on this commit, and only if nothing changed while checks ran
$gitDir = (git rev-parse --git-dir).Trim()
$sha = (git rev-parse HEAD).Trim()
$cleanAtStart = -not (git status --porcelain)
$py = Join-Path (Get-Location) '.venv\Scripts\python.exe'

function Step($name, [scriptblock]$cmd) {
    Write-Host "==> $name" -ForegroundColor Cyan
    $t = Get-Date
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $name" -ForegroundColor Red; exit 1 }
    $script:times += '  {0,-20} {1,4:N0} s' -f $name, ((Get-Date) - $t).TotalSeconds
}

# every PowerShell script in the repo must at least parse
Step 'scripts parse' {
    $bad = 0
    foreach ($f in Get-ChildItem scripts -Filter *.ps1 -ErrorAction SilentlyContinue) {
        $errors = $null
        [void][Management.Automation.Language.Parser]::ParseFile($f.FullName, [ref]$null, [ref]$errors)
        foreach ($e in $errors) { $bad++; Write-Host "$($f.Name):$($e.Extent.StartLineNumber): $($e.Message)" -ForegroundColor Red }
    }
    $global:LASTEXITCODE = [int]($bad -gt 0)
}

# the check tools live in .venv at the versions pinned in pyproject.toml; they are installed on
# the first run and again whenever pyproject.toml changes
Step 'tools' {
    $global:LASTEXITCODE = 0
    $marker = '.venv\.installed-for'
    $want = (Get-FileHash pyproject.toml).Hash
    if (-not (Test-Path $py)) { python -m venv .venv }
    if ($LASTEXITCODE -eq 0 -and "$(Get-Content $marker -ErrorAction SilentlyContinue)" -ne $want) {
        & $py -m pip install -q --disable-pip-version-check -e '.[dev]'
        if ($LASTEXITCODE -eq 0) { Set-Content -Path $marker -Value $want -Encoding ascii }
    }
}

Step 'format' { & $py -m ruff format --check . }
Step 'lint' { & $py -m ruff check . }
Step 'tests' { & $py -m pytest }

# the real run: the leak guard, started the way a person would start it. Clean data has to pass
# and a planted leak has to be caught, so the guard can never quietly stop working. A leak exits
# with 3; a crash exits with something else and does not count as catching it.
Step 'leak guard' {
    & $py -m minnow.guard tests\fixtures\train_clean.json tests\fixtures\dev.json | Out-Null
    $cleanPassed = $LASTEXITCODE -eq 0
    & $py -m minnow.guard tests\fixtures\train_leaky.json tests\fixtures\dev.json | Out-Null
    $leakCaught = $LASTEXITCODE -eq 3
    if (-not $cleanPassed) { Write-Host 'clean data was reported as a leak' -ForegroundColor Red }
    if (-not $leakCaught) { Write-Host 'the planted leak was not caught' -ForegroundColor Red }
    $global:LASTEXITCODE = [int](-not ($cleanPassed -and $leakCaught))
}

# every score in results.md, the chart and the README comes from the runs' own files; this fails
# when one of them is out of date or a run under results/ was never logged
Step 'results log' { & $py -m minnow.runlog check }

if ($Full) {
    # what someone who clones the repo gets: only the committed files, a new environment, the
    # pinned tools, the tests. Catches tests that only pass because of files that are not in git.
    Step 'fresh clone' {
        $fresh = Join-Path ([IO.Path]::GetTempPath()) ('minnow-verify-' + [guid]::NewGuid())
        $result = 1
        try {
            git clone -q --no-hardlinks . $fresh
            if ($LASTEXITCODE -eq 0) {
                Push-Location $fresh
                try {
                    python -m venv .venv
                    $freshPy = Join-Path $fresh '.venv\Scripts\python.exe'
                    & $freshPy -m pip install -q --disable-pip-version-check -e '.[dev]'
                    if ($LASTEXITCODE -eq 0) { & $freshPy -m pytest -p no:cacheprovider }
                    if ($LASTEXITCODE -eq 0) { & $freshPy -m minnow.runlog check }
                    $result = $LASTEXITCODE
                } finally {
                    Pop-Location
                }
            }
        } finally {
            Remove-Item $fresh -Recurse -Force -ErrorAction SilentlyContinue
        }
        $global:LASTEXITCODE = $result
    }
}

if (-not $cleanAtStart) {
    Write-Host 'All checks passed, but there were uncommitted changes: commit and run again to ship.' -ForegroundColor Yellow
} elseif ((git rev-parse HEAD).Trim() -ne $sha -or (git status --porcelain)) {
    Write-Host 'All checks passed, but the code changed while they ran: run again to ship.' -ForegroundColor Yellow
} else {
    Set-Content -Path (Join-Path $gitDir 'verified') -Value "$sha $(if ($Full) { 'full' } else { 'fast' })" -Encoding ascii
    Write-Host "All checks passed. Stamped $($sha.Substring(0, 7))$(if ($Full) { ' (full)' })." -ForegroundColor Green
}
$times | ForEach-Object { Write-Host $_ }
Write-Host ('  {0,-20} {1,4:N0} s' -f 'total', ((Get-Date) - $started).TotalSeconds)
