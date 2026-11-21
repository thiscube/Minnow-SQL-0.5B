# Ships the current branch:
# makes sure scripts/verify.ps1 passed on exactly this commit and that the branch already holds
# the latest default branch, pushes, opens or updates the PR, squash-merges exactly that commit,
# and brings the default branch up to date here. Never waits for remote CI.
#   powershell -File scripts/ship.ps1 [-Title "..."] [-Body "text, or a file path"] [-NeedFull]
# Title and body default to the last commit's message. Stamps live in the git folder:
#   verified  "<sha> fast|full", written by verify.ps1 when it passes on a clean tree
#   shipped   the merge commit's sha, written here (release scripts tag it)
param([string]$Title, [string]$Body, [switch]$NeedFull)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

function Run([scriptblock]$cmd, $what) {
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "ship: $what failed" }
}

$gitDir = (git rev-parse --git-dir).Trim()
$base = "$(git symbolic-ref --short refs/remotes/origin/HEAD)".Trim() -replace '^origin/', ''
if (-not $base) { $base = 'main' }
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -eq $base) { throw "ship: on $base; branch first" }
if (git status --porcelain) { throw 'ship: uncommitted changes' }
$sha = (git rev-parse HEAD).Trim()
$stamp = "$(Get-Content (Join-Path $gitDir 'verified') -ErrorAction SilentlyContinue)"
if (-not $stamp.StartsWith($sha)) { throw "ship: $($sha.Substring(0, 7)) wasn't verified; run scripts/verify.ps1" }
if ($NeedFull -and $stamp -notmatch 'full$') { throw 'ship: needs scripts/verify.ps1 -Full' }

# what gets merged must be what was checked: the branch has to sit on top of the latest base
Run { git fetch -q origin $base } "fetching $base"
git merge-base --is-ancestor "origin/$base" HEAD
if ($LASTEXITCODE -ne 0) { throw "ship: $base moved since this branch was cut; rebase onto origin/$base and verify again" }

# title and body go through a file: PowerShell 5.1 mangles quotes in arguments to other programs
if (-not $Title) { $Title = (git log -1 --format=%s).Trim() }
if ($Body -and [IO.File]::Exists($Body)) { $Body = [IO.File]::ReadAllText($Body) }
if (-not $Body) { $Body = (git log -1 --format=%b) -join "`n" }
if (-not $Body.Trim()) { $Body = $Title }
$bodyFile = Join-Path ([IO.Path]::GetTempPath()) ("ship-pr-" + [guid]::NewGuid() + '.md')
[IO.File]::WriteAllText($bodyFile, $Body)
try {
    Run { git push -q -u origin $branch } 'push'
    $pr = gh pr list --head $branch --state open --json number --jq '.[0].number'
    if ($pr) {
        Run { gh pr edit $pr --title $Title --body-file $bodyFile | Out-Null } 'updating the PR'
    } else {
        Run { gh pr create --base $base --title $Title --body-file $bodyFile | Out-Null } 'opening the PR'
    }
} finally {
    Remove-Item $bodyFile -ErrorAction SilentlyContinue
}

# a brand-new PR takes GitHub a moment to work out it's mergeable; only ever merge the
# commit that was checked
$merged = $false
foreach ($try in 1..15) {
    gh pr merge $branch --squash --match-head-commit $sha
    if ($LASTEXITCODE -eq 0) { $merged = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $merged) { throw 'ship: GitHub would not merge it' }
$mergeCommit = (gh pr view $branch --json mergeCommit --jq .mergeCommit.oid).Trim()
Set-Content -Path (Join-Path $gitDir 'shipped') -Value $mergeCommit -Encoding ascii

Run { git checkout -q $base } "switching to $base"
Run { git pull -q --ff-only } "updating $base"
Run { git branch -D $branch | Out-Null } 'deleting the local branch'
$ErrorActionPreference = 'Continue' # GitHub may have deleted it already; that's fine
git push -q origin --delete $branch 2>&1 | Out-Null
Write-Host "Shipped: $((git log --oneline -1).Trim())" -ForegroundColor Green
