$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$planPath = Join-Path $projectRoot 'project_management\cleanup_plan.json'
$proofPath = Join-Path $projectRoot 'project_management\cleanup_precheck.json'
$publishPath = Join-Path $projectRoot 'project_management\PUBLISH_RECEIPT.json'
$pythonPath = Join-Path $projectRoot 'rc3_learning\.venv\Scripts\python.exe'

# All targets are explicit entries; never enumerate across shells for deletion.
if (-not (Test-Path -LiteralPath $publishPath)) { throw 'Remote upload verification is required before cleanup.' }
$publication = Get-Content -LiteralPath $publishPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $publication.initial_remote_verified) { throw 'Remote project backup has not been verified.' }
& $pythonPath -B (Join-Path $projectRoot 'tools\check_cleanup_backup.py')
if ($LASTEXITCODE -ne 0) { throw 'Cleanup backup precheck failed.' }
$plan = Get-Content -LiteralPath $planPath -Raw -Encoding UTF8 | ConvertFrom-Json
$proof = Get-Content -LiteralPath $proofPath -Raw -Encoding UTF8 | ConvertFrom-Json
$currentPlanHash = (Get-FileHash -LiteralPath $planPath -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $proof.ready_for_cleanup -or $proof.plan_sha256 -ne $currentPlanHash) { throw 'Cleanup plan does not match verified backup.' }

$checkedTargets = @()
foreach ($entry in $plan.roots) {
    $target = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $entry.path))
    if (-not $target.StartsWith($projectRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Target outside project.' }
    $relativeParts = $target.Substring($projectRoot.Length + 1).Split('\')
    if (($relativeParts -contains '.git') -or ($relativeParts -contains '.venv') -or ($relativeParts -contains 'project_backups')) { throw 'Protected target.' }
    $item = Get-Item -LiteralPath $target -Force
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Link/reparse cleanup target rejected.' }
    $checkedTargets += [PSCustomObject]@{ absolute = $target; relative = $entry.path; files = $entry.files; bytes = $entry.bytes; reason = $entry.reason }
}

$removed = @()
foreach ($target in $checkedTargets) {
    Remove-Item -LiteralPath $target.absolute -Recurse -Force
    if (Test-Path -LiteralPath $target.absolute) { throw 'Cleanup target still exists.' }
    $removed += [PSCustomObject]@{ path = $target.relative; files = $target.files; bytes = $target.bytes; reason = $target.reason }
}
$record = [ordered]@{
    schema = 'rc3-cleanup-receipt-v1'
    completed_utc = [DateTime]::UtcNow.ToString('o')
    workspace = $projectRoot
    removed_targets = $removed
    removed_file_count = ($removed | Measure-Object -Property files -Sum).Sum
    removed_logical_bytes = ($removed | Measure-Object -Property bytes -Sum).Sum
    backup_sha256 = $proof.backup_sha256
    remote_verified_before_cleanup = $true
    outside_project_deletions = 0
    local_python_environment_retained = $true
    frozen_project_payloads_retained = $true
}
$receiptPath = Join-Path $projectRoot 'project_management\cleanup_receipt.json'
$receiptJson = $record | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText($receiptPath, $receiptJson + "`n", [System.Text.UTF8Encoding]::new($false))
Write-Output ($record | ConvertTo-Json -Depth 2 -Compress)
