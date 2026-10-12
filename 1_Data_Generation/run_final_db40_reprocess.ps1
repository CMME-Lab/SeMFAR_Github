param(
    [string]$DatasetRoot = 'D:\Woojeong_SR\data10\Paired_ver10',
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:MPLBACKEND = 'Agg'
$ProjectRoot = $PSScriptRoot
$env:PYTHONPATH = $ProjectRoot
$Processor = Join-Path $ProjectRoot 'postprocessing\final_db40_pair_reprocess.py'
$LogRoot = Join-Path $DatasetRoot '_final_db40_logs'
$TempRoot = Join-Path $DatasetRoot '_final_db40_temp'
$Status = Join-Path $LogRoot 'STATUS.txt'
$CompletedFile = Join-Path $LogRoot 'COMPLETED_SAMPLES.txt'
New-Item -ItemType Directory -Force -Path $LogRoot, $TempRoot | Out-Null
"STARTED $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $Status -Encoding utf8 -Append

$Ids = @(Get-ChildItem $DatasetRoot -File | ForEach-Object {
    if ($_.Name -match '^scan_benign_(\d+)_HR\.pkl$') { [int]$Matches[1] }
} | Sort-Object -Unique)
$Completed = @{}
if (Test-Path $CompletedFile) {
    Get-Content $CompletedFile | ForEach-Object {
        if ($_ -match '^benign_(\d+)$') { $Completed[$Matches[1]] = $true }
    }
}

foreach ($Id in $Ids) {
    if ($Completed.ContainsKey("$Id")) { continue }
    $Sample = "benign_$Id"
    $Temp = Join-Path $TempRoot $Sample
    if (Test-Path $Temp) { Remove-Item $Temp -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $Temp | Out-Null
    "PROCESSING $Sample $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $Status -Encoding utf8 -Append
    $OutLog = Join-Path $LogRoot "$Sample.out.log"
    $ErrLog = Join-Path $LogRoot "$Sample.err.log"
    & $Python $Processor --input-dir $DatasetRoot --output-dir $Temp --sample $Sample 1>>$OutLog 2>>$ErrLog
    $Names = @(
        "scan_${Sample}_HR.pkl", "scan_${Sample}_HR.png",
        "scan_${Sample}_LR.pkl", "scan_${Sample}_LR.png",
        "scan_${Sample}_HR_minus_LR_pixelwise.png"
    )
    $Valid = $LASTEXITCODE -eq 0
    foreach ($Name in $Names) {
        $File = Join-Path $Temp $Name
        if (!(Test-Path $File) -or (Get-Item $File).Length -le 0) { $Valid = $false }
    }
    if ($Valid) {
        foreach ($Name in $Names) {
            Move-Item (Join-Path $Temp $Name) (Join-Path $DatasetRoot $Name) -Force
        }
        Remove-Item $Temp -Recurse -Force -ErrorAction SilentlyContinue
        $Sample | Out-File $CompletedFile -Encoding utf8 -Append
        $Completed[$Id] = $true
        "COMPLETE $Sample $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $Status -Encoding utf8 -Append
    } else {
        "FAILED $Sample exit=$LASTEXITCODE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $Status -Encoding utf8 -Append
    }
}
"FINISHED $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') completed=$($Completed.Count)/$($Ids.Count)" | Out-File $Status -Encoding utf8 -Append
