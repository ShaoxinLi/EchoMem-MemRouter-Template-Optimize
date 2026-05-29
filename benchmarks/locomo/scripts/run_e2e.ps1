#Requires -Version 5.1
#
# LoCoMo + VikingBot + MemRouter + OpenViking E2E 一键测评脚本
# =============================================================================
# 使用说明：
#   1. 确保已安装依赖：pip install -r requirements.txt
#   2. 复制 configs/*.template 为 configs/*.yaml 并填入 API Key
#   3. 确保 OpenViking 代码仓库在指定路径（或修改下方 $ovRoot 变量）
#   4. 运行：.\benchmarks\locomo\scripts\run_e2e.ps1 -Judge -JudgeToken "sk-xxxxx"
# =============================================================================

[CmdletBinding()]
param(
    [switch]$ForceMemorySearch,
    [switch]$Judge,
    [string]$JudgeToken = "",
    [string]$JudgeBaseUrl = "https://api.deepseek.com/v1",
    [string]$JudgeModel = "deepseek-v4-flash",
    [int]$LimitSamples = 0,
    [int]$LimitQuestions = 0,
    [string]$Category = "",
    [switch]$SkipServerStart,
    [string]$FixedRunDir = "",
    [string]$CaseIdsFile = ""
)

$ErrorActionPreference = "Stop"

# ------------------------------------------------------------------
# 路径配置（根据实际环境修改）
# ------------------------------------------------------------------
$scriptDir    = $PSScriptRoot
$benchmarkRoot = Split-Path -Parent $scriptDir                  # benchmarks/locomo
$echomemRoot  = (Resolve-Path "$benchmarkRoot\..\..").Path     # EchoMem 仓库根目录
$ovRoot       = "D:\Code\cursorProject\OpenViking"              # OpenViking 仓库路径

# 基准包内路径
$dataDir      = "$benchmarkRoot\data"
$configsDir   = "$benchmarkRoot\configs"
$scriptsDir   = "$benchmarkRoot\scripts"

# 数据文件
$dataset      = "$dataDir\locomo10.json"
$routeLabels  = "$dataDir\locomo_e2e_route_labels.v2.jsonl"   # 默认使用 v2 标签

# 配置文件
$ovConf       = "$configsDir\ov.conf"
$memrouterCfg = "$configsDir\memrouter_eval.local.yaml"

# 检查必要文件存在
$missing = @()
foreach ($f in @($dataset, $routeLabels, $ovConf, $memrouterCfg)) {
    if (-not (Test-Path $f)) {
        $missing += $f
    }
}
if ($missing.Count -gt 0) {
    Write-Error "缺少必要文件，请检查配置：`n$($missing -join "`n")"
    exit 1
}

# ------------------------------------------------------------------
# 运行目录
# ------------------------------------------------------------------
if ($FixedRunDir) {
    $runDir = $FixedRunDir
} else {
    $ts     = Get-Date -Format "yyyyMMdd_HHmmss"
    $runDir = "$benchmarkRoot\runs\${ts}_locomo_e2e"
}
$logsDir = "$runDir\logs"
$resDir  = "$runDir\results"

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path $resDir  | Out-Null

# ------------------------------------------------------------------
# 环境变量
# ------------------------------------------------------------------
$env:PYTHONIOENCODING       = "utf-8"
$env:PYTHONUTF8             = "1"
$env:NO_COLOR               = "1"
$env:PYTHONPATH             = "$ovRoot\openviking\lib;$ovRoot;$ovRoot\bot;$echomemRoot"
$env:OPENVIKING_CONFIG_FILE = $ovConf
$env:MEMROUTER_ENABLED      = "true"
$env:MEMROUTER_CONFIG       = $memrouterCfg
$env:ECHOMEM_PATH           = $echomemRoot
$env:MEMROUTER_ROUTE_EVENTS = "$logsDir\route_events.jsonl"
$env:VIKING_SEARCH_USE_LOCAL_CLIENT = "true"

# 清理旧的路由事件文件
if (Test-Path $env:MEMROUTER_ROUTE_EVENTS) {
    Remove-Item $env:MEMROUTER_ROUTE_EVENTS -Force
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "LoCoMo E2E Benchmark Launcher" -ForegroundColor Cyan
Write-Host "========================================"
Write-Host "Benchmark dir : $benchmarkRoot"
Write-Host "EchoMem root  : $echomemRoot"
Write-Host "OpenViking    : $ovRoot"
Write-Host "Run directory : $runDir"
Write-Host "Logs directory: $logsDir"
Write-Host "Results dir   : $resDir"
Write-Host "Route events  : $($env:MEMROUTER_ROUTE_EVENTS)"
Write-Host "Route labels  : $routeLabels"
Write-Host ""

# ------------------------------------------------------------------
# Helper: 清理端口占用
# ------------------------------------------------------------------
function Stop-E2EServers {
    foreach ($port in @(1933, 18790)) {
        try {
            Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | ForEach-Object {
                Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
                Write-Host "  Killed process on port $port (PID $($_.OwningProcess))" -ForegroundColor DarkGray
            }
        } catch {}
    }
}

# ------------------------------------------------------------------
# Helper: 等待端口就绪
# ------------------------------------------------------------------
function Wait-ForPort {
    param([string]$HostName = "127.0.0.1", [int]$Port, [int]$TimeoutSec = 30, [string]$Label = "service")
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        try {
            $client = New-Object System.Net.Sockets.TcpClient
            $client.Connect($HostName, $Port)
            $client.Close()
            return $true
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    return $false
}

# ------------------------------------------------------------------
# 启动服务
# ------------------------------------------------------------------
$gwProc = $null
$ovProc = $null

if (-not $SkipServerStart) {
    Write-Host "Cleaning up any stale E2E processes..." -ForegroundColor DarkGray
    Stop-E2EServers

    Write-Host "[1/4] Starting VikingBot gateway..." -ForegroundColor Yellow
    $gwCmd = @"
cd "$ovRoot\bot"
`$env:PYTHONIOENCODING='utf-8'
`$env:PYTHONUTF8='1'
`$env:NO_COLOR='1'
`$env:PYTHONPATH='$ovRoot\openviking\lib;$ovRoot;$ovRoot\bot;$echomemRoot'
`$env:OPENVIKING_CONFIG_FILE='$ovConf'
`$env:MEMROUTER_ENABLED='true'
`$env:MEMROUTER_CONFIG='$memrouterCfg'
`$env:ECHOMEM_PATH='$echomemRoot'
`$env:MEMROUTER_ROUTE_EVENTS='$($env:MEMROUTER_ROUTE_EVENTS)'
python -m vikingbot gateway --config '$ovConf' --host 127.0.0.1 --port 18790 2>'$logsDir\vikingbot.gateway.stderr.log'
"@
    $gwProc = Start-Process powershell -ArgumentList "-Command", $gwCmd -PassThru -WindowStyle Hidden

    Write-Host "[2/4] Waiting for gateway port 18790 (max 30s)..." -ForegroundColor Yellow
    if (Wait-ForPort -Port 18790 -TimeoutSec 30 -Label "gateway") {
        Write-Host "      Gateway is ready." -ForegroundColor Green
    } else {
        Write-Warning "Gateway did not respond on 18790 within 30s."
    }

    Write-Host "[3/4] Starting OpenViking server..." -ForegroundColor Yellow
    $ovCmd = @"
cd "$ovRoot"
`$env:PYTHONIOENCODING='utf-8'
`$env:PYTHONUTF8='1'
`$env:NO_COLOR='1'
`$env:PYTHONPATH='$ovRoot\openviking\lib;$ovRoot;$ovRoot\bot;$echomemRoot'
`$env:OPENVIKING_CONFIG_FILE='$ovConf'
`$env:MEMROUTER_ENABLED='true'
`$env:MEMROUTER_CONFIG='$memrouterCfg'
`$env:ECHOMEM_PATH='$echomemRoot'
`$env:MEMROUTER_ROUTE_EVENTS='$($env:MEMROUTER_ROUTE_EVENTS)'
python -m uvicorn openviking.server.app:create_app --factory --host 127.0.0.1 --port 1933 2>'$logsDir\openviking.stderr.log'
"@
    $ovProc = Start-Process powershell -ArgumentList "-Command", $ovCmd -PassThru -WindowStyle Hidden

    Write-Host "[4/4] Waiting for OpenViking /health (max 30s)..." -ForegroundColor Yellow
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:1933/health" -Method GET -TimeoutSec 2 -ErrorAction Stop
            if ($resp.status -eq "ok" -or $resp.healthy -eq $true) {
                $ready = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $ready) {
        Write-Warning "OpenViking health check did not pass in 30s. Proceeding anyway..."
    } else {
        Write-Host "      OpenViking is ready." -ForegroundColor Green
    }
} else {
    Write-Host "[1/4] Skipping server start (assumes already running)." -ForegroundColor Yellow
}

# ------------------------------------------------------------------
# 构建测评参数
# ------------------------------------------------------------------
$argList = @(
    "--dataset", $dataset,
    "--route-labels", $routeLabels,
    "--route-events-path", $env:MEMROUTER_ROUTE_EVENTS,
    "--ov-config", $ovConf,
    "--memrouter-config", $memrouterCfg,
    "--openviking-root", $ovRoot,
    "--output-base", "$benchmarkRoot\runs",
    "--fixed-run-dir", $runDir,
    "--ov-chat-endpoint", "http://127.0.0.1:1933",
    "--ov-api-key", "ov-test-key-12345",
    "--ov-account", "default"
)

if ($ForceMemorySearch) {
    $argList += "--force-memory-search"
}
if ($LimitSamples -gt 0) {
    $argList += @("--limit-samples", $LimitSamples)
}
if ($LimitQuestions -gt 0) {
    $argList += @("--limit-questions", $LimitQuestions)
}
if ($Category) {
    $argList += @("--category", $Category)
}
if ($CaseIdsFile) {
    $argList += @("--case-ids-file", $CaseIdsFile)
}
if ($Judge) {
    if (-not $JudgeToken) {
        Write-Error "--Judge requires --JudgeToken"
        exit 1
    }
    $argList += @("--judge", "--judge-token", $JudgeToken, "--judge-base-url", $JudgeBaseUrl, "--judge-model", $JudgeModel)
}

# ------------------------------------------------------------------
# 运行测评
# ------------------------------------------------------------------
$evalScript = "$scriptsDir\eval_locomo_vikingbot_memrouter_e2e.py"
Write-Host "Running evaluator..." -ForegroundColor Yellow
Write-Host "      python $evalScript $argList"
Write-Host ""

cd $echomemRoot
python "$evalScript" @argList

$exitCode = $LASTEXITCODE

# ------------------------------------------------------------------
# 后处理汇总
# ------------------------------------------------------------------
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($exitCode -eq 0) {
    Write-Host "Evaluation completed successfully." -ForegroundColor Green
} else {
    Write-Host "Evaluator exited with code $exitCode" -ForegroundColor Red
}
Write-Host "Run directory : $runDir"
Write-Host "Logs directory: $logsDir"
Write-Host "Results dir   : $resDir"
Write-Host "Report        : $resDir\report.md"
Write-Host "========================================"

# ------------------------------------------------------------------
# 清理后台服务
# ------------------------------------------------------------------
if (-not $SkipServerStart) {
    Write-Host "Stopping background servers..." -ForegroundColor DarkGray
    if ($ovProc -ne $null) {
        Stop-Process -Id $ovProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($gwProc -ne $null) {
        Stop-Process -Id $gwProc.Id -Force -ErrorAction SilentlyContinue
    }
    Stop-E2EServers
    Write-Host "Servers stopped." -ForegroundColor DarkGray
}

exit $exitCode
