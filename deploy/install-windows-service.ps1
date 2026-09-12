<#
.SYNOPSIS
    把 CIDA 检测服务注册成 Windows 服务（基于 NSSM），并做一次健康检查。

.DESCRIPTION
    做三件事：创建服务 → 固定工作目录与日志重定向 → 启动并验证 /api/health。
    默认以「只观察 + 演练模式」启动（对应 deploy/cida-detect.env 的默认值），
    确认告警质量后再改环境变量打开真实处置。

.EXAMPLE
    .\deploy\install-windows-service.ps1 -ProjectRoot "D:\sdn_demo" -Port 5000
    .\deploy\install-windows-service.ps1 -ProjectRoot "D:\sdn_demo" -ProcessedDir "data\processed_sdn" -ModelTag "sdn_ctrl"

.NOTES
    需要先准备 nssm.exe：https://nssm.cc/download （解压后把 win64\nssm.exe 放进 PATH，
    或用 -NssmPath 指定绝对路径）。卸载：nssm remove cida-detect confirm
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ProjectRoot,
    [string]$ServiceName  = "cida-detect",
    [int]$Port            = 5000,
    [string]$HostAddress  = "127.0.0.1",
    [string]$ProcessedDir = "data\processed",
    [string]$ModelTag     = "",
    [string]$NssmPath     = "nssm.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) { throw "项目目录不存在：$ProjectRoot" }
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "找不到虚拟环境解释器：$python（请先按 DEPLOYMENT.md 第 1 节创建并安装依赖）" }

if (-not (Get-Command $NssmPath -ErrorAction SilentlyContinue)) {
    throw "找不到 nssm（$NssmPath）。请从 https://nssm.cc/download 下载后加入 PATH，或用 -NssmPath 指定绝对路径。"
}

$logDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$appArgs = @("app\app.py", "--host", $HostAddress, "--port", "$Port", "--processed-dir", $ProcessedDir)
if ($ModelTag) { $appArgs += @("--model-tag", $ModelTag) }

Write-Host "[1/4] 安装服务 $ServiceName ..." -ForegroundColor Cyan
& $NssmPath install $ServiceName $python
& $NssmPath set $ServiceName AppParameters ($appArgs -join " ")
& $NssmPath set $ServiceName AppDirectory $ProjectRoot          # 工作目录 = 项目根，模型与产物按相对路径解析
& $NssmPath set $ServiceName DisplayName "CIDA SDN 异常流量检测服务"
& $NssmPath set $ServiceName Description "CNN+BiLSTM 在线异常检测（在线告警级指标 + 处置策略 + 审计台账）"
& $NssmPath set $ServiceName Start SERVICE_AUTO_START
& $NssmPath set $ServiceName AppExit Default Restart            # 崩溃自动重启
& $NssmPath set $ServiceName AppRestartDelay 5000

Write-Host "[2/4] 配置日志与安全默认值 ..." -ForegroundColor Cyan
& $NssmPath set $ServiceName AppStdout (Join-Path $logDir "service_out.log")
& $NssmPath set $ServiceName AppStderr (Join-Path $logDir "service_err.log")
& $NssmPath set $ServiceName AppRotateFiles 1
& $NssmPath set $ServiceName AppRotateBytes 10485760

# 以「只观察 + 演练」启动：上线首周只看不动，确认无误后再改成 block / 0
& $NssmPath set $ServiceName AppEnvironmentExtra `
    "PYTHONIOENCODING=utf-8" "TF_CPP_MIN_LOG_LEVEL=2" `
    "CIDA_POLICY_ACTION=observe" "CIDA_DRY_RUN=1" "CIDA_CONFIRM_WINDOWS=3"

Write-Host "[3/4] 启动服务 ..." -ForegroundColor Cyan
& $NssmPath start $ServiceName
Start-Sleep -Seconds 12

Write-Host "[4/4] 健康检查 http://${HostAddress}:${Port}/api/health ..." -ForegroundColor Cyan
try {
    $hp = Invoke-RestMethod "http://${HostAddress}:${Port}/api/health" -TimeoutSec 15
    Write-Host "服务健康：$($hp | ConvertTo-Json -Compress)" -ForegroundColor Green
    Write-Host "大屏地址： http://${HostAddress}:${Port}/" -ForegroundColor Green
    Write-Host "策略查询： curl http://${HostAddress}:${Port}/api/policy" -ForegroundColor Green
} catch {
    Write-Warning "健康检查失败：$($_.Exception.Message)"
    Write-Warning "请看 $(Join-Path $logDir 'service_err.log')；常见原因：未跑 run_all.py、模型与 --processed-dir 口径不匹配、端口被占用。"
    Write-Warning "排查清单见 DEPLOYMENT.md 第 6 节。"
}
