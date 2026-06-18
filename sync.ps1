$env:PATH = "C:\Program Files\Git\bin;" + $env:PATH
Set-Location "d:\NAS项目"

Write-Host "拉取远端更新..." -ForegroundColor Cyan
git pull origin main
if ($LASTEXITCODE -ne 0) { Write-Host "拉取失败，请检查网络或冲突" -ForegroundColor Red; exit 1 }

$changed = git status --porcelain
if (-not $changed) { Write-Host "无本地变更，已是最新" -ForegroundColor Green; exit 0 }

Write-Host "变更文件：" -ForegroundColor Yellow
git status --short

$msg = Read-Host "提交说明（直接回车使用时间戳）"
if (-not $msg) { $msg = "sync: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" }

git add -A
git commit -m $msg
git push origin main

if ($LASTEXITCODE -eq 0) { Write-Host "同步完成" -ForegroundColor Green }
else { Write-Host "推送失败" -ForegroundColor Red }
