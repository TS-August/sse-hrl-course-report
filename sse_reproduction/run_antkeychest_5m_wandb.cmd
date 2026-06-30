@echo off
setlocal
powershell.exe -ExecutionPolicy Bypass -File "%~dp0run_antkeychest_5m_wandb.ps1" %*
