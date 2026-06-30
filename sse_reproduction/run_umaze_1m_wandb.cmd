@echo off
setlocal
powershell.exe -ExecutionPolicy Bypass -File "%~dp0run_umaze_1m_wandb.ps1" %*
