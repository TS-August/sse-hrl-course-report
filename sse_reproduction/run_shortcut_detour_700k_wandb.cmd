@echo off
setlocal
powershell.exe -ExecutionPolicy Bypass -File "%~dp0run_shortcut_detour_700k_wandb.ps1" %*
