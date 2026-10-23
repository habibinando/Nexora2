@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0gradlew-bootstrap.ps1" %*
exit /b %ERRORLEVEL%
