@echo off
title Claude Code Launcher

echo ========================================
echo  Checking for Claude Code updates...
echo ========================================

:: Get current version
for /f "tokens=*" %%i in ('npm list -g @anthropic-ai/claude-code --depth=0 2^>nul ^| findstr claude-code') do set CURRENT=%%i

:: Get latest version from npm
for /f "tokens=*" %%i in ('npm view @anthropic-ai/claude-code version 2^>nul') do set LATEST=%%i

echo Current: %CURRENT%
echo Latest available: %LATEST%

echo %CURRENT% | findstr "%LATEST%" >nul 2>&1
if %errorlevel%==0 (
    echo Already up to date. No update needed.
) else (
    echo Update found! Updating Claude Code...
    npm install -g @anthropic-ai/claude-code@latest
    if %errorlevel%==0 (
        echo Update successful!
    ) else (
        echo Update failed! Continuing with current version...
    )
)

echo.
echo ========================================
echo  Starting Claude Code (emergency bypass)
echo ========================================
echo.

claude --dangerously-skip-permissions
pause
