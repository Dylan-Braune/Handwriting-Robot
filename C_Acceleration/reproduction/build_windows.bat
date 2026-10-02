@echo off
REM Build the Windows accel.dll from accel.c using the MinGW-w64 gcc.
REM Run this script from anywhere; it builds into this same directory.

set GCC="C:\Users\braun\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\gcc.exe"
set SRC=%~dp0accel.c
set OUT=%~dp0libaccel.dll

%GCC% -O3 -shared -o %OUT% %SRC% -lm

if %ERRORLEVEL% EQU 0 (
    echo Built %OUT%
) else (
    echo Build FAILED
    exit /b 1
)
