@echo off
REM Build the Windows test artifact (libaccel.dll) using the MinGW-w64 gcc.
REM This DLL is for local dev/testing only -- the ODROID deployment build
REM uses build_linux.sh on its own native gcc.

set GCC="C:\Users\braun\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\gcc.exe"

%GCC% -O3 -shared -o "%~dp0libaccel.dll" "%~dp0accel.c" -lm

if %ERRORLEVEL% EQU 0 (
    echo Built %~dp0libaccel.dll
) else (
    echo Build FAILED
    exit /b 1
)
