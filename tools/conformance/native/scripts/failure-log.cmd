@echo off
setlocal
set EXE=%~dp0..\build\Release\conform_native.exe
set ROOT=%~dp0..\out\failures
set LOG=%~dp0..\out\failures.log
if exist "%ROOT%" rmdir /s /q "%ROOT%"
mkdir "%ROOT%"

> "%LOG%" echo === output path is an existing file
echo not a directory> "%ROOT%\blocked"
"%EXE%" "%ROOT%\blocked">> "%LOG%" 2>&1
>> "%LOG%" echo EXITCODE %errorlevel%
del "%ROOT%\blocked"

>> "%LOG%" echo.
>> "%LOG%" echo === shader directory with no .cso files
mkdir "%ROOT%\no-shaders"
"%EXE%" "%ROOT%\shader-failure" --shaders "%ROOT%\no-shaders">> "%LOG%" 2>&1
>> "%LOG%" echo EXITCODE %errorlevel%

>> "%LOG%" echo.
>> "%LOG%" echo === PNG path occupied by a directory
mkdir "%ROOT%\png-blocked"
mkdir "%ROOT%\png-blocked\clear.png"
"%EXE%" "%ROOT%\png-blocked">> "%LOG%" 2>&1
>> "%LOG%" echo EXITCODE %errorlevel%

type "%LOG%"
