@echo off
REM ============================================
REM Environment setup script for LLaMA3 + TornadoVM (Windows)
REM ============================================

REM Resolve the absolute path to this script's directory
set "LLAMA_ROOT=%~dp0"
set "LLAMA_ROOT=%LLAMA_ROOT:~0,-1%"

REM Set TornadoVM root and SDK paths
set "TORNADO_ROOT=%LLAMA_ROOT%\external\tornadovm"
set "TORNADOVM_HOME=%TORNADO_ROOT%\bin\sdk"

REM Add TornadoVM SDK and LLaMA3 bin to PATH
set "PATH=%TORNADOVM_HOME%;%LLAMA_ROOT%;%PATH%"

REM Optional: Set JAVA_HOME if needed
REM set "JAVA_HOME=C:\Path\To\GraalVM"
REM set "PATH=%JAVA_HOME%\bin;%PATH%"

echo [INFO] Environment configured for LLaMA3 with TornadoVM at: %TORNADO_ROOT%

REM ===== Notes =====
REM After running this script:
REM 1. TornadoVM will be available for GPU computation
REM 2. LLaMA3 command-line tools will be in your PATH
REM 3. You can run LLaMA3 with GPU acceleration using TornadoVM
REM
REM To use this script: call set_paths.cmd
