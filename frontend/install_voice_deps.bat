@echo off
REM Text2Loc Visionary 语音交互依赖安装脚本 (Windows版本)
REM 安装语音识别所需的依赖

echo ==========================================
echo Text2Loc Visionary 语音交互依赖安装
echo ==========================================

REM 获取脚本所在目录
cd /d "%~dp0"

REM 检查Node.js环境
echo.
echo [1/3] 检查Node.js环境...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Node.js，请先安装Node.js
    echo 下载地址: https://nodejs.org/
    pause
    exit /b 1
)
for /f "tokens=*" %%a in ('node --version') do set NODE_VERSION=%%a
echo Node.js版本: %NODE_VERSION%

REM 检查npm
echo.
echo [2/3] 检查npm...
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到npm
    pause
    exit /b 1
)
for /f "tokens=*" %%a in ('npm --version') do set NPM_VERSION=%%a
echo npm版本: %NPM_VERSION%

REM 安装依赖
echo.
echo [3/3] 安装语音交互依赖...

REM 检查是否已有node_modules
if not exist "node_modules" (
    echo 首次安装，安装所有依赖...
    call npm install
) else (
    echo node_modules已存在
)

REM 安装语音识别相关依赖
echo 安装语音识别依赖...
call npm install react-speech-recognition regenerator-runtime --save

echo.
echo ==========================================
echo 依赖安装完成！
echo ==========================================
echo.
echo 现在您可以运行以下命令启动语音交互版本：
echo cd "C:\Users\yaoyingliang\Desktop\Text2Loc-main\Text2Loc visionary"
echo start_local.bat
echo.
echo 然后在浏览器中打开：
echo http://localhost:5173/voice.html
echo.
echo 或者访问文本版（也支持语音输入）：
echo http://localhost:5173/
echo.

pause
