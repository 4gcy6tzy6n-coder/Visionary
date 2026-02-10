#!/bin/bash

# Text2Loc Visionary 语音交互依赖安装脚本
# 安装语音识别所需的依赖

echo "=========================================="
echo "Text2Loc Visionary 语音交互依赖安装"
echo "=========================================="

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查Node.js环境
echo -e "\n${BLUE}[1/3] 检查Node.js环境...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}错误: 未找到Node.js，请先安装Node.js${NC}"
    echo "下载地址: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version)
echo -e "${GREEN}Node.js版本: $NODE_VERSION${NC}"

# 检查npm
echo -e "\n${BLUE}[2/3] 检查npm...${NC}"
if ! command -v npm &> /dev/null; then
    echo -e "${RED}错误: 未找到npm${NC}"
    exit 1
fi

NPM_VERSION=$(npm --version)
echo -e "${GREEN}npm版本: $NPM_VERSION${NC}"

# 安装依赖
echo -e "\n${BLUE}[3/3] 安装语音交互依赖...${NC}"

# 检查是否已有node_modules
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}首次安装，安装所有依赖...${NC}"
    npm install
else
    echo -e "${GREEN}node_modules已存在${NC}"
fi

# 安装语音识别相关依赖
echo -e "\n${YELLOW}安装语音识别依赖...${NC}"
npm install react-speech-recognition regenerator-runtime --save

echo -e "\n${GREEN}=========================================="
echo "依赖安装完成！"
echo "==========================================${NC}"
echo ""
echo "现在您可以运行以下命令启动语音交互版本："
echo -e "${BLUE}cd \"/Users/yaoyingliang/Desktop/Text2Loc-main/Text2Loc visionary\"${NC}"
echo -e "${BLUE}./start_local.sh${NC}"
echo ""
echo "然后在浏览器中打开："
echo -e "${BLUE}http://localhost:5173/voice.html${NC}"
echo ""
echo "或者访问文本版（也支持语音输入）："
echo -e "${BLUE}http://localhost:5173/${NC}"
