"""
前端模块 - Text2Loc Visionary

提供前端资源文件
"""

import os

# 前端配置
FRONTEND_CONFIG = {
    'dev_port': int(os.environ.get('FRONTEND_DEV_PORT', 6001)),  # 前端开发服务器端口
    'prod_port': int(os.environ.get('FRONTEND_PROD_PORT', 6002)),  # 前端生产服务器端口
    'api_port': int(os.environ.get('TEXT2LOC_PORT', 8080)),  # API服务端口
    'auto_open_browser': True  # 自动打开浏览器
}


def get_frontend_path():
    """获取前端资源路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, 'frontend')


def get_index_path():
    """获取首页HTML路径"""
    return os.path.join(get_frontend_path(), 'index.html')


def check_frontend_exists():
    """检查前端资源是否存在"""
    index_path = get_index_path()
    return os.path.exists(index_path)


if __name__ == '__main__':
    print("Frontend Configuration:")
    print(f"  Development port: {FRONTEND_CONFIG['dev_port']}")
    print(f"  Production port: {FRONTEND_CONFIG['prod_port']}")
    print(f"  API port: {FRONTEND_CONFIG['api_port']}")
    print(f"  Index file: {get_index_path()}")
    print(f"  Exists: {check_frontend_exists()}")
