"""
Flask后端服务 - Text2Loc Visionary

提供REST API接口，端口使用6000+避免冲突
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, Any
import time

# 加载环境变量（无日志）
try:
    import dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
            dotenv.load_dotenv(env_path)
except ImportError:
    pass

from .monitoring import get_monitor, setup_logging

setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)

# 启动日志显示环境变量状态
deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
if deepseek_key:
    logger.info(f"DeepSeek API Key: {deepseek_key[:10]}...")
    logger.info("使用 DeepSeek AI 模型进行解析")
else:
    logger.info("未配置 DeepSeek API Key，将使用规则解析")


class Text2LocServer:
    """Text2Loc Visionary服务器"""
    
    def __init__(self, host='0.0.0.0', port=8080, debug=False):
        """
        初始化服务器
        
        Args:
            host: 主机地址
            port: 端口号（默认8080，使用6000+端口避免冲突）
            debug: 调试模式
        """
        self.host = host
        self.port = port
        self.debug = debug
        
        # 从环境变量读取端口配置（支持自定义）
        self.port = int(os.environ.get('TEXT2LOC_PORT', port))
        
        # 创建Flask应用
        self.app = Flask(__name__)
        self.app.config['JSON_SORT_KEYS'] = False
        
        # 启用CORS - 允许所有来源（用于iPhone远程麦克风）
        CORS(self.app, resources={
            r"/api/*": {
                "origins": "*",
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"]
            }
        })
        
        # 初始化API
        self.api = None
        
        # 注册路由
        self._register_routes()
        
        logger.info(f"Text2Loc服务器初始化完成，端口: {self.port}")
    
    def set_api(self, api):
        """设置API实例"""
        self.api = api
    
    def _register_routes(self):
        """注册API路由"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """健康检查"""
            return jsonify({
                'status': 'healthy',
                'service': 'text2loc-visionary',
                'port': self.port,
                'version': '1.0.0'
            })
        
        @self.app.route('/api/v1/query', methods=['POST'])
        def process_query():
            """
            处理查询请求
            
            请求体:
            {
                "query": "自然语言描述",
                "top_k": 5,
                "enable_enhanced": true,
                "return_debug_info": false
            }
            """
            try:
                start_time = time.time()
                
                # 解析请求
                data = request.get_json()
                if not data:
                    return jsonify({'error': '请求体为空'}), 400
                
                query = data.get('query', '')
                if not query:
                    return jsonify({'error': '查询内容为空'}), 400
                
                top_k = data.get('top_k', 5)
                enable_enhanced = data.get('enable_enhanced', True)
                return_debug = data.get('return_debug_info', False)
                
                # 处理查询
                from .text2loc_api import QueryRequest
                request_obj = QueryRequest(
                    query=query,
                    top_k=top_k,
                    enable_enhanced=enable_enhanced,
                    return_debug_info=return_debug
                )
                
                response = self.api.process_query(request_obj)
                
                # 转换为字典
                result = response.to_dict()
                processing_time_ms = (time.time() - start_time) * 1000
                result['processing_time_ms'] = processing_time_ms
                
                # 记录监控指标
                get_monitor().record_query(
                    query_id=result.get('query_id', f'query_{int(time.time())}'),
                    duration_ms=processing_time_ms,
                    success=result.get('status') == 'success',
                    details={'enable_enhanced': enable_enhanced}
                )
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"查询处理失败: {e}")
                get_monitor().track_error(
                    error_type=type(e).__name__,
                    message=str(e),
                    module='api',
                    details={'query': data.get('query', '') if 'data' in dir() else ''}
                )
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500
        
        @self.app.route('/api/v1/status', methods=['GET'])
        def get_status():
            """获取服务状态"""
            if self.api:
                return jsonify(self.api.get_status())
            return jsonify({
                'status': 'running',
                'adapter': 'not initialized',
                'port': self.port
            })
        
        @self.app.route('/api/v1/config', methods=['GET'])
        def get_config():
            """获取当前模型配置"""
            try:
                from .config_api import get_config_manager
                config_manager = get_config_manager()
                config = config_manager.get_config()
                
                return jsonify({
                    'status': 'success',
                    'provider': config.get('provider'),
                    'model': config.get('model'),
                    'url': config.get('base_url'),
                    'api_key': config.get('api_key'),
                    'is_configured': bool(config.get('model') and config.get('base_url'))
                })
            except Exception as e:
                logger.error(f"获取配置失败: {e}")
                return jsonify({'status': 'error', 'error': str(e)}), 500
        
        @self.app.route('/api/v1/config', methods=['POST'])
        def update_config():
            """更新模型配置"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': '请求体为空'}), 400
                
                from .config_api import get_config_manager
                config_manager = get_config_manager()
                config_manager.update_config(data)
                
                # 重新初始化 NLU 引擎
                if self.api:
                    self.api._init_nlu_engine()
                
                return jsonify({'status': 'success', 'message': '配置已更新'})
            except Exception as e:
                logger.error(f"更新配置失败: {e}")
                return jsonify({'status': 'error', 'error': str(e)}), 500
        
        @self.app.route('/api/v1/config', methods=['DELETE'])
        def reset_config():
            """重置模型配置"""
            try:
                from .config_api import get_config_manager
                config_manager = get_config_manager()
                config_manager.reset_config()
                
                # 重新初始化 NLU 引擎
                if self.api:
                    self.api._init_nlu_engine()
                
                return jsonify({'status': 'success', 'message': '配置已重置'})
            except Exception as e:
                logger.error(f"重置配置失败: {e}")
                return jsonify({'status': 'error', 'error': str(e)}), 500
        
        @self.app.route('/api/v1/config/test', methods=['POST'])
        def test_config():
            """测试模型配置"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': '请求体为空'}), 400
                
                from .config_api import ConfigManager, ModelConfig
                
                # 创建临时配置管理器进行测试
                temp_config = ModelConfig(
                    provider=data.get('provider', 'deepseek'),
                    api_key=data.get('api_key', ''),
                    base_url=data.get('url', ''),
                    model=data.get('model', '')
                )
                
                # 使用临时配置测试
                temp_manager = ConfigManager()
                temp_manager.config = temp_config
                result = temp_manager.test_connection()
                
                return jsonify(result)
            except Exception as e:
                logger.error(f"测试配置失败: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/v1/metrics', methods=['GET'])
        def get_metrics():
            """获取性能指标"""
            return jsonify(get_monitor().get_system_status())
        
        @self.app.route('/api/v1/errors', methods=['GET'])
        def get_errors():
            """获取错误信息"""
            return jsonify(get_monitor().error_tracker.get_error_summary())
        
        @self.app.route('/api/v1/feedback', methods=['POST'])
        def submit_feedback():
            """提交用户反馈"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': '请求体为空'}), 400
                
                query_id = data.get('query_id', '')
                rating = data.get('rating', 3)
                comment = data.get('comment', '')
                query_text = data.get('query_text', '')
                result_quality = data.get('result_quality', 'neutral')
                
                if not 1 <= rating <= 5:
                    return jsonify({'error': '评分必须在1-5之间'}), 400
                
                get_monitor().collect_feedback(
                    query_id=query_id,
                    rating=rating,
                    comment=comment,
                    query_text=query_text,
                    result_quality=result_quality
                )
                
                return jsonify({'status': 'success', 'message': '感谢您的反馈'})
                
            except Exception as e:
                logger.error(f"反馈提交失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/', methods=['GET'])
        def index():
            """首页"""
            return jsonify({
                'service': 'Text2Loc Visionary API',
                'version': '1.0.0',
                'documentation': '/api/v1/docs',
                'endpoints': {
                    'health': '/health',
                    'query': '/api/v1/query',
                    'status': '/api/v1/status',
                    'config': '/api/v1/config',
                    'remote_voice': '/api/v1/remote-voice',
                    'iphone_mic': '/iphone-remote-mic.html'
                }
            })
        
        @self.app.route('/iphone-remote-mic.html', methods=['GET'])
        def iphone_remote_mic():
            """提供iPhone远程麦克风页面"""
            from flask import send_from_directory
            import os
            frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
            return send_from_directory(frontend_dir, 'iphone-remote-mic.html')
        
        @self.app.route('/advanced-voice.html', methods=['GET'])
        def advanced_voice_recognition():
            """提供高精度语音识别页面"""
            from flask import send_from_directory
            import os
            frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
            return send_from_directory(frontend_dir, 'advanced-voice-recognition.html')
        
        @self.app.route('/test-connection.html', methods=['GET'])
        def connection_test():
            """提供连接测试页面"""
            from flask import send_from_directory
            import os
            frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
            return send_from_directory(frontend_dir, 'connection-test.html')
        
        @self.app.route('/test-iphone-voice.html', methods=['GET'])
        def test_iphone_voice():
            """iPhone远程语音功能测试页面"""
            from flask import send_from_directory
            import os
            frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
            return send_from_directory(frontend_dir, 'test-iphone-voice.html')
        
        @self.app.route('/api/v1/remote-voice', methods=['POST'])
        def receive_remote_voice():
            """
            接收iPhone远程语音输入
            
            请求体:
            {
                "text": "识别到的文本",
                "timestamp": "时间戳",
                "device": "iPhone",
                "engine": "识别引擎" (browser/whisper/azure/aliyun)
            }
            
            响应:
            {
                "status": "success",
                "query_id": "生成的查询ID",
                "result": "定位结果"
            }
            """
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': '请求体为空'}), 400
                
                voice_text = data.get('text', '').strip()
                if not voice_text:
                    return jsonify({'error': '语音文本为空'}), 400
                
                engine = data.get('engine', 'browser')
                
                # 保存到全局变量（临时存储）
                if not hasattr(Text2LocServer, '_remote_voice_queue'):
                    Text2LocServer._remote_voice_queue = []
                
                query_id = f"remote_{int(time.time() * 1000)}"
                
                # 存储语音文本和查询ID
                Text2LocServer._remote_voice_queue.append({
                    'query_id': query_id,
                    'text': voice_text,
                    'timestamp': data.get('timestamp', ''),
                    'device': data.get('device', 'iPhone'),
                    'engine': engine,
                    'processed': False,
                    'result': None
                })
                
                logger.info(f"收到语音输入 [引擎: {engine}]: {voice_text}")
                
                # 直接处理查询
                try:
                    from .text2loc_api import QueryRequest
                    request_obj = QueryRequest(
                        query=voice_text,
                        top_k=5,
                        enable_enhanced=True,
                        return_debug_info=False
                    )
                    
                    response = self.api.process_query(request_obj)
                    result = response.to_dict()
                    
                    # 更新队列中的结果
                    for item in Text2LocServer._remote_voice_queue:
                        if item['query_id'] == query_id:
                            item['processed'] = True
                            item['result'] = result
                            break
                    
                    return jsonify({
                        'status': 'success',
                        'query_id': query_id,
                        'result': result,
                        'processing_time_ms': result.get('processing_time_ms', 0),
                        'engine_used': engine
                    })
                    
                except Exception as api_error:
                    logger.error(f"处理远程语音查询失败: {api_error}")
                    return jsonify({
                        'status': 'success',
                        'query_id': query_id,
                        'message': '文本已接收，处理失败',
                        'error': str(api_error)
                    })
                
            except Exception as e:
                logger.error(f"接收远程语音失败: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/remote-voice/result/<query_id>', methods=['GET'])
        def get_remote_voice_result(query_id):
            """
            获取远程语音处理结果（轮询）
            """
            try:
                if not hasattr(Text2LocServer, '_remote_voice_queue'):
                    return jsonify({'status': 'pending', 'message': '等待语音输入'})
                
                for item in Text2LocServer._remote_voice_queue:
                    if item['query_id'] == query_id:
                        if item['processed']:
                            return jsonify({
                                'status': 'success',
                                'query_id': query_id,
                                'text': item['text'],
                                'result': item['result']
                            })
                        else:
                            return jsonify({'status': 'processing'})
                
                return jsonify({'status': 'not_found'}), 404
                
            except Exception as e:
                logger.error(f"获取远程语音结果失败: {e}")
                return jsonify({'error': str(e)}), 500
    
    def run(self):
        """启动服务器"""
        logger.info(f"启动Text2Loc服务器，端口: {self.port}")
        self.app.run(
            host=self.host,
            port=self.port,
            debug=self.debug
        )


def create_server(host='0.0.0.0', port=8080, debug=False) -> Text2LocServer:
    """
    创建服务器实例
    
    Args:
        host: 主机地址
        port: 端口号（使用6000+端口）
        debug: 调试模式
        
    Returns:
        Text2LocServer实例
    """
    server = Text2LocServer(host=host, port=port, debug=debug)
    
    # 创建API并设置到服务器
    from .text2loc_api import create_api
    api = create_api()
    server.set_api(api)
    
    return server


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Text2Loc Visionary Server')
    parser.add_argument('--host', default='0.0.0.0', help='主机地址')
    parser.add_argument('--port', type=int, default=8080, help='端口号（使用6000+端口避免冲突）')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 创建并启动服务器
    server = create_server(host=args.host, port=args.port, debug=args.debug)
    server.run()
