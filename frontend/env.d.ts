
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      // ============ 基础配置 ============
      /** 环境模式: development | production | test */
      NODE_ENV: string;

      // ============ 应用配置 ============
      /** 应用名称 */
      VITE_APP_NAME: string;
      /** 应用版本 */
      VITE_APP_VERSION: string;
      /** 应用描述 */
      VITE_APP_DESCRIPTION: string;
      /** 基础路径，用于部署在子目录下 */
      VITE_BASE_PATH: string;

      // ============ API配置 ============
      /** API服务器地址 */
      VITE_API_BASE_URL: string;
      /** WebSocket服务器地址 */
      VITE_WS_BASE_URL: string;
      /** API超时时间（毫秒） */
      VITE_API_TIMEOUT: string;
      /** API密钥（可选） */
      VITE_API_KEY?: string;
      /** 是否启用API模拟模式 */
      VITE_API_MOCK_MODE: string;

      // ============ 服务器配置 ============
      /** 开发服务器端口 */
      VITE_PORT: string;
      /** 预览服务器端口 */
      VITE_PREVIEW_PORT: string;
      /** 服务器主机名 */
      VITE_HOST: string;
      /** 是否自动打开浏览器 */
      VITE_OPEN_BROWSER: string;

      // ============ 功能开关 ============
      /** 是否启用增强模式 */
      VITE_ENABLE_ENHANCED_MODE: string;
      /** 是否启用3D可视化 */
      VITE_ENABLE_3D_VISUALIZATION: string;
      /** 是否启用语音输入 */
      VITE_ENABLE_VOICE_INPUT: string;
      /** 是否启用离线模式 */
      VITE_ENABLE_OFFLINE_MODE: string;
      /** 是否启用实时协作 */
      VITE_ENABLE_REALTIME_COLLAB: string;

      // ============ 性能配置 ============
      /** 3D渲染质量: low | medium | high | ultra */
      VITE_3D_QUALITY: string;
      /** 点云细节级别: 1-10 */
      VITE_POINTCLOUD_LOD: string;
      /** 是否启用GPU加速 */
      VITE_ENABLE_GPU_ACCELERATION: string;
      /** 最大并发请求数 */
      VITE_MAX_CONCURRENT_REQUESTS: string;
      /** 缓存大小限制（MB） */
      VITE_CACHE_SIZE_LIMIT: string;

      // ============ 监控与调试 ============
      /** Sentry DSN（错误监控） */
      VITE_SENTRY_DSN?: string;
      /** Google Analytics ID */
      VITE_GA_ID?: string;
      /** 是否启用性能监控 */
      VITE_ENABLE_PERFORMANCE_MONITORING: string;
      /** 是否启用详细日志 */
      VITE_ENABLE_VERBOSE_LOGGING: string;
      /** 调试模式开关 */
      VITE_DEBUG_MODE: string;

      // ============ 第三方服务 ============
      /** Mapbox访问令牌 */
      VITE_MAPBOX_TOKEN?: string;
      /** OpenStreetMap配置 */
      VITE_OSM_CONFIG?: string;
      /** 地理编码服务URL */
      VITE_GEOCODING_SERVICE_URL?: string;

      // ============ 安全配置 ============
      /** 内容安全策略 */
      VITE_CSP_CONFIG?: string;
      /** 是否启用严格模式 */
      VITE_STRICT_MODE: string;
      /** 是否启用CORS */
      VITE_ENABLE_CORS: string;

      // ============ 构建优化 ============
      /** 是否启用包分析 */
      VITE_BUNDLE_ANALYSIS: string;
      /** 是否启用代码压缩 */
      VITE_ENABLE_COMPRESSION: string;
      /** 是否启用Tree Shaking */
      VITE_ENABLE_TREE_SHAKING: string;
      /** 是否启用Source Map */
      VITE_ENABLE_SOURCE_MAP: string;

      // ============ 实验性功能 ============
      /** 是否启用WebAssembly */
      VITE_ENABLE_WASM: string;
      /** 是否启用WebGPU（实验性） */
      VITE_ENABLE_WEBGPU: string;
      /** 是否启用AR模式（实验性） */
      VITE_ENABLE_AR_MODE: string;
      /** 是否启用机器学习推理（实验性） */
      VITE_ENABLE_ML_INFERENCE: string;

      // ============ 本地化配置 ============
      /** 默认语言 */
      VITE_DEFAULT_LANGUAGE: string;
      /** 支持的语言列表 */
      VITE_SUPPORTED_LANGUAGES: string;
      /** 是否启用RTL布局 */
      VITE_ENABLE_RTL: string;

      // ============ 无障碍访问 ============
      /** 是否启用无障碍模式 */
      VITE_ENABLE_ACCESSIBILITY_MODE: string;
      /** 高对比度模式 */
      VITE_HIGH_CONTRAST_MODE: string;
      /** 字体大小缩放因子 */
      VITE_FONT_SIZE_SCALE: string;
    }
  }
}

// 增强 import.meta.env 的类型定义
interface ImportMetaEnv {
  // ============ 基础配置 ============
  /** 环境模式: development | production | test */
  readonly MODE: string;
  /** 构建版本 */
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly SSR: boolean;

  // ============ 应用配置 ============
  /** 应用名称 */
  readonly VITE_APP_NAME: string;
  /** 应用版本 */
  readonly VITE_APP_VERSION: string;
  /** 应用描述 */
  readonly VITE_APP_DESCRIPTION: string;
  /** 基础路径，用于部署在子目录下 */
  readonly VITE_BASE_PATH: string;

  // ============ API配置 ============
  /** API服务器地址 */
  readonly VITE_API_BASE_URL: string;
  /** WebSocket服务器地址 */
  readonly VITE_WS_BASE_URL: string;
  /** API超时时间（毫秒） */
  readonly VITE_API_TIMEOUT: string;
  /** API密钥（可选） */
  readonly VITE_API_KEY?: string;
  /** 是否启用API模拟模式 */
  readonly VITE_API_MOCK_MODE: string;

  // ============ 服务器配置 ============
  /** 开发服务器端口 */
  readonly VITE_PORT: string;
  /** 预览服务器端口 */
  readonly VITE_PREVIEW_PORT: string;
  /** 服务器主机名 */
  readonly VITE_HOST: string;
  /** 是否自动打开浏览器 */
  readonly VITE_OPEN_BROWSER: string;

  // ============ 功能开关 ============
  /** 是否启用增强模式 */
  readonly VITE_ENABLE_ENHANCED_MODE: string;
  /** 是否启用3D可视化 */
  readonly VITE_ENABLE_3D_VISUALIZATION: string;
  /** 是否启用语音输入 */
  readonly VITE_ENABLE_VOICE_INPUT: string;
  /** 是否启用离线模式 */
  readonly VITE_ENABLE_OFFLINE_MODE: string;
  /** 是否启用实时协作 */
  readonly VITE_ENABLE_REALTIME_COLLAB: string;

  // ============ 性能配置 ============
  /** 3D渲染质量: low | medium | high | ultra */
  readonly VITE_3D_QUALITY: string;
  /** 点云细节级别: 1-10 */
  readonly VITE_POINTCLOUD_LOD: string;
  /** 是否启用GPU加速 */
  readonly VITE_ENABLE_GPU_ACCELERATION: string;
  /** 最大并发请求数 */
  readonly VITE_MAX_CONCURRENT_REQUESTS: string;
  /** 缓存大小限制（MB） */
  readonly VITE_CACHE_SIZE_LIMIT: string;

  // ============ 监控与调试 ============
  /** Sentry DSN（错误监控） */
  readonly VITE_SENTRY_DSN?: string;
  /** Google Analytics ID */
  readonly VITE_GA_ID?: string;
  /** 是否启用性能监控 */
  readonly VITE_ENABLE_PERFORMANCE_MONITORING: string;
  /** 是否启用详细日志 */
  readonly VITE_ENABLE_VERBOSE_LOGGING: string;
  /** 调试模式开关 */
  readonly VITE_DEBUG_MODE: string;

  // ============ 第三方服务 ============
  /** Mapbox访问令牌 */
  readonly VITE_MAPBOX_TOKEN?: string;
  /** OpenStreetMap配置 */
  readonly VITE_OSM_CONFIG?: string;
  /** 地理编码服务URL */
  readonly VITE_GEOCODING_SERVICE_URL?: string;

  // ============ 安全配置 ============
  /** 内容安全策略 */
  readonly VITE_CSP_CONFIG?: string;
  /** 是否启用严格模式 */
  readonly VITE_STRICT_MODE: string;
  /** 是否启用CORS */
  readonly VITE_ENABLE_CORS: string;

  // ============ 构建优化 ============
  /** 是否启用包分析 */
  readonly VITE_BUNDLE_ANALYSIS: string;
  /** 是否启用代码压缩 */
  readonly VITE_ENABLE_COMPRESSION: string;
  /** 是否启用Tree Shaking */
  readonly VITE_ENABLE_TREE_SHAKING: string;
  /** 是否启用Source Map */
  readonly VITE_ENABLE_SOURCE_MAP: string;

  // ============ 实验性功能 ============
  /** 是否启用WebAssembly */
  readonly VITE_ENABLE_WASM: string;
  /** 是否启用WebGPU（实验性） */
  readonly VITE_ENABLE_WEBGPU: string;
  /** 是否启用AR模式（实验性） */
  readonly VITE_ENABLE_AR_MODE: string;
  /** 是否启用机器学习推理（实验性） */
  readonly VITE_ENABLE_ML_INFERENCE: string;

  // ============ 本地化配置 ============
  /** 默认语言 */
  readonly VITE_DEFAULT_LANGUAGE: string;
  /** 支持的语言列表 */
  readonly VITE_SUPPORTED_LANGUAGES: string;
  /** 是否启用RTL布局 */
  readonly VITE_ENABLE_RTL: string;

  // ============ 无障碍访问 ============
  /** 是否启用无障碍模式 */
  readonly VITE_ENABLE_ACCESSIBILITY_MODE: string;
  /** 高对比度模式 */
  readonly VITE_HIGH_CONTRAST_MODE: string;
  /** 字体大小缩放因子 */
  readonly VITE_FONT_SIZE_SCALE: string;

  // ============ 其他环境变量 ============
  [key: `VITE_${string}`]: string | undefined;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// 导出全局类型，确保模块化
export {};
