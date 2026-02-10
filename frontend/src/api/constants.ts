// API常量配置 - Text2Loc Visionary
// 包含所有API相关的配置常量

// ============ 基础配置 ============
export const API_CONFIG = {
  // API服务器地址
  baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080',

  // WebSocket服务器地址
  wsBaseUrl: import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8080',

  // API请求超时时间（毫秒）
  timeout: parseInt(import.meta.env.VITE_API_TIMEOUT || '30000'),

  // 最大并发请求数
  maxConcurrentRequests: parseInt(import.meta.env.VITE_MAX_CONCURRENT_REQUESTS || '10'),

  // 最大内容长度（字节）
  maxContentLength: 50 * 1024 * 1024, // 50MB

  // 最大请求体长度（字节）
  maxBodyLength: 50 * 1024 * 1024, // 50MB

  // 是否发送凭据（cookies）
  withCredentials: false,

  // API密钥（可选）
  apiKey: import.meta.env.VITE_API_KEY || '',

  // 客户端版本
  clientVersion: import.meta.env.VITE_APP_VERSION || '1.0.0',

  // 缓存TTL（秒）
  cacheTTL: parseInt(import.meta.env.VITE_CACHE_TTL || '300'),

  // 是否启用API模拟模式
  mockMode: import.meta.env.VITE_API_MOCK_MODE === 'true',

  // 是否启用详细日志
  verboseLogging: import.meta.env.VITE_ENABLE_VERBOSE_LOGGING === 'true',
} as const;

// ============ 认证配置 ============
export const AUTH_CONFIG = {
  // 认证令牌存储键名
  tokenKey: 'text2loc_auth_token',
  tokenExpiryKey: 'text2loc_auth_token_expiry',
  refreshTokenKey: 'text2loc_auth_refresh_token',

  // 认证端点
  loginEndpoint: '/api/v1/auth/login',
  logoutEndpoint: '/api/v1/auth/logout',
  refreshEndpoint: '/api/v1/auth/refresh',
  registerEndpoint: '/api/v1/auth/register',
  verifyEndpoint: '/api/v1/auth/verify',

  // 登录重定向URL
  loginUrl: '/login',

  // 令牌刷新阈值（毫秒，令牌过期前多久开始刷新）
  refreshThreshold: 5 * 60 * 1000, // 5分钟

  // 令牌过期时间（秒）
  tokenExpiry: 24 * 60 * 60, // 24小时
  refreshTokenExpiry: 7 * 24 * 60 * 60, // 7天
} as const;

// ============ 重试策略配置 ============
export const RETRY_CONFIG = {
  // 最大重试次数
  maxRetries: 3,

  // 基础重试延迟（毫秒）
  baseDelay: 1000,

  // 最大重试延迟（毫秒）
  maxDelay: 10000,

  // 需要重试的状态码
  retryStatusCodes: [
    408, // Request Timeout
    429, // Too Many Requests
    500, // Internal Server Error
    502, // Bad Gateway
    503, // Service Unavailable
    504, // Gateway Timeout
  ],

  // 需要重试的网络错误
  retryNetworkErrors: [
    'ECONNABORTED',
    'ETIMEDOUT',
    'ENETDOWN',
    'ENETUNREACH',
    'ENETRESET',
    'ECONNRESET',
    'ECONNREFUSED',
    'EHOSTDOWN',
    'EHOSTUNREACH',
  ],
} as const;

// ============ 错误代码定义 ============
export const ERROR_CODES = {
  // 网络错误
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT_ERROR: 'TIMEOUT_ERROR',
  CONNECTION_ERROR: 'CONNECTION_ERROR',

  // 认证错误
  UNAUTHORIZED: 'UNAUTHORIZED',
  FORBIDDEN: 'FORBIDDEN',
  INVALID_TOKEN: 'INVALID_TOKEN',
  EXPIRED_TOKEN: 'EXPIRED_TOKEN',
  INVALID_CREDENTIALS: 'INVALID_CREDENTIALS',

  // 请求错误
  BAD_REQUEST: 'BAD_REQUEST',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  INVALID_PARAMETERS: 'INVALID_PARAMETERS',
  MISSING_PARAMETERS: 'MISSING_PARAMETERS',

  // 服务器错误
  INTERNAL_SERVER_ERROR: 'INTERNAL_SERVER_ERROR',
  SERVICE_UNAVAILABLE: 'SERVICE_UNAVAILABLE',

  // 业务错误
  LOCATION_NOT_FOUND: 'LOCATION_NOT_FOUND',
  QUERY_PARSING_ERROR: 'QUERY_PARSING_ERROR',
  POINTCLOUD_LOAD_ERROR: 'POINTCLOUD_LOAD_ERROR',
  MODEL_INFERENCE_ERROR: 'MODEL_INFERENCE_ERROR',

  // 限制错误
  RATE_LIMIT_EXCEEDED: 'RATE_LIMIT_EXCEEDED',
  QUOTA_EXCEEDED: 'QUOTA_EXCEEDED',

  // 版本错误
  VERSION_MISMATCH: 'VERSION_MISMATCH',
  DEPRECATED_API: 'DEPRECATED_API',
} as const;

// ============ 错误消息定义 ============
export const ERROR_MESSAGES = {
  [ERROR_CODES.NETWORK_ERROR]: '网络连接错误，请检查网络设置',
  [ERROR_CODES.TIMEOUT_ERROR]: '请求超时，请稍后重试',
  [ERROR_CODES.CONNECTION_ERROR]: '服务器连接失败',

  [ERROR_CODES.UNAUTHORIZED]: '未授权访问，请先登录',
  [ERROR_CODES.FORBIDDEN]: '没有权限访问该资源',
  [ERROR_CODES.INVALID_TOKEN]: '无效的认证令牌',
  [ERROR_CODES.EXPIRED_TOKEN]: '认证令牌已过期',
  [ERROR_CODES.INVALID_CREDENTIALS]: '用户名或密码错误',

  [ERROR_CODES.BAD_REQUEST]: '请求参数错误',
  [ERROR_CODES.VALIDATION_ERROR]: '数据验证失败',
  [ERROR_CODES.INVALID_PARAMETERS]: '无效的请求参数',
  [ERROR_CODES.MISSING_PARAMETERS]: '缺少必要的请求参数',

  [ERROR_CODES.INTERNAL_SERVER_ERROR]: '服务器内部错误',
  [ERROR_CODES.SERVICE_UNAVAILABLE]: '服务暂时不可用',

  [ERROR_CODES.LOCATION_NOT_FOUND]: '未找到匹配的位置',
  [ERROR_CODES.QUERY_PARSING_ERROR]: '查询解析失败',
  [ERROR_CODES.POINTCLOUD_LOAD_ERROR]: '点云数据加载失败',
  [ERROR_CODES.MODEL_INFERENCE_ERROR]: '模型推理失败',

  [ERROR_CODES.RATE_LIMIT_EXCEEDED]: '请求频率过高，请稍后重试',
  [ERROR_CODES.QUOTA_EXCEEDED]: '超出使用配额',

  [ERROR_CODES.VERSION_MISMATCH]: '客户端版本不兼容',
  [ERROR_CODES.DEPRECATED_API]: '该API接口已废弃',
} as const;

// ============ API端点定义 ============
export const API_ENDPOINTS = {
  // 健康检查
  HEALTH: '/api/v1/health',
  STATUS: '/api/v1/status',

  // 核心定位功能
  LOCATE: '/api/v1/locate',
  BATCH_LOCATE: '/api/v1/batch/locate',

  // 增强功能
  ENHANCED_CAPABILITIES: '/api/v1/enhanced/capabilities',
  ENHANCED_PARSE: '/api/v1/enhanced/parse',
  ENHANCED_CONFIG: '/api/v1/enhanced/config',
  ENHANCED_STATS: '/api/v1/enhanced/stats',
  ENHANCED_FEEDBACK: '/api/v1/enhanced/feedback',

  // 点云数据
  POINTCLOUD_LIST: '/api/v1/pointclouds',
  POINTCLOUD_DETAIL: '/api/v1/pointclouds/{id}',
  POINTCLOUD_DOWNLOAD: '/api/v1/pointclouds/{id}/download',
  POINTCLOUD_METADATA: '/api/v1/pointclouds/{id}/metadata',

  // 查询历史
  QUERY_HISTORY: '/api/v1/queries/history',
  QUERY_DETAIL: '/api/v1/queries/{id}',
  QUERY_DELETE: '/api/v1/queries/{id}',
  QUERY_EXPORT: '/api/v1/queries/export',

  // 监控指标
  METRICS: '/api/v1/metrics',
  PERFORMANCE: '/api/v1/performance',
  USAGE_STATS: '/api/v1/usage/stats',

  // 配置管理
  SYSTEM_CONFIG: '/api/v1/config/system',
  USER_CONFIG: '/api/v1/config/user',
  MODEL_CONFIG: '/api/v1/config/model',

  // 用户管理
  USER_PROFILE: '/api/v1/user/profile',
  USER_PREFERENCES: '/api/v1/user/preferences',
  USER_API_KEYS: '/api/v1/user/api-keys',

  // 实时通信
  WS_LOCATE: '/ws/locate',
  SSE_UPDATES: '/sse/updates',
} as const;

// ============ 缓存配置 ============
export const CACHE_CONFIG = {
  // 缓存键前缀
  PREFIX: {
    API: 'api_cache_',
    POINTCLOUD: 'pointcloud_cache_',
    QUERY: 'query_cache_',
    USER: 'user_cache_',
  },

  // 缓存过期时间（秒）
  TTL: {
    SHORT: 60,           // 1分钟
    MEDIUM: 300,         // 5分钟
    LONG: 3600,          // 1小时
    VERY_LONG: 86400,    // 24小时
    PERMANENT: 604800,   // 7天
  },

  // 缓存大小限制（条目数）
  SIZE_LIMIT: {
    API: 100,
    POINTCLOUD: 10,
    QUERY: 50,
    USER: 20,
  },

  // 存储类型
  STORAGE: {
    SESSION: 'sessionStorage',
    LOCAL: 'localStorage',
    INDEXED_DB: 'indexedDB',
  },
} as const;

// ============ 请求优先级定义 ============
export const REQUEST_PRIORITY = {
  HIGHEST: 1,    // 最高优先级：用户交互、实时数据
  HIGH: 2,       // 高优先级：核心功能、重要数据
  NORMAL: 3,     // 普通优先级：常规请求
  LOW: 4,        // 低优先级：后台任务、预加载
  LOWEST: 5,     // 最低优先级：统计上报、日志记录
} as const;

// ============ 内容类型定义 ============
export const CONTENT_TYPES = {
  JSON: 'application/json',
  FORM_DATA: 'multipart/form-data',
  FORM_URLENCODED: 'application/x-www-form-urlencoded',
  TEXT: 'text/plain',
  HTML: 'text/html',
  XML: 'application/xml',
  BINARY: 'application/octet-stream',
  POINTCLOUD: 'application/pointcloud+json',
  GEOJSON: 'application/geo+json',
} as const;

// ============ 功能开关 ============
export const FEATURE_FLAGS = {
  // 是否启用增强模式
  ENHANCED_MODE: import.meta.env.VITE_ENABLE_ENHANCED_MODE === 'true',

  // 是否启用3D可视化
  VISUALIZATION_3D: import.meta.env.VITE_ENABLE_3D_VISUALIZATION === 'true',

  // 是否启用语音输入
  VOICE_INPUT: import.meta.env.VITE_ENABLE_VOICE_INPUT === 'true',

  // 是否启用离线模式
  OFFLINE_MODE: import.meta.env.VITE_ENABLE_OFFLINE_MODE === 'true',

  // 是否启用实时协作
  REALTIME_COLLAB: import.meta.env.VITE_ENABLE_REALTIME_COLLAB === 'true',

  // 是否启用GPU加速
  GPU_ACCELERATION: import.meta.env.VITE_ENABLE_GPU_ACCELERATION === 'true',

  // 是否启用性能监控
  PERFORMANCE_MONITORING: import.meta.env.VITE_ENABLE_PERFORMANCE_MONITORING === 'true',

  // 是否启用无障碍模式
  ACCESSIBILITY_MODE: import.meta.env.VITE_ENABLE_ACCESSIBILITY_MODE === 'true',
} as const;

// ============ 性能配置 ============
export const PERFORMANCE_CONFIG = {
  // 3D渲染质量
  QUALITY_3D: import.meta.env.VITE_3D_QUALITY || 'medium',

  // 点云细节级别
  POINTCLOUD_LOD: parseInt(import.meta.env.VITE_POINTCLOUD_LOD || '5'),

  // 点云分块大小（点数）
  POINTCLOUD_CHUNK_SIZE: 100000,

  // 最大点云大小（MB）
  MAX_POINTCLOUD_SIZE: 500,

  // 图片压缩质量（0-1）
  IMAGE_COMPRESSION_QUALITY: 0.8,

  // 纹理最大尺寸（像素）
  MAX_TEXTURE_SIZE: 2048,

  // 帧率限制（FPS）
  FPS_LIMIT: 60,

  // 内存使用警告阈值（MB）
  MEMORY_WARNING_THRESHOLD: 512,
  MEMORY_CRITICAL_THRESHOLD: 768,
} as const;

// ============ 类型定义 ============

// 请求优先级类型
export type RequestPriority = typeof REQUEST_PRIORITY[keyof typeof REQUEST_PRIORITY];

// 错误代码类型
export type ErrorCode = typeof ERROR_CODES[keyof typeof ERROR_CODES];

// 缓存TTL类型
export type CacheTTL = typeof CACHE_CONFIG.TTL[keyof typeof CACHE_CONFIG.TTL];

// 内容类型类型
export type ContentType = typeof CONTENT_TYPES[keyof typeof CONTENT_TYPES];

// API端点类型
export type ApiEndpoint = typeof API_ENDPOINTS[keyof typeof API_ENDPOINTS];

// ============ 工具函数 ============

/**
 * 获取完整的API URL
 * @param endpoint API端点
 * @returns 完整的URL
 */
export function getApiUrl(endpoint: string): string {
  return `${API_CONFIG.baseUrl}${endpoint}`;
}

/**
 * 获取WebSocket URL
 * @param endpoint WebSocket端点
 * @returns 完整的WebSocket URL
 */
export function getWsUrl(endpoint: string): string {
  const baseUrl = API_CONFIG.wsBaseUrl.replace(/^http/, 'ws');
  return `${baseUrl}${endpoint}`;
}

/**
 * 生成缓存键
 * @param prefix 缓存前缀
 * @param key 缓存键
 * @returns 完整的缓存键
 */
export function generateCacheKey(prefix: string, key: string): string {
  return `${prefix}${key}`;
}

/**
 * 检查是否为可重试的错误
 * @param error 错误对象
 * @returns 是否可以重试
 */
export function isRetryableError(error: any): boolean {
  if (!error) return false;

  // 检查状态码
  if (error.status && RETRY_CONFIG.retryStatusCodes.includes(error.status)) {
    return true;
  }

  // 检查错误代码
  if (error.code && RETRY_CONFIG.retryNetworkErrors.includes(error.code)) {
    return true;
  }

  // 检查是否为网络错误
  if (!error.status && !error.response) {
    return true;
  }

  return false;
}

/**
 * 获取错误消息
 * @param code 错误代码
 * @param defaultMessage 默认错误消息
 * @returns 错误消息
 */
export function getErrorMessage(code: string, defaultMessage?: string): string {
  return ERROR_MESSAGES[code as ErrorCode] || defaultMessage || '未知错误';
}

/**
 * 验证API配置
 * @throws 如果配置无效则抛出错误
 */
export function validateApiConfig(): void {
  const errors: string[] = [];

  if (!API_CONFIG.baseUrl) {
    errors.push('API基础URL未配置');
  }

  if (API_CONFIG.timeout <= 0) {
    errors.push('API超时时间必须大于0');
  }

  if (API_CONFIG.maxConcurrentRequests <= 0) {
    errors.push('最大并发请求数必须大于0');
  }

  if (errors.length > 0) {
    throw new Error(`API配置错误: ${errors.join(', ')}`);
  }
}

// 导出所有配置
export default {
  API_CONFIG,
  AUTH_CONFIG,
  RETRY_CONFIG,
  ERROR_CODES,
  ERROR_MESSAGES,
  API_ENDPOINTS,
  CACHE_CONFIG,
  REQUEST_PRIORITY,
  CONTENT_TYPES,
  FEATURE_FLAGS,
  PERFORMANCE_CONFIG,

  // 工具函数
  getApiUrl,
  getWsUrl,
  generateCacheKey,
  isRetryableError,
  getErrorMessage,
  validateApiConfig,
};
