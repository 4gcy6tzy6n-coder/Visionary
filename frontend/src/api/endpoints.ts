// API端点定义 - Text2Loc Visionary
// 包含所有API端点的类型定义和路径生成函数

import { ApiEndpoint } from './constants';

// ============ 类型定义 ============

/**
 * 查询请求参数
 */
export interface QueryRequest {
  /** 自然语言查询文本 */
  query: string;

  /** 地图ID */
  map_id: string;

  /** 查询选项 */
  options?: QueryOptions;

  /** 用户ID（可选） */
  user_id?: string;

  /** 会话ID（可选） */
  session_id?: string;
}

/**
 * 查询选项
 */
export interface QueryOptions {
  /** 查询模式：enhanced | legacy | auto */
  mode?: 'enhanced' | 'legacy' | 'auto';

  /** 是否启用调试模式 */
  debug?: boolean;

  /** 是否启用可视化 */
  visualization?: boolean;

  /** 返回结果数量限制 */
  top_k?: number;

  /** 置信度阈值 */
  confidence_threshold?: number;

  /** 是否启用缓存 */
  cache?: boolean;

  /** 语言设置 */
  language?: 'zh' | 'en';

  /** 超时时间（毫秒） */
  timeout?: number;
}

/**
 * 定位结果
 */
export interface LocationResult {
  /** 结果ID */
  result_id: string;

  /** 查询ID */
  query_id: string;

  /** 定位位置 */
  location: GeoLocation;

  /** 置信度（0-1） */
  confidence: number;

  /** 使用的模式 */
  mode: 'enhanced' | 'legacy';

  /** 处理时间（毫秒） */
  processing_time: number;

  /** 时间戳 */
  timestamp: string;

  /** 解析详情 */
  parsing_details?: ParsingDetails;

  /** 检索详情 */
  retrieval_details?: RetrievalDetails;

  /** 可视化数据（可选） */
  visualization_data?: VisualizationData;
}

/**
 * 地理位置
 */
export interface GeoLocation {
  /** 经度 */
  longitude: number;

  /** 纬度 */
  latitude: number;

  /** 海拔（可选） */
  altitude?: number;

  /** 坐标系 */
  coordinate_system: 'wgs84' | 'utm' | 'cartesian';

  /** 精度（米） */
  accuracy?: number;

  /** 单元格ID */
  cell_id?: string;

  /** 位姿ID */
  pose_id?: string;
}

/**
 * 解析详情
 */
export interface ParsingDetails {
  /** 解析的方向信息 */
  directions?: DirectionInfo[];

  /** 解析的颜色信息 */
  colors?: ColorInfo[];

  /** 解析的对象信息 */
  objects?: ObjectInfo[];

  /** 解析的空间关系 */
  spatial_relations?: SpatialRelation[];

  /** 解析的距离信息 */
  distances?: DistanceInfo[];

  /** 整体置信度 */
  overall_confidence: number;

  /** 解析使用的模型 */
  model_used: string;

  /** 解析时间（毫秒） */
  parsing_time: number;
}

/**
 * 方向信息
 */
export interface DirectionInfo {
  /** 方向文本 */
  text: string;

  /** 标准化方向 */
  normalized: string;

  /** 置信度 */
  confidence: number;

  /** 方位角（度） */
  bearing?: number;

  /** 是否为相对方向 */
  is_relative: boolean;
}

/**
 * 颜色信息
 */
export interface ColorInfo {
  /** 颜色文本 */
  text: string;

  /** 标准化颜色名称 */
  normalized: string;

  /** RGB值 */
  rgb?: [number, number, number];

  /** HSV值 */
  hsv?: [number, number, number];

  /** 置信度 */
  confidence: number;
}

/**
 * 对象信息
 */
export interface ObjectInfo {
  /** 对象文本 */
  text: string;

  /** 标准化对象名称 */
  normalized: string;

  /** 对象类别 */
  category: string;

  /** 对象ID（如果已知） */
  object_id?: string;

  /** 置信度 */
  confidence: number;

  /** 是否为开放词汇识别 */
  is_open_vocab: boolean;
}

/**
 * 空间关系
 */
export interface SpatialRelation {
  /** 关系文本 */
  text: string;

  /** 标准化关系 */
  normalized: string;

  /** 关系类型 */
  type: 'proximity' | 'direction' | 'topology';

  /** 置信度 */
  confidence: number;
}

/**
 * 距离信息
 */
export interface DistanceInfo {
  /** 距离文本 */
  text: string;

  /** 距离值（米） */
  value: number;

  /** 距离单位 */
  unit: 'meters' | 'kilometers' | 'feet' | 'miles';

  /** 置信度 */
  confidence: number;

  /** 是否为近似距离 */
  is_approximate: boolean;
}

/**
 * 检索详情
 */
export interface RetrievalDetails {
  /** 检索模式 */
  mode: 'hybrid' | 'vector_only' | 'template_only';

  /** 向量检索得分 */
  vector_score?: number;

  /** 模板匹配得分 */
  template_score?: number;

  /** 最终得分 */
  final_score: number;

  /** 候选数量 */
  candidates_considered: number;

  /** 检索时间（毫秒） */
  retrieval_time: number;

  /** 使用的索引 */
  index_used: string;
}

/**
 * 可视化数据
 */
export interface VisualizationData {
  /** 点云URL */
  pointcloud_url?: string;

  /** 缩略图URL */
  thumbnail_url?: string;

  /** 高亮对象 */
  highlighted_objects?: HighlightedObject[];

  /** 相机视角 */
  camera_view?: CameraView;

  /** 标注信息 */
  annotations?: Annotation[];
}

/**
 * 高亮对象
 */
export interface HighlightedObject {
  /** 对象ID */
  object_id: string;

  /** 对象名称 */
  name: string;

  /** 位置 */
  position: [number, number, number];

  /** 边界框 */
  bounding_box?: BoundingBox;

  /** 颜色 */
  color: string;
}

/**
 * 边界框
 */
export interface BoundingBox {
  /** 最小点 */
  min: [number, number, number];

  /** 最大点 */
  max: [number, number, number];

  /** 中心点 */
  center: [number, number, number];

  /** 尺寸 */
  size: [number, number, number];
}

/**
 * 相机视角
 */
export interface CameraView {
  /** 位置 */
  position: [number, number, number];

  /** 目标点 */
  target: [number, number, number];

  /** 上方向 */
  up: [number, number, number];

  /** 视野角度（度） */
  fov: number;
}

/**
 * 标注
 */
export interface Annotation {
  /** 标注ID */
  id: string;

  /** 位置 */
  position: [number, number, number];

  /** 文本 */
  text: string;

  /** 类型 */
  type: 'info' | 'warning' | 'error' | 'success';
}

/**
 * 批量定位请求
 */
export interface BatchLocateRequest {
  /** 查询列表 */
  queries: QueryRequest[];

  /** 批量选项 */
  options?: BatchOptions;
}

/**
 * 批量选项
 */
export interface BatchOptions {
  /** 并发数 */
  concurrency?: number;

  /** 超时时间（毫秒） */
  timeout?: number;

  /** 回调URL（可选） */
  callback_url?: string;

  /** 是否启用进度通知 */
  progress_notifications?: boolean;
}

/**
 * 批量定位响应
 */
export interface BatchLocateResponse {
  /** 批量ID */
  batch_id: string;

  /** 状态 */
  status: 'pending' | 'processing' | 'completed' | 'failed';

  /** 总查询数 */
  total_queries: number;

  /** 已完成查询数 */
  completed_queries: number;

  /** 失败查询数 */
  failed_queries: number;

  /** 开始时间 */
  start_time?: string;

  /** 结束时间 */
  end_time?: string;

  /** 结果（仅当状态为completed时） */
  results?: BatchResult[];

  /** 错误（如果有） */
  error?: string;
}

/**
 * 批量结果
 */
export interface BatchResult {
  /** 查询索引 */
  query_index: number;

  /** 查询文本 */
  query_text: string;

  /** 状态 */
  status: 'success' | 'failed';

  /** 定位结果（成功时） */
  result?: LocationResult;

  /** 错误信息（失败时） */
  error?: string;

  /** 处理时间（毫秒） */
  processing_time: number;
}

/**
 * 增强功能列表
 */
export interface EnhancedCapabilities {
  /** 支持的解析功能 */
  parsing: {
    /** 是否支持方向解析 */
    directions: boolean;

    /** 是否支持颜色解析 */
    colors: boolean;

    /** 是否支持对象解析 */
    objects: boolean;

    /** 是否支持空间关系解析 */
    spatial_relations: boolean;

    /** 是否支持距离解析 */
    distances: boolean;
  };

  /** 支持的检索模式 */
  retrieval_modes: Array<'hybrid' | 'vector_only' | 'template_only'>;

  /** 支持的查询语言 */
  languages: string[];

  /** 可用模型 */
  available_models: {
    /** NLU模型 */
    nlu: string[];

    /** 嵌入模型 */
    embedding: string[];

    /** 点云处理模型 */
    pointcloud: string[];
  };

  /** 性能指标 */
  performance: {
    /** 平均响应时间（毫秒） */
    avg_response_time: number;

    /** 最大并发数 */
    max_concurrency: number;

    /** 支持的点云最大尺寸 */
    max_pointcloud_size_mb: number;
  };

  /** 系统状态 */
  system_status: {
    /** 状态 */
    status: 'healthy' | 'degraded' | 'unhealthy';

    /** 版本 */
    version: string;

    /** 运行时间（秒） */
    uptime_seconds: number;

    /** 内存使用（MB） */
    memory_usage_mb: number;

    /** GPU使用率（%） */
    gpu_utilization?: number;
  };
}

/**
 * 系统配置
 */
export interface SystemConfig {
  /** 增强功能开关 */
  enhancement: {
    /** 是否启用增强功能 */
    enabled: boolean;

    /** 回退阈值 */
    fallback_threshold: number;
  };

  /** 模块配置 */
  modules: {
    /** NLU模块配置 */
    nlu: NLUConfig;

    /** 检索模块配置 */
    retrieval: RetrievalConfig;

    /** 可视化模块配置 */
    visualization: VisualizationConfig;
  };

  /** 性能配置 */
  performance: PerformanceConfig;

  /** 缓存配置 */
  cache: CacheConfig;
}

/**
 * NLU配置
 */
export interface NLUConfig {
  /** 是否启用 */
  enabled: boolean;

  /** 使用的模型 */
  model: string;

  /** 置信度阈值 */
  confidence_threshold: number;

  /** 缓存TTL（秒） */
  cache_ttl: number;

  /** 超时时间（毫秒） */
  timeout: number;

  /** 批处理大小 */
  batch_size: number;
}

/**
 * 检索配置
 */
export interface RetrievalConfig {
  /** 检索模式 */
  mode: 'hybrid' | 'vector_only' | 'template_only';

  /** 权重配置 */
  weights: {
    /** 模板匹配权重 */
    template: number;

    /** 向量检索权重 */
    vector: number;
  };

  /** 向量索引配置 */
  vector_index: {
    /** 索引类型 */
    type: string;

    /** 向量维度 */
    dimension: number;

    /** 是否启用GPU加速 */
    gpu_accelerated: boolean;
  };

  /** 返回结果数量 */
  top_k: number;
}

/**
 * 可视化配置
 */
export interface VisualizationConfig {
  /** 是否显示增强信息 */
  show_enhanced_info: boolean;

  /** 是否启用比较模式 */
  compare_mode: boolean;

  /** 渲染质量 */
  render_quality: 'low' | 'medium' | 'high' | 'ultra';

  /** 点云细节级别 */
  lod_levels: number;
}

/**
 * 性能配置
 */
export interface PerformanceConfig {
  /** 最大并发请求数 */
  max_concurrent_requests: number;

  /** 请求超时时间（毫秒） */
  request_timeout: number;

  /** 批处理大小 */
  batch_size: number;

  /** 内存限制（MB） */
  memory_limit_mb: number;

  /** GPU内存限制（MB） */
  gpu_memory_limit_mb?: number;
}

/**
 * 缓存配置
 */
export interface CacheConfig {
  /** 是否启用缓存 */
  enabled: boolean;

  /** 最大缓存大小（条目数） */
  max_size: number;

  /** 默认TTL（秒） */
  default_ttl: number;

  /** 缓存类型 */
  type: 'memory' | 'redis' | 'disk';
}

/**
 * 系统统计信息
 */
export interface SystemStats {
  /** 查询统计 */
  queries: {
    /** 总查询数 */
    total: number;

    /** 成功查询数 */
    successful: number;

    /** 失败查询数 */
    failed: number;

    /** 平均响应时间（毫秒） */
    avg_response_time: number;

    /** P95响应时间（毫秒） */
    p95_response_time: number;

    /** P99响应时间（毫秒） */
    p99_response_time: number;
  };

  /** 增强功能使用统计 */
  enhancement_usage: {
    /** 增强模式使用次数 */
    enhanced_mode_count: number;

    /** 传统模式使用次数 */
    legacy_mode_count: number;

    /** 回退次数 */
    fallback_count: number;

    /** 增强功能成功率 */
    success_rate: number;
  };

  /** 资源使用统计 */
  resource_usage: {
    /** CPU使用率（%） */
    cpu_usage: number;

    /** 内存使用（MB） */
    memory_usage_mb: number;

    /** GPU使用率（%） */
    gpu_utilization?: number;

    /** GPU内存使用（MB） */
    gpu_memory_usage_mb?: number;
  };

  /** 缓存统计 */
  cache_stats: {
    /** 缓存命中数 */
    hits: number;

    /** 缓存未命中数 */
    misses: number;

    /** 命中率 */
    hit_rate: number;

    /** 缓存大小（条目数） */
    size: number;
  };

  /** 时间范围 */
  time_range: {
    /** 开始时间 */
    start_time: string;

    /** 结束时间 */
    end_time: string;
  };
}

/**
 * 用户反馈
 */
export interface UserFeedback {
  /** 查询ID */
  query_id: string;

  /** 评分（1-5） */
  rating: number;

  /** 评论（可选） */
  comment?: string;

  /** 结果质量评价 */
  result_quality: 'excellent' | 'good' | 'fair' | 'poor';

  /** 是否愿意分享用于改进 */
  allow_sharing: boolean;

  /** 用户ID（可选） */
  user_id?: string;
}

/**
 * 点云元数据
 */
export interface PointCloudMetadata {
  /** 点云ID */
  id: string;

  /** 点云名称 */
  name: string;

  /** 描述 */
  description?: string;

  /** 位置范围 */
  bounds: {
    /** 最小点 */
    min: [number, number, number];

    /** 最大点 */
    max: [number, number, number];
  };

  /** 点数 */
  point_count: number;

  /** 文件大小（字节） */
  file_size: number;

  /** 格式 */
  format: 'las' | 'laz' | 'ply' | 'pcd' | 'xyz';

  /** 坐标系 */
  coordinate_system: string;

  /** 创建时间 */
  created_at: string;

  /** 更新时间 */
  updated_at: string;

  /** 预览图URL */
  preview_url?: string;

  /** 标签 */
  tags?: string[];
}

/**
 * 查询历史项
 */
export interface QueryHistoryItem {
  /** 查询ID */
  query_id: string;

  /** 查询文本 */
  query_text: string;

  /** 时间戳 */
  timestamp: string;

  /** 结果数量 */
  result_count: number;

  /** 最佳结果置信度 */
  best_confidence: number;

  /** 处理时间（毫秒） */
  processing_time: number;

  /** 使用的模式 */
  mode: 'enhanced' | 'legacy';

  /** 是否成功 */
  success: boolean;
}

// ============ 端点路径生成函数 ============

/**
 * 获取健康检查端点路径
 */
export function getHealthEndpoint(): ApiEndpoint {
  return '/api/v1/health';
}

/**
 * 获取系统状态端点路径
 */
export function getStatusEndpoint(): ApiEndpoint {
  return '/api/v1/status';
}

/**
 * 获取定位端点路径
 */
export function getLocateEndpoint(): ApiEndpoint {
  return '/api/v1/locate';
}

/**
 * 获取批量定位端点路径
 */
export function getBatchLocateEndpoint(): ApiEndpoint {
  return '/api/v1/batch/locate';
}

/**
 * 获取批量结果端点路径
 * @param batchId 批量ID
 */
export function getBatchResultsEndpoint(batchId: string): string {
  return `/api/v1/batch/results/${batchId}`;
}

/**
 * 获取增强功能列表端点路径
 */
export function getEnhancedCapabilitiesEndpoint(): ApiEndpoint {
  return '/api/v1/enhanced/capabilities';
}

/**
 * 获取增强解析端点路径
 */
export function getEnhancedParseEndpoint(): ApiEndpoint {
  return '/api/v1/enhanced/parse';
}

/**
 * 获取增强配置端点路径
 */
export function getEnhancedConfigEndpoint(): ApiEndpoint {
  return '/api/v1/enhanced/config';
}

/**
 * 获取系统统计端点路径
 */
export function getEnhancedStatsEndpoint(): ApiEndpoint {
  return '/api/v1/enhanced/stats';
}

/**
 * 获取用户反馈端点路径
 */
export function getEnhancedFeedbackEndpoint(): ApiEndpoint {
  return '/api/v1/enhanced/feedback';
}

/**
 * 获取点云列表端点路径
 */
export function getPointCloudListEndpoint(): ApiEndpoint {
  return '/api/v1/pointclouds';
}

/**
 * 获取点云详情端点路径
 * @param pointcloudId 点云ID
 */
export function getPointCloudDetailEndpoint(pointcloudId: string): string {
  return `/api/v1/pointclouds/${pointcloudId}`;
}

/**
 * 获取点云下载端点路径
 * @param pointcloudId 点云ID
 */
export function getPointCloudDownloadEndpoint(pointcloudId: string): string {
  return `/api/v1/pointclouds/${pointcloudId}/download`;
}

/**
 * 获取点云元数据端点路径
 * @param pointcloudId 点云ID
 */
export function getPointCloudMetadataEndpoint(pointcloudId: string): string {
  return `/api/v1/pointclouds/${pointcloudId}/metadata`;
}

/**
 * 获取查询历史端点路径
 * @param params 查询参数
 */
export function getQueryHistoryEndpoint(params?: {
  limit?: number;
  offset?: number;
  start_date?: string;
  end_date?: string;
}): string {
  const basePath = '/api/v1/queries/history';

  if (!params) {
    return basePath;
  }

  const queryParams = new URLSearchParams();

  if (params.limit !== undefined) {
    queryParams.append('limit', params.limit.toString());
  }

  if (params.offset !== undefined) {
    queryParams.append('offset', params.offset.toString());
  }

  if (params.start_date) {
    queryParams.append('start_date', params.start_date);
  }

  if (params.end_date) {
    queryParams.append('end_date', params.end_date);
  }

  const queryString = queryParams.toString();
  return queryString ? `${basePath}?${queryString}` : basePath;
}

/**
 * 获取查询详情端点路径
 * @param queryId 查询ID
 */
export function getQueryDetailEndpoint(queryId: string): string {
  return `/api/v1/queries/${queryId}`;
}

/**
 * 获取查询删除端点路径
 * @param queryId 查询ID
 */
export function getQueryDeleteEndpoint(queryId: string): string {
  return `/api/v1/queries/${queryId}`;
}

/**
 * 获取查询导出端点路径
 */
export function getQueryExportEndpoint(): ApiEndpoint {
  return '/api/v1/queries/export';
}

/**
 * 获取监控指标端点路径
 */
export function getMetricsEndpoint(): ApiEndpoint {
  return '/api/v1/metrics';
}

/**
 * 获取性能指标端点路径
 */
export function getPerformanceEndpoint(): ApiEndpoint {
  return '/api/v1/performance';
}

/**
 * 获取使用统计端点路径
 */
export function getUsageStatsEndpoint(): ApiEndpoint {
  return '/api/v1/usage/stats';
}

/**
 * 获取系统配置端点路径
 */
export function getSystemConfigEndpoint(): ApiEndpoint {
  return '/api/v1/config/system';
}

/**
 * 获取用户配置端点路径
 */
export function getUserConfigEndpoint(): ApiEndpoint {
  return '/api/v1/config/user';
}

/**
 * 获取模型配置端点路径
 */
export function getModelConfigEndpoint(): ApiEndpoint {
  return '/api/v1/config/model';
}

/**
 * 获取用户资料端点路径
 */
export function getUserProfileEndpoint(): ApiEndpoint {
  return '/api/v1/user/profile';
}

/**
 * 获取用户偏好端点路径
 */
export function getUserPreferencesEndpoint(): ApiEndpoint {
  return '/api/v1/user/preferences';
}

/**
 * 获取用户API密钥端点路径
 */
export function getUserApiKeysEndpoint(): ApiEndpoint {
  return '/api/v1/user/api-keys';
}

/**
 * 获取WebSocket定位端点路径
 */
export function getWebSocketLocateEndpoint(): string {
  return '/ws/locate';
}

/**
 * 获取SSE更新端点路径
 */
export function getServerSentEventsEndpoint(): string {
  return '/sse/updates';
}

// ============ 端点映射 ============

/**
 * API端点映射
 * 提供从端点名称到路径生成函数的映射
 */
export const API_ENDPOINTS_MAP = {
  // 健康检查
  health: getHealthEndpoint,
  status: getStatusEndpoint,

  // 核心定位功能
  locate: getLocateEndpoint,
  batchLocate: getBatchLocateEndpoint,
  batchResults: getBatchResultsEndpoint,

  // 增强功能
  enhancedCapabilities: getEnhancedCapabilitiesEndpoint,
  enhancedParse: getEnhancedParseEndpoint,
  enhancedConfig: getEnhancedConfigEndpoint,
  enhancedStats: getEnhancedStatsEndpoint,
  enhancedFeedback: getEnhancedFeedbackEndpoint,

  // 点云数据
  pointCloudList: getPointCloudListEndpoint,
  pointCloudDetail: getPointCloudDetailEndpoint,
  pointCloudDownload: getPointCloudDownloadEndpoint,
  pointCloudMetadata: getPointCloudMetadataEndpoint,

  // 查询历史
  queryHistory: getQueryHistoryEndpoint,
  queryDetail: getQueryDetailEndpoint,
  queryDelete: getQueryDeleteEndpoint,
  queryExport: getQueryExportEndpoint,

  // 监控指标
  metrics: getMetricsEndpoint,
  performance: getPerformanceEndpoint,
  usageStats: getUsageStatsEndpoint,

  // 配置管理
  systemConfig: getSystemConfigEndpoint,
  userConfig: getUserConfigEndpoint,
  modelConfig: getModelConfigEndpoint,

  // 用户管理
  userProfile: getUserProfileEndpoint,
  userPreferences: getUserPreferencesEndpoint,
  userApiKeys: getUserApiKeysEndpoint,

  // 实时通信
  webSocketLocate: getWebSocketLocateEndpoint,
  serverSentEvents: getServerSentEventsEndpoint,
} as const;

/**
 * API端点类型
 */
export type ApiEndpointKey = keyof typeof API_ENDPOINTS_MAP;

/**
 * 获取API端点路径
 * @param endpointKey 端点键名
 * @param params 参数（可选）
 * @returns 完整的端点路径
 */
export function getApiEndpoint(
  endpointKey: ApiEndpointKey,
  ...params: any[]
): string {
  const endpointFunc = API_ENDPOINTS_MAP[endpointKey];

  if (typeof endpointFunc !== 'function') {
    throw new Error(`Invalid API endpoint key: ${endpointKey}`);
  }

  return endpointFunc(...params);
}

// ============ 导出所有类型和函数 ============

export type {
  QueryRequest,
  QueryOptions,
  LocationResult,
  GeoLocation,
  ParsingDetails,
  DirectionInfo,
  ColorInfo,
  ObjectInfo,
  SpatialRelation,
  DistanceInfo,
  RetrievalDetails,
  VisualizationData,
  HighlightedObject,
  BoundingBox,
  CameraView,
  Annotation,
  BatchLocateRequest,
  BatchOptions,
  BatchLocateResponse,
  BatchResult,
  EnhancedCapabilities,
  SystemConfig,
  NLUConfig,
  RetrievalConfig,
  VisualizationConfig,
  PerformanceConfig,
  CacheConfig,
  SystemStats,
  UserFeedback,
  PointCloudMetadata,
  QueryHistoryItem,
};
