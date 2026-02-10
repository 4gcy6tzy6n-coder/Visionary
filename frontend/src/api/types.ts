// API类型定义 - Text2Loc Visionary
// 包含所有API相关的TypeScript类型定义

// ============ 基础类型 ============

/**
 * API响应包装类型
 */
export interface ApiResponse<T = any> {
  /** 请求是否成功 */
  success: boolean;

  /** 响应数据 */
  data?: T;

  /** 错误信息 */
  error?: ApiError;

  /** 响应时间戳 */
  timestamp: string;

  /** 请求ID */
  requestId: string;
}

/**
 * API错误类型
 */
export interface ApiError {
  /** 错误名称 */
  name: string;

  /** 错误消息 */
  message: string;

  /** HTTP状态码 */
  status?: number;

  /** 错误代码 */
  code?: string;

  /** 错误时间戳 */
  timestamp?: string;

  /** 请求路径 */
  path?: string;

  /** 错误详情 */
  details?: Record<string, any>;
}

/**
 * 分页响应类型
 */
export interface PaginatedResponse<T> {
  /** 数据列表 */
  items: T[];

  /** 总记录数 */
  total: number;

  /** 当前页码 */
  page: number;

  /** 每页大小 */
  pageSize: number;

  /** 总页数 */
  totalPages: number;

  /** 是否有上一页 */
  hasPrevious: boolean;

  /** 是否有下一页 */
  hasNext: boolean;
}

/**
 * 分页请求参数
 */
export interface PaginationParams {
  /** 页码（从1开始） */
  page?: number;

  /** 每页大小 */
  pageSize?: number;

  /** 排序字段 */
  sortBy?: string;

  /** 排序方向 */
  sortDirection?: 'asc' | 'desc';

  /** 搜索关键词 */
  search?: string;

  /** 筛选条件 */
  filters?: Record<string, any>;
}

// ============ 核心功能类型 ============

/**
 * 查询请求类型
 */
export interface QueryRequest {
  /** 自然语言查询文本 */
  query: string;

  /** 地图ID */
  mapId: string;

  /** 查询选项 */
  options?: QueryOptions;

  /** 用户ID（可选） */
  userId?: string;

  /** 会话ID（可选） */
  sessionId?: string;

  /** 查询模式 */
  mode?: 'enhanced' | 'legacy' | 'auto';
}

/**
 * 查询选项类型
 */
export interface QueryOptions {
  /** 查询模式：enhanced | legacy | auto */
  mode?: 'enhanced' | 'legacy' | 'auto';

  /** 是否启用调试模式 */
  debug?: boolean;

  /** 是否启用可视化 */
  visualization?: boolean;

  /** 返回结果数量限制 */
  topK?: number;

  /** 置信度阈值 */
  confidenceThreshold?: number;

  /** 是否启用缓存 */
  cache?: boolean;

  /** 语言设置 */
  language?: 'zh' | 'en';

  /** 超时时间（毫秒） */
  timeout?: number;

  /** 是否启用增强解析 */
  enableEnhancedParsing?: boolean;

  /** 是否启用向量检索 */
  enableVectorSearch?: boolean;

  /** 是否启用混合检索 */
  enableHybridRetrieval?: boolean;
}

/**
 * 定位结果类型
 */
export interface LocationResult {
  /** 结果ID */
  resultId: string;

  /** 查询ID */
  queryId: string;

  /** 定位位置 */
  location: GeoLocation;

  /** 置信度（0-1） */
  confidence: number;

  /** 使用的模式 */
  mode: 'enhanced' | 'legacy';

  /** 处理时间（毫秒） */
  processingTime: number;

  /** 时间戳 */
  timestamp: string;

  /** 解析详情 */
  parsingDetails?: ParsingDetails;

  /** 检索详情 */
  retrievalDetails?: RetrievalDetails;

  /** 可视化数据（可选） */
  visualizationData?: VisualizationData;

  /** 查询文本 */
  queryText: string;
}

/**
 * 地理位置类型
 */
export interface GeoLocation {
  /** 经度 */
  longitude: number;

  /** 纬度 */
  latitude: number;

  /** 海拔（可选） */
  altitude?: number;

  /** 坐标系 */
  coordinateSystem: 'wgs84' | 'utm' | 'cartesian';

  /** 精度（米） */
  accuracy?: number;

  /** 单元格ID */
  cellId?: string;

  /** 位姿ID */
  poseId?: string;

  /** 描述信息 */
  description?: string;

  /** 地址 */
  address?: string;

  /** 城市 */
  city?: string;

  /** 国家 */
  country?: string;
}

// ============ 解析相关类型 ============

/**
 * 解析详情类型
 */
export interface ParsingDetails {
  /** 解析的方向信息 */
  directions?: DirectionInfo[];

  /** 解析的颜色信息 */
  colors?: ColorInfo[];

  /** 解析的对象信息 */
  objects?: ObjectInfo[];

  /** 解析的空间关系 */
  spatialRelations?: SpatialRelation[];

  /** 解析的距离信息 */
  distances?: DistanceInfo[];

  /** 整体置信度 */
  overallConfidence: number;

  /** 解析使用的模型 */
  modelUsed: string;

  /** 解析时间（毫秒） */
  parsingTime: number;

  /** 是否使用增强解析 */
  enhancedParsingUsed: boolean;

  /** 解析是否成功 */
  success: boolean;

  /** 解析错误信息 */
  error?: string;
}

/**
 * 方向信息类型
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
  isRelative: boolean;

  /** 方向类型 */
  type: 'cardinal' | 'intercardinal' | 'relative' | 'other';

  /** 原始文本位置 */
  textPosition?: TextPosition;
}

/**
 * 颜色信息类型
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

  /** 十六进制颜色值 */
  hex?: string;

  /** 置信度 */
  confidence: number;

  /** 颜色类别 */
  category?: 'primary' | 'secondary' | 'accent';

  /** 原始文本位置 */
  textPosition?: TextPosition;
}

/**
 * 对象信息类型
 */
export interface ObjectInfo {
  /** 对象文本 */
  text: string;

  /** 标准化对象名称 */
  normalized: string;

  /** 对象类别 */
  category: string;

  /** 对象ID（如果已知） */
  objectId?: string;

  /** 置信度 */
  confidence: number;

  /** 是否为开放词汇识别 */
  isOpenVocab: boolean;

  /** 对象属性 */
  attributes?: Record<string, any>;

  /** 原始文本位置 */
  textPosition?: TextPosition;
}

/**
 * 空间关系类型
 */
export interface SpatialRelation {
  /** 关系文本 */
  text: string;

  /** 标准化关系 */
  normalized: string;

  /** 关系类型 */
  type: 'proximity' | 'direction' | 'topology' | 'containment';

  /** 置信度 */
  confidence: number;

  /** 涉及的对象 */
  objects?: string[];

  /** 原始文本位置 */
  textPosition?: TextPosition;
}

/**
 * 距离信息类型
 */
export interface DistanceInfo {
  /** 距离文本 */
  text: string;

  /** 距离值（米） */
  value: number;

  /** 距离单位 */
  unit: 'meters' | 'kilometers' | 'feet' | 'miles' | 'yards';

  /** 置信度 */
  confidence: number;

  /** 是否为近似距离 */
  isApproximate: boolean;

  /** 距离范围 */
  range?: {
    min: number;
    max: number;
  };

  /** 原始文本位置 */
  textPosition?: TextPosition;
}

/**
 * 文本位置类型
 */
export interface TextPosition {
  /** 开始位置 */
  start: number;

  /** 结束位置 */
  end: number;

  /** 文本内容 */
  text: string;
}

// ============ 检索相关类型 ============

/**
 * 检索详情类型
 */
export interface RetrievalDetails {
  /** 检索模式 */
  mode: 'hybrid' | 'vector_only' | 'template_only';

  /** 向量检索得分 */
  vectorScore?: number;

  /** 模板匹配得分 */
  templateScore?: number;

  /** 最终得分 */
  finalScore: number;

  /** 候选数量 */
  candidatesConsidered: number;

  /** 检索时间（毫秒） */
  retrievalTime: number;

  /** 使用的索引 */
  indexUsed: string;

  /** 检索算法 */
  algorithm?: string;

  /** 检索参数 */
  parameters?: Record<string, any>;

  /** 检索步骤 */
  steps?: RetrievalStep[];
}

/**
 * 检索步骤类型
 */
export interface RetrievalStep {
  /** 步骤名称 */
  name: string;

  /** 步骤描述 */
  description: string;

  /** 处理时间（毫秒） */
  processingTime: number;

  /** 输入数据 */
  input?: any;

  /** 输出数据 */
  output?: any;

  /** 是否成功 */
  success: boolean;

  /** 错误信息 */
  error?: string;
}

// ============ 可视化相关类型 ============

/**
 * 可视化数据类型
 */
export interface VisualizationData {
  /** 点云URL */
  pointcloudUrl?: string;

  /** 缩略图URL */
  thumbnailUrl?: string;

  /** 高亮对象 */
  highlightedObjects?: HighlightedObject[];

  /** 相机视角 */
  cameraView?: CameraView;

  /** 标注信息 */
  annotations?: Annotation[];

  /** 路径信息 */
  paths?: Path[];

  /** 测量结果 */
  measurements?: Measurement[];

  /** 渲染配置 */
  renderConfig?: RenderConfig;
}

/**
 * 高亮对象类型
 */
export interface HighlightedObject {
  /** 对象ID */
  objectId: string;

  /** 对象名称 */
  name: string;

  /** 位置 */
  position: [number, number, number];

  /** 边界框 */
  boundingBox?: BoundingBox;

  /** 颜色 */
  color: string;

  /** 透明度 */
  opacity?: number;

  /** 标签 */
  label?: string;

  /** 描述 */
  description?: string;

  /** 类别 */
  category?: string;

  /** 置信度 */
  confidence?: number;
}

/**
 * 边界框类型
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

  /** 旋转角度 */
  rotation?: [number, number, number, number];
}

/**
 * 相机视角类型
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

  /** 近平面 */
  near: number;

  /** 远平面 */
  far: number;

  /** 相机类型 */
  type: 'perspective' | 'orthographic';
}

/**
 * 标注类型
 */
export interface Annotation {
  /** 标注ID */
  id: string;

  /** 位置 */
  position: [number, number, number];

  /** 文本 */
  text: string;

  /** 类型 */
  type: 'info' | 'warning' | 'error' | 'success' | 'highlight';

  /** 颜色 */
  color?: string;

  /** 大小 */
  size?: number;

  /** 图标 */
  icon?: string;

  /** 创建时间 */
  createdAt?: string;

  /** 创建者 */
  createdBy?: string;
}

/**
 * 路径类型
 */
export interface Path {
  /** 路径ID */
  id: string;

  /** 路径点 */
  points: [number, number, number][];

  /** 颜色 */
  color: string;

  /** 线宽 */
  width: number;

  /** 是否闭合 */
  closed: boolean;

  /** 描述 */
  description?: string;
}

/**
 * 测量结果类型
 */
export interface Measurement {
  /** 测量ID */
  id: string;

  /** 测量类型 */
  type: 'distance' | 'angle' | 'area' | 'volume';

  /** 测量值 */
  value: number;

  /** 单位 */
  unit: string;

  /** 测量点 */
  points: [number, number, number][];

  /** 描述 */
  description?: string;

  /** 颜色 */
  color?: string;
}

/**
 * 渲染配置类型
 */
export interface RenderConfig {
  /** 渲染质量 */
  quality: 'low' | 'medium' | 'high' | 'ultra';

  /** 是否启用阴影 */
  shadows: boolean;

  /** 是否启用抗锯齿 */
  antialiasing: boolean;

  /** 是否启用雾效 */
  fog: boolean;

  /** 是否启用后期处理 */
  postprocessing: boolean;

  /** 背景颜色 */
  backgroundColor: string;

  /** 环境光强度 */
  ambientLightIntensity: number;

  /** 直射光强度 */
  directionalLightIntensity: number;
}

// ============ 批量处理类型 ============

/**
 * 批量定位请求类型
 */
export interface BatchLocateRequest {
  /** 查询列表 */
  queries: QueryRequest[];

  /** 批量选项 */
  options?: BatchOptions;
}

/**
 * 批量选项类型
 */
export interface BatchOptions {
  /** 并发数 */
  concurrency?: number;

  /** 超时时间（毫秒） */
  timeout?: number;

  /** 回调URL（可选） */
  callbackUrl?: string;

  /** 是否启用进度通知 */
  progressNotifications?: boolean;

  /** 是否启用结果聚合 */
  aggregateResults?: boolean;

  /** 聚合策略 */
  aggregationStrategy?: 'average' | 'weighted' | 'best';
}

/**
 * 批量定位响应类型
 */
export interface BatchLocateResponse {
  /** 批量ID */
  batchId: string;

  /** 状态 */
  status: 'pending' | 'processing' | 'completed' | 'failed';

  /** 总查询数 */
  totalQueries: number;

  /** 已完成查询数 */
  completedQueries: number;

  /** 失败查询数 */
  failedQueries: number;

  /** 开始时间 */
  startTime?: string;

  /** 结束时间 */
  endTime?: string;

  /** 进度百分比 */
  progress?: number;

  /** 结果（仅当状态为completed时） */
  results?: BatchResult[];

  /** 聚合结果 */
  aggregatedResult?: LocationResult;

  /** 错误（如果有） */
  error?: string;

  /** 统计信息 */
  statistics?: BatchStatistics;
}

/**
 * 批量结果类型
 */
export interface BatchResult {
  /** 查询索引 */
  queryIndex: number;

  /** 查询文本 */
  queryText: string;

  /** 状态 */
  status: 'success' | 'failed';

  /** 定位结果（成功时） */
  result?: LocationResult;

  /** 错误信息（失败时） */
  error?: string;

  /** 处理时间（毫秒） */
  processingTime: number;

  /** 使用的模式 */
  mode?: 'enhanced' | 'legacy';
}

/**
 * 批量统计类型
 */
export interface BatchStatistics {
  /** 总处理时间（毫秒） */
  totalProcessingTime: number;

  /** 平均处理时间（毫秒） */
  averageProcessingTime: number;

  /** 成功率 */
  successRate: number;

  /** 增强模式使用率 */
  enhancedModeUsage: number;

  /** 平均置信度 */
  averageConfidence: number;

  /** 性能指标 */
  performanceMetrics: {
    /** 最小处理时间 */
    minProcessingTime: number;

    /** 最大处理时间 */
    maxProcessingTime: number;

    /** P50处理时间 */
    p50ProcessingTime: number;

    /** P95处理时间 */
    p95ProcessingTime: number;

    /** P99处理时间 */
    p99ProcessingTime: number;
  };
}

// ============ 增强功能类型 ============

/**
 * 增强功能列表类型
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
    spatialRelations: boolean;

    /** 是否支持距离解析 */
    distances: boolean;

    /** 支持的语言 */
    languages: string[];

    /** 支持的方言 */
    dialects?: string[];
  };

  /** 支持的检索模式 */
  retrievalModes: Array<'hybrid' | 'vector_only' | 'template_only'>;

  /** 支持的查询语言 */
  languages: string[];

  /** 可用模型 */
  availableModels: {
    /** NLU模型 */
    nlu: string[];

    /** 嵌入模型 */
    embedding: string[];

    /** 点云处理模型 */
    pointcloud: string[];

    /** 可视化模型 */
    visualization: string[];
  };

  /** 性能指标 */
  performance: {
    /** 平均响应时间（毫秒） */
    avgResponseTime: number;

    /** 最大并发数 */
    maxConcurrency: number;

    /** 支持的点云最大尺寸 */
    maxPointcloudSizeMb: number;

    /** 最大查询长度 */
    maxQueryLength: number;

    /** 支持的批量大小 */
    maxBatchSize: number;
  };

  /** 系统状态 */
  systemStatus: {
    /** 状态 */
    status: 'healthy' | 'degraded' | 'unhealthy';

    /** 版本 */
    version: string;

    /** 运行时间（秒） */
    uptimeSeconds: number;

    /** 内存使用（MB） */
    memoryUsageMb: number;

    /** CPU使用率（%） */
    cpuUtilization?: number;

    /** GPU使用率（%） */
    gpuUtilization?: number;

    /** GPU内存使用（MB） */
    gpuMemoryUsageMb?: number;

    /** 磁盘使用（GB） */
    diskUsageGb?: number;

    /** 网络延迟（毫秒） */
    networkLatencyMs?: number;
  };

  /** 功能特性 */
  features: {
    /** 是否支持批量处理 */
    batchProcessing: boolean;

    /** 是否支持实时流式处理 */
    streaming: boolean;

    /** 是否支持离线模式 */
    offlineMode: boolean;

    /** 是否支持多语言 */
    multilingual: boolean;

    /** 是否支持自定义模型 */
    customModels: boolean;

    /** 是否支持API扩展 */
    apiExtensions: boolean;
  };
}

// ============ 配置管理类型 ============

/**
 * 系统配置类型
 */
export interface SystemConfig {
  /** 增强功能开关 */
  enhancement: {
    /** 是否启用增强功能 */
    enabled: boolean;

    /** 回退阈值 */
    fallbackThreshold: number;

    /** 自动回退 */
    autoFallback: boolean;

    /** 回退原因记录 */
    logFallbackReasons: boolean;
  };

  /** 模块配置 */
  modules: {
    /** NLU模块配置 */
    nlu: NLUConfig;

    /** 检索模块配置 */
    retrieval: RetrievalConfig;

    /** 可视化模块配置 */
    visualization: VisualizationConfig;

    /** 缓存模块配置 */
    cache: CacheConfig;
  };

  /** 性能配置 */
  performance: PerformanceConfig;

  /** 安全配置 */
  security: SecurityConfig;

  /** 日志配置 */
  logging: LoggingConfig;

  /** 监控配置 */
  monitoring: MonitoringConfig;
}

/**
 * NLU配置类型
 */
export interface NLUConfig {
  /** 是否启用 */
  enabled: boolean;

  /** 使用的模型 */
  model: string;

  /** 置信度阈值 */
  confidenceThreshold: number;

  /** 缓存TTL（秒） */
  cacheTtl: number;

  /** 超时时间（毫秒） */
  timeout: number;

  /** 批处理大小 */
  batchSize: number;

  /** 是否启用并行处理 */
  parallelProcessing: boolean;

  /** 语言设置 */
  language: string;

  /** 模型参数 */
  modelParameters?: Record<string, any>;
}

/**
 * 检索配置类型
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
  vectorIndex: {
    /** 索引类型 */
    type: string;

    /** 向量维度 */
    dimension: number;

    /** 是否启用GPU加速 */
    gpuAccelerated: boolean;

    /** 相似度度量 */
    similarityMetric: 'cosine' | 'euclidean' | 'dot';
  };

  /** 返回结果数量 */
  topK: number;

  /** 阈值配置 */
  thresholds: {
    /** 最小相似度阈值 */
    minSimilarity: number;

    /** 最大距离阈值 */
    maxDistance: number;

    /** 置信度阈值 */
    confidenceThreshold: number;
  };
}

/**
 * 可视化配置类型
 */
export interface VisualizationConfig {
  /** 是否显示增强信息 */
  showEnhancedInfo: boolean;

  /** 是否启用比较模式 */
  compareMode: boolean;

  /** 渲染质量 */
  renderQuality: 'low' | 'medium' | 'high' | 'ultra';

  /** 点云细节级别 */
  lodLevels: number;

  /** 默认相机视角 */
  defaultCameraView: CameraView;

  /** 颜色方案 */
  colorScheme: {
    /** 成功颜色 */
    success: string;

    /** 警告颜色 */
    warning: string;

    /** 错误颜色 */
    error: string;

    /** 信息颜色 */
    info: string;

    /** 高亮颜色 */
    highlight: string;
  };

  /** 交互配置 */
  interaction: {
    /** 是否启用鼠标交互 */
    mouseInteraction: boolean;

    /** 是否启用触摸交互 */
    touchInteraction: boolean;

    /** 是否启用键盘快捷键 */
    keyboardShortcuts: boolean;

    /** 是否启用手势控制 */
    gestureControl: boolean;
  };
}

/**
 * 缓存配置类型
 */
export interface CacheConfig {
  /** 是否启用缓存 */
  enabled: boolean;

  /** 最大缓存大小（条目数） */
  maxSize: number;

  /** 默认TTL（秒） */
  defaultTtl: number;

  /** 缓存类型 */
  type: 'memory' | 'redis' | 'disk';

  /** 缓存策略 */
  strategy: 'lru' | 'lfu' | 'fifo';

  /** 压缩配置 */
  compression: {
    /** 是否启用压缩 */
    enabled: boolean;

    /** 压缩算法 */
    algorithm: 'gzip' | 'brotli' | 'zstd';

    /** 压缩级别 */
    level: number;
  };

  /** 持久化配置 */
  persistence: {
    /** 是否启用持久化 */
    enabled: boolean;

    /** 持久化路径 */
    path?: string;

    /** 持久化间隔（秒） */
    interval: number;
  };
}

/**
 * 性能配置类型
 */
export interface PerformanceConfig {
  /** 最大并发请求数 */
  maxConcurrentRequests: number;

  /** 请求超时时间（毫秒） */
  requestTimeout: number;

  /** 批处理大小 */
  batchSize: number;

  /** 内存限制（MB） */
  memoryLimitMb: number;

  /** GPU内存限制（MB） */
  gpuMemoryLimitMb?: number;

  /** 线程池配置 */
  threadPool: {
    /** 最小线程数 */
    minThreads: number;

    /** 最大线程数 */
    maxThreads: number;

    /** 线程空闲时间（秒） */
    idleTimeout: number;
  };

  /** 连接池配置 */
  connectionPool: {
    /** 最小连接数 */
    minConnections: number;

    /** 最大连接数 */
    maxConnections: number;

    /** 连接超时（秒） */
    connectionTimeout: number;

    /** 连接空闲时间（秒） */
    idleTimeout: number;
  };
}

/**
 * 安全配置类型
 */
export interface SecurityConfig {
  /** 是否启用认证 */
  authentication: boolean;

  /** 是否启用授权 */
  authorization: boolean;

  /** 是否启用加密 */
  encryption: boolean;

  /** 是否启用速率限制 */
  rateLimiting: boolean;

  /** 速率限制配置 */
  rateLimits: {
    /** 每分钟请求数限制 */
    requestsPerMinute: number;

    /** 每小時请求数限制 */
    requestsPerHour: number;

    /** 每天请求数限制 */
    requestsPerDay: number;

    /** 突发请求限制 */
    burstLimit: number;
  };

  /** CORS配置 */
  cors: {
    /** 允许的来源 */
    allowedOrigins: string[];

    /** 允许的方法 */
    allowedMethods: string[];

    /** 允许的头部 */
    allowedHeaders: string[];

    /** 是否允许凭据 */
    allowCredentials: boolean;

    /** 最大年龄（秒） */
    maxAge: number;
  };
}

/**
 * 日志配置类型
 */
export interface LoggingConfig {
  /** 日志级别 */
  level: 'debug' | 'info' | 'warn' | 'error';

  /** 日志格式 */
  format: 'json' | 'text' | 'simple';

  /** 是否输出到控制台 */
  consoleOutput: boolean;

  /** 是否输出到文件 */
  fileOutput: boolean;

  /** 文件输出路径 */
  filePath?: string;

  /** 文件最大大小（MB） */
  maxFileSize: number;

  /** 文件保留天数 */
  retentionDays: number;

  /** 是否启用结构化日志 */
  structuredLogging: boolean;

  /** 是否记录性能指标 */
  performanceMetrics: boolean;

  /** 是否记录用户行为 */
  userBehavior: boolean;
}

/**
 * 监控配置类型
 */
export interface MonitoringConfig {
  /** 是否启用性能监控 */
  performanceMonitoring: boolean;

  /** 是否启用错误监控 */
  errorMonitoring: boolean;

  /** 是否启用业务监控 */
  businessMonitoring: boolean;

  /** 监控采样率（0-1） */
  samplingRate: number;

  /** 监控端点 */
  endpoints: {
    /** 健康检查端点 */
    healthCheck: string;

    /** 性能指标端点 */
    metrics: string;

    /** 跟踪端点 */
    tracing: string;
  };

  /** 告警配置 */
  alerts: {
    /** 是否启用告警 */
    enabled: boolean;

    /** 告警渠道 */
    channels: string[];

    /** 告警阈值 */
    thresholds: Record<string, number>;
  };
}

// ============ 系统状态类型 ============

/**
 * 系统统计信息类型
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
    avgResponseTime: number;

    /** P95响应时间（毫秒） */
    p95ResponseTime: number;

    /** P99响应时间（毫秒） */
    p99ResponseTime: number;

    /** 每秒查询数 */
    qps: number;

    /** 并发查询数 */
    concurrentQueries: number;
  };

  /** 增强功能使用统计 */
  enhancementUsage: {
    /** 增强模式使用次数 */
    enhancedModeCount: number;

    /** 传统模式使用次数 */
    legacyModeCount: number;

    /** 回退次数 */
    fallbackCount: number;

    /** 增强功能成功率 */
    successRate: number;

    /** 平均增强处理时间（毫秒） */
    avgEnhancedProcessingTime: number;

    /** 增强功能准确率 */
    accuracy: number;
  };

  /** 资源使用统计 */
  resourceUsage: {
    /** CPU使用率（%） */
    cpuUsage: number;

    /** 内存使用（MB） */
    memoryUsageMb: number;

    /** 磁盘使用（GB） */
    diskUsageGb: number;

    /** GPU使用率（%） */
    gpuUtilization?: number;

    /** GPU内存使用（MB） */
    gpuMemoryUsageMb?: number;

    /** 网络带宽使用（MB/s） */
    networkBandwidthUsage: number;
  };

  /** 缓存统计 */
  cacheStats: {
    /** 缓存命中数 */
    hits: number;

    /** 缓存未命中数 */
    misses: number;

    /** 命中率 */
    hitRate: number;

    /** 缓存大小（条目数） */
    size: number;

    /** 缓存内存使用（MB） */
    memoryUsageMb: number;

    /** 缓存平均访问时间（毫秒） */
    avgAccessTime: number;
  };

  /** 模型使用统计 */
  modelStats: {
    /** NLU模型调用次数 */
    nluCalls: number;

    /** 嵌入模型调用次数 */
    embeddingCalls: number;

    /** 点云模型调用次数 */
    pointcloudCalls: number;

    /** 模型平均推理时间（毫秒） */
    avgInferenceTime: number;

    /** 模型成功率 */
    successRate: number;
  };

  /** 时间范围 */
  timeRange: {
    /** 开始时间 */
    startTime: string;

    /** 结束时间 */
    endTime: string;

    /** 时间粒度 */
    granularity: 'minute' | 'hour' | 'day' | 'week' | 'month';
  };
}

// ============ 用户相关类型 ============

/**
 * 用户反馈类型
 */
export interface UserFeedback {
  /** 查询ID */
  queryId: string;

  /** 评分（1-5） */
  rating: number;

  /** 评论（可选） */
  comment?: string;

  /** 结果质量评价 */
  resultQuality: 'excellent' | 'good' | 'fair' | 'poor';

  /** 是否愿意分享用于改进 */
  allowSharing: boolean;

  /** 用户ID（可选） */
  userId?: string;

  /** 反馈时间 */
  timestamp: string;

  /** 反馈类型 */
  type: 'general' | 'bug' | 'feature_request' | 'improvement';

  /** 反馈标签 */
  tags?: string[];

  /** 附件URL */
  attachmentUrl?: string;
}

/**
 * 用户配置类型
 */
export interface UserConfig {
  /** 用户ID */
  userId: string;

  /** 偏好设置 */
  preferences: {
    /** 默认查询模式 */
    defaultQueryMode: 'enhanced' | 'legacy' | 'auto';

    /** 默认语言 */
    defaultLanguage: string;

    /** 可视化偏好 */
    visualization: {
      /** 默认渲染质量 */
      defaultQuality: 'low' | 'medium' | 'high' | 'ultra';

      /** 默认颜色方案 */
      colorScheme: string;

      /** 是否启用动画 */
      animations: boolean;

      /** 是否启用声音反馈 */
      soundFeedback: boolean;
    };

    /** 通知偏好 */
    notifications: {
      /** 是否启用查询完成通知 */
      queryComplete: boolean;

      /** 是否启用系统更新通知 */
      systemUpdates: boolean;

      /** 是否启用错误通知 */
      errorNotifications: boolean;
    };
  };

  /** 历史记录设置 */
  history: {
    /** 是否保存查询历史 */
    saveQueryHistory: boolean;

    /** 历史记录保留天数 */
    retentionDays: number;

    /** 是否启用自动清理 */
    autoCleanup: boolean;
  };

  /** 隐私设置 */
  privacy: {
    /** 是否启用匿名化 */
    anonymization: boolean;

    /** 是否共享使用数据 */
    shareUsageData: boolean;

    /** 是否共享错误报告 */
    shareErrorReports: boolean;
  };
}

// ============ 点云相关类型 ============

/**
 * 点云元数据类型
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
  pointCount: number;

  /** 文件大小（字节） */
  fileSize: number;

  /** 格式 */
  format: 'las' | 'laz' | 'ply' | 'pcd' | 'xyz';

  /** 坐标系 */
  coordinateSystem: string;

  /** 创建时间 */
  createdAt: string;

  /** 更新时间 */
  updatedAt: string;

  /** 预览图URL */
  previewUrl?: string;

  /** 标签 */
  tags?: string[];

  /** 属性信息 */
  attributes?: PointCloudAttribute[];

  /** 空间参考系统 */
  spatialReference?: string;

  /** 压缩信息 */
  compression?: {
    /** 是否压缩 */
    compressed: boolean;

    /** 压缩算法 */
    algorithm?: string;

    /** 压缩率 */
    ratio?: number;
  };

  /** 质量指标 */
  qualityMetrics?: {
    /** 点密度（点/平方米） */
    pointDensity?: number;

    /** 噪声水平 */
    noiseLevel?: number;

    /** 完整性 */
    completeness?: number;
  };
}

/**
 * 点云属性类型
 */
export interface PointCloudAttribute {
  /** 属性名称 */
  name: string;

  /** 属性类型 */
  type: 'int' | 'float' | 'double' | 'string' | 'boolean';

  /** 最小值 */
  min?: number;

  /** 最大值 */
  max?: number;

  /** 平均值 */
  mean?: number;

  /** 标准差 */
  stdDev?: number;

  /** 是否标准化 */
  normalized: boolean;

  /** 描述 */
  description?: string;
}

// ============ 查询历史类型 ============

/**
 * 查询历史项类型
 */
export interface QueryHistoryItem {
  /** 查询ID */
  queryId: string;

  /** 查询文本 */
  queryText: string;

  /** 时间戳 */
  timestamp: string;

  /** 结果数量 */
  resultCount: number;

  /** 最佳结果置信度 */
  bestConfidence: number;

  /** 处理时间（毫秒） */
  processingTime: number;

  /** 使用的模式 */
  mode: 'enhanced' | 'legacy';

  /** 是否成功 */
  success: boolean;

  /** 错误信息（如果失败） */
  error?: string;

  /** 定位结果 */
  locationResult?: LocationResult;

  /** 用户评分 */
  userRating?: number;

  /** 标签 */
  tags?: string[];

  /** 是否已收藏 */
  favorite: boolean;
}

// ============ 实时通信类型 ============

/**
 * WebSocket消息类型
 */
export interface WebSocketMessage {
  /** 消息类型 */
  type: 'query_progress' | 'result_ready' | 'system_update' | 'error';

  /** 消息ID */
  messageId: string;

  /** 时间戳 */
  timestamp: string;

  /** 数据 */
  data: any;

  /** 查询ID（如果相关） */
  queryId?: string;

  /** 用户ID（如果相关） */
  userId?: string;

  /** 会话ID（如果相关） */
  sessionId?: string;
}

/**
 * 查询进度消息类型
 */
export interface QueryProgressMessage {
  /** 查询ID */
  queryId: string;

  /** 进度百分比 */
  progress: number;

  /** 当前步骤 */
  currentStep: string;

  /** 步骤描述 */
  stepDescription: string;

  /** 预计剩余时间（毫秒） */
  estimatedRemainingTime?: number;

  /** 已用时间（毫秒） */
  elapsedTime: number;
}

/**
 * 结果就绪消息类型
 */
export interface ResultReadyMessage {
  /** 查询ID */
  queryId: string;

  /** 结果ID */
  resultId: string;

  /** 处理时间（毫秒） */
  processingTime: number;

  /** 结果概要 */
  resultSummary: {
    /** 置信度 */
    confidence: number;

    /** 位置描述 */
    locationDescription?: string;

    /** 是否成功 */
    success: boolean;
  };
}

// ============ 实用工具类型 ============

/**
 * 键值对类型
 */
export interface KeyValuePair<T = any> {
  key: string;
  value: T;
}

/**
 * 时间范围类型
 */
export interface TimeRange {
  start: string;
  end: string;
}

/**
 * 地理范围类型
 */
export interface GeoBounds {
  north: number;
  south: number;
  east: number;
  west: number;
}

/**
 * 文件上传类型
 */
export interface FileUpload {
  file: File;
  name: string;
  size: number;
  type: string;
  progress: number;
  status: 'pending' | 'uploading' | 'completed' | 'failed';
  error?: string;
  url?: string;
}

// ============ 泛型类型 ============

/**
 * 可选字段类型
 */
export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;

/**
 * 必需字段类型
 */
export type Required<T, K extends keyof T> = T & Required<Pick<T, K>>;

/**
 * 只读字段类型
 */
export type Readonly<T, K extends keyof T> = T & Readonly<Pick<T, K>>;

/**
 * 可写字段类型
 */
export type Writable<T, K extends keyof T> = T & { -readonly [P in K]: T[P] };

/**
 * 深度只读类型
 */
export type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

/**
 * 深度可选类型
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * 深度必需类型
 */
export type DeepRequired<T> = {
  [P in keyof T]-?: T[P] extends object ? DeepRequired<T[P]> : T[P];
};

// ============ 导出所有类型 ============

export type {
  ApiResponse,
  ApiError,
  PaginatedResponse,
  PaginationParams,
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
  TextPosition,
  RetrievalDetails,
  RetrievalStep,
  VisualizationData,
  HighlightedObject,
  BoundingBox,
  CameraView,
  Annotation,
  Path,
  Measurement,
  RenderConfig,
  BatchLocateRequest,
  BatchOptions,
  BatchLocateResponse,
  BatchResult,
  BatchStatistics,
  EnhancedCapabilities,
  SystemConfig,
  NLUConfig,
  RetrievalConfig,
  VisualizationConfig,
  CacheConfig,
  PerformanceConfig,
  SecurityConfig,
  LoggingConfig,
  MonitoringConfig,
  SystemStats,
  UserFeedback,
  UserConfig,
  PointCloudMetadata,
  PointCloudAttribute,
  QueryHistoryItem,
  WebSocketMessage,
  QueryProgressMessage,
  ResultReadyMessage,
  KeyValuePair,
  TimeRange,
  GeoBounds,
  FileUpload,
  Optional,
  Required,
  Readonly,
  Writable,
  DeepReadonly,
  DeepPartial,
  DeepRequired,
};
