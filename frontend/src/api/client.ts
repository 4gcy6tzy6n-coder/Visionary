// API客户端配置 - Text2Loc Visionary
// 基于Axios的HTTP客户端，提供统一的API请求管理

import axios, {
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  AxiosError,
  InternalAxiosRequestConfig,
  AxiosHeaders,
} from 'axios';
import { queryClient } from '@api/queryClient';
import { API_CONFIG, RETRY_CONFIG, AUTH_CONFIG } from '@api/constants';

// API响应错误接口
export interface ApiError extends Error {
  status?: number;
  code?: string;
  timestamp?: string;
  path?: string;
}

// API响应包装接口
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: ApiError;
  timestamp: string;
  requestId: string;
}

// 请求配置扩展
export interface Text2LocRequestConfig extends AxiosRequestConfig {
  /**
   * 是否在请求失败时自动重试
   * @default true
   */
  retry?: boolean;

  /**
   * 重试次数
   * @default 3
   */
  retryCount?: number;

  /**
   * 是否显示加载状态
   * @default true
   */
  showLoading?: boolean;

  /**
   * 是否在错误时显示提示
   * @default true
   */
  showError?: boolean;

  /**
   * 是否在401错误时重定向到登录页
   * @default true
   */
  redirectOnUnauthorized?: boolean;

  /**
   * 是否使用缓存
   * @default false
   */
  useCache?: boolean;

  /**
   * 缓存过期时间（秒）
   * @default 300
   */
  cacheTTL?: number;

  /**
   * 请求优先级（1-5，1为最高）
   * @default 3
   */
  priority?: number;
}

// 请求队列项
interface RequestQueueItem {
  config: Text2LocRequestConfig;
  resolve: (value: any) => void;
  reject: (reason?: any) => void;
}

/**
 * Text2Loc Visionary API客户端
 * 提供统一的HTTP请求管理，包含拦截器、重试、缓存等功能
 */
class ApiClient {
  private static instance: ApiClient;
  private client: AxiosInstance;
  private requestQueue: RequestQueueItem[] = [];
  private isProcessingQueue = false;
  private concurrentRequests = 0;
  private maxConcurrentRequests = API_CONFIG.maxConcurrentRequests;
  private requestTimeoutMap = new Map<string, NodeJS.Timeout>();

  /**
   * 私有构造函数，实现单例模式
   */
  private constructor() {
    // 创建axios实例
    this.client = axios.create({
      baseURL: API_CONFIG.baseUrl,
      timeout: API_CONFIG.timeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'X-Client-Version': API_CONFIG.clientVersion,
        'X-Client-Platform': 'web',
      },
      withCredentials: API_CONFIG.withCredentials,
      maxContentLength: API_CONFIG.maxContentLength,
      maxBodyLength: API_CONFIG.maxBodyLength,
      validateStatus: (status) => status >= 200 && status < 500,
    });

    // 设置请求拦截器
    this.setupRequestInterceptors();

    // 设置响应拦截器
    this.setupResponseInterceptors();

    // 设置全局错误处理
    this.setupGlobalErrorHandling();
  }

  /**
   * 获取ApiClient单例实例
   */
  public static getInstance(): ApiClient {
    if (!ApiClient.instance) {
      ApiClient.instance = new ApiClient();
    }
    return ApiClient.instance;
  }

  /**
   * 设置请求拦截器
   */
  private setupRequestInterceptors(): void {
    this.client.interceptors.request.use(
      async (config: InternalAxiosRequestConfig) => {
        const requestConfig = config as InternalAxiosRequestConfig & Text2LocRequestConfig;

        // 生成请求ID
        const requestId = this.generateRequestId();
        config.headers = config.headers || new AxiosHeaders();
        config.headers.set('X-Request-Id', requestId);

        // 添加时间戳
        config.headers.set('X-Request-Timestamp', Date.now().toString());

        // 添加认证令牌
        const token = await this.getAuthToken();
        if (token) {
          config.headers.set('Authorization', `Bearer ${token}`);
        }

        // 添加API密钥（如果有）
        if (API_CONFIG.apiKey) {
          config.headers.set('X-API-Key', API_CONFIG.apiKey);
        }

        // 添加会话ID
        const sessionId = this.getSessionId();
        if (sessionId) {
          config.headers.set('X-Session-Id', sessionId);
        }

        // 记录请求开始时间
        requestConfig.metadata = {
          ...requestConfig.metadata,
          startTime: Date.now(),
          requestId,
        };

        // 显示加载状态
        if (requestConfig.showLoading !== false) {
          this.showLoading();
        }

        console.debug(`[API Request] ${config.method?.toUpperCase()} ${config.url}`, {
          requestId,
          headers: config.headers,
        });

        return config;
      },
      (error: AxiosError) => {
        console.error('[API Request Error]', error);
        this.hideLoading();
        return Promise.reject(this.normalizeError(error));
      }
    );
  }

  /**
   * 设置响应拦截器
   */
  private setupResponseInterceptors(): void {
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        const config = response.config as InternalAxiosRequestConfig & Text2LocRequestConfig;
        const requestId = config.headers?.get('X-Request-Id') as string;
        const startTime = config.metadata?.startTime as number;
        const duration = startTime ? Date.now() - startTime : undefined;

        // 隐藏加载状态
        if (config.showLoading !== false) {
          this.hideLoading();
        }

        // 记录请求完成
        console.debug(`[API Response] ${response.status} ${config.method?.toUpperCase()} ${config.url}`, {
          requestId,
          duration: `${duration}ms`,
          data: response.data,
        });

        // 处理缓存
        if (config.useCache && response.status === 200) {
          this.cacheResponse(config, response.data);
        }

        // 统一响应格式
        return this.normalizeResponse(response);
      },
      async (error: AxiosError) => {
        const config = error.config as (InternalAxiosRequestConfig & Text2LocRequestConfig) | undefined;
        const requestId = config?.headers?.get('X-Request-Id') as string;

        // 隐藏加载状态
        if (config?.showLoading !== false) {
          this.hideLoading();
        }

        console.error(`[API Error] ${error.response?.status || 'Network'} ${config?.method?.toUpperCase()} ${config?.url}`, {
          requestId,
          error: error.response?.data || error.message,
        });

        // 处理错误
        const normalizedError = this.normalizeError(error);

        // 重试逻辑
        if (config?.retry !== false && this.shouldRetry(error)) {
          const retryCount = config?.retryCount || RETRY_CONFIG.maxRetries;
          const currentRetry = config?.metadata?.retryCount || 0;

          if (currentRetry < retryCount) {
            const retryDelay = this.calculateRetryDelay(currentRetry);

            console.warn(`[API Retry] Retrying request (${currentRetry + 1}/${retryCount}) after ${retryDelay}ms`, {
              requestId,
              url: config.url,
            });

            return new Promise((resolve) => {
              setTimeout(() => {
                config.metadata = {
                  ...config.metadata,
                  retryCount: currentRetry + 1,
                };
                resolve(this.client(config));
              }, retryDelay);
            });
          }
        }

        // 显示错误提示
        if (config?.showError !== false) {
          this.showError(normalizedError);
        }

        // 处理认证错误
        if (error.response?.status === 401 && config?.redirectOnUnauthorized !== false) {
          await this.handleUnauthorizedError();
        }

        // 处理网络错误
        if (!error.response) {
          this.handleNetworkError();
        }

        return Promise.reject(normalizedError);
      }
    );
  }

  /**
   * 设置全局错误处理
   */
  private setupGlobalErrorHandling(): void {
    // 全局错误事件监听
    window.addEventListener('unhandledrejection', (event) => {
      if (event.reason?.isAxiosError) {
        console.error('[Global API Error]', event.reason);
        event.preventDefault();
      }
    });
  }

  /**
   * 发送HTTP请求
   */
  public async request<T = any>(config: Text2LocRequestConfig): Promise<T> {
    // 检查缓存
    if (config.useCache) {
      const cachedData = this.getCachedResponse(config);
      if (cachedData) {
        console.debug('[API Cache] Returning cached response', {
          url: config.url,
          method: config.method,
        });
        return cachedData as T;
      }
    }

    // 处理并发限制
    if (this.concurrentRequests >= this.maxConcurrentRequests) {
      return new Promise((resolve, reject) => {
        this.requestQueue.push({
          config,
          resolve,
          reject,
        });

        if (!this.isProcessingQueue) {
          this.processRequestQueue();
        }
      });
    }

    this.concurrentRequests++;

    try {
      const response = await this.client(config);
      return response.data;
    } finally {
      this.concurrentRequests--;
      this.processRequestQueue();
    }
  }

  /**
   * 发送GET请求
   */
  public async get<T = any>(url: string, config?: Text2LocRequestConfig): Promise<T> {
    return this.request<T>({
      method: 'GET',
      url,
      ...config,
    });
  }

  /**
   * 发送POST请求
   */
  public async post<T = any>(url: string, data?: any, config?: Text2LocRequestConfig): Promise<T> {
    return this.request<T>({
      method: 'POST',
      url,
      data,
      ...config,
    });
  }

  /**
   * 发送PUT请求
   */
  public async put<T = any>(url: string, data?: any, config?: Text2LocRequestConfig): Promise<T> {
    return this.request<T>({
      method: 'PUT',
      url,
      data,
      ...config,
    });
  }

  /**
   * 发送PATCH请求
   */
  public async patch<T = any>(url: string, data?: any, config?: Text2LocRequestConfig): Promise<T> {
    return this.request<T>({
      method: 'PATCH',
      url,
      data,
      ...config,
    });
  }

  /**
   * 发送DELETE请求
   */
  public async delete<T = any>(url: string, config?: Text2LocRequestConfig): Promise<T> {
    return this.request<T>({
      method: 'DELETE',
      url,
      ...config,
    });
  }

  /**
   * 上传文件
   */
  public async upload<T = any>(url: string, file: File, config?: Text2LocRequestConfig): Promise<T> {
    const formData = new FormData();
    formData.append('file', file);

    return this.request<T>({
      method: 'POST',
      url,
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      ...config,
    });
  }

  /**
   * 批量请求
   */
  public async batch<T = any>(requests: Text2LocRequestConfig[]): Promise<T[]> {
    return Promise.all(requests.map((config) => this.request<T>(config)));
  }

  /**
   * 取消请求
   */
  public cancelRequest(requestId: string): void {
    // 清除超时定时器
    const timeoutId = this.requestTimeoutMap.get(requestId);
    if (timeoutId) {
      clearTimeout(timeoutId);
      this.requestTimeoutMap.delete(requestId);
    }

    // TODO: 实现axios cancel token取消
    console.debug(`[API Cancel] Request cancelled: ${requestId}`);
  }

  /**
   * 取消所有请求
   */
  public cancelAllRequests(): void {
    // 清除所有超时定时器
    this.requestTimeoutMap.forEach((timeoutId) => {
      clearTimeout(timeoutId);
    });
    this.requestTimeoutMap.clear();

    // 清空请求队列
    this.requestQueue = [];

    // TODO: 实现axios cancel token取消所有请求
    console.debug('[API Cancel] All requests cancelled');
  }

  /**
   * 更新认证令牌
   */
  public updateAuthToken(token: string): void {
    localStorage.setItem(AUTH_CONFIG.tokenKey, token);
    console.debug('[API Auth] Token updated');
  }

  /**
   * 清除认证令牌
   */
  public clearAuthToken(): void {
    localStorage.removeItem(AUTH_CONFIG.tokenKey);
    localStorage.removeItem(AUTH_CONFIG.tokenExpiryKey);
    console.debug('[API Auth] Token cleared');
  }

  /**
   * 获取当前会话ID
   */
  private getSessionId(): string {
    let sessionId = localStorage.getItem('session_id');
    if (!sessionId) {
      sessionId = this.generateSessionId();
      localStorage.setItem('session_id', sessionId);
    }
    return sessionId;
  }

  /**
   * 生成会话ID
   */
  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * 生成请求ID
   */
  private generateRequestId(): string {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * 获取认证令牌
   */
  private async getAuthToken(): Promise<string | null> {
    const token = localStorage.getItem(AUTH_CONFIG.tokenKey);
    const expiry = localStorage.getItem(AUTH_CONFIG.tokenExpiryKey);

    if (!token || !expiry) {
      return null;
    }

    // 检查令牌是否过期
    if (Date.now() > parseInt(expiry)) {
      console.warn('[API Auth] Token expired');
      this.clearAuthToken();
      return null;
    }

    // 检查是否需要刷新令牌
    const refreshThreshold = Date.now() + AUTH_CONFIG.refreshThreshold;
    if (parseInt(expiry) < refreshThreshold) {
      await this.refreshAuthToken();
    }

    return token;
  }

  /**
   * 刷新认证令牌
   */
  private async refreshAuthToken(): Promise<void> {
    try {
      const refreshToken = localStorage.getItem(AUTH_CONFIG.refreshTokenKey);
      if (!refreshToken) {
        throw new Error('No refresh token available');
      }

      const response = await this.post<{ token: string; expires_in: number }>(
        AUTH_CONFIG.refreshEndpoint,
        { refresh_token: refreshToken },
        { showLoading: false, showError: false }
      );

      const newToken = response.token;
      const expiryTime = Date.now() + response.expires_in * 1000;

      localStorage.setItem(AUTH_CONFIG.tokenKey, newToken);
      localStorage.setItem(AUTH_CONFIG.tokenExpiryKey, expiryTime.toString());

      console.debug('[API Auth] Token refreshed successfully');
    } catch (error) {
      console.error('[API Auth] Failed to refresh token:', error);
      this.clearAuthToken();
      throw error;
    }
  }

  /**
   * 处理请求队列
   */
  private processRequestQueue(): void {
    if (this.requestQueue.length === 0 || this.isProcessingQueue) {
      return;
    }

    this.isProcessingQueue = true;

    const processNext = () => {
      if (this.concurrentRequests >= this.maxConcurrentRequests || this.requestQueue.length === 0) {
        this.isProcessingQueue = false;
        return;
      }

      const item = this.requestQueue.shift();
      if (!item) {
        this.isProcessingQueue = false;
        return;
      }

      this.concurrentRequests++;

      this.request(item.config)
        .then(item.resolve)
        .catch(item.reject)
        .finally(() => {
          this.concurrentRequests--;
          processNext();
        });
    };

    processNext();
  }

  /**
   * 判断是否应该重试
   */
  private shouldRetry(error: AxiosError): boolean {
    if (!error.config) {
      return false;
    }

    // 只在特定状态码下重试
    const retryStatusCodes = RETRY_CONFIG.retryStatusCodes;
    if (error.response && retryStatusCodes.includes(error.response.status)) {
      return true;
    }

    // 网络错误也重试
    if (!error.response) {
      return true;
    }

    return false;
  }

  /**
   * 计算重试延迟
   */
  private calculateRetryDelay(retryCount: number): number {
    const baseDelay = RETRY_CONFIG.baseDelay;
    const maxDelay = RETRY_CONFIG.maxDelay;
    const delay = Math.min(baseDelay * Math.pow(2, retryCount), maxDelay);

    // 添加随机抖动
    const jitter = delay * 0.1 * Math.random();
    return delay + jitter;
  }

  /**
   * 标准化错误
   */
  private normalizeError(error: AxiosError): ApiError {
    const apiError: ApiError = {
      name: 'ApiError',
      message: 'Unknown error occurred',
    };

    if (error.response) {
      // 服务器响应错误
      const data = error.response.data as any;
      apiError.status = error.response.status;
      apiError.code = data?.code || `HTTP_${error.response.status}`;
      apiError.message = data?.message || error.response.statusText || 'Server error';
      apiError.timestamp = data?.timestamp || new Date().toISOString();
    } else if (error.request) {
      // 请求发送但没有响应
      apiError.code = 'NETWORK_ERROR';
      apiError.message = 'Network error: No response received from server';
    } else {
      // 请求配置错误
      apiError.code = 'REQUEST_ERROR';
      apiError.message = error.message || 'Request configuration error';
    }

    return apiError;
  }

  /**
   * 标准化响应
   */
  private normalizeResponse(response: AxiosResponse): AxiosResponse {
    // 确保响应数据有统一的格式
    if (response.data && typeof response.data === 'object') {
      if (!response.data.success && !response.data.error) {
        response.data = {
          success: response.status >= 200 && response.status < 300,
          data: response.data,
          timestamp: new Date().toISOString(),
          requestId: response.config.headers?.get('X-Request-Id') as string,
        };
      }
    }

    return response;
  }

  /**
   * 缓存响应
   */
  private cacheResponse(config: Text2LocRequestConfig, data: any): void {
    const cacheKey = this.generateCacheKey(config);
    const cacheData = {
      data,
      timestamp: Date.now(),
      ttl: config.cacheTTL || API_CONFIG.cacheTTL,
    };

    try {
      sessionStorage.setItem(cacheKey, JSON.stringify(cacheData));
      console.debug('[API Cache] Response cached', { cacheKey });
    } catch (error) {
      console.warn('[API Cache] Failed to cache response:', error);
    }
  }

  /**
   * 获取缓存的响应
   */
  private getCachedResponse(config: Text2LocRequestConfig): any | null {
    const cacheKey = this.generateCacheKey(config);

    try {
      const cached = sessionStorage.getItem(cacheKey);
      if (!cached) {
        return null;
      }

      const cacheData = JSON.parse(cached);
      const now = Date.now();

      // 检查缓存是否过期
      if (now - cacheData.timestamp > cacheData.ttl * 1000) {
        sessionStorage.removeItem(cacheKey);
        console.debug('[API Cache] Cache expired', { cacheKey });
        return null;
      }

      console.debug('[API Cache] Cache hit', { cacheKey });
      return cacheData.data;
    } catch (error) {
      console.warn('[API Cache] Failed to read cache:', error);
      return null;
    }
  }

  /**
   * 生成缓存键
   */
  private generateCacheKey(config: Text2LocRequestConfig): string {
    const { url, method, params, data } = config;
    const keyParts = [
      method,
      url,
      JSON.stringify(params || {}),
      JSON.stringify(data || {}),
    ];
    return `api_cache_${btoa(keyParts.join('|')).replace(/[^a-zA-Z0-9]/g, '_')}`;
  }

  /**
   * 显示加载状态
   */
  private showLoading(): void {
    // 这里可以集成全局加载状态管理
    // 例如：使用zustand store或事件总线
    document.dispatchEvent(new CustomEvent('api:loading:start'));
  }

  /**
   * 隐藏加载状态
   */
  private hideLoading(): void {
    document.dispatchEvent(new CustomEvent('api:loading:end'));
  }

  /**
   * 显示错误提示
   */
  private showError(error: ApiError): void {
    // 这里可以集成全局错误提示系统
    // 例如：使用toast通知或状态管理
    document.dispatchEvent(new CustomEvent('api:error', {
      detail: error,
    }));
  }

  /**
   * 处理未授权错误
   */
  private async handleUnauthorizedError(): Promise<void> {
    console.warn('[API Auth] Unauthorized access, redirecting to login');

    // 清除认证信息
    this.clearAuthToken();

    // 清除查询缓存
    queryClient.clear();

    // 重定向到登录页
    const loginUrl = AUTH_CONFIG.loginUrl;
    if (loginUrl && window.location.pathname !== loginUrl) {
      window.location.href = `${loginUrl}?redirect=${encodeURIComponent(window.location.pathname)}`;
    }
  }

  /**
   * 处理网络错误
   */
  private handleNetworkError(): void {
    console.error('[API Network] Network connection error');

    // 显示网络错误提示
    document.dispatchEvent(new CustomEvent('api:network:error', {
      detail: {
        message: 'Network connection error. Please check your internet connection.',
        code: 'NETWORK_ERROR',
      },
    }));
  }
}

// 导出ApiClient单例
export const apiClient = ApiClient.getInstance();

// 导出便捷方法
export const http = {
  get: apiClient.get.bind(apiClient),
  post: apiClient.post.bind(apiClient),
  put: apiClient.put.bind(apiClient),
  patch: apiClient.patch.bind(apiClient),
  delete: apiClient.delete.bind(apiClient),
  upload: apiClient.upload.bind(apiClient),
  batch: apiClient.batch.bind(apiClient),
  request: apiClient.request.bind(apiClient),
  cancelRequest: apiClient.cancelRequest.bind(apiClient),
  cancelAllRequests: apiClient.cancelAllRequests.bind(apiClient),
  updateAuthToken: apiClient.updateAuthToken.bind(apiClient),
  clearAuthToken: apiClient.clearAuthToken.bind(apiClient),
};

// 默认导出
export default apiClient;
