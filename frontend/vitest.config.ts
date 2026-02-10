import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react-swc';
import { resolve } from 'path';
import type { UserConfig } from 'vitest';

// Vitest配置扩展
export default defineConfig({
  // 共享Vite配置
  plugins: [
    react({
      jsxImportSource: '@emotion/react',
      tsDecorators: true,
    }),
  ],

  // 测试专用配置
  test: {
    /* ============ 环境配置 ============ */
    // 测试环境
    environment: 'jsdom',

    // 全局测试文件
    setupFiles: [
      // 全局测试设置
      resolve(__dirname, 'src/test/setup.ts'),
      // DOM测试库扩展
      resolve(__dirname, 'src/test/setup-dom.ts'),
    ],

    // 全局变量
    globals: true,

    /* ============ 测试运行配置 ============ */
    // 测试超时时间（毫秒）
    testTimeout: 10000,

    // 钩子超时时间（毫秒）
    hookTimeout: 30000,

    // 测试运行器配置
    watch: false,
    threads: true,
    maxThreads: 4,
    minThreads: 1,

    // 测试隔离
    isolate: true,
    singleThread: false,

    /* ============ 覆盖率配置 ============ */
    coverage: {
      // 覆盖率提供者
      provider: 'istanbul',

      // 启用覆盖率
      enabled: true,

      // 覆盖率报告器
      reporter: ['text', 'json', 'html', 'lcov'],

      // 报告输出目录
      reportsDirectory: './coverage',

      // 覆盖率阈值
      thresholds: {
        statements: 70,
        branches: 60,
        functions: 65,
        lines: 70,
      },

      // 包含的文件
      include: ['src/**/*.{ts,tsx}'],

      // 排除的文件
      exclude: [
        '**/*.d.ts',
        '**/*.config.ts',
        '**/*.config.js',
        'src/test/**',
        'src/types/**',
        'src/**/*.stories.{ts,tsx}',
        'src/**/*.test.{ts,tsx}',
        'src/**/*.spec.{ts,tsx}',
      ],

      // 是否包含所有文件（包括未测试的）
      all: false,

      // 覆盖率计算配置
      clean: true,
      cleanOnRerun: true,
      skipFull: false,

      // 每行覆盖率
      perFile: true,

      // 100%覆盖率文件
      lines: 100,
      statements: 100,
      functions: 100,
      branches: 100,

      // 水印
      watermarks: {
        statements: [50, 80],
        functions: [50, 80],
        branches: [50, 80],
        lines: [50, 80],
      },
    },

    /* ============ 测试匹配配置 ============ */
    // 包含的文件
    include: [
      'src/**/*.{test,spec}.{ts,tsx,js,jsx}',
    ],

    // 排除的文件
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/build/**',
      '**/coverage/**',
      '**/.{git,svn,hg}/**',
      '**/__snapshots__/**',
      '**/*.config.*',
    ],

    // 测试名称模式
    name: 'Text2Loc Visionary Tests',

    /* ============ 输出配置 ============ */
    // 输出模式
    outputFile: './test-results/vitest-results.json',

    // 静默模式
    silent: false,

    // 详细输出
    verbose: true,

    // 堆栈跟踪
    stackTrace: true,

    // 日志级别
    logLevel: 'info',

    // 控制台输出
    console: {
      log: true,
      info: true,
      warn: true,
      error: true,
    },

    /* ============ UI配置 ============ */
    // UI模式配置
    ui: false,

    // UI基础路径
    uiBase: '/__vitest__/',

    /* ============ 浏览器测试配置 ============ */
    // 浏览器配置（如果需要）
    browser: {
      // 启用浏览器测试
      enabled: false,

      // 浏览器名称
      name: 'chrome',

      // 浏览器提供者
      provider: 'webdriverio',

      // 浏览器实例
      instances: 1,
    },

    /* ============ 快照测试配置 ============ */
    // 快照选项
    snapshotFormat: {
      printBasicPrototype: false,
      escapeString: true,
    },

    // 快照目录
    resolveSnapshotPath: (testPath, snapshotExtension) => {
      return testPath.replace(/\.test\.([tj]sx?)/, `.test$1${snapshotExtension}`);
    },

    /* ============ 类型检查配置 ============ */
    // TypeScript检查
    typecheck: {
      enabled: true,
      checker: 'tsc',
      include: ['src/**/*.{ts,tsx}'],
      exclude: ['**/*.d.ts', '**/node_modules/**'],
    },

    /* ============ 性能配置 ============ */
    // 缓存配置
    cache: {
      dir: './node_modules/.vitest',
    },

    // 序列化器
    sequence: {
      shuffle: false,
      concurrent: false,
      seed: undefined,
    },

    /* ============ 自定义配置 ============ */
    // 自定义匹配器
    // 在这里可以添加自定义的expect扩展
  },

  // 解析配置（与Vite保持一致）
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@api': resolve(__dirname, 'src/api'),
      '@components': resolve(__dirname, 'src/components'),
      '@features': resolve(__dirname, 'src/features'),
      '@hooks': resolve(__dirname, 'src/hooks'),
      '@stores': resolve(__dirname, 'src/stores'),
      '@types': resolve(__dirname, 'src/types'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@styles': resolve(__dirname, 'src/styles'),
      '@assets': resolve(__dirname, 'src/assets'),
    },
    extensions: [
      '.mjs',
      '.js',
      '.ts',
      '.jsx',
      '.tsx',
      '.json',
      '.css',
      '.scss',
      '.svg',
    ],
  },

  // 依赖优化
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      '@testing-library/react',
      '@testing-library/user-event',
      '@testing-library/jest-dom',
    ],
  },
} as UserConfig);
