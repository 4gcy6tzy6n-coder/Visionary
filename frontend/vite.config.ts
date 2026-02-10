import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // 加载环境变量
  const env = loadEnv(mode, process.cwd(), "");

  return {
    // 项目根目录
    root: resolve(__dirname, "."),

    // 基础路径
    base: env.VITE_BASE_PATH || "/",

    // 开发服务器配置
    server: {
      // 服务器主机名
      host: true,
      // 端口号
      port: parseInt(env.VITE_PORT || "5173"),
      // 自动打开浏览器
      open: false,
      // 启用热更新
      hmr: true,
      // 代理配置，将API请求转发到后端
      proxy: {
        "/api": {
          target: env.VITE_API_BASE_URL || "http://localhost:8080",
          changeOrigin: true,
          secure: false,
        },
        "/ws": {
          target: env.VITE_API_BASE_URL || "http://localhost:8080",
          changeOrigin: true,
          secure: false,
          ws: true,
        },
      },
      // 跨域配置
      cors: true,
    },

    // 预览服务器配置
    preview: {
      port: parseInt(env.VITE_PREVIEW_PORT || "4173"),
      host: true,
    },

    // 插件配置 - 简化版本
    plugins: [
      // 使用基础React插件，避免SWC插件依赖问题
      react({
        jsxImportSource: "@emotion/react",
        babel: {
          plugins: ["@emotion/babel-plugin"],
        },
      }),
    ],

    // 解析配置
    resolve: {
      // 别名配置
      alias: {
        "@": resolve(__dirname, "src"),
        "@api": resolve(__dirname, "src/api"),
        "@components": resolve(__dirname, "src/components"),
        "@assets": resolve(__dirname, "src/assets"),
      },
      // 扩展名配置
      extensions: [
        ".mjs",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".json",
        ".css",
        ".scss",
        ".svg",
      ],
    },

    // 构建配置
    build: {
      // 输出目录
      outDir: "dist",
      // 静态资源目录
      assetsDir: "assets",
      // 是否生成sourcemap
      sourcemap: mode !== "production",
      // 最小化配置
      minify: mode === "production" ? "terser" : false,
      // 代码分割配置
      rollupOptions: {
        output: {
          // 代码分割策略
          manualChunks: {
            // 第三方库分离
            "vendor-react": ["react", "react-dom", "react-router-dom"],
            "vendor-state": ["zustand", "@tanstack/react-query"],
            "vendor-ui": [
              "@chakra-ui/react",
              "@emotion/react",
              "@emotion/styled",
            ],
            "vendor-viz": ["three", "@react-three/fiber", "@react-three/drei"],
            "vendor-utils": ["axios", "lodash", "dayjs", "clsx", "uuid"],
          },
        },
        // 外部依赖
        external: [],
      },
      // 目标浏览器
      target: "es2020",
      // CSS代码分割
      cssCodeSplit: true,
      // 资源大小限制（警告）
      assetsInlineLimit: 4096,
      // 块大小警告限制
      chunkSizeWarningLimit: 1000,
    },

    // 优化配置
    optimizeDeps: {
      include: [
        "react",
        "react-dom",
        "react-router-dom",
        "axios",
        "@tanstack/react-query",
        "zustand",
        "@chakra-ui/react",
        "@emotion/react",
        "@emotion/styled",
        "three",
        "@react-three/fiber",
        "react-hook-form",
      ],
      exclude: ["framer-motion"], // 暂时排除framer-motion
    },

    // 环境变量前缀
    envPrefix: "VITE_",
  };
});
