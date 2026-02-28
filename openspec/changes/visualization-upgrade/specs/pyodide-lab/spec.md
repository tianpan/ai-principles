# Pyodide Lab Spec

## 功能描述

在浏览器中运行 Mini Transformer，支持：
1. 用户输入文本
2. 实时计算 Attention
3. 显示可视化结果
4. 训练/推理两种模式

## 技术要求

### Pyodide 加载 (usePyodide.ts)
- 首页开始预加载
- 显示加载进度条
- 加载完成后缓存到 IndexedDB
- 降级方案：加载失败时使用预计算模式

### Web Worker (transformer.worker.ts)
- 后台运行模型计算
- 不阻塞 UI 线程
- 支持取消计算

### 运行模式

**推理模式：**
1. 用户输入文本
2. 分词 + Embedding
3. 计算 Attention
4. 显示热力图 + 输出概率

**训练模式：**
1. 显示训练数据
2. 开始训练
3. 实时更新 Loss 曲线
4. 完成后可测试生成

## 验收标准

- [ ] Pyodide 加载时间 < 10 秒（首次）
- [ ] 用户输入后 5 秒内显示结果
- [ ] 支持 Chrome/Firefox/Safari 最新版
- [ ] 移动端可用（简化模式）
