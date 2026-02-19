/**
 * Towngas Manus Electron 预加载脚本
 *
 * 在渲染进程中暴露安全的 API 接口
 */

const { contextBridge, ipcRenderer } = require('electron');

/**
 * 验证 URL 是否安全
 * 只允许 http 和 https 协议，阻止危险协议
 * @param {string} url - 要验证的 URL
 * @returns {Object} 验证结果 { valid: boolean, error?: string }
 */
function validateUrl(url) {
    if (typeof url !== 'string') {
        return { valid: false, error: 'URL must be a string' };
    }

    // 去除前后空白
    const trimmedUrl = url.trim();

    if (!trimmedUrl) {
        return { valid: false, error: 'URL cannot be empty' };
    }

    // 危险协议列表
    const dangerousProtocols = [
        'javascript:',
        'data:',
        'vbscript:',
        'file:',
        'about:',
        'blob:',
        'filesystem:',
    ];

    // 检查是否包含危险协议（不区分大小写）
    const lowerUrl = trimmedUrl.toLowerCase();
    for (const protocol of dangerousProtocols) {
        if (lowerUrl.startsWith(protocol)) {
            return { valid: false, error: `Dangerous protocol not allowed: ${protocol}` };
        }
    }

    // 只允许 http 和 https 协议
    if (!lowerUrl.startsWith('http://') && !lowerUrl.startsWith('https://')) {
        return { valid: false, error: 'Only http and https protocols are allowed' };
    }

    try {
        // 使用 URL 构造函数进行额外验证
        const parsedUrl = new URL(trimmedUrl);

        // 确保 protocol 是 http: 或 https:
        if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
            return { valid: false, error: `Invalid protocol: ${parsedUrl.protocol}` };
        }

        return { valid: true, url: trimmedUrl };
    } catch (e) {
        return { valid: false, error: `Invalid URL format: ${e.message}` };
    }
}

// 暴露安全的 API 到渲染进程
contextBridge.exposeInMainWorld('electronAPI', {
    /**
     * 获取应用版本
     * @returns {Promise<string>} 应用版本号
     */
    getVersion: () => ipcRenderer.invoke('get-version'),

    /**
     * 获取后端服务状态
     * @returns {Promise<Object>} 后端状态信息
     */
    getBackendStatus: () => ipcRenderer.invoke('get-backend-status'),

    /**
     * 重启后端服务
     * @returns {Promise<Object>} 操作结果
     */
    restartBackend: () => ipcRenderer.invoke('restart-backend'),

    /**
     * 在默认浏览器中打开外部链接
     * @param {string} url - 要打开的 URL
     * @returns {Promise<Object>} 操作结果 { success: boolean, error?: string }
     */
    openExternal: (url) => {
        const validation = validateUrl(url);
        if (!validation.valid) {
            return Promise.resolve({ success: false, error: validation.error });
        }
        return ipcRenderer.invoke('open-external', validation.url);
    },

    /**
     * 监听主进程发送的消息
     * @param {string} channel - 频道名称
     * @param {Function} callback - 回调函数
     */
    on: (channel, callback) => {
        ipcRenderer.on(channel, (event, ...args) => callback(...args));
    },

    /**
     * 移除消息监听
     * @param {string} channel - 频道名称
     */
    removeListener: (channel) => {
        ipcRenderer.removeAllListeners(channel);
    }
});

// 打印日志确认预加载脚本已执行
console.log('Electron preload script loaded');
