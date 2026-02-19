/**
 * Towngas Manus Electron 主进程
 *
 * 负责创建窗口、管理应用生命周期、启动后端服务
 */

const { app, BrowserWindow, ipcMain, Tray, Menu, nativeImage } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

// 保持对窗口对象的全局引用
let mainWindow = null;
let backendProcess = null;
let trayIcon = null;

// 判断是否是开发模式
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

// 后端服务配置
const BACKEND_PORT = 8000;
const FRONTEND_PORT = 5173;

/**
 * 启动后端 Python 服务
 */
function startBackend() {
    console.log('正在启动后端服务...');

    // 获取后端目录路径
    const backendDir = isDev
        ? path.join(__dirname, '..', 'backend')
        : path.join(process.resourcesPath, 'backend');

    // Python 可执行文件路径 - 开发模式使用虚拟环境
    const pythonPath = isDev
        ? path.join(backendDir, 'venv', 'bin', 'python')
        : path.join(process.resourcesPath, 'python', 'bin', 'python3');

    // 启动后端进程
    backendProcess = spawn(pythonPath, [
        '-m', 'uvicorn',
        'app.main:app',
        '--host', '127.0.0.1',
        '--port', String(BACKEND_PORT)
    ], {
        cwd: backendDir,
        env: {
            ...process.env,
            PYTHONUNBUFFERED: '1'
        }
    });

    // 处理后端输出
    backendProcess.stdout.on('data', (data) => {
        console.log(`[Backend] ${data.toString()}`);
    });

    backendProcess.stderr.on('data', (data) => {
        console.error(`[Backend Error] ${data.toString()}`);
    });

    backendProcess.on('close', (code) => {
        console.log(`后端服务已停止，退出码: ${code}`);
        backendProcess = null;
    });
}

/**
 * 停止后端服务
 */
function stopBackend() {
    if (backendProcess) {
        console.log('正在停止后端服务...');
        backendProcess.kill();
        backendProcess = null;
    }
}

/**
 * 创建主窗口
 */
function createMainWindow() {
    // 创建浏览器窗口
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1000,
        minHeight: 700,
        title: 'Towngas Manus',
        icon: path.join(__dirname, 'assets', 'icon.png'),
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js')
        },
        show: false // 先隐藏，等加载完成后再显示
    });

    // 加载前端页面
    const frontendUrl = isDev
        ? `http://localhost:${FRONTEND_PORT}`
        : `file://${path.join(__dirname, 'renderer', 'index.html')}`;

    mainWindow.loadURL(frontendUrl);

    // 窗口准备就绪时显示
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();

        // 开发模式下打开开发者工具
        if (isDev) {
            mainWindow.webContents.openDevTools();
        }
    });

    // 窗口关闭时隐藏而不是退出（支持托盘）
    mainWindow.on('close', (event) => {
        if (!app.isQuitting) {
            event.preventDefault();
            mainWindow.hide();
        }
    });

    // 窗口关闭时的清理
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

/**
 * 创建系统托盘图标
 */
function createTrayIcon() {
    // 创建托盘图标
    const iconPath = path.join(__dirname, 'assets', 'tray-icon.png');
    const trayImage = nativeImage.createFromPath(iconPath);

    trayIcon = new Tray(trayImage.resize({ width: 16, height: 16 }));

    // 托盘菜单
    const contextMenu = Menu.buildFromTemplate([
        {
            label: '显示主窗口',
            click: () => {
                if (mainWindow) {
                    mainWindow.show();
                    mainWindow.focus();
                }
            }
        },
        {
            label: '新建会话',
            click: () => {
                if (mainWindow) {
                    mainWindow.show();
                    mainWindow.webContents.send('action', 'new-session');
                }
            }
        },
        { type: 'separator' },
        {
            label: '重启后端',
            click: () => {
                stopBackend();
                setTimeout(startBackend, 1000);
            }
        },
        { type: 'separator' },
        {
            label: '退出',
            click: () => {
                app.isQuitting = true;
                app.quit();
            }
        }
    ]);

    trayIcon.setToolTip('Towngas Manus');
    trayIcon.setContextMenu(contextMenu);

    // 点击托盘图标显示窗口
    trayIcon.on('click', () => {
        if (mainWindow) {
            if (mainWindow.isVisible()) {
                mainWindow.hide();
            } else {
                mainWindow.show();
                mainWindow.focus();
            }
        }
    });
}

/**
 * 应用就绪时的处理
 */
app.whenReady().then(async () => {
    // 启动后端服务
    startBackend();

    // 等待后端启动
    await new Promise(resolve => setTimeout(resolve, 2000));

    // 创建窗口
    createMainWindow();

    // 创建托盘图标
    createTrayIcon();

    // macOS 激活应用时重新创建窗口
    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createMainWindow();
        }
    });
});

/**
 * 所有窗口关闭时的处理
 */
app.on('window-all-closed', () => {
    // macOS 上通常不会退出应用
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

/**
 * 应用退出前的清理
 */
app.on('before-quit', () => {
    app.isQuitting = true;
    stopBackend();

    if (trayIcon) {
        trayIcon.destroy();
    }
});

/**
 * IPC 通信处理
 */

// 获取应用版本
ipcMain.handle('get-version', () => {
    return app.getVersion();
});

// 获取后端状态
ipcMain.handle('get-backend-status', async () => {
    try {
        const response = await fetch(`http://localhost:${BACKEND_PORT}/api/health`);
        const data = await response.json();
        return { running: true, ...data };
    } catch (error) {
        return { running: false, error: error.message };
    }
});

// 重启后端
ipcMain.handle('restart-backend', () => {
    stopBackend();
    setTimeout(startBackend, 1000);
    return { success: true };
});

// 打开外部链接
ipcMain.handle('open-external', async (event, url) => {
    const { shell } = require('electron');

    // 服务端 URL 验证（defense in depth）
    // 验证 URL 是否为字符串
    if (typeof url !== 'string') {
        return { success: false, error: 'URL must be a string' };
    }

    const trimmedUrl = url.trim();

    // 检查 URL 是否为空
    if (!trimmedUrl) {
        return { success: false, error: 'URL cannot be empty' };
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
            console.error(`[Security] Blocked dangerous protocol in URL: ${protocol}`);
            return { success: false, error: `Dangerous protocol not allowed: ${protocol}` };
        }
    }

    // 只允许 http 和 https 协议
    if (!lowerUrl.startsWith('http://') && !lowerUrl.startsWith('https://')) {
        return { success: false, error: 'Only http and https protocols are allowed' };
    }

    try {
        // 使用 URL 构造函数进行额外验证
        const parsedUrl = new URL(trimmedUrl);

        // 确保 protocol 是 http: 或 https:
        if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
            return { success: false, error: `Invalid protocol: ${parsedUrl.protocol}` };
        }

        // 打开外部链接
        await shell.openExternal(trimmedUrl);
        return { success: true };
    } catch (e) {
        console.error(`[Security] Invalid URL rejected: ${e.message}`);
        return { success: false, error: `Invalid URL format: ${e.message}` };
    }
});
