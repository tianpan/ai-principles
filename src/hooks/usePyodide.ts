import { useState, useEffect, useCallback, useRef } from 'react';

interface UsePyodideOptions {
  indexURL?: string;
  packages?: string[];
  onProgress?: (progress: number, stage: string) => void;
}

interface PyodideInterface {
  runPython: (code: string) => unknown;
  runPythonAsync: (code: string) => Promise<unknown>;
  loadPackage: (packages: string | string[]) => Promise<void>;
  globals: {
    get: (name: string) => unknown;
    set: (name: string, value: unknown) => void;
  };
  FS: {
    writeFile: (path: string, data: string | Uint8Array) => void;
    readFile: (path: string, options?: { encoding: string }) => string | Uint8Array;
    mkdir: (path: string) => void;
    readdir: (path: string) => string[];
  };
}

type LoadingState = 'idle' | 'loading' | 'ready' | 'error';

declare global {
  interface Window {
    loadPyodide: (config: { indexURL: string }) => Promise<PyodideInterface>;
  }
}

export function usePyodide(options: UsePyodideOptions = {}) {
  const { indexURL = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/', packages = [], onProgress } = options;

  const [pyodide, setPyodide] = useState<PyodideInterface | null>(null);
  const [loadingState, setLoadingState] = useState<LoadingState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [loadedPackages, setLoadedPackages] = useState<string[]>([]);
  const loadingRef = useRef(false);

  const loadPyodide = useCallback(async () => {
    if (loadingRef.current || pyodide) return;

    loadingRef.current = true;
    setLoadingState('loading');
    setError(null);

    try {
      onProgress?.(10, '加载 Pyodide 运行时...');

      // Load Pyodide script if not already loaded
      if (!window.loadPyodide) {
        await new Promise<void>((resolve, reject) => {
          const script = document.createElement('script');
          script.src = `${indexURL}pyodide.js`;
          script.onload = () => resolve();
          script.onerror = () => reject(new Error('Failed to load Pyodide script'));
          document.head.appendChild(script);
        });
      }

      onProgress?.(30, '初始化 Python 环境...');

      const pyodideInstance = await window.loadPyodide({ indexURL });

      onProgress?.(60, 'Python 环境就绪');

      // Load additional packages if specified
      if (packages.length > 0) {
        onProgress?.(70, `加载依赖包: ${packages.join(', ')}...`);
        await pyodideInstance.loadPackage(packages);
        setLoadedPackages(packages);
      }

      onProgress?.(100, '准备就绪');

      setPyodide(pyodideInstance);
      setLoadingState('ready');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load Pyodide';
      setError(message);
      setLoadingState('error');
      console.error('Pyodide loading error:', err);
    } finally {
      loadingRef.current = false;
    }
  }, [indexURL, packages, pyodide, onProgress]);

  const runPython = useCallback(
    async (code: string): Promise<unknown> => {
      if (!pyodide) {
        throw new Error('Pyodide not loaded');
      }
      return pyodide.runPythonAsync(code);
    },
    [pyodide]
  );

  const setGlobal = useCallback(
    (name: string, value: unknown) => {
      if (!pyodide) {
        throw new Error('Pyodide not loaded');
      }
      pyodide.globals.set(name, value);
    },
    [pyodide]
  );

  const getGlobal = useCallback(
    (name: string): unknown => {
      if (!pyodide) {
        throw new Error('Pyodide not loaded');
      }
      return pyodide.globals.get(name);
    },
    [pyodide]
  );

  const writeFile = useCallback(
    (path: string, content: string) => {
      if (!pyodide) {
        throw new Error('Pyodide not loaded');
      }
      pyodide.FS.mkdir(path.split('/').slice(0, -1).join('/'));
      pyodide.FS.writeFile(path, content);
    },
    [pyodide]
  );

  const readFile = useCallback(
    (path: string): string => {
      if (!pyodide) {
        throw new Error('Pyodide not loaded');
      }
      return pyodide.FS.readFile(path, { encoding: 'utf8' }) as string;
    },
    [pyodide]
  );

  // Auto-load on mount
  useEffect(() => {
    loadPyodide();
  }, []);

  return {
    pyodide,
    loadingState,
    error,
    loadedPackages,
    isLoading: loadingState === 'loading',
    isReady: loadingState === 'ready',
    hasError: loadingState === 'error',
    loadPyodide,
    runPython,
    setGlobal,
    getGlobal,
    writeFile,
    readFile,
  };
}
