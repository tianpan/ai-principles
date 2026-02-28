// Pyodide CDN Configuration
// Centralized configuration for Pyodide loading with fallback support

export interface PyodideConfig {
  /** Primary CDN URL */
  primaryCDN: string;
  /** Fallback CDN URLs */
  fallbackCDNs: string[];
  /** Pyodide version */
  version: string;
  /** Whether to use cache */
  useCache: boolean;
  /** Cache key prefix for IndexedDB */
  cacheKeyPrefix: string;
  /** Load timeout in ms */
  timeout: number;
  /** Retry attempts */
  retryAttempts: number;
}

// CDN options with their base URLs
const CDN_BASES = {
  jsdelivr: 'https://cdn.jsdelivr.net/pyodide',
  unpkg: 'https://unpkg.com/pyodide',
  cdnjs: 'https://cdnjs.cloudflare.com/ajax/libs/pyodide',
};

// Default configuration
export const defaultPyodideConfig: PyodideConfig = {
  version: 'v0.24.1',
  get primaryCDN() {
    return `${CDN_BASES.jsdelivr}/${this.version}/full/`;
  },
  get fallbackCDNs() {
    return [
      `${CDN_BASES.unpkg}@${this.version}/full/`,
    ];
  },
  useCache: true,
  cacheKeyPrefix: 'pyodide-cache',
  timeout: 60000, // 60 seconds
  retryAttempts: 2,
};

/**
 * Get Pyodide configuration with environment overrides
 */
export function getPyodideConfig(): PyodideConfig {
  // Allow environment variable override (for development/testing)
  const envVersion = typeof import.meta !== 'undefined'
    ? (import.meta as { env?: { VITE_PYODIDE_VERSION?: string } }).env?.VITE_PYODIDE_VERSION
    : undefined;

  if (envVersion) {
    return {
      ...defaultPyodideConfig,
      version: envVersion,
    };
  }

  return defaultPyodideConfig;
}

/**
 * Build Pyodide module URL
 */
export function getPyodideModuleURL(baseURL: string): string {
  return `${baseURL}pyodide.mjs`;
}

/**
 * Check if a CDN URL is accessible
 */
export async function checkCDNHealth(url: string, timeout = 5000): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);

    const response = await fetch(url, {
      method: 'HEAD',
      signal: controller.signal,
      mode: 'no-cors', // Allow cross-origin requests
    });

    clearTimeout(timeoutId);
    return true; // If no error thrown, CDN is accessible
  } catch {
    return false;
  }
}

/**
 * Get the best available CDN URL
 */
export async function getBestCDN(config: PyodideConfig = getPyodideConfig()): Promise<string> {
  // Try primary CDN first
  const primaryHealth = await checkCDNHealth(config.primaryCDN);
  if (primaryHealth) {
    return config.primaryCDN;
  }

  // Try fallback CDNs
  for (const fallback of config.fallbackCDNs) {
    const health = await checkCDNHealth(fallback);
    if (health) {
      console.warn(`Primary CDN unavailable, using fallback: ${fallback}`);
      return fallback;
    }
  }

  // Return primary as last resort (will show error if truly unavailable)
  console.warn('All CDNs appear unavailable, attempting primary anyway');
  return config.primaryCDN;
}

/**
 * Preload Pyodide assets for faster loading
 */
export function preloadPyodideAssets(baseURL: string): void {
  if (typeof document === 'undefined') return;

  // Preload the main Pyodide script
  const moduleURL = getPyodideModuleURL(baseURL);

  const link = document.createElement('link');
  link.rel = 'modulepreload';
  link.href = moduleURL;
  document.head.appendChild(link);
}

export type { PyodideConfig as PyodideConfigType };
