// Pyodide IndexedDB Cache Service
// Caches Pyodide binaries for faster subsequent loads

const DB_NAME = 'pyodide-cache';
const DB_VERSION = 1;
const STORE_NAME = 'binaries';

interface CacheEntry {
  url: string;
  data: Blob;
  timestamp: number;
  version: string;
}

interface CacheMetadata {
  url: string;
  size: number;
  timestamp: number;
  version: string;
}

/**
 * Open IndexedDB connection
 */
function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = () => reject(request.error);

    request.onsuccess = () => resolve(request.result);

    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;

      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'url' });
        store.createIndex('timestamp', 'timestamp', { unique: false });
        store.createIndex('version', 'version', { unique: false });
      }
    };
  });
}

/**
 * Get cached Pyodide binary
 */
export async function getCachedPyodide(url: string): Promise<Blob | null> {
  try {
    const db = await openDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.get(url);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const entry = request.result as CacheEntry | undefined;
        resolve(entry?.data || null);
      };

      transaction.oncomplete = () => db.close();
    });
  } catch {
    return null;
  }
}

/**
 * Cache Pyodide binary
 */
export async function cachePyodide(url: string, data: Blob, version: string): Promise<void> {
  try {
    const db = await openDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);

      const entry: CacheEntry = {
        url,
        data,
        timestamp: Date.now(),
        version,
      };

      const request = store.put(entry);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();

      transaction.oncomplete = () => db.close();
    });
  } catch (error) {
    console.warn('Failed to cache Pyodide:', error);
  }
}

/**
 * Check if cached version is valid
 */
export async function isCacheValid(url: string, version: string): Promise<boolean> {
  try {
    const db = await openDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.get(url);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const entry = request.result as CacheEntry | undefined;
        resolve(entry?.version === version);
      };

      transaction.oncomplete = () => db.close();
    });
  } catch {
    return false;
  }
}

/**
 * Clear all cached Pyodide data
 */
export async function clearPyodideCache(): Promise<void> {
  try {
    const db = await openDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.clear();

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();

      transaction.oncomplete = () => db.close();
    });
  } catch (error) {
    console.warn('Failed to clear Pyodide cache:', error);
  }
}

/**
 * Get cache metadata (for debugging)
 */
export async function getCacheMetadata(): Promise<CacheMetadata[]> {
  try {
    const db = await openDB();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readonly');
      const store = transaction.objectStore(STORE_NAME);
      const request = store.getAll();

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const entries = request.result as CacheEntry[];
        const metadata: CacheMetadata[] = entries.map(entry => ({
          url: entry.url,
          size: entry.data.size,
          timestamp: entry.timestamp,
          version: entry.version,
        }));
        resolve(metadata);
      };

      transaction.oncomplete = () => db.close();
    });
  } catch {
    return [];
  }
}

/**
 * Get total cache size in bytes
 */
export async function getCacheSize(): Promise<number> {
  const metadata = await getCacheMetadata();
  return metadata.reduce((total, entry) => total + entry.size, 0);
}

/**
 * Clean old cache entries (older than maxAge days)
 */
export async function cleanOldCache(maxAgeDays: number = 30): Promise<number> {
  try {
    const db = await openDB();
    const cutoffTime = Date.now() - maxAgeDays * 24 * 60 * 60 * 1000;

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, 'readwrite');
      const store = transaction.objectStore(STORE_NAME);
      const index = store.index('timestamp');
      const range = IDBKeyRange.upperBound(cutoffTime);
      const request = index.openCursor(range);
      let deletedCount = 0;

      request.onerror = () => reject(request.error);
      request.onsuccess = (event) => {
        const cursor = (event.target as IDBRequest).result;
        if (cursor) {
          cursor.delete();
          deletedCount++;
          cursor.continue();
        }
      };

      transaction.oncomplete = () => {
        db.close();
        resolve(deletedCount);
      };
    });
  } catch {
    return 0;
  }
}

export type { CacheEntry, CacheMetadata };
