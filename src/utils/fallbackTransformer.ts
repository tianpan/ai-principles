// Fallback Transformer for when Pyodide fails to load
// Uses pre-computed values and simulated attention patterns

// Pre-computed attention patterns for common phrases
const PRECOMPUTED_ATTENTION: Record<string, number[][]> = {
  'default': [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.3, 0.7, 0.0, 0.0, 0.0],
    [0.2, 0.3, 0.5, 0.0, 0.0],
    [0.1, 0.2, 0.3, 0.4, 0.0],
    [0.1, 0.1, 0.2, 0.3, 0.3],
  ],
  'the': [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.6, 0.4, 0.0, 0.0, 0.0],
    [0.3, 0.4, 0.3, 0.0, 0.0],
    [0.2, 0.3, 0.3, 0.2, 0.0],
    [0.15, 0.2, 0.25, 0.2, 0.2],
  ],
};

// Character frequency for simplified generation
const CHAR_FREQUENCY: Record<string, number> = {
  '的': 0.05, '是': 0.04, '在': 0.035, '了': 0.03, '不': 0.025,
  '和': 0.025, '有': 0.02, '大': 0.02, '这': 0.018, '主': 0.018,
  '为': 0.017, '人': 0.016, '中': 0.015, '国': 0.015, '年': 0.014,
  'AI': 0.012, '学': 0.012, '习': 0.011, '技': 0.01, '术': 0.01,
};

// Common phrase continuations
const PHRASE_CONTINUATIONS: Record<string, string[]> = {
  '我爱': ['学习', 'AI', '编程', '技术'],
  'AI': ['改变世界', '技术', '学习', '智能'],
  '学习': ['使人进步', 'AI', '技术', '编程'],
  '技术': ['改变生活', '发展', '进步', '创新'],
};

export interface FallbackResult {
  text: string;
  attention: number[][][];
  confidence: number;
  isFallback: true;
}

/**
 * Generate a simple attention pattern for given tokens
 */
function generateAttention(tokens: string[]): number[][] {
  const n = tokens.length;

  // Check if we have a pre-computed pattern for first token
  const firstToken = tokens[0].toLowerCase();
  const basePattern = PRECOMPUTED_ATTENTION[firstToken] || PRECOMPUTED_ATTENTION['default'];

  // Scale pattern to match token count
  if (n <= basePattern.length) {
    return basePattern.slice(0, n).map(row => row.slice(0, n));
  }

  // Extend pattern for longer sequences
  const result: number[][] = [];

  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      if (j > i) {
        row.push(0); // Causal mask
      } else if (i < basePattern.length && j < basePattern.length) {
        row.push(basePattern[i][j]);
      } else {
        // Decay pattern for longer sequences
        const dist = i - j;
        row.push(Math.exp(-dist * 0.3) * 0.8 + Math.random() * 0.2);
      }
    }

    // Normalize row
    const sum = row.reduce((a, b) => a + b, 0);
    if (sum > 0) {
      for (let j = 0; j < row.length; j++) {
        row[j] /= sum;
      }
    }

    result.push(row);
  }

  return result;
}

/**
 * Generate text continuation based on prompt
 */
function generateContinuation(prompt: string, maxTokens: number): string {
  // Check for phrase continuations
  for (const [prefix, continuations] of Object.entries(PHRASE_CONTINUATIONS)) {
    if (prompt.includes(prefix)) {
      const continuation = continuations[Math.floor(Math.random() * continuations.length)];
      return prompt + continuation.slice(0, maxTokens);
    }
  }

  // Character-level generation
  let result = prompt;
  const chars = Object.keys(CHAR_FREQUENCY);
  const weights = Object.values(CHAR_FREQUENCY);

  for (let i = 0; i < maxTokens && result.length < prompt.length + maxTokens; i++) {
    // Weighted random selection
    const totalWeight = weights.reduce((a, b) => a + b, 0);
    let random = Math.random() * totalWeight;

    for (let j = 0; j < chars.length; j++) {
      random -= weights[j];
      if (random <= 0) {
        result += chars[j];
        break;
      }
    }
  }

  return result;
}

/**
 * Simulate training loss curve
 */
export function simulateTrainingLoss(epochs: number): number[] {
  const losses: number[] = [];
  let currentLoss = 2.5 + Math.random() * 0.5;

  for (let i = 0; i < epochs; i++) {
    // Loss decreases with some noise
    currentLoss *= 0.85 + Math.random() * 0.1;
    currentLoss += Math.random() * 0.05;
    losses.push(Math.max(0.1, currentLoss));
  }

  return losses;
}

/**
 * Fallback transformer for when Pyodide is unavailable
 */
export class FallbackTransformer {
  private trained = false;
  private trainingLoss: number[] = [];

  async train(data: string, epochs: number, _learningRate: number): Promise<void> {
    // Simulate training delay
    await new Promise(resolve => setTimeout(resolve, 100 * epochs));

    this.trainingLoss = simulateTrainingLoss(epochs);
    this.trained = true;
  }

  async generate(prompt: string, maxTokens: number, _temperature: number): Promise<FallbackResult> {
    // Simulate generation delay
    await new Promise(resolve => setTimeout(resolve, 200 + Math.random() * 300));

    const tokens = prompt.split('');
    const attentionLayers: number[][][] = [];

    // Generate attention for each layer
    for (let layer = 0; layer < 2; layer++) {
      attentionLayers.push(generateAttention(tokens));
    }

    // Generate continuation
    const text = generateContinuation(prompt, maxTokens);

    return {
      text,
      attention: attentionLayers,
      confidence: this.trained ? 0.7 : 0.4,
      isFallback: true,
    };
  }

  getTrainingProgress(epoch: number, totalEpochs: number): { loss: number } {
    if (this.trainingLoss.length > 0 && epoch <= this.trainingLoss.length) {
      return { loss: this.trainingLoss[epoch - 1] };
    }
    return { loss: simulateTrainingLoss(1)[0] };
  }

  isTrained(): boolean {
    return this.trained;
  }

  getState() {
    return {
      vocabSize: 1000,
      dModel: 32,
      nHeads: 2,
      trained: this.trained,
      trainingLoss: this.trainingLoss,
    };
  }
}

/**
 * Check if fallback mode should be used
 */
export function shouldUseFallback(): boolean {
  // Check if WebAssembly is supported
  if (typeof WebAssembly !== 'object') {
    return true;
  }

  // Check if Web Workers are supported
  if (typeof Worker !== 'function') {
    return true;
  }

  // Check device memory (if available)
  const nav = navigator as { deviceMemory?: number };
  if (nav.deviceMemory !== undefined && nav.deviceMemory < 4) {
    return true;
  }

  return false;
}

/**
 * Get fallback status message
 */
export function getFallbackStatusMessage(): string {
  if (typeof WebAssembly !== 'object') {
    return '您的浏览器不支持 WebAssembly，正在使用简化模式';
  }
  if (typeof Worker !== 'function') {
    return '您的浏览器不支持 Web Workers，正在使用简化模式';
  }
  const nav = navigator as { deviceMemory?: number };
  if (nav.deviceMemory !== undefined && nav.deviceMemory < 4) {
    return '检测到设备内存较低，正在使用简化模式';
  }
  return '正在使用简化演示模式';
}
