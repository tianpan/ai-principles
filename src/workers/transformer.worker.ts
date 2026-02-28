/// <reference lib="webworker" />

// Transformer Web Worker for background computation
// This worker handles Pyodide-based transformer operations

declare let self: DedicatedWorkerGlobalScope;

interface WorkerMessage {
  type: 'init' | 'run' | 'train' | 'generate' | 'status';
  payload?: unknown;
}

interface InitPayload {
  indexURL?: string;
}

interface RunPayload {
  code: string;
}

interface TrainPayload {
  epochs: number;
  learningRate: number;
  data: string;
}

interface GeneratePayload {
  prompt: string;
  maxTokens: number;
  temperature: number;
}

// CDN configuration with fallbacks
const PYODIDE_VERSION = 'v0.24.1';
const CDN_URLS = [
  `https://cdn.jsdelivr.net/pyodide/${PYODIDE_VERSION}/full/`,
  `https://unpkg.com/pyodide@${PYODIDE_VERSION}/full/`,
];

let pyodide: Awaited<ReturnType<typeof importPyodide>> | null = null;
let currentCDN = CDN_URLS[0];

async function importPyodide(indexURL?: string) {
  const cdnURL = indexURL || currentCDN;

  // @ts-expect-error - Pyodide is loaded dynamically
  return import(`${cdnURL}pyodide.mjs`).then(
    (mod) => mod.default.loadPyodide({ indexURL: cdnURL })
  ).catch(async (error) => {
    // Try fallback CDN
    console.warn(`Failed to load from ${cdnURL}, trying fallback...`);
    const fallbackURL = CDN_URLS.find(url => url !== cdnURL);
    if (fallbackURL) {
      currentCDN = fallbackURL;
      // @ts-expect-error - Pyodide is loaded dynamically
      return import(`${fallbackURL}pyodide.mjs`).then(
        (mod) => mod.default.loadPyodide({ indexURL: fallbackURL })
      );
    }
    throw error;
  });
}

async function initPyodide(payload: InitPayload) {
  try {
    self.postMessage({ type: 'progress', stage: 'loading', progress: 10 });

    pyodide = await importPyodide();

    self.postMessage({ type: 'progress', stage: 'initializing', progress: 50 });

    // Define mini transformer code
    const transformerCode = `
import math
import json
from typing import List, Dict, Any

class SimpleTokenizer:
    """Simple character-level tokenizer for demonstration."""

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def train(self, text: str):
        chars = sorted(list(set(text)))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join([self.id_to_char.get(i, '') for i in ids])


class MiniTransformer:
    """Simplified transformer for educational purposes."""

    def __init__(self, vocab_size: int, d_model: int = 32, n_heads: int = 2):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Initialize weights (simplified)
        self.embedding = [[math.sin(i * j * 0.01) for j in range(d_model)] for i in range(vocab_size)]
        self.pos_encoding = [[math.sin(pos / (10000 ** (2 * i / d_model)))
                             if i % 2 == 0 else math.cos(pos / (10000 ** (2 * i / d_model)))
                             for i in range(d_model)] for pos in range(512)]

        # Attention weights
        self.W_q = [[0.1 * (i + j) for j in range(d_model)] for i in range(d_model)]
        self.W_k = [[0.1 * (i + j) for j in range(d_model)] for i in range(d_model)]
        self.W_v = [[0.1 * (i + j) for j in range(d_model)] for i in range(d_model)]
        self.W_o = [[0.1 * (i + j) for j in range(d_model)] for i in range(d_model)]

        # Output projection
        self.W_out = [[0.1 * (i + j) for j in range(vocab_size)] for i in range(d_model)]

        self.trained = False
        self.training_loss = []

    def attention(self, Q, K, V):
        """Simplified scaled dot-product attention."""
        seq_len = len(Q)
        scores = [[sum(q * k for q, k in zip(Q[i], K[j])) / math.sqrt(self.d_model)
                   for j in range(seq_len)] for i in range(seq_len)]

        # Softmax (simplified)
        attention_weights = []
        for row in scores:
            exp_row = [math.exp(s - max(row)) for s in row]
            sum_exp = sum(exp_row)
            attention_weights.append([e / sum_exp for e in exp_row])

        # Weighted sum
        output = [[sum(attention_weights[i][j] * V[j][d]
                  for j in range(seq_len)) for d in range(self.d_model)]
                  for i in range(seq_len)]

        return output, attention_weights

    def forward(self, token_ids):
        """Forward pass through the transformer."""
        seq_len = len(token_ids)

        # Embedding + positional encoding
        x = [[self.embedding[tid][d] + self.pos_encoding[pos][d]
              for d in range(self.d_model)]
             for pos, tid in enumerate(token_ids)]

        # Compute Q, K, V
        Q = [[sum(x[i][d] * self.W_q[d][j] for d in range(self.d_model))
              for j in range(self.d_model)] for i in range(seq_len)]
        K = [[sum(x[i][d] * self.W_k[d][j] for d in range(self.d_model))
              for j in range(self.d_model)] for i in range(seq_len)]
        V = [[sum(x[i][d] * self.W_v[d][j] for d in range(self.d_model))
              for j in range(self.d_model)] for i in range(seq_len)]

        # Self-attention
        attn_out, attention_weights = self.attention(Q, K, V)

        # Output projection
        logits = [[sum(attn_out[i][d] * self.W_out[d][v]
                   for d in range(self.d_model)) for v in range(self.vocab_size)]
                  for i in range(seq_len)]

        return logits, attention_weights

    def generate(self, prompt_ids, max_tokens=20, temperature=1.0):
        """Generate tokens autoregressively."""
        generated = list(prompt_ids)
        all_attention = []

        for _ in range(max_tokens):
            logits, attention = self.forward(generated[-64:])  # Limit context
            next_logits = logits[-1]

            # Temperature scaling and softmax
            scaled = [l / temperature for l in next_logits]
            exp_scaled = [math.exp(s - max(scaled)) for s in scaled]
            probs = [e / sum(exp_scaled) for e in exp_scaled]

            # Sample (simplified - use argmax for determinism in demo)
            next_token = probs.index(max(probs))
            generated.append(next_token)
            all_attention.append(attention)

            # Stop on EOS (assume token 0 is EOS for demo)
            if next_token == 0 and len(generated) > len(prompt_ids) + 5:
                break

        return generated, all_attention

    def train_step(self, data_ids, learning_rate=0.01):
        """Simplified training step."""
        # This is a demo - real training would use backprop
        loss = 0.0
        for i in range(len(data_ids) - 1):
            input_ids = data_ids[:i+1]
            target = data_ids[i+1]

            logits, _ = self.forward(input_ids)
            pred_logits = logits[-1]

            # Simple cross-entropy loss (numerical approximation)
            max_logit = max(pred_logits)
            exp_logits = [math.exp(l - max_logit) for l in pred_logits]
            sum_exp = sum(exp_logits)
            log_prob = pred_logits[target] - math.log(sum_exp) - max_logit
            loss -= log_prob

        loss /= max(1, len(data_ids) - 1)
        self.training_loss.append(loss)

        # Simplified weight update (just add noise for demo)
        import random
        for i in range(len(self.W_q)):
            for j in range(len(self.W_q[0])):
                self.W_q[i][j] += learning_rate * (random.random() - 0.5) * 0.01

        return loss

    def get_state(self):
        """Get model state for serialization."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'trained': self.trained,
            'training_loss': self.training_loss[-10:] if self.training_loss else [],
        }
`;

    await pyodide.runPythonAsync(transformerCode);

    self.postMessage({ type: 'progress', stage: 'ready', progress: 100 });
    self.postMessage({ type: 'ready' });

  } catch (error) {
    self.postMessage({
      type: 'error',
      message: error instanceof Error ? error.message : 'Initialization failed'
    });
  }
}

async function runCode(payload: RunPayload) {
  if (!pyodide) {
    self.postMessage({ type: 'error', message: 'Pyodide not initialized' });
    return;
  }

  try {
    const result = await pyodide.runPythonAsync(payload.code);
    self.postMessage({ type: 'result', result: result?.toString() ?? 'None' });
  } catch (error) {
    self.postMessage({
      type: 'error',
      message: error instanceof Error ? error.message : 'Execution failed'
    });
  }
}

async function trainModel(payload: TrainPayload) {
  if (!pyodide) {
    self.postMessage({ type: 'error', message: 'Pyodide not initialized' });
    return;
  }

  try {
    // Initialize tokenizer and model
    await pyodide.runPythonAsync(`
tokenizer = SimpleTokenizer()
tokenizer.train("""${payload.data}""")

model = MiniTransformer(tokenizer.vocab_size, d_model=32, n_heads=2)
`);

    const dataIds = await pyodide.runPythonAsync(`tokenizer.encode("""${payload.data}""")`);
    const ids = dataIds.toJs();

    for (let epoch = 0; epoch < payload.epochs; epoch++) {
      const loss = await pyodide.runPythonAsync(
        `model.train_step(tokenizer.encode("""${payload.data}"""), ${payload.learningRate})`
      );

      self.postMessage({
        type: 'training_progress',
        epoch: epoch + 1,
        totalEpochs: payload.epochs,
        loss: parseFloat(loss.toString())
      });
    }

    const state = await pyodide.runPythonAsync(`model.get_state()`);
    self.postMessage({
      type: 'training_complete',
      state: state.toJs()
    });

  } catch (error) {
    self.postMessage({
      type: 'error',
      message: error instanceof Error ? error.message : 'Training failed'
    });
  }
}

async function generateText(payload: GeneratePayload) {
  if (!pyodide) {
    self.postMessage({ type: 'error', message: 'Pyodide not initialized' });
    return;
  }

  try {
    await pyodide.runPythonAsync(`
prompt_ids = tokenizer.encode("""${payload.prompt}""")
generated_ids, attention = model.generate(prompt_ids, max_tokens=${payload.maxTokens}, temperature=${payload.temperature})
generated_text = tokenizer.decode(generated_ids)
`);

    const text = await pyodide.runPythonAsync(`generated_text`);
    const attention = await pyodide.runPythonAsync(`attention`);

    self.postMessage({
      type: 'generation_complete',
      text: text.toString(),
      attention: attention.toJs()
    });

  } catch (error) {
    self.postMessage({
      type: 'error',
      message: error instanceof Error ? error.message : 'Generation failed'
    });
  }
}

// Message handler
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'init':
      await initPyodide(payload as InitPayload);
      break;
    case 'run':
      await runCode(payload as RunPayload);
      break;
    case 'train':
      await trainModel(payload as TrainPayload);
      break;
    case 'generate':
      await generateText(payload as GeneratePayload);
      break;
    case 'status':
      self.postMessage({
        type: 'status',
        ready: pyodide !== null
      });
      break;
  }
};

export {};
