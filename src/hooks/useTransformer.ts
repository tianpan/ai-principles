import { useState, useEffect, useRef, useCallback } from 'react';
import { getPyodideConfig } from '../config/pyodide';

// Timeout for Pyodide initialization (30 seconds)
const INIT_TIMEOUT_MS = 30000;

interface TrainingProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
}

interface GenerationResult {
  text: string;
  attention: number[][][];
}

interface TransformerState {
  vocabSize: number;
  dModel: number;
  nHeads: number;
  trained: boolean;
  trainingLoss: number[];
}

type WorkerStatus = 'idle' | 'loading' | 'ready' | 'training' | 'generating' | 'error';

interface WorkerMessage {
  type: string;
  [key: string]: unknown;
}

export function useTransformer() {
  const [status, setStatus] = useState<WorkerStatus>('idle');
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [modelState, setModelState] = useState<TransformerState | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [generationResult, setGenerationResult] = useState<GenerationResult | null>(null);

  const workerRef = useRef<Worker | null>(null);
  const messageHandlerRef = useRef<((message: WorkerMessage) => void) | null>(null);

  // Initialize worker
  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout> | null = null;
    let worker: Worker | null = null;

    try {
      const workerUrl = new URL('../workers/transformer.worker.ts', import.meta.url);
      worker = new Worker(workerUrl, { type: 'module' });
      workerRef.current = worker;

      worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
        const { type, ...data } = event.data;

        // Clear timeout on any successful message
        if (timeoutId) {
          clearTimeout(timeoutId);
          timeoutId = null;
        }

        switch (type) {
          case 'progress':
            setProgress(data.progress as number);
            setStage(data.stage as string);
            break;

          case 'ready':
            setStatus('ready');
            setProgress(100);
            setStage('ready');
            break;

          case 'error':
            setStatus('error');
            setError(data.message as string);
            break;

          case 'result':
            if (messageHandlerRef.current) {
              messageHandlerRef.current(event.data);
            }
            break;

          case 'training_progress':
            setStatus('training');
            setTrainingProgress({
              epoch: data.epoch as number,
              totalEpochs: data.totalEpochs as number,
              loss: data.loss as number,
            });
            break;

          case 'training_complete':
            setStatus('ready');
            setModelState(data.state as TransformerState);
            setTrainingProgress(null);
            break;

          case 'generation_complete':
            setStatus('ready');
            setGenerationResult({
              text: data.text as string,
              attention: data.attention as number[][][],
            });
            break;

          case 'status':
            if (data.ready) {
              setStatus('ready');
            }
            break;
        }
      };

      worker.onerror = (event) => {
        if (timeoutId) {
          clearTimeout(timeoutId);
          timeoutId = null;
        }
        setStatus('error');
        setError(event.message || 'Worker initialization failed');
      };

      // Start initialization with centralized config
      setStatus('loading');
      const config = getPyodideConfig();
      worker.postMessage({
        type: 'init',
        payload: { indexURL: config.primaryCDN },
      });

      // Set timeout for initialization
      timeoutId = setTimeout(() => {
        console.warn('Pyodide initialization timeout, falling back to error state');
        setStatus('error');
        setError('Pyodide 加载超时，请尝试刷新页面或使用简化模式');
      }, INIT_TIMEOUT_MS);

    } catch (err) {
      setStatus('error');
      setError(err instanceof Error ? err.message : 'Worker creation failed');
    }

    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      worker?.terminate();
    };
  }, []);

  const runCode = useCallback(
    (code: string): Promise<string> => {
      return new Promise((resolve, reject) => {
        if (!workerRef.current || status === 'loading') {
          reject(new Error('Worker not ready'));
          return;
        }

        messageHandlerRef.current = (message) => {
          if (message.type === 'result') {
            resolve(message.result as string);
          } else if (message.type === 'error') {
            reject(new Error(message.message as string));
          }
        };

        workerRef.current.postMessage({ type: 'run', payload: { code } });
      });
    },
    [status]
  );

  const trainModel = useCallback(
    (data: string, epochs: number = 10, learningRate: number = 0.01) => {
      if (!workerRef.current || status === 'loading') {
        throw new Error('Worker not ready');
      }

      setTrainingProgress(null);
      workerRef.current.postMessage({
        type: 'train',
        payload: { data, epochs, learningRate },
      });
    },
    [status]
  );

  const generateText = useCallback(
    (prompt: string, maxTokens: number = 20, temperature: number = 1.0) => {
      if (!workerRef.current || status === 'loading') {
        throw new Error('Worker not ready');
      }

      setGenerationResult(null);
      setStatus('generating');
      workerRef.current.postMessage({
        type: 'generate',
        payload: { prompt, maxTokens, temperature },
      });
    },
    [status]
  );

  const reset = useCallback(() => {
    setError(null);
    setTrainingProgress(null);
    setGenerationResult(null);
  }, []);

  return {
    status,
    progress,
    stage,
    error,
    modelState,
    trainingProgress,
    generationResult,
    isLoading: status === 'loading',
    isReady: status === 'ready',
    isTraining: status === 'training',
    isGenerating: status === 'generating',
    hasError: status === 'error',
    runCode,
    trainModel,
    generateText,
    reset,
  };
}
