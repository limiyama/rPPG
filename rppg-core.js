/*
 *   1. Captura frames da webcam via MediaPipe Face Mesh
 *   2. Extrai ROI (testa + bochechas)
 *   3. Calcula médias RGB de cada ROI por frame
 *   4. Aplica algoritmo POS para separar sinal de pulso do ruído de movimento
 *   5. Aplica filtro passa-banda (0.7–3.5 Hz = 42–210 BPM)
 *   6. FFT → pico dominante = BPM
 *   7. Derivação de HRV (RMSSD) a partir dos intervalos R-R
 *   8. Estimativa de SpO₂ via razão R/IR simulada com canais R e G
 */

const CONFIG = {
  FPS: 30,               // frames por segundo da câmera
  WINDOW_SECONDS: 30,    // tempo de escaneamento
  BPM_MIN: 42,           // 0.7 Hz
  BPM_MAX: 210,          // 3.5 Hz
  FREQ_MIN: 0.7,
  FREQ_MAX: 3.5,
  MIN_FRAMES: 150,       // mínimo de frames para análise confiável (5s)
  SPO2_CALIBRATION_A: 110,  // constantes empíricas para SpO₂ (modelo linear)
  SPO2_CALIBRATION_B: 25,
};

const ROI_LANDMARKS = {
  // landmarks da testa
  forehead: [10, 67, 69, 104, 108, 109, 151, 299, 337, 338],

  // landmarks bochecha esquerda
  leftCheek: [234, 227, 116, 123, 147, 213, 192, 214],

  // landmarks bochecha direita
  rightCheek: [454, 447, 345, 352, 376, 433, 416, 434],
};

// acumula  sinais rgb?
class FrameBuffer {
  constructor(maxFrames) {
    this.maxFrames = maxFrames;
    this.frames = []; // cada entry: { r, g, b, timestamp }
  }

  push(rgbMean, timestamp) {
    this.frames.push({ ...rgbMean, timestamp });
    if (this.frames.length > this.maxFrames) {
      this.frames.shift();
    }
  }

  get length() { return this.frames.length; }
  get isFull() { return this.frames.length >= this.maxFrames; }

  getChannels() {
    return {
      r: this.frames.map(f => f.r),
      g: this.frames.map(f => f.g),
      b: this.frames.map(f => f.b),
      timestamps: this.frames.map(f => f.timestamp),
    };
  }

  clear() { this.frames = []; }
}

// extração das ROI e cálculo da média dos pixels RGB das landmarks 
function extractROIMean(imageData, landmarks) { // imageData - pixels brutos
  const { width, height, data } = imageData;

  let totalR = 0, totalG = 0, totalB = 0, totalCount = 0;

  for (const [roiName, indices] of Object.entries(ROI_LANDMARKS)) {
    const points = indices.map(i => ({
      x: Math.round(landmarks[i].x * width),
      y: Math.round(landmarks[i].y * height),
    }));

    const xMin = Math.max(0, Math.min(...points.map(p => p.x)));
    const xMax = Math.min(width - 1, Math.max(...points.map(p => p.x)));
    const yMin = Math.max(0, Math.min(...points.map(p => p.y)));
    const yMax = Math.min(height - 1, Math.max(...points.map(p => p.y)));

    for (let y = yMin; y <= yMax; y++) {
      for (let x = xMin; x <= xMax; x++) {
        if (pointInPolygon({ x, y }, points)) {
          const idx = (y * width + x) * 4;
          totalR += data[idx];
          totalG += data[idx + 1];
          totalB += data[idx + 2];
          totalCount++;
        }
      }
    }
  }

  if (totalCount === 0) return { r: 0, g: 0, b: 0 };

  return {
    r: totalR / totalCount,
    g: totalG / totalCount,
    b: totalB / totalCount,
  };
}

/**
 * Ray casting algorithm — verifica se ponto está dentro do polígono da ROI
 */
function pointInPolygon(point, polygon) {
  let inside = false;
  const n = polygon.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i].x, yi = polygon[i].y;
    const xj = polygon[j].x, yj = polygon[j].y;
    const intersect = ((yi > point.y) !== (yj > point.y)) &&
      (point.x < (xj - xi) * (point.y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}


// POS (Plane-Orthogonal-to-Skin) do Wang et al. 2017 para diminuir ruíd
function applyPOS(r, g, b) {
  const n = r.length;

  // Normaliza cada canal pela sua média temporal (C_n = C / mean(C))
  const meanR = mean(r), meanG = mean(g), meanB = mean(b);
  const rn = r.map(v => v / (meanR || 1));
  const gn = g.map(v => v / (meanG || 1));
  const bn = b.map(v => v / (meanB || 1));

  // Projeção POS: S1 = Gn - Bn ; S2 = Gn + Bn - 2*Rn
  const s1 = gn.map((g, i) => g - bn[i]);
  const s2 = gn.map((g, i) => g + bn[i] - 2 * rn[i]);

  // Componente final H = S1 + (std(S1)/std(S2)) * S2
  const alpha = stdDev(s1) / (stdDev(s2) || 1e-10);
  const h = s1.map((v, i) => v + alpha * s2[i]);

  // Remove tendência linear (detrend)
  return detrend(h);
}


// ---------------------------------------------------------------------------
// 5. FILTRO PASSA-BANDA — Butterworth de ordem 2 (implementação biquad)
// ---------------------------------------------------------------------------

/**
 * Filtro Butterworth passa-banda digital (ordem 2, implementação biquad IIR).
 * Mantém apenas frequências entre freqMin e freqMax Hz.
 *
 * @param {number[]} signal
 * @param {number} fs - taxa de amostragem (FPS)
 * @param {number} freqMin
 * @param {number} freqMax
 * @returns {number[]}
 */
function bandpassFilter(signal, fs, freqMin = CONFIG.FREQ_MIN, freqMax = CONFIG.FREQ_MAX) {
  // Coeficientes biquad para passa-banda — calculados via transformação bilinear
  const nyq = fs / 2;
  const low = freqMin / nyq;
  const high = freqMax / nyq;

  // Pré-warp das frequências
  const w1 = Math.tan(Math.PI * low);
  const w2 = Math.tan(Math.PI * high);
  const bw = w2 - w1;
  const wc2 = w1 * w2;

  // Coeficientes do filtro passa-banda biquad
  const b0 = bw;
  const b1 = 0;
  const b2 = -bw;
  const a0 = 1 + bw + wc2;
  const a1 = 2 * (wc2 - 1);
  const a2 = 1 - bw + wc2;

  // Normaliza
  const B = [b0 / a0, b1 / a0, b2 / a0];
  const A = [1, a1 / a0, a2 / a0];

  // Aplica filtro IIR (forward pass)
  const out = new Array(signal.length).fill(0);
  for (let i = 0; i < signal.length; i++) {
    out[i] = B[0] * signal[i]
      + (i >= 1 ? B[1] * signal[i - 1] - A[1] * out[i - 1] : 0)
      + (i >= 2 ? B[2] * signal[i - 2] - A[2] * out[i - 2] : 0);
  }

  return out;
}


// ---------------------------------------------------------------------------
// 6. FFT — Fast Fourier Transform (Cooley-Tukey radix-2)
// ---------------------------------------------------------------------------

/**
 * FFT recursiva in-place sobre array de tamanho potência de 2.
 * Retorna {magnitude, freqs} para análise espectral.
 */
function fft(signal) {
  // Pad para próxima potência de 2
  const n = nextPow2(signal.length);
  const padded = [...signal, ...new Array(n - signal.length).fill(0)];

  // Parte real e imaginária
  const re = [...padded];
  const im = new Array(n).fill(0);

  fftInPlace(re, im, n);

  // Magnitude (apenas metade positiva do espectro)
  const half = Math.floor(n / 2);
  const magnitude = re.slice(0, half).map((r, i) => Math.sqrt(r * r + im[i] * im[i]));

  return { magnitude, n };
}

function fftInPlace(re, im, n) {
  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  // Cooley-Tukey butterfly
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe; im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe; im[i + k + len / 2] = uIm - vIm;
        const newRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newRe;
      }
    }
  }
}

// extrai BPM dominante
function extractBPM(signal, fs) { // signal - sinal filtrado
  const { magnitude, n } = fft(signal);

  const freqResolution = fs / n;
  const freqs = magnitude.map((_, i) => i * freqResolution);

  const validIndices = freqs.reduce((acc, f, i) => {
    if (f >= CONFIG.FREQ_MIN && f <= CONFIG.FREQ_MAX) acc.push(i);
    return acc;
  }, []);

  if (validIndices.length === 0) return { bpm: 0, confidence: 0 };

  let peakIdx = validIndices[0];
  let peakMag = 0;
  let totalPower = 0;

  for (const i of validIndices) {
    totalPower += magnitude[i] * magnitude[i];
    if (magnitude[i] > peakMag) {
      peakMag = magnitude[i];
      peakIdx = i;
    }
  }

  const dominantFreq = freqs[peakIdx];
  const bpm = Math.round(dominantFreq * 60);

  // confidence: razão entre potência do pico e potência total na faixa
  const peakPower = peakMag * peakMag;
  const confidence = Math.min(1, peakPower / (totalPower || 1));

  return {
    bpm,
    confidence: Math.round(confidence * 100),
    spectrum: { freqs: freqs.slice(0, freqs.length), magnitude },
  };
}


// detecção de pcos para cálculo HRV
function detectPeaks(signal, fs, bpm) {
  const minDistFrames = Math.floor((60 / (bpm * 1.5)) * fs); // distância mínima entre batimentos
  const threshold = mean(signal) + 0.3 * stdDev(signal);

  const peaks = [];
  for (let i = 1; i < signal.length - 1; i++) {
    if (
      signal[i] > signal[i - 1] &&
      signal[i] > signal[i + 1] &&
      signal[i] > threshold
    ) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDistFrames) {
        peaks.push(i);
      }
    }
  }
  return peaks;
}

// cálculo métricas HRV
function calculateHRV(peakIndices, fs) {
  if (peakIndices.length < 3) {
    return { rmssd: 0, sdnn: 0, meanRR: 0, rrIntervals: [] };
  }

  // Intervalos R-R em milissegundos
  const rrIntervals = [];
  for (let i = 1; i < peakIndices.length; i++) {
    const rr = ((peakIndices[i] - peakIndices[i - 1]) / fs) * 1000;
    // Filtra intervalos fisiologicamente impossíveis (< 300ms ou > 2000ms)
    if (rr >= 300 && rr <= 2000) rrIntervals.push(rr);
  }

  if (rrIntervals.length < 2) return { rmssd: 0, sdnn: 0, meanRR: 0, rrIntervals: [] };

  const meanRR = mean(rrIntervals);

  // SDNN: desvio padrão dos intervalos R-R (variabilidade total)
  const sdnn = stdDev(rrIntervals);

  // RMSSD: raiz quadrada da média dos quadrados das diferenças sucessivas
  // Métrica mais relevante para estresse e atividade parassimpática
  let sumSquaredDiff = 0;
  for (let i = 1; i < rrIntervals.length; i++) {
    const diff = rrIntervals[i] - rrIntervals[i - 1];
    sumSquaredDiff += diff * diff;
  }
  const rmssd = Math.sqrt(sumSquaredDiff / (rrIntervals.length - 1));

  return {
    rmssd: Math.round(rmssd),
    sdnn: Math.round(sdnn),
    meanRR: Math.round(meanRR),
    rrIntervals,
  };
}


// estimativa SpO2
function estimateSpO2(r, g) {
  // AC = amplitude do sinal (max - min após detrend)
  // DC = média do sinal
  const rDetrend = detrend(r);
  const gDetrend = detrend(g);

  const acR = Math.max(...rDetrend) - Math.min(...rDetrend);
  const dcR = mean(r) || 1;
  const acG = Math.max(...gDetrend) - Math.min(...gDetrend);
  const dcG = mean(g) || 1;

  // razão de perfusão (proxy) ???????????
  const ratio = (acR / dcR) / (acG / dcG);

  const spo2 = Math.min(100, Math.max(85, CONFIG.SPO2_CALIBRATION_A - CONFIG.SPO2_CALIBRATION_B * ratio));

  const reliable = ratio > 0.3 && ratio < 3.0;

  return { spo2: Math.round(spo2 * 10) / 10, reliable };
}


//score bem estar
function calculateWellnessScore(metrics) {
  const { bpm, confidence, rmssd, sdnn, spo2 } = metrics;

  // score de FC: penaliza extremos (bradicardia / taquicardia)
  let bpmScore = 100;
  if (bpm < 50 || bpm > 110) bpmScore = 30;
  else if (bpm < 60 || bpm > 100) bpmScore = 70;
  else if (bpm >= 60 && bpm <= 80) bpmScore = 100;
  else bpmScore = 85; // 80–100 BPM — dentro do normal mas acima do ideal

  // score de HRV baseado em RMSSD
  // < 20ms = muito baixo (estresse alto); > 60ms = excelente
  let hrvScore;
  if (rmssd <= 0) hrvScore = 50; // sem dados suficientes
  else if (rmssd < 15) hrvScore = 20;
  else if (rmssd < 25) hrvScore = 45;
  else if (rmssd < 40) hrvScore = 65;
  else if (rmssd < 60) hrvScore = 85;
  else hrvScore = 100;

  // score de SpO₂
  let spo2Score;
  if (spo2 >= 97) spo2Score = 100;
  else if (spo2 >= 95) spo2Score = 85;
  else if (spo2 >= 92) spo2Score = 60;
  else spo2Score = 30;

  // score levando em conta a confiança
  const confidenceFactor = confidence / 100;
  const rawScore = (bpmScore * 0.35) + (hrvScore * 0.45) + (spo2Score * 0.20);
  const overall = Math.round(rawScore * confidenceFactor + 50 * (1 - confidenceFactor));

  let interpretation;
  if (overall >= 80) interpretation = 'Ótimo — sinais cardiovasculares dentro dos parâmetros ideais';
  else if (overall >= 65) interpretation = 'Bom — pequenas variações, monitoramento recomendado';
  else if (overall >= 50) interpretation = 'Atenção — indicadores sugerem possível estresse ou fadiga';
  else interpretation = 'Alerta — recomenda-se consulta com profissional de saúde';

  return {
    overall,
    bpmScore: Math.round(bpmScore),
    hrvScore: Math.round(hrvScore),
    spo2Score: Math.round(spo2Score),
    interpretation,
  };
}

class RppgAnalyzer {
  constructor(config = {}) {
    this.config = { ...CONFIG, ...config };
    this.buffer = new FrameBuffer(this.config.FPS * this.config.WINDOW_SECONDS);
    this.isRunning = false;
    this.videoElement = null;
    this.canvasElement = null;
    this.faceMesh = null;
    this.animFrameId = null;
    this.startTime = null;

    // callbacks
    this.onProgress = null;   // (percent: number, partialMetrics?: object) => void
    this.onResult = null;     // (result: object) => void
    this.onError = null;      // (error: Error) => void
    this.onFaceDetected = null; // (detected: boolean) => void
  }

  async init(videoEl, canvasEl) {
    this.videoElement = videoEl;
    this.canvasElement = canvasEl;

    // solicita acesso à câmera
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: this.config.FPS },
          facingMode: 'user',
        },
        audio: false,
      });
      this.videoElement.srcObject = stream;
      await new Promise(resolve => { this.videoElement.onloadedmetadata = resolve; });
      await this.videoElement.play();
    } catch (err) {
      this._error(new Error(`Câmera indisponível: ${err.message}`));
      return false;
    }

    // inicializa MediaPipe Face Mesh
    // (MediaPipe deve estar carregado globalmente via <script> no HTML)
    try {
      this.faceMesh = new FaceMesh({
        locateFile: file =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });
      this.faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      this.faceMesh.onResults(results => this._onFaceMeshResults(results));
      await this.faceMesh.initialize();
    } catch (err) {
      this._error(new Error(`MediaPipe não carregado: ${err.message}`));
      return false;
    }

    return true;
  }

  async start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.buffer.clear();
    this.startTime = performance.now();
    this._captureLoop();
  }

  stop() {
    this.isRunning = false;
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);
    if (this.videoElement?.srcObject) {
      this.videoElement.srcObject.getTracks().forEach(t => t.stop());
    }
  }

  _captureLoop() {
    if (!this.isRunning) return;

    const elapsed = (performance.now() - this.startTime) / 1000;
    const targetFrames = this.config.FPS * this.config.WINDOW_SECONDS;

    const progress = Math.min(100, Math.round((elapsed / this.config.WINDOW_SECONDS) * 100));
    if (this.onProgress) {
      // métricas parciais
      const partialMetrics = (this.buffer.length >= this.config.MIN_FRAMES && elapsed % 5 < 0.1)
        ? this._computePartialMetrics()
        : null;
      this.onProgress(progress, partialMetrics);
    }

    if (elapsed >= this.config.WINDOW_SECONDS || this.buffer.length >= targetFrames) {
      this._finalizeAnalysis();
      return;
    }

    this.faceMesh.send({ image: this.videoElement });
    this.animFrameId = requestAnimationFrame(() => this._captureLoop());
  }

  // aciona mediapipe a cada frame
  _onFaceMeshResults(results) {
    const faceDetected = results.multiFaceLandmarks?.length > 0;
    if (this.onFaceDetected) this.onFaceDetected(faceDetected);

    if (!faceDetected || !this.isRunning) return;

    // renderiza e extrai frames brutos
    const ctx = this.canvasElement.getContext('2d');
    ctx.drawImage(this.videoElement, 0, 0,
      this.canvasElement.width, this.canvasElement.height);
    const imageData = ctx.getImageData(0, 0,
      this.canvasElement.width, this.canvasElement.height);

    const landmarks = results.multiFaceLandmarks[0];
    const rgbMean = extractROIMean(imageData, landmarks);

    this.buffer.push(rgbMean, performance.now());
  }

  // feedback parcial na hora
  _computePartialMetrics() {
    try {
      const { r, g, b } = this.buffer.getChannels();
      const posSignal = applyPOS(r, g, b);
      const filtered = bandpassFilter(posSignal, this.config.FPS);
      const { bpm, confidence } = extractBPM(filtered, this.config.FPS);
      return { bpm, confidence };
    } catch {
      return null;
    }
  }

  _finalizeAnalysis() {
    this.isRunning = false;
    if (this.animFrameId) cancelAnimationFrame(this.animFrameId);

    try {
      const result = this._runPipeline();
      if (this.onResult) this.onResult(result);
    } catch (err) {
      this._error(err);
    }
  }

  _runPipeline() {
    if (this.buffer.length < this.config.MIN_FRAMES) {
      throw new Error(`Frames insuficientes: ${this.buffer.length}. Mantenha o rosto visível.`);
    }

    const { r, g, b } = this.buffer.getChannels();

    // 1 — POS
    const posSignal = applyPOS(r, g, b);

    // 2 — Filtro passa-banda
    const filtered = bandpassFilter(posSignal, this.config.FPS);

    // 3 — FFT → BPM
    const { bpm, confidence, spectrum } = extractBPM(filtered, this.config.FPS);

    // 4 — Picos R-R → HRV
    const peaks = detectPeaks(filtered, this.config.FPS, bpm);
    const hrv = calculateHRV(peaks, this.config.FPS);

    // 5 — SpO₂
    const { spo2, reliable: spo2Reliable } = estimateSpO2(r, g);

    // 6 — Score composto
    const metrics = { bpm, confidence, rmssd: hrv.rmssd, sdnn: hrv.sdnn, spo2 };
    const wellness = calculateWellnessScore(metrics);

    return {
      bpm,
      confidence,           // 0–100 — qualidade do sinal

      hrv: {
        rmssd: hrv.rmssd,   // ms — estresse/parassimpático
        sdnn: hrv.sdnn,     // ms — variabilidade total
        meanRR: hrv.meanRR, // ms — intervalo médio
        rrIntervals: hrv.rrIntervals,
      },

      spo2: {
        value: spo2,
        reliable: spo2Reliable,  // false = câmera com limitações para SpO₂
      },

      // score
      wellness,

      raw: {
        posSignal,
        filteredSignal: filtered,
        spectrum,
        framesAnalyzed: this.buffer.length,
        durationSeconds: this.buffer.length / this.config.FPS,
      },

      timestamp: new Date().toISOString(),
      disclaimer: 'Resultado indicativo. Não substitui avaliação médica.',
    };
  }

  _error(err) {
    if (this.onError) this.onError(err);
    else console.error('[rPPG]', err);
  }
}


// cálculos matemáticos
function mean(arr) {
  return arr.reduce((s, v) => s + v, 0) / (arr.length || 1);
}

function stdDev(arr) {
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length || 1));
}

// remove least-squares detrend - evitar vazamento
function detrend(signal) {
  const n = signal.length;
  const x = Array.from({ length: n }, (_, i) => i);
  const mx = mean(x), my = mean(signal);
  const num = x.reduce((s, xi, i) => s + (xi - mx) * (signal[i] - my), 0);
  const den = x.reduce((s, xi) => s + (xi - mx) ** 2, 0);
  const slope = num / (den || 1);
  const intercept = my - slope * mx;
  return signal.map((v, i) => v - (slope * i + intercept));
}

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    RppgAnalyzer,
    extractROIMean,
    applyPOS,
    bandpassFilter,
    extractBPM,
    detectPeaks,
    calculateHRV,
    estimateSpO2,
    calculateWellnessScore,
    fft,
    detrend,
    mean,
    stdDev,
  };
}