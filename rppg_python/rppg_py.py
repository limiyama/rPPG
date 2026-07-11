import time
from datetime import datetime
import numpy as np
import cv2

CONFIG = {
    "FPS": 30,                  # frames por segundo da câmera
    "WINDOW_SECONDS": 30,       # tempo de escaneamento
    "BPM_MIN": 42,              # 0.7 Hz
    "BPM_MAX": 210,             # 3.5 Hz
    "FREQ_MIN": 0.7,
    "FREQ_MAX": 3.5,
    "MIN_FRAMES": 150,          # mínimo de frames para análise confiável (5s)
    "SPO2_CALIBRATION_A": 110.0,# constantes empíricas para SpO₂ (modelo linear)
    "SPO2_CALIBRATION_B": 25.0,
}

ROI_LANDMARKS = {
    # landmarks da testa
    "forehead": [10, 67, 69, 104, 108, 109, 151, 299, 337, 338],
    # landmarks bochecha esquerda
    "leftCheek": [234, 227, 116, 123, 147, 213, 192, 214],
    # landmarks bochecha direita
    "rightCheek": [454, 447, 345, 352, 376, 433, 416, 434],
}

# buffer de frames?
class FrameBuffer:
    def __init__(self, max_frames):
        self.max_frames = max_frames
        self.frames = []  # cada entry: {"r": v, "g": v, "b": v, "timestamp": v}

    def push(self, rgb_mean, timestamp):
        self.frames.append({**rgb_mean, "timestamp": timestamp})
        if len(self.frames) > self.max_frames:
            self.frames.pop(0)

    @property
    def length(self):
        return len(self.frames)

    @property
    def is_full(self):
        return len(self.frames) >= self.max_frames

    def get_channels(self):
        return {
            "r": np.array([f["r"] for f in self.frames]),
            "g": np.array([f["g"] for f in self.frames]),
            "b": np.array([f["b"] for f in self.frames]),
            "timestamps": np.array([f["timestamp"] for f in self.frames]),
        }

    def clear(self):
        self.frames = []


# extração de ROIs
def extract_roi_mean(image_data, landmarks):
    height, width, _ = image_data.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for indices in ROI_LANDMARKS.values():
        points = []
        for idx in indices:
            lm = landmarks[idx]
            px = getattr(lm, 'x', None)
            py = getattr(lm, 'y', None)
            if px is None:
                px = lm.get('x', 0)
                py = lm.get('y', 0)
            points.append((
                int(round(px * width)),
                int(round(py * height)),
            ))

        if len(points) < 3:
            continue

        poly = np.array([points], dtype=np.int32)
        cv2.fillPoly(mask, poly, 1)

    total_count = int(np.count_nonzero(mask))
    if total_count == 0:
        return {"r": 0.0, "g": 0.0, "b": 0.0}

    total_r = float(np.sum(image_data[:, :, 0][mask == 1]))
    total_g = float(np.sum(image_data[:, :, 1][mask == 1]))
    total_b = float(np.sum(image_data[:, :, 2][mask == 1]))

    return {
        "r": total_r / total_count,
        "g": total_g / total_count,
        "b": total_b / total_count,
    }

def detrend(signal):
    n = len(signal)
    x = np.arange(n)
    mean_x = np.mean(x)
    mean_y = np.mean(signal)
    
    num = np.sum((x - mean_x) * (signal - mean_y))
    den = np.sum((x - mean_x) ** 2)
    slope = num / den if den != 0 else 0
    intercept = mean_y - slope * mean_x
    
    return signal - (slope * x + intercept)

def apply_pos(r, g, b):
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    
    rn = r / (mean_r if mean_r != 0 else 1.0)
    gn = g / (mean_g if mean_g != 0 else 1.0)
    bn = b / (mean_b if mean_b != 0 else 1.0)

    s1 = gn - bn
    s2 = gn + bn - 2 * rn

    std_s1 = np.std(s1)
    std_s2 = np.std(s2)
    alpha = std_s1 / (std_s2 if std_s2 != 0 else 1e-10)
    
    h = s1 + alpha * s2
    return detrend(h)

def bandpass_filter(signal, fs, freq_min=CONFIG["FREQ_MIN"], freq_max=CONFIG["FREQ_MAX"]):
    nyq = fs / 2.0
    low = freq_min / nyq
    high = freq_max / nyq

    w1 = np.tan(np.pi * low)
    w2 = np.tan(np.pi * high)
    bw = w2 - w1
    wc2 = w1 * w2

    b0 = bw
    b1 = 0.0
    b2 = -bw
    a0 = 1.0 + bw + wc2
    a1 = 2.0 * (wc2 - 1.0)
    a2 = 1.0 - bw + wc2

    B = [b0 / a0, b1 / a0, b2 / a0]
    A = [1.0, a1 / a0, a2 / a0]

    out = np.zeros_like(signal)
    for i in range(len(signal)):
        term_b1 = B[1] * signal[i - 1] - A[1] * out[i - 1] if i >= 1 else 0.0
        term_b2 = B[2] * signal[i - 2] - A[2] * out[i - 2] if i >= 2 else 0.0
        out[i] = B[0] * signal[i] + term_b1 + term_b2

    return out

def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p

def fft_analysis(signal):
    n = next_pow2(len(signal))
    padded = np.zeros(n)
    padded[:len(signal)] = signal

    # FFT nativo do NumPy
    fft_res = np.fft.fft(padded)
    half = n // 2
    magnitude = np.abs(fft_res[:half])
    
    return magnitude, n

def extract_bpm(signal, fs):
    magnitude, n = fft_analysis(signal)
    freq_resolution = fs / n
    freqs = np.arange(len(magnitude)) * freq_resolution

    valid_indices = [i for i, f in enumerate(freqs) if CONFIG["FREQ_MIN"] <= f <= CONFIG["FREQ_MAX"]]

    if not valid_indices:
        return {"bpm": 0, "confidence": 0, "spectrum": {"freqs": freqs.tolist(), "magnitude": magnitude.tolist()}}

    peak_idx = valid_indices[0]
    peak_mag = 0.0
    total_power = 0.0

    for i in valid_indices:
        power = magnitude[i] * magnitude[i]
        total_power += power
        if magnitude[i] > peak_mag:
            peak_mag = magnitude[i]
            peak_idx = i

    dominant_freq = freqs[peak_idx]
    bpm = int(round(dominant_freq * 60))

    peak_power = peak_mag * peak_mag
    confidence = min(1.0, peak_power / total_power) if total_power != 0 else 0.0

    return {
        "bpm": bpm,
        "confidence": int(round(confidence * 100)),
        "spectrum": {"freqs": freqs.tolist(), "magnitude": magnitude.tolist()},
    }


# HRV
def detect_peaks(signal, fs, bpm):
    if bpm <= 0:
        bpm = 70
    min_dist_frames = int(np.floor((60.0 / (bpm * 1.5)) * fs))
    threshold = np.mean(signal) + 0.3 * np.std(signal)

    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1] and signal[i] > threshold:
            if not peaks or (i - peaks[-1]) >= min_dist_frames:
                peaks.append(i)
    return peaks

def calculate_hrv(peak_indices, fs):
    if len(peak_indices) < 3:
        return {"rmssd": 0, "sdnn": 0, "meanRR": 0, "rrIntervals": []}

    rr_intervals = []
    for i in range(1, len(peak_indices)):
        rr = ((peak_indices[i] - peak_indices[i - 1]) / fs) * 1000.0
        if 300.0 <= rr <= 2000.0:
            rr_intervals.append(rr)

    if len(rr_intervals) < 2:
        return {"rmssd": 0, "sdnn": 0, "meanRR": 0, "rrIntervals": []}

    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)

    sum_squared_diff = 0.0
    for i in range(1, len(rr_intervals)):
        diff = rr_intervals[i] - rr_intervals[i - 1]
        sum_squared_diff += diff * diff
    
    rmssd = np.sqrt(sum_squared_diff / (len(rr_intervals) - 1))

    return {
        "rmssd": int(round(rmssd)),
        "sdnn": int(round(sdnn)),
        "meanRR": int(round(mean_rr)),
        "rrIntervals": rr_intervals,
    }

# OXIMETRIA (SpO2)
def estimate_spo2(r, g):
    r_detrend = detrend(r)
    g_detrend = detrend(g)

    ac_r = np.max(r_detrend) - np.min(r_detrend)
    dc_r = np.mean(r) if np.mean(r) != 0 else 1.0
    ac_g = np.max(g_detrend) - np.min(g_detrend)
    dc_g = np.mean(g) if np.mean(g) != 0 else 1.0

    ratio = (ac_r / dc_r) / (ac_g / dc_g) if (ac_g / dc_g) != 0 else 0.0

    spo2 = min(100.0, max(85.0, CONFIG["SPO2_CALIBRATION_A"] - CONFIG["SPO2_CALIBRATION_B"] * ratio))
    reliable = 0.3 < ratio < 3.0

    return {"spo2": round(spo2, 1), "reliable": reliable}

def calculate_wellness_score(metrics):
    bpm = metrics["bpm"]
    confidence = metrics["confidence"]
    rmssd = metrics["rmssd"]
    spo2 = metrics["spo2"]

    if bpm < 50 or bpm > 110: bpm_score = 30
    elif bpm < 60 or bpm > 100: bpm_score = 70
    elif 60 <= bpm <= 80: bpm_score = 100
    else: bpm_score = 85

    if rmssd <= 0: hrv_score = 50
    elif rmssd < 15: hrv_score = 20
    elif rmssd < 25: hrv_score = 45
    elif rmssd < 40: hrv_score = 65
    elif rmssd < 60: hrv_score = 85
    else: hrv_score = 100

    if spo2 >= 97: spo2_score = 100
    elif spo2 >= 95: spo2_score = 85
    elif spo2 >= 92: spo2_score = 60
    else: spo2_score = 30

    confidence_factor = confidence / 100.0
    raw_score = (bpm_score * 0.35) + (hrv_score * 0.45) + (spo2_score * 0.20)
    overall = int(round(raw_score * confidence_factor + 50.0 * (1.0 - confidence_factor)))

    if overall >= 80: interpretation = 'Ótimo — sinais cardiovasculares dentro dos parâmetros ideais'
    elif overall >= 65: interpretation = 'Bom — pequenas variações, monitoramento recomendado'
    elif overall >= 50: interpretation = 'Atenção — indicadores sugerem possível estresse ou fadiga'
    else: interpretation = 'Alerta — recomenda-se consulta com profissional de saúde'

    return {
        "overall": overall,
        "bpmScore": int(round(bpm_score)),
        "hrvScore": int(round(hrv_score)),
        "spo2Score": int(round(spo2_score)),
        "interpretation": interpretation,
    }


class RppgAnalyzer:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.config = {**CONFIG, **config}
        self.buffer = FrameBuffer(self.config["FPS"] * self.config["WINDOW_SECONDS"])
        self.is_running = False
        self.start_time = None

        self.on_progress = None   # lambda pct, partial_metrics: ...
        self.on_result = None     # lambda result: ...
        self.on_error = None      # lambda err_msg: ...

    def start(self):
        self.is_running = True
        self.buffer.clear()
        self.start_time = time.perf_counter()

    def process_frame(self, image_data, landmarks):
        if not self.is_running:
            return

        elapsed = time.perf_counter() - self.start_time
        target_frames = self.config["FPS"] * self.config["WINDOW_SECONDS"]

        # 1. extrai a média RGB e joga no buffer
        rgb_mean = extract_roi_mean(image_data, landmarks)
        self.buffer.push(rgb_mean, time.perf_counter())

        # 2. mostra o progresso
        progress = min(100, int(round((elapsed / self.config["WINDOW_SECONDS"]) * 100)))
        
        partial_metrics = None
        if self.buffer.length >= self.config["MIN_FRAMES"] and int(elapsed * 10) % 50 == 0:
            partial_metrics = self._compute_partial_metrics()

        if self.on_progress:
            self.on_progress(progress, partial_metrics)

        # terminar
        if elapsed >= self.config["WINDOW_SECONDS"] or self.buffer.length >= target_frames:
            self._finalize_analysis()

    def _compute_partial_metrics(self):
        try:
            channels = self.buffer.get_channels()
            pos_signal = apply_pos(channels["r"], channels["g"], channels["b"])
            filtered = bandpass_filter(pos_signal, self.config["FPS"])
            bpm_data = extract_bpm(filtered, self.config["FPS"])
            return {"bpm": bpm_data["bpm"], "confidence": bpm_data["confidence"]}
        except Exception:
            return None

    def _finalize_analysis(self):
        self.is_running = False
        try:
            result = self._run_pipeline()
            if self.on_result:
                self.on_result(result)
        except Exception as e:
            if self.on_error:
                self.on_error(str(e))
            else:
                print(f"[rPPG Error]: {e}")

    def _run_pipeline(self):
        if self.buffer.length < self.config["MIN_FRAMES"]:
            raise ValueError(f"Frames insuficientes: {self.buffer.length}. Mantenha o rosto visível.")

        channels = self.buffer.get_channels()
        r, g, b = channels["r"], channels["g"], channels["b"]

        pos_signal = apply_pos(r, g, b)
        filtered = bandpass_filter(pos_signal, self.config["FPS"])
        
        bpm_data = extract_bpm(filtered, self.config["FPS"])
        bpm = bpm_data["bpm"]
        confidence = bpm_data["confidence"]
        spectrum = bpm_data["spectrum"]

        peaks = detect_peaks(filtered, self.config["FPS"], bpm)
        hrv = calculate_hrv(peaks, self.config["FPS"])

        spo2_data = estimate_spo2(r, g)

        metrics = {
            "bpm": bpm,
            "confidence": confidence,
            "rmssd": hrv["rmssd"],
            "spo2": spo2_data["spo2"]
        }
        wellness = calculate_wellness_score(metrics)

        return {
            "bpm": bpm,
            "confidence": confidence,
            "hrv": {
                "rmssd": hrv["rmssd"],
                "sdnn": hrv["sdnn"],
                "meanRR": hrv["meanRR"],
                "rrIntervals": hrv["rrIntervals"],
            },
            "spo2": {
                "value": spo2_data["spo2"],
                "reliable": spo2_data["reliable"],
            },
            "wellness": wellness,
            "raw": {
                "posSignal": pos_signal.tolist(),
                "filteredSignal": filtered.tolist(),
                "spectrum": spectrum,
                "framesAnalyzed": self.buffer.length,
                "durationSeconds": self.buffer.length / self.config["FPS"],
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": "Resultado indicativo. Não substitui avaliação médica.",
        }