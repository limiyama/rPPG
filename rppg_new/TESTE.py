import time
import sys

import cv2
import numpy as np
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )
except ImportError:
    print("Instale o mediapipe: pip install mediapipe")
    sys.exit(1)

MODEL_PATH = 'assets/face_landmarker.task'

# configuração das ROIs usando MediaPipe
ROI_LANDMARKS = {
    "testa": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
              361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
              176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
              162, 21, 54, 103, 67, 109],
}

# dividindo ROIs por polígonos menores para maior precisão
ROI_POINTS = {
    "testa":            [67, 109, 10, 338, 297, 332, 284, 251, 301, 71],
    "bochecha_esquerda": [50, 187, 205, 36, 142, 126, 209, 49, 129, 203],
    "bochecha_direita":  [280, 411, 425, 266, 371, 355, 429, 279, 358, 423],
}

def get_roi_mask(landmarks_px, roi_indices, frame_shape):
    pts = np.array([landmarks_px[i] for i in roi_indices if i < len(landmarks_px)],
                    dtype=np.int32)
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(pts)
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

# captura de vídeo + detecção da face e das ROIs
class RPPGCapture:
    def __init__(self, camera_index=0, duration_s=30):
        self.camera_index = camera_index
        self.duration_s = duration_s

        base_options = BaseOptions(model_asset_path=MODEL_PATH)
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.face_landmarker = FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

        # buffers de sinal RGB por frame, por ROI (média)
        self.roi_signals = {roi: [] for roi in ROI_POINTS}
        self.timestamps = []
        self.n_frames_descartados = 0

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError("Não foi possível acessar a câmera.")

        print(f"Escaneamento facial iniciado ({self.duration_s}s). "
              f"Mantenha o rosto estável e bem iluminado.")
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - t_start
            if elapsed > self.duration_s:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((time.time() - t_start) * 1000)
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_valid = False
            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0] 
                h, w = frame.shape[:2]
                landmarks_px = [(int(lm.x * w), int(lm.y * h))
                                 for lm in face_landmarks]

                frame_valid = True
                frame_means = {}
                for roi_name, idxs in ROI_POINTS.items():
                    mask = get_roi_mask(landmarks_px, idxs, frame.shape)
                    if mask is None or cv2.countNonZero(mask) == 0:
                        frame_valid = False
                        break
                    
                    mean_rgb = cv2.mean(rgb_frame, mask=mask)[:3] 
                    frame_means[roi_name] = mean_rgb

                    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contour, -1, (0, 255, 0), 1)

                if frame_valid:
                    for roi_name in ROI_POINTS:
                        self.roi_signals[roi_name].append(frame_means[roi_name])
                    self.timestamps.append(time.time())

            if not frame_valid:
                self.n_frames_descartados += 1

            remaining = max(0, self.duration_s - elapsed)
            cv2.putText(frame, f"Escaneando... {remaining:0.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("rPPG - Escaneamento Facial", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.face_landmarker.close()

        n_frames = len(self.timestamps)
        if n_frames < 2:
            raise RuntimeError("Poucos frames válidos capturados. "
                                "Verifique iluminação e posicionamento do rosto.")

        fps_medio = n_frames / (self.timestamps[-1] - self.timestamps[0])
        print(f"Captura concluída: {n_frames} frames válidos, "
              f"{self.n_frames_descartados} descartados (rosto/ROI não detectados), "
              f"fps médio ≈ {fps_medio:.2f}")

        signals = {roi: np.array(vals, dtype=np.float64) 
                   for roi, vals in self.roi_signals.items()}
        return signals, np.array(self.timestamps, dtype=np.float64)


# reamostragem para grade de tempo uniforme
def resample_to_uniform_grid(roi_signals, timestamps, fps_target=None):
    timestamps = np.asarray(timestamps, dtype=np.float64)

    if fps_target is None:
        dt_mediano = np.median(np.diff(timestamps))
        fps_target = 1.0 / dt_mediano

    t_uniform = np.arange(timestamps[0], timestamps[-1], 1.0 / fps_target)

    resampled = {}
    for roi_name, rgb_raw in roi_signals.items():
        interp = interp1d(timestamps, rgb_raw, axis=0, kind='linear')
        resampled[roi_name] = interp(t_uniform)

    return resampled, t_uniform, fps_target


# pré-processamento
def preprocess_rgb_signal(rgb_signal):
    detrended = sig.detrend(rgb_signal, axis=0, type='linear')
    means = np.mean(rgb_signal, axis=0)
    means[means == 0] = 1e-6
    normalized = detrended / means + 1.0 
    return normalized


# suaviza artefatos de movimento da câmera
def moving_average_smooth(x, window=3):
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.array([np.convolve(x[:, c], kernel, mode='same')
                      for c in range(x.shape[1])]).T


# extração do rPPG por CHROM
def chrom_algorithm(rgb_norm):
    R, G, B = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]

    X = 3 * R - 2 * G
    Y = 1.5 * R + G - 1.5 * B

    std_x = np.std(X)
    std_y = np.std(Y)
    alpha = std_x / std_y if std_y != 0 else 1.0

    S = X - alpha * Y
    return S


# extração do rPPG por POS
def pos_algorithm(rgb_norm, fps, window_s=1.6):
    n = rgb_norm.shape[0]
    window_len = max(3, int(round(window_s * fps)))
    S = np.zeros(n)

    for start in range(0, n - window_len + 1):
        end = start + window_len
        segment = rgb_norm[start:end]

        mean_seg = np.mean(segment, axis=0)
        mean_seg[mean_seg == 0] = 1e-6
        Cn = segment / mean_seg

        S1 = Cn[:, 1] - Cn[:, 2]              
        S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]  

        std1, std2 = np.std(S1), np.std(S2)
        alpha = std1 / std2 if std2 != 0 else 1.0
        h = S1 + alpha * S2

        h = (h - np.mean(h)) / (np.std(h) + 1e-8)
        S[start:end] += h

    return S


# processa, aplica CHROM e POS separadamente, normaliza os resultados e faz a média
def combine_roi_and_methods(roi_signals, fps):
    combined_per_roi = []

    for roi_name, rgb_raw in roi_signals.items():
        rgb_smooth = moving_average_smooth(rgb_raw, window=3)
        rgb_norm = preprocess_rgb_signal(rgb_smooth)

        s_chrom = chrom_algorithm(rgb_norm)
        s_pos = pos_algorithm(rgb_norm, fps)

        z_chrom = (s_chrom - np.mean(s_chrom)) / (np.std(s_chrom) + 1e-8)
        z_pos = (s_pos - np.mean(s_pos)) / (np.std(s_pos) + 1e-8)

        s_final_roi = 0.5 * z_chrom + 0.5 * z_pos
        combined_per_roi.append(s_final_roi)

    min_len = min(len(s) for s in combined_per_roi)
    combined_per_roi = [s[:min_len] for s in combined_per_roi]
    rppg_signal = np.mean(np.vstack(combined_per_roi), axis=0)

    return rppg_signal


# filtro butterworth passa-banda entre 0.7 Hz e 4.0 Hz
def bandpass_filter(x, fps, low_hz=0.7, high_hz=4.0, order=4):
    nyq = 0.5 * fps
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    b, a = sig.butter(order, [low, high], btype='band')
    y = sig.filtfilt(b, a, x)
    return y


# calcula frequência cardíaca (HR) por FFT
def compute_hr_fft(filtered_signal, fps):
    n = len(filtered_signal)
    windowed = filtered_signal * np.hanning(n)

    freqs = rfftfreq(n, d=1.0 / fps)
    fft_vals = np.abs(rfft(windowed))

    valid = (freqs >= 0.7) & (freqs <= 4.0)
    freqs_valid = freqs[valid]
    fft_valid = fft_vals[valid]

    if len(freqs_valid) == 0:
        raise RuntimeError("Não foi possível estimar HR: faixa espectral vazia.")

    peak_idx = np.argmax(fft_valid)
    peak_freq_hz = freqs_valid[peak_idx]
    hr_bpm = peak_freq_hz * 60.0

    return hr_bpm


# cálculo HRV
def compute_hrv(filtered_signal, fps, t_uniform, max_hr_bpm=200.0, min_prominence_std=0.3):
    min_distance = max(1, int(round(fps * 60.0 / max_hr_bpm)))
    prominence = min_prominence_std * np.std(filtered_signal)

    peaks, _ = sig.find_peaks(filtered_signal, distance=min_distance, prominence=prominence)

    if len(peaks) < 3:
        return {
            "SDNN_ms": None, "RMSSD_ms": None, "pNN50_%": None,
            "n_batimentos_detectados": len(peaks),
            "aviso": "Batimentos insuficientes para HRV confiável."
        }, peaks, np.array([])

    peak_times_s = t_uniform[peaks]
    ibi_ms = np.diff(peak_times_s) * 1000.0  

    ibi_ms = ibi_ms[(ibi_ms > 250) & (ibi_ms < 1500)]

    if len(ibi_ms) < 2:
        return {
            "SDNN_ms": None, "RMSSD_ms": None, "pNN50_%": None,
            "n_batimentos_detectados": len(peaks),
            "aviso": "IBIs insuficientes após remoção de outliers."
        }, peaks, ibi_ms

    sdnn = np.std(ibi_ms, ddof=1)
    diffs = np.diff(ibi_ms)
    rmssd = np.sqrt(np.mean(diffs ** 2))
    nn50 = np.sum(np.abs(diffs) > 50)
    pnn50 = 100.0 * nn50 / len(diffs) if len(diffs) > 0 else 0.0

    metrics = {
        "SDNN_ms": round(sdnn, 2),
        "RMSSD_ms": round(rmssd, 2),
        "pNN50_%": round(pnn50, 2),
        "n_batimentos_detectados": len(peaks),
    }
    return metrics


# main
def main():
    DURATION_S = 30.0  
    CAMERA_INDEX = 0
    MAX_HR = 200.0
    MIN_PROMINENCE = 0.3

    capture = RPPGCapture(camera_index=CAMERA_INDEX, duration_s=DURATION_S)
    roi_signals_raw, timestamps = capture.run()

    roi_signals, t_uniform, fps = resample_to_uniform_grid(roi_signals_raw, timestamps)
    print(f"Sinal reamostrado para grade uniforme: fps={fps:.2f}, {len(t_uniform)} amostras")

    rppg_signal = combine_roi_and_methods(roi_signals, fps)

    filtered = bandpass_filter(rppg_signal, fps, low_hz=0.7, high_hz=4.0)

    hr_bpm = compute_hr_fft(filtered, fps)

    hrv_metrics = compute_hrv(
        filtered, fps, t_uniform[:len(filtered)],
        max_hr_bpm=MAX_HR, min_prominence_std=MIN_PROMINENCE,
    )

    duracao_medida_s = t_uniform[len(filtered) - 1] - t_uniform[0]
    n_batimentos = hrv_metrics["n_batimentos_detectados"]
    bpm_por_contagem_picos = (n_batimentos / (duracao_medida_s / 60.0)
                               if n_batimentos and duracao_medida_s > 0 else None)

    print("\n================ RESULTADOS ================")
    print(f"HR estimado por FFT:                 {hr_bpm:.1f} bpm")
    print(f"Duração efetivamente analisada:      {duracao_medida_s:.2f} s")
    if bpm_por_contagem_picos is not None:
        print(f"HR estimado por contagem de picos:   {bpm_por_contagem_picos:.1f} bpm")
    print("\nVariabilidade da Frequência Cardíaca (HRV):")
    for k, v in hrv_metrics.items():
        print(f"  {k}: {v}")
    print("==============================================\n")

    return {"hr_bpm": hr_bpm, "hrv": hrv_metrics}


if __name__ == "__main__":
    main()