import argparse
import time
import sys

import cv2
import numpy as np
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq

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

            if result.face_landmarks:
                face_landmarks = result.face_landmarks[0]  # lista de NormalizedLandmark
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
                    # média espacial dos canais R, G, B dentro da ROI
                    mean_rgb = cv2.mean(rgb_frame, mask=mask)[:3]  # (R, G, B)
                    frame_means[roi_name] = mean_rgb

                    # contorno da ROI
                    contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contour, -1, (0, 255, 0), 1)

                if frame_valid:
                    for roi_name in ROI_POINTS:
                        self.roi_signals[roi_name].append(frame_means[roi_name])
                    self.timestamps.append(time.time())

            # feedback visual em tempo real
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

        fps_estimado = n_frames / (self.timestamps[-1] - self.timestamps[0])
        print(f"Captura concluída: {n_frames} frames válidos, "
              f"fps efetivo ≈ {fps_estimado:.2f}")

        signals = {roi: np.array(vals, dtype=np.float64)  # shape (N, 3) RGB
                   for roi, vals in self.roi_signals.items()}
        return signals, fps_estimado


# pré-processamento
def preprocess_rgb_signal(rgb_signal):
    # detrend - remove tendência linear e normaliza cada canal RGB pela sua média
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

        S1 = Cn[:, 1] - Cn[:, 2]              # G - B
        S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]  # G + B - 2R

        std1, std2 = np.std(S1), np.std(S2)
        alpha = std1 / std2 if std2 != 0 else 1.0
        h = S1 + alpha * S2

        # overlap-add com normalização de variância (sobreposição-adição)
        h = (h - np.mean(h)) / (np.std(h) + 1e-8)
        S[start:end] += h

    return S

# processa, aplica CHROM e POS separadamente, normaliza os resultados e faz a média entre eles
def combine_roi_and_methods(roi_signals, fps):
    combined_per_roi = []

    for roi_name, rgb_raw in roi_signals.items():
        rgb_smooth = moving_average_smooth(rgb_raw, window=3)
        rgb_norm = preprocess_rgb_signal(rgb_smooth)

        s_chrom = chrom_algorithm(rgb_norm)
        s_pos = pos_algorithm(rgb_norm, fps)

        # z-score de cada sinal antes de combinar (mesma escala)
        z_chrom = (s_chrom - np.mean(s_chrom)) / (np.std(s_chrom) + 1e-8)
        z_pos = (s_pos - np.mean(s_pos)) / (np.std(s_pos) + 1e-8)

        # média CHROM + POS
        s_final_roi = 0.5 * z_chrom + 0.5 * z_pos
        combined_per_roi.append(s_final_roi)

    # combina as ROIs (média) em um único sinal rPPG final.
    min_len = min(len(s) for s in combined_per_roi)
    combined_per_roi = [s[:min_len] for s in combined_per_roi]
    rppg_signal = np.mean(np.vstack(combined_per_roi), axis=0)

    return rppg_signal


# fitro butterworth passa-banda entre 0.7 Hz (42 bpm) e 4.0 Hz (240 bpm)
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
    # janela de Hann reduz vazamento espectral (spectral leakage) whatever that meanss
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

    return hr_bpm, freqs_valid, fft_valid


# cálculo HRV (diferença entre intervalos R-R)
def compute_hrv(filtered_signal, fps):
    min_distance = int(fps * 60.0 / 240.0)
    peaks, _ = sig.find_peaks(filtered_signal, distance=min_distance)

    if len(peaks) < 3:
        return {
            "SDNN_ms": None, "RMSSD_ms": None, "pNN50_%": None,
            # SDNN (Standard Deviation of Normal-to-Normal intervals) = desvio padrão de todos os intervalos entre os batimentos, medido em milissegundos
            # RMSSD (Root Mean Square of Successive Differences)= diferença sucessiva entre os batimentos - medir a atividade do sistema nervoso parassimpático?
            # pNN50 = porcentagem de batimentos consecutivos que tiveram uma diferença de tempo maior que 50 milissegundos entre eles
            "n_batimentos_detectados": len(peaks),
            "aviso": "Batimentos insuficientes para HRV confiável."
        }

    peak_times_s = peaks / fps
    ibi_ms = np.diff(peak_times_s) * 1000.0  # intervalos inter-batimento em ms

    # remove outliers
    ibi_ms = ibi_ms[(ibi_ms > 250) & (ibi_ms < 1500)]

    if len(ibi_ms) < 2:
        return {
            "SDNN_ms": None, "RMSSD_ms": None, "pNN50_%": None,
            "n_batimentos_detectados": len(peaks),
            "aviso": "IBIs insuficientes após remoção de outliers."
        }

    sdnn = np.std(ibi_ms, ddof=1)
    diffs = np.diff(ibi_ms)
    rmssd = np.sqrt(np.mean(diffs ** 2))
    nn50 = np.sum(np.abs(diffs) > 50)
    pnn50 = 100.0 * nn50 / len(diffs) if len(diffs) > 0 else 0.0

    return {
        "SDNN_ms": round(sdnn, 2),
        "RMSSD_ms": round(rmssd, 2),
        "pNN50_%": round(pnn50, 2),
        "n_batimentos_detectados": len(peaks),
    }

# main
def main():
    parser = argparse.ArgumentParser(description="Medição de HR/HRV via rPPG")
    parser.add_argument("--duration", type=float, default=30.0,
                         help="Duração do escaneamento em segundos")
    parser.add_argument("--camera", type=int, default=0,
                         help="Índice da câmera (0 = padrão)")
    args = parser.parse_args()

    capture = RPPGCapture(camera_index=args.camera, duration_s=args.duration)
    roi_signals, fps = capture.run()

    rppg_signal = combine_roi_and_methods(roi_signals, fps)

    filtered = bandpass_filter(rppg_signal, fps, low_hz=0.7, high_hz=4.0)

    hr_bpm, freqs, spectrum = compute_hr_fft(filtered, fps)

    hrv_metrics = compute_hrv(filtered, fps)

    print("\n================ RESULTADOS ================")
    print(f"Frequência Cardíaca estimada: {hr_bpm:.1f} bpm")
    print("Variabilidade da Frequência Cardíaca (HRV):")
    for k, v in hrv_metrics.items():
        print(f"  {k}: {v}")
    print("==============================================\n")

    return {"hrv": hrv_metrics}


if __name__ == "__main__":
    main()