import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from rppg_python.rppg_py import RppgAnalyzer, ROI_LANDMARKS

MODEL_PATH = 'assets/face_landmarker.task'

def create_face_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def run_rppg_camera_app():
    landmarker = create_face_landmarker()

    analyzer = RppgAnalyzer()

    current_progress = 0
    current_bpm = 0
    current_confidence = 0
    final_results = None

    def on_progress(progress, partial_metrics):
        nonlocal current_progress, current_bpm, current_confidence
        current_progress = progress
        if partial_metrics:
            current_bpm = partial_metrics["bpm"]
            current_confidence = partial_metrics["confidence"]

    def on_result(result):
        nonlocal final_results
        final_results = result
        print("\n--- Análise Concluída com Sucesso ---")
        print(f"BPM: {result['bpm']} (Confiança: {result['confidence']}%)")
        print(f"SpO2: {result['spo2']['value']}%")
        print(f"HRV (RMSSD): {result['hrv']['rmssd']} ms")
        print(f"Score: {result['wellness']['overall']} -> {result['wellness']['interpretation']}")

    def on_error(err_msg):
        print(f"[Erro no Pipeline]: {err_msg}")

    analyzer.on_progress = on_progress
    analyzer.on_result = on_result
    analyzer.on_error = on_error

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: não foi possível abrir a câmera (índice 0). "
              "Verifique se ela está conectada e não está em uso por outro app.")
        landmarker.close()
        return

    print("Controles na janela de vídeo:")
    print(" -> Pressione 's' para INICIAR o escaneamento.")
    print(" -> Pressione 'q' para FECHAR o aplicativo.")

    start_time = time.perf_counter()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Erro ao acessar a câmera.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms = int((time.perf_counter() - start_time) * 1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            # result.face_landmarks[0] é uma lista de NormalizedLandmark
            landmarks_list = result.face_landmarks[0]

            if analyzer.is_running:
                analyzer.process_frame(frame_rgb, landmarks_list)

            height, width, _ = frame.shape
            all_roi_indices = ROI_LANDMARKS["forehead"] + ROI_LANDMARKS["leftCheek"] + ROI_LANDMARKS["rightCheek"]
            for idx in all_roi_indices:
                lm = landmarks_list[idx]
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        if analyzer.is_running:
            cv2.putText(frame, f"Escaneando: {current_progress}%", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, f"BPM Atual: {current_bpm} ({current_confidence}%)", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        elif final_results:
            cv2.putText(frame, "Resultado Pronto!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"BPM: {final_results['bpm']} | SpO2: {final_results['spo2']['value']}%", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"HRV: {final_results['hrv']['rmssd']}ms | Score: {final_results['wellness']['overall']}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Pressione 's' para comecar", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('rPPG Monitor - OpenCV & MediaPipe', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not analyzer.is_running:
            final_results = None
            analyzer.start()
            print("Escaneamento iniciado... Mantenha o rosto imóvel.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    run_rppg_camera_app()