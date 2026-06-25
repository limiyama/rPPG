document.addEventListener('DOMContentLoaded', async () => {
    const videoEl = document.getElementById('webcam');
    const canvasEl = document.getElementById('output_canvas');
    const statusEl = document.getElementById('status');
    const bpmEl = document.getElementById('bpm');
    const spo2El = document.getElementById('spo2');
    const wellnessEl = document.getElementById('wellness');

    // 1. Instancia o analisador do rppg-core.js
    const analyzer = new RppgAnalyzer();

    // 2. Configura os Callbacks de atualização da UI
    analyzer.onProgress = (percent, partialMetrics) => {
        statusEl.innerText = `Análise em andamento: ${percent}%`;
        if (partialMetrics) {
            bpmEl.innerText = `BPM (Tempo Real): ${partialMetrics.bpm} (Confiança: ${partialMetrics.confidence}%)`;
        }
    };

    analyzer.onFaceDetected = (detected) => {
        if (!detected) {
            statusEl.innerText = "Rosto não detectado! Posicione-se em frente à câmera.";
            statusEl.style.color = "red";
        } else {
            statusEl.style.color = "lightgreen";
        }
    };

    analyzer.onResult = (result) => {
        statusEl.innerText = "Análise concluída!";
        bpmEl.innerText = `BPM Final: ${result.bpm} (Confiança: ${result.confidence}%)`;
        spo2El.innerText = `SpO₂: ${result.spo2.value}% (${result.spo2.reliable ? 'Confiável' : 'Sinal Fraco'})`;
        wellnessEl.innerText = `Score: ${result.wellness.overall}/100 - ${result.wellness.interpretation}`;
        console.log("Resultado Completo:", result);
    };

    analyzer.onError = (err) => {
        statusEl.innerText = `Erro: ${err.message}`;
        statusEl.style.color = "red";
    };

    // 3. Inicializa a câmera e o MediaPipe
    statusEl.innerText = "Carregando câmera e inteligência artificial...";
    const initSuccess = await analyzer.init(videoEl, canvasEl);

    if (initSuccess) {
        statusEl.innerText = "Pronto! Iniciando contagem de 30 segundos...";
        // Começa a gravar os frames
        analyzer.start();
    }
});