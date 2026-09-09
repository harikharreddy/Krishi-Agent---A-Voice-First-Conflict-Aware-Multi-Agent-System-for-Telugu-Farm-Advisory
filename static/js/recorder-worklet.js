// Runs on the audio render thread; forwards raw Float32 PCM frames from the
// mic back to the main thread, where they're buffered and WAV-encoded.
// Recording at a 16000 Hz AudioContext (see app.js) means no server-side
// resampling is needed -- transcribe_audio() requires exactly 16000 Hz.
class PCMRecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length > 0) {
      this.port.postMessage(input[0].slice());
    }
    return true;
  }
}

registerProcessor("pcm-recorder", PCMRecorderProcessor);
