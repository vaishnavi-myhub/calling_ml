// AudioWorkletProcessor that downsamples the mic's native sample rate to a
// target rate (16kHz, matching the backend's Whisper pipeline) and emits
// 16-bit PCM chunks to the main thread. Runs on the audio rendering thread,
// so it stays dependency-free and allocation-light.
class PCMWorklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetSampleRate = options.processorOptions?.targetSampleRate ?? 16000;
    this.ratio = sampleRate / this.targetSampleRate;
    this.buffer = [];
    this.cursor = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i += 1) this.buffer.push(channel[i]);

    const outSamples = [];
    while (this.cursor + this.ratio <= this.buffer.length) {
      outSamples.push(this.buffer[Math.floor(this.cursor)]);
      this.cursor += this.ratio;
    }

    if (outSamples.length > 0) {
      const consumed = Math.floor(this.cursor);
      this.buffer = this.buffer.slice(consumed);
      this.cursor -= consumed;

      const pcm16 = new Int16Array(outSamples.length);
      for (let i = 0; i < outSamples.length; i += 1) {
        const sample = Math.max(-1, Math.min(1, outSamples[i]));
        pcm16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      }
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-worklet", PCMWorklet);
