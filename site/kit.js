// The real Roland TR-606 / TR-808 / TR-909 one-shots (engine/samples, served under /kit/): fetched, decoded at the
// AudioContext's rate and handed to the worklet's engine (Engine.setSamples). Without them the engine plays its
// synthesized drum models.
export async function loadKit(ctx, node) {
  try {
    const index = await (await fetch('/kit/index.json')).json();
    const kit = {};
    await Promise.all(Object.entries(index).flatMap(([dev, voices]) => voices.map(async (voice) => {
      const buf = await (await fetch(`/kit/${dev}/${voice}.wav`)).arrayBuffer();
      const audio = await ctx.decodeAudioData(buf);
      (kit[dev] ||= {})[voice] = audio.getChannelData(0).slice();
    })));
    node.port.postMessage({ type: 'samples', kit, sampleRate: ctx.sampleRate });
    return kit;
  } catch (err) {
    console.warn('drum samples not loaded, using the synthesized models:', err);
    return null;
  }
}
