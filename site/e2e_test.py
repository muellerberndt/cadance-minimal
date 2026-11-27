# End-to-end test of the page in a separate headless Chromium (Playwright's own browser, not the owner's Chrome),
# against the TEST instance of the server (port 8793). Taps the audio at the destination with an analyser and reads
# band levels: kick 40-120 Hz, bass 120-400, mids 400-4k, highs 4-12k. Steps: open, play, listen; switch brain;
# listen; reload, play again.
import asyncio, json, sys, time
from playwright.async_api import async_playwright
URL = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 8793}/"   # run against a TEST instance: CADANCE_PORT=8793 python site/serve.py
TAP = """
(() => {
  const orig = AudioNode.prototype.connect;
  AudioNode.prototype.connect = function (dest, ...rest) {
    if (dest instanceof AudioDestinationNode && !this.__tapped) {
      const an = this.context.createAnalyser(); an.fftSize = 4096; orig.call(this, an); window.__an = an; window.__sr = this.context.sampleRate; this.__tapped = true;
    }
    return orig.call(this, dest, ...rest);
  };
  window.__bands = [];
  setInterval(() => {
    const an = window.__an; if (!an) return;
    const f = new Float32Array(an.frequencyBinCount); an.getFloatFrequencyData(f);
    const hz = (i) => i * window.__sr / an.fftSize;
    const band = (lo, hi) => { let s = 0, n = 0; for (let i = 0; i < f.length; i++) { const h = hz(i); if (h >= lo && h < hi && isFinite(f[i])) { s += Math.pow(10, f[i] / 10); n++; } } return n ? 10 * Math.log10(s / n + 1e-20) : -200; };
    window.__bands.push({ t: performance.now(), kick: band(40, 120), bass: band(120, 400), mid: band(400, 4000), high: band(4000, 12000), status: (document.getElementById('status') || {}).textContent || '' });
  }, 25);
})();
"""
def summary(rows, label):
    if not rows:
        print(f"{label}: NO AUDIO DATA"); return
    import statistics as st
    k = [r["kick"] for r in rows]; b = [r["bass"] for r in rows]; m = [r["mid"] for r in rows]; h = [r["high"] for r in rows]
    loud = sum(1 for r in rows if max(r["kick"], r["bass"], r["mid"]) > -90) / len(rows)
    print(f"{label}: {len(rows)} reads, sound present in {loud:.0%}; median dB kick {st.median(k):.0f} (spread p10-p90 {sorted(k)[len(k)//10]:.0f}..{sorted(k)[9*len(k)//10]:.0f}), bass {st.median(b):.0f}, mids {st.median(m):.0f}, highs {st.median(h):.0f}; status '{rows[-1]['status']}'")


async def verify_checkpoint(browser, name):
    page = await browser.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    await page.add_init_script(TAP)
    await page.goto(URL)
    await page.wait_for_function("document.querySelector('#status').textContent.startsWith('Ready to play')", timeout=95000)
    started = time.monotonic()
    await page.select_option("#checkpoint", name)
    await page.wait_for_function(
        "name => document.querySelector('#status').textContent === `Ready to play · ${name}` && !document.querySelector('#play').disabled",
        arg=name, timeout=95000,
    )
    print(f"{name} ready in {time.monotonic() - started:.1f}s", flush=True)
    await page.click("#play")
    await page.wait_for_function(
        "name => window.__bands.some(b => b.status.includes(name) && b.status.startsWith('bar ') && Math.max(b.kick, b.bass, b.mid) > -85)",
        arg=name, timeout=30000,
    )
    print("Confirmed audible output from the selected checkpoint", flush=True)
    await page.click("#play")
    await page.evaluate("""() => {
      const menu = document.querySelector('#checkpoint');
      menu.add(new Option('Missing checkpoint', 'missing_checkpoint'));
      menu.value = 'missing_checkpoint';
      menu.dispatchEvent(new Event('change'));
    }""")
    await page.wait_for_function(
        "name => document.querySelector('#status').textContent.includes('Could not load missing_checkpoint') && "
        "document.querySelector('#status').textContent.includes(`Ready to play · ${name}`) && !document.querySelector('#play').disabled",
        arg=name, timeout=12000,
    )
    print("Confirmed a failed selection reports the error and restores Play", flush=True)
    assert not errors, errors


async def main():
    async with async_playwright() as p:
        if len(sys.argv) > 2:
            br = await p.chromium.launch(headless=True)
            await verify_checkpoint(br, sys.argv[2])
            await br.close()
            return
        br = await p.chromium.launch(headless=True, args=["--autoplay-policy=no-user-gesture-required"])
        page = await br.new_page()
        logs = []
        page.on("console", lambda m: logs.append(f"console {m.type}: {m.text}"))
        page.on("pageerror", lambda e: logs.append(f"PAGE ERROR: {e}"))
        await page.add_init_script(TAP)
        await page.goto(URL); await page.wait_for_timeout(3000)
        opts = await page.eval_on_selector_all("#checkpoint option", "els => els.map(e => [e.value, e.textContent])")
        print("menu:", [o[0] for o in opts][:6], "| selected:", await page.eval_on_selector("#checkpoint", "e => e.value"))
        t0 = time.time(); await page.click("#play")
        await page.wait_for_timeout(16000)
        rows = await page.evaluate("window.__bands"); summary(rows[len(rows)//4:], f"1. play ({time.time() - t0:.0f} s)")
        cur = await page.eval_on_selector("#checkpoint", "e => e.value")
        other = next(o[0] for o in opts if o[0] not in ("demo", cur))
        n0 = len(rows); t1 = time.time()
        await page.select_option("#checkpoint", other)
        first_new = None
        for _ in range(240):
            await page.wait_for_timeout(250)
            stt = await page.eval_on_selector("#status", "e => e.textContent")
            if other in stt and first_new is None: first_new = time.time() - t1; break
        await page.wait_for_timeout(10000)
        rows = await page.evaluate("window.__bands")
        print(f"2. switched to {other}: its bars play {first_new if first_new is None else round(first_new, 1)} s after the switch")
        summary(rows[n0 + 40:], "   after the switch")
        sel = await page.eval_on_selector("#checkpoint", "e => e.value")
        await page.reload(); await page.wait_for_timeout(3000)
        await page.click("#play"); await page.wait_for_timeout(12000)
        rows = await page.evaluate("window.__bands"); summary(rows[len(rows)//4:], "3. reload + play")
        for l in logs[-12:]: print("  ", l[:220])
        await br.close()
asyncio.run(main())
