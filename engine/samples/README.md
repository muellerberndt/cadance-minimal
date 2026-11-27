# Drum machine one-shots

One-shot hits of the Roland TR-606, TR-808 and TR-909, one per device and voice, played by the
engine's drum voices (`Engine.setSamples`). Without this folder the engine and the player use the
engine's own models of the same voices, and the player's rack says which source plays.

- Source: the community collection tidal-drum-machines, https://github.com/geikha/tidal-drum-machines
  (folders `machines/RolandTR606`, `machines/RolandTR808`, `machines/RolandTR909`), downloaded on
  23 November 2026. That collection states no licence.
- Changes: converted to mono, 44.1 kHz, 16 bit; leading and trailing silence trimmed; peak
  normalised to 0.89; a 10 ms fade at the end.
- The 808 hits are the classic tone and decay settings at 50 and 50.
- `index.json` lists the voices per device. `engine/samples.mjs` reads the folder for the Node
  tools, and the player serves it to the page under `/kit/`.

To use other one-shots, replace the WAV files and keep the names in `index.json`.
