// FFR -> colour (SPEC §6.6). Pure functions, no imports, so they can be tested with plain node.
export const STOPS = [
  [0.7, "#d73027"],
  [0.8, "#fee08b"],
  [0.9, "#1a9850"],
  [1.0, "#4575b4"],
];
export const SMALL_VESSEL_COLOR = "#9e9e9e";
export const SMALL_DIAMETER_MM = 1.5;

export function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function rgbToHex([r, g, b]) {
  return "#" + [r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
}

// Linear in RGB between stops, clamped outside [0.70, 1.00].
export function ffrColor(ffr) {
  if (ffr <= STOPS[0][0]) return STOPS[0][1];
  if (ffr >= STOPS[STOPS.length - 1][0]) return STOPS[STOPS.length - 1][1];
  for (let i = 1; i < STOPS.length; i++) {
    const [v1, c1] = STOPS[i];
    if (ffr <= v1) {
      const [v0, c0] = STOPS[i - 1];
      const t = (ffr - v0) / (v1 - v0);
      const a = hexToRgb(c0);
      const b = hexToRgb(c1);
      return rgbToHex(a.map((x, k) => x + (b[k] - x) * t));
    }
  }
  return STOPS[STOPS.length - 1][1];
}

// Colour of one centreline sample: grey for vessels thinner than 1.5 mm.
export function sampleColor(ffr, radiusMm) {
  return 2 * radiusMm < SMALL_DIAMETER_MM ? SMALL_VESSEL_COLOR : ffrColor(ffr);
}
