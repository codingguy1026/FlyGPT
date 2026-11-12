export const THEMES = [
  { id: "bio", name: "Papiti", accent: "#26765a", accent2: "#c5ef8b", glow: "38,118,90" },
  { id: "violet", name: "Iris", accent: "#6e63ff", accent2: "#9f7cff", glow: "110,99,255" },
  { id: "aqua", name: "Aqua", accent: "#2bbfd3", accent2: "#62e0d1", glow: "43,191,211" },
  { id: "sun", name: "Solar", accent: "#f0a83a", accent2: "#ffd06b", glow: "240,168,58" },
  { id: "rose", name: "Rose", accent: "#e96a9f", accent2: "#ff9fbd", glow: "233,106,159" },
] as const;

export type FlightThemeId = (typeof THEMES)[number]["id"];
