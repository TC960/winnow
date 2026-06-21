import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          base: "#06070b",
          panel: "#0c0e15",
          raised: "#11141d",
        },
        ink: {
          DEFAULT: "#e8ecf4",
          dim: "#9aa3b6",
          faint: "#5b6479",
        },
        raw: {
          DEFAULT: "#ff5fb1",
          glow: "rgba(255,95,177,0.18)",
        },
        keep: {
          DEFAULT: "#36f1a3",
          glow: "rgba(54,241,163,0.18)",
        },
        cyan: {
          accent: "#6ee7ff",
        },
        amber: {
          accent: "#ffd166",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      keyframes: {
        "pulse-glow": {
          "0%, 100%": { opacity: "0.4" },
          "50%": { opacity: "1" },
        },
        "fade-in-up": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        drift: {
          "0%, 100%": { transform: "translate3d(0%, 0%, 0)" },
          "33%":      { transform: "translate3d(6%, -4%, 0)" },
          "66%":      { transform: "translate3d(-4%, 5%, 0)" },
        },
      },
      animation: {
        "pulse-glow": "pulse-glow 2s ease-in-out infinite",
        "fade-in-up": "fade-in-up 0.35s ease-out",
        "drift-slow":   "drift 22s ease-in-out infinite",
        "drift-slower": "drift 32s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
