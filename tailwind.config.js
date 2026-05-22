/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#0B0D0F',
        'bg-surface': '#141619',
        'bg-surface-alt': '#1A1C20',
        'border-default': '#262830',
        'accent-copper': '#C8926A',
        'accent-copper-bright': '#E8B88A',
        'accent-solder': '#D4A843',
        'accent-signal': '#5C9A6B',
        'text-primary': '#E8E4DF',
        'text-secondary': '#8A8680',
        'text-muted': '#5C5955',
      },
      fontFamily: {
        heading: ['"Barlow Condensed"', 'sans-serif'],
        body: ['Inter', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
};
