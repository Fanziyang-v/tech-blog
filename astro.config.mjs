import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkAlerts from './src/plugins/remark-alerts.mjs';

export default defineConfig({
  site: 'https://fanziyang-v.github.io',
  base: '/tech-blog/',
  trailingSlash: 'always',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath, remarkAlerts],
      rehypePlugins: [rehypeKatex],
    }),
    shikiConfig: { theme: 'github-dark' },
  },
});
