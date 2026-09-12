import { defineConfig } from 'astro/config';
export default defineConfig({ site: 'https://fanziyang-v.github.io', base: '/tech-blog/', trailingSlash: 'always', markdown: { shikiConfig: { theme: 'github-dark' } } });
