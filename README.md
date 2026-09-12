# 中文技术博客

简洁的 Astro 静态站点，支持手机排版和跟随系统的深色模式。

## 本地运行

使用 Node.js 22 或更高版本，在本目录执行：

```bash
npm ci
npm run dev
```

发布前运行 `npm run build`，使用 `npm run preview` 检查生产构建。

## 内容维护

文章存放在 `src/content/posts/`。复制 `welcome.md`，使用英文文件名（如 `understanding-attention.md`），修改 Frontmatter 后写正文。

```yaml
title: "文章标题"
description: "简短摘要"
date: 2026-09-12
tags: ["深度学习"]
draft: false
```

`draft: true` 会同时从列表和静态文章路由中排除文章，但 Markdown 源文件仍在仓库中；公开仓库中的草稿不是私密内容。日期越新的文章排在越前面。

图片可放在 `public/images/`，Markdown 中使用 `![说明](/tech-blog/images/example.png)`。支持标题、表格、引用和代码高亮；数学公式和 Mermaid 暂未启用。

`welcome.md` 是排版示例，可替换或删除。

## GitHub Pages

预期仓库：`Fanziyang-v/tech-blog`。

`astro.config.mjs` 已设置域名及 `/tech-blog/` 子路径。改仓库名称时需同步修改 base、跨站导航和文章图片路径。

1. 将项目根目录文件推送到目标仓库的 `main` 分支。
2. 在仓库 Settings → Pages → Build and deployment 中选择 GitHub Actions。
3. 在 Actions 页面检查 Deploy to GitHub Pages 工作流；必要时手动运行。

`.github/workflows/deploy.yml` 会安装锁定依赖、构建并发布 `dist/`。

导航中的个人主页和博客链接是预期线上地址，在目标站点发布后可互访。
