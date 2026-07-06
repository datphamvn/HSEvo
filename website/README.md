# HSEvo project page

Static project page for the HSEvo paper, built with [Roman Hauksson's academic project Astro template](https://github.com/RomanHauksson/academic-project-astro-template).

## Local development

```bash
cd website
npm install
npm run dev
```

Open `http://localhost:4321` to preview. Edit content in `src/paper.mdx`.

## Deployment

The site deploys automatically to GitHub Pages on pushes to `main` via `.github/workflows/astro.yml`.

To enable hosting:

1. Open **Settings → Pages** for this repository.
2. Set **Source** to **GitHub Actions**.

After the first successful deploy, the page will be available at `https://datphamvn.github.io/HSEvo/`.
