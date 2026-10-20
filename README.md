# AI Engineering, From Zero

A zero-to-professional learning curriculum for AI, machine learning, LLMs, AI engineering and on-device AI, published as a website with GitHub Pages.

**Live site:** https://manjulkadyan.github.io/ai-leaning/

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve        # http://127.0.0.1:8000
```

## Publishing

Every push to `main` builds the site with MkDocs Material and deploys it through GitHub Actions. To turn this on once: **Settings → Pages → Build and deployment → Source: GitHub Actions**.

## Layout

```
docs/            all site content (Markdown)
mkdocs.yml       site config and navigation
.github/workflows/deploy.yml   build + deploy to Pages
```
