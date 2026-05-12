# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Interview preparation web app for ML topics (Transformers, Diffusion, Reinforcement Learning). Content is authored as Markdown, converted to JSON, and rendered by a single-page `index.html` deployed to GitHub Pages.

Live site: https://lens-lab-ai.github.io/Interview/

## Key Commands

```bash
# Convert all markdown sources to QA.json (run after editing any .md files in Topics/)
bash update.sh

# Convert a single topic
python scripts/convert_to_json.py --input-dir Topics/Transformers -v

# Local dev server (serves on port 8080)
python serve.py
```

There is no build step, test suite, or linter.

## Content Pipeline

Markdown files in `Topics/<Subject>/` → `scripts/convert_to_json.py` → `Topics/<Subject>/QA.json` → fetched at runtime by `index.html`.

The converter supports four Q&A markdown formats (detected by regex in priority order):
1. `## Question Title` + `**T:** tag` + `**A:** answer` (Transformers theory/deployment style)
2. `### Q1. Question` + `**T:** tag` + `**A:** answer` (Transformers model/learning style)
3. `**Q. question**` + `**A:** answer` (Diffusion style)
4. `* **question**` + indented `* bullet` answers (RL bullet style)

When adding new markdown content, match the format already used in that topic's folder. The type label is extracted from the first `# Heading` line (author names after `—` or `-` are stripped).

## Architecture

`index.html` is the entire frontend — no framework, no bundler. jQuery for DOM, MathJax for LaTeX, hash-based routing (`#home`, `#topic/<Subject>`).

Two modes in the topic view:
- **All Mode**: expandable accordion of all Q&A cards
- **Practice Mode**: randomized flashcard quiz with an optional LLM answer-verification feature (OpenRouter API, `allenai/molmo-2-8b:free`)

LLM verification uses two API key strategies (tried in order):
1. **Cloudflare Worker proxy** (`worker/`) — holds the OpenRouter key server-side. The proxy URL placeholder `%%PROXY_URL%%` is injected at deploy time from the `PROXY_URL` GitHub Actions variable.
2. **User-provided key** — entered via the settings modal (gear icon), stored in `localStorage`.

The deploy workflow (`.github/workflows/deploy.yml`) also auto-converts markdown to JSON and commits updated `QA.json` files before deploying.

## Adding a New Topic

1. Create `Topics/<NewTopic>/` with markdown files following one of the supported formats
2. Add the conversion command to `update.sh`
3. Add a topic card in `index.html` (inside `.cards-grid`) and an entry in the `subjects` JS object
4. Run `bash update.sh` to generate `QA.json`
