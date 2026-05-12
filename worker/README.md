# OpenRouter Proxy Worker

A Cloudflare Worker that proxies requests to the OpenRouter API, keeping your API key server-side so it's never exposed in the browser.

## Setup

1. Sign up at [dash.cloudflare.com](https://dash.cloudflare.com) (free tier works)
2. Go to **Workers & Pages** → **Create application** → **Create Worker**
3. Name it (e.g. `interview-qa-proxy`) and click **Deploy**
4. Click **Edit Code**, replace the default code with the contents of `worker.js`, and click **Deploy**
5. Go to the worker's **Settings** → **Variables and Secrets** → **Add**
   - Type: **Secret**
   - Name: `OPENROUTER_API_KEY`
   - Value: your OpenRouter API key (get one at [openrouter.ai/keys](https://openrouter.ai/keys))
   - Click **Deploy**

Your worker is now live at `https://<worker-name>.<your-account>.workers.dev`.

## Connecting to the site

Add your worker URL as a GitHub repository **variable** (not secret):

1. Go to repo **Settings** → **Secrets and variables** → **Actions** → **Variables** tab
2. Add variable: `PROXY_URL` = `https://<worker-name>.<your-account>.workers.dev`

The deploy workflow injects this URL into the site. On the next push to `main`, the LLM verification feature will route through your worker.

## How it works

- The frontend sends a POST request to the worker with the model and prompt
- The worker adds the `Authorization` header (using the secret API key) and forwards the request to OpenRouter
- The response is returned to the frontend with CORS headers

The worker only accepts requests from allowed origins (`https://lens-lab-ai.github.io` and `localhost` for local dev). Edit the `ALLOWED_ORIGINS` array in `worker.js` to change this.

## CLI deployment (alternative)

If you prefer the command line over the dashboard:

```bash
npm install -g wrangler
wrangler login
cd worker
wrangler deploy
wrangler secret put OPENROUTER_API_KEY
```
