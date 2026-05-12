const ALLOWED_ORIGINS = [
  'https://lens-lab-ai.github.io',
];

function isAllowedOrigin(origin) {
  if (!origin) return false;
  if (origin.startsWith('http://localhost')) return true;
  return ALLOWED_ORIGINS.some(allowed => origin.startsWith(allowed));
}

function corsHeaders(origin) {
  return {
    'Access-Control-Allow-Origin': origin || '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
  };
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') || '';

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders(origin) });
    }

    if (request.method !== 'POST') {
      return new Response('Method not allowed', { status: 405 });
    }

    if (!isAllowedOrigin(origin)) {
      return new Response('Forbidden', { status: 403 });
    }

    try {
      const body = await request.json();
      body.model = body.model || 'allenai/molmo-2-8b:free';

      const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${env.OPENROUTER_API_KEY}`,
          'HTTP-Referer': 'https://lens-lab-ai.github.io/Interview/',
          'X-Title': 'Interview Preparation',
        },
        body: JSON.stringify(body),
      });

      const data = await response.text();

      return new Response(data, {
        status: response.status,
        headers: {
          'Content-Type': 'application/json',
          ...corsHeaders(origin),
        },
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: err.message }), {
        status: 500,
        headers: {
          'Content-Type': 'application/json',
          ...corsHeaders(origin),
        },
      });
    }
  },
};
