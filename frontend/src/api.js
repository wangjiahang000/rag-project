const BASE = '';

export async function healthCheck() {
  const r = await fetch(`${BASE}/health`);
  return r.json();
}

export async function sendChat(question, sessionId) {
  const r = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  return r.json();
}

export async function* streamChat(question, sessionId) {
  const r = await fetch(`${BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  if (!r.ok) {
    yield { type: 'error', data: `HTTP ${r.status}` };
    return;
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith('data: ')) continue;
      try {
        const data = JSON.parse(trimmed.slice(6));
        yield data;
      } catch { /* skip malformed */ }
    }
  }

  // Process remaining buffer
  if (buffer.trim().startsWith('data: ')) {
    try {
      const data = JSON.parse(buffer.trim().slice(6));
      yield data;
    } catch { /* skip */ }
  }
}

export async function getProfile(sessionId) {
  const r = await fetch(`${BASE}/profile/${sessionId}`);
  return r.json();
}

export async function deleteProfile(sessionId) {
  const r = await fetch(`${BASE}/profile/${sessionId}`, { method: 'DELETE' });
  return r.json();
}

export async function getInterests(sessionId) {
  const r = await fetch(`${BASE}/profile/${sessionId}/interests`);
  return r.json();
}

export async function getStats() {
  const r = await fetch(`${BASE}/stats`);
  return r.json();
}

export async function getMetrics() {
  const r = await fetch(`${BASE}/metrics`);
  return r.json();
}

export async function resetMetrics() {
  const r = await fetch(`${BASE}/metrics/reset`, { method: 'POST' });
  return r.json();
}
