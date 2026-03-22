export interface StreamEvent {
  type: 'status' | 'token' | 'clear_tokens';
  message?: string;
  content?: string;
}

const API_URL = 'http://127.0.0.1:8000/api';

export async function* streamMessage(
  query: string,
  sessionId: string
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Connection Error: ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (line.trim()) {
        try {
          yield JSON.parse(line.trim()) as StreamEvent;
        } catch {
          console.warn('Failed to parse SSE line:', line);
        }
      }
    }
  }
}
