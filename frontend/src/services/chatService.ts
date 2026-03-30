export interface SourceDoc {
  pmid: string;
  title: string;
  journal: string;
  year: number | null;
  first_author: string;
  publication_type: string;
  rerank_score: number | null;
  section: string;
  snippet: string;
}

export interface EvidenceTableRow {
  pmid: string;
  study: string;
  population: string;
  intervention: string;
  comparator: string;
  outcome: string;
  study_type: string;
}

export interface StreamEvent {
  type: 'status' | 'token' | 'clear_tokens' | 'sources' | 'evidence_table' | 'safety';
  message?: string;
  content?: string;
  docs?: SourceDoc[];
  rows?: EvidenceTableRow[];
  level?: string;
  disclaimer?: string;
}

export interface ChatFilters {
  year_min?: number;
  year_max?: number;
  publication_types?: string[];
  human_only?: boolean;
}

const API_URL = 'http://127.0.0.1:8000/api';

export async function* streamMessage(
  query: string,
  sessionId: string,
  filters?: ChatFilters
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId, ...filters }),
  });

  if (!response.ok || !response.body) {
    throw new Error(`Connection Error: ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
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
  } finally {
    reader.cancel();
  }
}
