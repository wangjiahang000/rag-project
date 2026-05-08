// 开发环境通过 Vite proxy 转发，生产环境同域部署
const BASE_URL = '';

export async function chat(
  question: string,
  history: [string, string][]
): Promise<string> {
  const resp = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, history }),
  });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`Chat API error (${resp.status}): ${err}`);
  }
  const data = await resp.json();
  return data.answer ?? '';
}

export async function uploadFile(file: File): Promise<{
  status: string;
  chunks: number;
  message?: string;
}> {
  const form = new FormData();
  form.append('file', file);
  const resp = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    body: form,
  });
  return resp.json();
}

export async function searchArxiv(
  query: string,
  maxResults: number = 10
): Promise<{
  status: string;
  papers: any[];
  count: number;
}> {
  const resp = await fetch(`${BASE_URL}/arxiv/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, max_results: maxResults }),
  });
  return resp.json();
}

export async function addArxivPaper(
  paperId: string,
  title: string,
  category: string = 'manual'
): Promise<{ status: string; message: string; chunks: number }> {
  const resp = await fetch(`${BASE_URL}/arxiv/add`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      paper_id: paperId,
      paper_title: title,
      category,
    }),
  });
  return resp.json();
}
