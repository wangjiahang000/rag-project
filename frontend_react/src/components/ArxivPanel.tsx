import { useState } from 'react';
import './ArxivPanel.css';
import { searchArxiv, addArxivPaper } from '../api/client';

interface ArxivPanelProps {
  onPaperAdded: (title: string) => void;
}

export default function ArxivPanel({ onPaperAdded }: ArxivPanelProps) {
  const [query, setQuery] = useState('');
  const [papers, setPapers] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [adding, setAdding] = useState<string | null>(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setStatus('');
    try {
      const res = await searchArxiv(query);
      setPapers(res.papers ?? []);
      setStatus(`找到 ${res.count ?? 0} 篇论文`);
    } catch (e: any) {
      setStatus(`搜索失败: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleAdd = async (paper: any) => {
    setAdding(paper.id);
    try {
      const res = await addArxivPaper(paper.id, paper.title);
      if (res.status === 'success') {
        onPaperAdded(paper.title);
      } else {
        setStatus(`添加失败: ${res.message}`);
      }
    } catch (e: any) {
      setStatus(`添加失败: ${e.message}`);
    } finally {
      setAdding(null);
    }
  };

  return (
    <div className="arxiv-panel">
      <div className="arxiv-header">arXiv 文献检索</div>

      <div className="arxiv-search-row">
        <input
          className="arxiv-input"
          placeholder="输入关键词搜索论文..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
        />
        <button className="arxiv-search-btn" onClick={handleSearch} disabled={loading || !query.trim()}>
          {loading ? '...' : '搜索'}
        </button>
      </div>

      {status && <div className="arxiv-status">{status}</div>}

      <div className="arxiv-results">
        {papers.map((p) => (
          <div key={p.id} className="arxiv-card">
            <div className="arxiv-card-title">{p.title}</div>
            <div className="arxiv-card-meta">
              <span>{p.authors_display}</span>
              <span>{p.published}</span>
            </div>
            <div className="arxiv-card-summary">{p.summary}</div>
            <button
              className="arxiv-add-btn"
              onClick={() => handleAdd(p)}
              disabled={adding === p.id}
            >
              {adding === p.id ? '导入中...' : '添加到知识库'}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
