import React, { useState, useEffect, useCallback } from 'react';
import { getStats, getMetrics } from '../api';

export default function Stats() {
  const [stats, setStats] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchAll = useCallback(async () => {
    try {
      const [s, m] = await Promise.all([getStats(), getMetrics()]);
      setStats(s);
      setMetrics(m);
    } catch {
      // ignore
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    fetchAll();
    const timer = setInterval(fetchAll, 5000);
    return () => clearInterval(timer);
  }, [fetchAll]);

  if (loading) {
    return (
      <div className="stats-container">
        <p style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: 40 }}>加载中…</p>
      </div>
    );
  }

  const uptime = metrics?.uptime_seconds
    ? formatUptime(metrics.uptime_seconds)
    : 'N/A';

  const counters = metrics?.counters || {};
  const latencies = metrics?.latencies || {};

  // Extract key counters
  const totalRequests = sumValues(Object.entries(counters)
    .filter(([k]) => k.startsWith('http_requests_total'))
    .map(([, v]) => v));
  const cacheHits = counters['cache_hit:endpoint=chat'] || 0;
  const cacheMisses = counters['cache_miss:endpoint=chat'] || 0;
  const llmCalls = counters['llm_calls:endpoint=chat'] || 0;
  const llmCallsStream = counters['llm_calls:endpoint=stream'] || 0;
  const totalLlm = llmCalls + llmCallsStream;
  const cacheTotal = cacheHits + cacheMisses;
  const hitRate = cacheTotal > 0 ? (cacheHits / cacheTotal * 100).toFixed(1) : 'N/A';

  return (
    <div className="stats-container">
      <h2>📊 系统监控</h2>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-value">{stats?.active_sessions ?? 0}</div>
          <div className="metric-label">活跃会话</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{totalRequests}</div>
          <div className="metric-label">总请求数</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{totalLlm}</div>
          <div className="metric-label">LLM 调用次数</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{hitRate}{hitRate !== 'N/A' ? '%' : ''}</div>
          <div className="metric-label">缓存命中率</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{uptime}</div>
          <div className="metric-label">运行时间</div>
        </div>
      </div>

      {Object.keys(latencies).length > 0 && (
        <>
          <h3 style={{ fontSize: 16, marginBottom: 12, color: 'var(--text-secondary)' }}>
            ⏱ 延迟分布
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table className="latency-table">
              <thead>
                <tr>
                  <th>端点</th>
                  <th>请求数</th>
                  <th>平均 (ms)</th>
                  <th>P50 (ms)</th>
                  <th>最大 (ms)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(latencies).map(([key, val]) => (
                  <tr key={key}>
                    <td>{key}</td>
                    <td>{val.count}</td>
                    <td>{val.avg_ms.toFixed(0)}</td>
                    <td>{val.p50_ms.toFixed(0)}</td>
                    <td>{val.max_ms.toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {stats?.session_ids?.length > 0 && (
        <details style={{ marginTop: 24 }}>
          <summary style={{ cursor: 'pointer', color: 'var(--text-secondary)', fontSize: 14 }}>
            活跃会话列表 ({stats.session_ids.length})
          </summary>
          <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {stats.session_ids.map(sid => (
              <code key={sid} style={{
                fontSize: 11, color: 'var(--text-muted)',
                background: 'var(--bg-tertiary)', padding: '2px 8px', borderRadius: 4,
              }}>{sid.slice(0, 20)}…</code>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}

function formatUptime(seconds) {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function sumValues(values) {
  return values.reduce((a, b) => a + b, 0);
}
