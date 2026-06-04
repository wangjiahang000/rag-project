import React, { useState, useEffect } from 'react';
import Chat from './components/Chat';
import Profile from './components/Profile';
import Stats from './components/Stats';
import './App.css';

function generateId() {
  return 'sess_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => {
    const saved = localStorage.getItem('myrag_session_id');
    if (saved) return saved;
    const id = generateId();
    localStorage.setItem('myrag_session_id', id);
    return id;
  });
  const [activeTab, setActiveTab] = useState('chat');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch('/health')
      .then(r => r.json())
      .then(d => setHealth(d.status))
      .catch(() => setHealth('offline'));
  }, []);

  const resetSession = () => {
    const id = generateId();
    localStorage.setItem('myrag_session_id', id);
    setSessionId(id);
  };

  const tabs = [
    { key: 'chat',      label: '💬 问答',   icon: '💬' },
    { key: 'profile',   label: '👤 画像',   icon: '👤' },
    { key: 'stats',     label: '📊 监控',   icon: '📊' },
  ];

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>MyRAG</h1>
          <span className={`status-dot ${health === 'ok' ? 'online' : 'offline'}`} />
        </div>
        <p className="sidebar-subtitle">学术论文智能问答</p>

        <nav className="sidebar-nav">
          {tabs.map(t => (
            <button
              key={t.key}
              className={`nav-btn ${activeTab === t.key ? 'active' : ''}`}
              onClick={() => setActiveTab(t.key)}
            >
              <span className="nav-icon">{t.icon}</span>
              <span className="nav-label">{t.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="session-info">
            <span className="session-label">会话 ID</span>
            <code className="session-id">{sessionId.slice(0, 16)}…</code>
          </div>
          <button className="btn-new-session" onClick={resetSession}>
            🔄 新会话
          </button>
        </div>
      </aside>

      <main className="main-content">
        {activeTab === 'chat' && <Chat sessionId={sessionId} />}
        {activeTab === 'profile' && <Profile sessionId={sessionId} />}
        {activeTab === 'stats' && <Stats />}
      </main>
    </div>
  );
}
