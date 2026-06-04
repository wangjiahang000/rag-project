import React, { useState, useEffect } from 'react';
import { getProfile, deleteProfile, getInterests } from '../api';

export default function Profile({ sessionId }) {
  const [profile, setProfile] = useState(null);
  const [interests, setInterests] = useState({});
  const [deleted, setDeleted] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchProfile = async () => {
    setLoading(true);
    try {
      const [p, i] = await Promise.all([
        getProfile(sessionId),
        getInterests(sessionId),
      ]);
      setProfile(p);
      setInterests(i.interests || {});
    } catch {
      // ignore
    }
    setLoading(false);
  };

  useEffect(() => {
    if (!deleted) fetchProfile();
  }, [sessionId, deleted]);

  const handleDelete = async () => {
    if (!confirm('确定清除当前会话的所有数据？')) return;
    await deleteProfile(sessionId);
    setDeleted(true);
    setProfile(null);
    setInterests({});
  };

  if (deleted) {
    return (
      <div className="profile-container">
        <div className="profile-card" style={{ textAlign: 'center', padding: '40px' }}>
          <p style={{ color: 'var(--text-secondary)', marginBottom: 16 }}>会话数据已清除</p>
          <button className="btn-danger" onClick={() => setDeleted(false)}>
            刷新
          </button>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="profile-container">
        <p style={{ color: 'var(--text-muted)', textAlign: 'center', marginTop: 40 }}>加载中…</p>
      </div>
    );
  }

  return (
    <div className="profile-container">
      <div className="profile-header">
        <h2>👤 用户画像</h2>
        <p className="session-id-full">会话: {sessionId}</p>
      </div>

      <div className="profile-card">
        <h3>📊 基本统计</h3>
        <div className="stat-row">
          <span className="label">总提问数</span>
          <span className="value">{profile?.total_queries ?? 0}</span>
        </div>
        <div className="stat-row">
          <span className="label">对话轮数</span>
          <span className="value">{profile?.turns ?? 0}</span>
        </div>
      </div>

      <div className="profile-card">
        <h3>🏷️ 兴趣标签</h3>
        {Object.keys(interests).length > 0 ? (
          <div className="tag-list">
            {Object.entries(interests).map(([tag, weight]) => (
              <span key={tag} className="tag">
                {tag}<span className="weight">{weight.toFixed(1)}</span>
              </span>
            ))}
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>暂无兴趣标签</p>
        )}
      </div>

      <div className="profile-card">
        <h3>🎯 意图分布</h3>
        {profile?.top_intents && profile.top_intents.length > 0 ? (
          <div className="tag-list">
            {profile.top_intents.map(([intent, count]) => (
              <span key={intent} className="tag">
                {intent}<span className="weight">{count}次</span>
              </span>
            ))}
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>暂无意图数据</p>
        )}
      </div>

      <button className="btn-danger" onClick={handleDelete}>
        🗑️ 清除会话数据
      </button>
    </div>
  );
}
