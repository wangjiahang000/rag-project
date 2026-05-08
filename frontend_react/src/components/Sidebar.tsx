import './Sidebar.css';
import type { Conversation } from '../types';

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onClearAll: () => void;
}

export default function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onClearAll,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>RAG 助手</h2>
      </div>

      <button className="new-chat-btn" onClick={onNew}>
        + 新对话
      </button>

      <div className="conv-list">
        {conversations.length === 0 && (
          <p className="empty-hint">暂无对话记录</p>
        )}
        {[...conversations]
          .sort((a, b) => b.createdAt - a.createdAt)
          .map((conv) => (
            <div
              key={conv.id}
              className={`conv-item ${conv.id === activeId ? 'active' : ''}`}
              onClick={() => onSelect(conv.id)}
            >
              <div className="conv-title">{conv.title || '新对话'}</div>
              <button
                className="conv-del"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(conv.id);
                }}
              >
                ✕
              </button>
            </div>
          ))}
      </div>

      {conversations.length > 0 && (
        <button className="clear-btn" onClick={onClearAll}>
          清空所有对话
        </button>
      )}
    </aside>
  );
}
