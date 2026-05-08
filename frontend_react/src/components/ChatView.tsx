import { useState, useRef, useEffect } from 'react';
import './ChatView.css';
import { chat, uploadFile as apiUploadFile } from '../api/client';
import type { Message } from '../types';
import ArxivPanel from './ArxivPanel';

interface ChatViewProps {
  messages: Message[];
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

export default function ChatView({ messages, setMessages }: ChatViewProps) {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showArxiv, setShowArxiv] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput('');

    const userMsg: Message = { role: 'user', content: text };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setLoading(true);

    try {
      const history: [string, string][] = [];
      for (let i = 0; i < updated.length - 1; i += 2) {
        const u = updated[i];
        const a = updated[i + 1];
        if (u?.role === 'user' && a?.role === 'assistant') {
          history.push([u.content, a.content]);
        }
      }
      const answer = await chat(text, history);
      setMessages((prev) => [...prev, { role: 'assistant', content: answer }]);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `❌ 请求失败: ${e.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileUpload = async () => {
    if (!file) return;
    setUploadStatus('上传中...');
    try {
      const res = await apiUploadFile(file);
      if (res.status === 'success') {
        setUploadStatus(`✅ 上传成功，${res.chunks} 个文本块`);
      } else {
        setUploadStatus(`❌ 上传失败: ${res.message}`);
      }
    } catch {
      setUploadStatus('❌ 上传失败');
    }
    setFile(null);
    if (fileRef.current) fileRef.current.value = '';
  };

  return (
    <div className="chat-view">
      {/* Header */}
      <div className="chat-header">
        <span className="chat-title">RAG 智能问答</span>
        <button
          className={`header-btn ${showArxiv ? 'active' : ''}`}
          onClick={() => setShowArxiv(!showArxiv)}
        >
          📄 arXiv 检索
        </button>
      </div>

      {/* Arxiv Panel */}
      {showArxiv && (
        <ArxivPanel
          onPaperAdded={(title) => {
            setMessages((prev) => [
              ...prev,
              {
                role: 'assistant',
                content: `✅ 已导入论文: ${title}`,
              },
            ]);
          }}
        />
      )}

      {/* Messages */}
      <div className="messages-area">
        {messages.length === 0 && (
          <div className="welcome">
            <h2>RAG 智能问答助手</h2>
            <p>上传 PDF/TXT 文档或搜索 arXiv 论文，然后基于文献进行问答</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message-row ${msg.role}`}>
            <div className="avatar">{msg.role === 'user' ? '👤' : '🤖'}</div>
            <div className="bubble">
              <div className="bubble-text">{msg.content}</div>
            </div>
          </div>
        ))}
        {loading && (
          <div className="message-row assistant">
            <div className="avatar">🤖</div>
            <div className="bubble thinking">
              <span className="dot-pulse" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* File upload status */}
      {uploadStatus && (
        <div className="upload-status-bar">
          <span>{uploadStatus}</span>
          <button onClick={() => setUploadStatus('')}>✕</button>
        </div>
      )}

      {/* Input area */}
      <div className="input-area">
        <div className="input-toolbar">
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.txt"
            hidden
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <button
            className="toolbar-btn"
            onClick={() => fileRef.current?.click()}
            title="上传 PDF/TXT"
          >
            📎
          </button>
          {file && (
            <span className="file-name" onClick={handleFileUpload}>
              {file.name} — 点击上传
            </span>
          )}
        </div>
        <div className="input-row">
          <textarea
            className="chat-input"
            rows={1}
            placeholder="输入问题，按 Enter 发送..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            className="send-btn"
            onClick={handleSend}
            disabled={loading || !input.trim()}
          >
            ➤
          </button>
        </div>
      </div>
    </div>
  );
}
