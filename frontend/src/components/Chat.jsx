import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { streamChat } from '../api';

export default function Chat({ sessionId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [citations, setCitations] = useState([]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const q = input.trim();
    if (!q || loading) return;

    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: q }]);
    setLoading(true);
    setStreamingText('');
    setCitations([]);

    let fullAnswer = '';
    let finalCitations = [];
    let hasContent = false;

    try {
      for await (const event of streamChat(q, sessionId)) {
        switch (event.type) {
          case 'token':
            fullAnswer += event.data;
            setStreamingText(fullAnswer);
            hasContent = true;
            break;
          case 'references':
            // references already included in full text
            break;
          case 'done':
            finalCitations = event.citations || [];
            setCitations(finalCitations);
            break;
          case 'error':
            setStreamingText(prev => prev + (prev ? '\n\n' : '') + `[错误: ${event.data}]`);
            break;
        }
      }
    } catch (err) {
      setStreamingText('[连接错误: 请求失败，请检查后端服务]');
    }

    setMessages(prev => [
      ...prev,
      {
        role: 'assistant',
        content: fullAnswer || streamingText,
        citations: finalCitations,
      },
    ]);
    setStreamingText('');
    setLoading(false);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && !loading && (
          <div className="chat-welcome">
            <h2>📚 学术论文智能问答</h2>
            <p>
              上传并索引学术论文后，您可以向我提问论文内容相关问题。<br />
              系统会通过检索增强生成（RAG）技术，结合检索到的文献给出带引用的回答。
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <Message key={i} message={msg} />
        ))}

        {loading && streamingText && (
          <div className="message assistant streaming">
            <div className="message-avatar">🤖</div>
            <div className="message-bubble">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {streamingText}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {loading && !streamingText && (
          <div className="message assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-bubble">
              <span className="thinking-dots">思考中<span>.</span><span>.</span><span>.</span></span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <form className="chat-input-form" onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            className="chat-input"
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入您的问题… (Enter 发送, Shift+Enter 换行)"
            disabled={loading}
          />
          <button className="btn-send" type="submit" disabled={loading || !input.trim()}>
            {loading ? '生成中…' : '发送'}
          </button>
        </form>
      </div>
    </div>
  );
}

function Message({ message }) {
  const isUser = message.role === 'user';
  return (
    <div className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="message-avatar">{isUser ? '👤' : '🤖'}</div>
      <div className="message-bubble">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {message.content}
        </ReactMarkdown>

        {message.citations && message.citations.length > 0 && (
          <details className="citations">
            <summary>📖 参考文献 ({message.citations.length} 篇)</summary>
            <div className="citations-list">
              {message.citations.map((c, i) => (
                <div key={i} className="citation-item">
                  <span className="cit-index">[{c.index}]</span>
                  {c.source}
                  {c.year && <span> ({c.year})</span>}
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
