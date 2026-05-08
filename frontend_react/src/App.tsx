import { useState, useCallback } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import ChatView from './components/ChatView';
import type { Conversation, Message } from './types';

function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function firstLine(text: string): string {
  const line = text.replace(/\n.*$/s, '').trim();
  return line.length > 40 ? line.slice(0, 40) + '…' : line;
}

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  const activeConv = conversations.find((c) => c.id === activeId) ?? null;
  const activeMessages = activeConv?.messages ?? [];

  const setMessages = useCallback(
    (updater: Message[] | ((prev: Message[]) => Message[])) => {
      setConversations((prev) => {
        const conv = prev.find((c) => c.id === activeId);
        if (!conv) return prev;
        const prevMsgs = conv.messages;
        const newMsgs =
          typeof updater === 'function' ? updater(prevMsgs) : updater;
        const title =
          newMsgs.length > 0
            ? firstLine(newMsgs[0]?.content ?? '')
            : '新对话';
        return prev.map((c) =>
          c.id === activeId ? { ...c, messages: newMsgs, title } : c
        );
      });
    },
    [activeId]
  );

  const handleNew = () => {
    const id = genId();
    const conv: Conversation = {
      id,
      title: '新对话',
      messages: [],
      createdAt: Date.now(),
    };
    setConversations((prev) => [...prev, conv]);
    setActiveId(id);
  };

  const handleSelect = (id: string) => setActiveId(id);

  const handleDelete = (id: string) => {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (activeId === id) setActiveId(null);
  };

  const handleClearAll = () => {
    setConversations([]);
    setActiveId(null);
  };

  return (
    <div className="app-layout">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={handleSelect}
        onNew={handleNew}
        onDelete={handleDelete}
        onClearAll={handleClearAll}
      />
      <main className="main-panel">
        {activeId ? (
          <ChatView messages={activeMessages} setMessages={setMessages} />
        ) : (
          <div className="welcome-screen">
            <h1>RAG 智能问答</h1>
            <p>选择左侧已有对话，或点击「+ 新对话」开始</p>
          </div>
        )}
      </main>
    </div>
  );
}
