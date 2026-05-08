export interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
}

export interface ArxivPaper {
  id: string;
  title: string;
  authors: string[];
  authors_display: string;
  summary: string;
  published: string;
  pdf_url?: string;
}

export interface UploadResponse {
  status: string;
  chunks: number;
  message?: string;
}
