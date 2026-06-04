export type UserProfile = {
  id: string;
  email?: string | null;
  name: string;
  role: string;
  accountTier?: 'admin' | 'member' | 'normal' | string;
  institution: string;
  avatarUrl: string;
  stats: {
    liked: number;
    streak: number;
    folders: number;
  };
  interests: string[];
  researchFields?: string[];
};

export type SessionSummary = {
  session_id: string;
  title?: string | null;
  event_count?: number;
  last_content?: string;
};

export type ChatEvent = {
  role: 'user' | 'assistant' | string;
  content: string;
  metadata?: {
    references?: AnswerReferences;
    [key: string]: unknown;
  };
};

export type Source = {
  doc_id: string;
  source_file?: string | null;
  chunk_index?: number | null;
  score?: number | null;
  heading_paths?: string[];
  metadata?: Record<string, unknown>;
};

export type MemoryContext = {
  role?: string;
  type?: string;
  content?: string;
  score?: number | null;
  importance?: number | null;
};

export type AskResult = {
  answer: string;
  sources: Source[];
  trace?: unknown;
  memories: MemoryContext[];
  session_title?: string;
};

export type AnswerReferences = {
  sources?: Source[];
  memories?: MemoryContext[];
  citations?: string[];
};

export type UploadedRagFile = {
  file_id: string;
  filename: string;
  path: string;
  size: number;
  created_at: number;
};

export type BackgroundTask = {
  type?: string;
  status?: string;
  progress?: string;
  message?: string;
  error?: string;
  result_json?: string;
  created_at?: string;
  updated_at?: string;
};

export type Message = {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isThinking?: boolean;
  sources?: Source[];
  memories?: MemoryContext[];
  trace?: unknown;
  citations?: string[];
};
