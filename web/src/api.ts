import type { AskResult, BackgroundTask, ChatEvent, SessionSummary, UploadedRagFile, UserProfile } from './types';

const USER_ID_KEY = 'user_id';
const USER_PROFILE_KEY = 'user_profile';
const API_BASE = normalizeBasePath(import.meta.env.VITE_API_BASE || '/api');

function normalizeBasePath(value: string): string {
  const trimmed = value.trim();
  if (!trimmed || trimmed === '/') return '';
  return `/${trimmed.replace(/^\/+|\/+$/g, '')}`;
}

function apiPath(path: string): string {
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

function storeUser(profile: UserProfile) {
  localStorage.setItem(USER_ID_KEY, profile.id);
  localStorage.setItem(USER_PROFILE_KEY, JSON.stringify(profile));
}

function authHeaders(extra?: HeadersInit): HeadersInit {
  const headers: Record<string, string> = {};
  const profile = getStoredUserProfile();
  const userId = profile?.id || getStoredUserId();
  if (userId) headers['X-User-Id'] = userId;
  if (profile?.email) headers['X-User-Email'] = profile.email;
  if (profile?.name) headers['X-User-Name'] = profile.name;
  return { ...headers, ...(extra || {}) };
}

export function getStoredUserId(): string | null {
  return localStorage.getItem(USER_ID_KEY);
}

export function getStoredUserProfile(): UserProfile | null {
  const raw = localStorage.getItem(USER_PROFILE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as UserProfile;
  } catch {
    return null;
  }
}

export function clearStoredUser() {
  localStorage.removeItem(USER_ID_KEY);
  localStorage.removeItem(USER_PROFILE_KEY);
}

async function parseResponse<T>(response: Response): Promise<T> {
  const text = await response.text();
  const data = text ? safeJson(text) : {};
  if (!response.ok) {
    throw new Error(errorMessage(response.status, data, text));
  }
  return data as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function errorMessage(status: number, data: unknown, fallback: string): string {
  if (typeof data === 'object' && data !== null) {
    const record = data as Record<string, unknown>;
    const message = record.message || record.error || record.detail;
    if (typeof message === 'string' && message.trim()) {
      if (status === 401) return '账号或密码错误；如果还没有账号，请先注册。';
      return message;
    }
  }
  if (status === 401) return '账号或密码错误；如果还没有账号，请先注册。';
  if (status === 413) return '文件过大，请压缩后再上传。';
  if (status === 429) return '请求过于频繁，请稍后再试。';
  return fallback || `请求失败：HTTP ${status}`;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(apiPath(path), {
    credentials: 'include',
    headers: authHeaders(),
  });
  return parseResponse<T>(response);
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(apiPath(path), {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    credentials: 'include',
    body: JSON.stringify(body ?? {}),
  });
  return parseResponse<T>(response);
}

async function deleteJson<T>(path: string): Promise<T> {
  const response = await fetch(apiPath(path), {
    method: 'DELETE',
    credentials: 'include',
    headers: authHeaders(),
  });
  return parseResponse<T>(response);
}

function normalizeAskResult(data: unknown): AskResult {
  const record = (data || {}) as Record<string, unknown>;
  const result = (record.result || record) as Record<string, unknown>;
  const metadata = (result.metadata || result.trace || {}) as Record<string, unknown>;
  return {
    answer: String(result.answer || ''),
    sources: (result.sources || result.documents || []) as AskResult['sources'],
    trace: result.trace || result.metadata || null,
    memories: (result.memories || metadata.memory_contexts || []) as AskResult['memories'],
    session_title: result.session_title as string | undefined,
  };
}

function normalizeSessions(data: unknown): SessionSummary[] {
  const record = (data || {}) as Record<string, unknown>;
  const sessions = record.sessions || record.data || [];
  return Array.isArray(sessions) ? sessions as SessionSummary[] : [];
}

function normalizeSessionEvents(data: unknown): ChatEvent[] {
  const record = (data || {}) as Record<string, unknown>;
  const events = record.events || record.messages || record.data || [];
  return Array.isArray(events) ? events as ChatEvent[] : [];
}

function normalizeUploads(data: unknown): UploadedRagFile[] {
  const record = (data || {}) as Record<string, unknown>;
  const files = record.files || record.data || [];
  return Array.isArray(files) ? files as UploadedRagFile[] : [];
}

function normalizeTask(data: unknown): BackgroundTask {
  const record = (data || {}) as Record<string, unknown>;
  return (record.task || record) as BackgroundTask;
}

export const api = {
  async login(payload: { email: string; password: string }): Promise<UserProfile> {
    const data = await postJson<{ user?: UserProfile }>('/auth/login', payload);
    if (!data.user) throw new Error('登录响应缺少用户信息。');
    storeUser(data.user);
    return data.user;
  },

  async register(payload: {
    name: string;
    email: string;
    phone?: string;
    password: string;
    avatarUrl?: string | null;
  }): Promise<UserProfile> {
    const data = await postJson<{ user?: UserProfile }>('/auth/register', payload);
    if (!data.user) throw new Error('注册响应缺少用户信息。');
    storeUser(data.user);
    return data.user;
  },

  async logout(): Promise<void> {
    await postJson('/auth/logout', {});
    clearStoredUser();
  },

  async ask(question: string, sessionId: string, trace = true): Promise<AskResult> {
    const data = await postJson('/ask', { question, session_id: sessionId, trace });
    return normalizeAskResult(data);
  },

  async listSessions(): Promise<SessionSummary[]> {
    return normalizeSessions(await getJson('/sessions'));
  },

  async loadSession(sessionId: string): Promise<ChatEvent[]> {
    return normalizeSessionEvents(await getJson(`/sessions/${encodeURIComponent(sessionId)}`));
  },

  async renameSession(sessionId: string, title: string): Promise<void> {
    await postJson(`/sessions/${encodeURIComponent(sessionId)}/rename`, { title });
  },

  async refreshSummary(sessionId: string): Promise<void> {
    await postJson(`/sessions/${encodeURIComponent(sessionId)}/summary`, {});
  },

  async deleteSession(sessionId: string): Promise<void> {
    await deleteJson(`/sessions/${encodeURIComponent(sessionId)}`);
  },

  async listRagUploads(): Promise<UploadedRagFile[]> {
    return normalizeUploads(await getJson('/rag/uploads'));
  },

  async uploadRagFiles(files: FileList): Promise<UploadedRagFile[]> {
    const formData = new FormData();
    Array.from(files).forEach((file) => formData.append('files', file));
    const response = await fetch(apiPath('/rag/uploads'), {
      method: 'POST',
      credentials: 'include',
      headers: authHeaders(),
      body: formData,
    });
    return normalizeUploads(await parseResponse(response));
  },

  async deleteRagUpload(fileId: string): Promise<void> {
    await deleteJson(`/rag/uploads/${encodeURIComponent(fileId)}`);
  },

  async ingestUploadedRagFiles(): Promise<{ task_id: string; upload_dir?: string }> {
    const data = await postJson<Record<string, unknown>>('/rag/uploads/ingest', {});
    return {
      task_id: String(data.task_id || ''),
      upload_dir: typeof data.upload_dir === 'string' ? data.upload_dir : undefined,
    };
  },

  async getTask(taskId: string): Promise<BackgroundTask> {
    return normalizeTask(await getJson(`/tasks/${encodeURIComponent(taskId)}`));
  },
};

export async function askStream(
  question: string,
  sessionId: string,
  trace: boolean,
  onToken: (token: string) => void,
): Promise<AskResult> {
  const response = await fetch(apiPath('/ask/stream'), {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    credentials: 'include',
    body: JSON.stringify({ question, session_id: sessionId, trace }),
  });
  if (!response.ok) {
    return parseResponse(response);
  }
  if (!response.body) {
    throw new Error('浏览器不支持流式读取。');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResult: AskResult | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const event = safeJson(trimmed) as Record<string, unknown>;
      const eventKind = String(event.type || event.event || '');
      if (eventKind === 'token') {
        onToken(String(event.content || event.token || ''));
      } else if (eventKind === 'final' || eventKind === 'result' || eventKind === 'answer') {
        finalResult = normalizeAskResult(event.result || event.data || event);
      } else if (eventKind === 'error') {
        throw new Error(String(event.message || event.error || '流式请求失败'));
      }
    }
  }

  if (buffer.trim()) {
    const event = safeJson(buffer.trim()) as Record<string, unknown>;
    const eventKind = String(event.type || event.event || '');
    if (eventKind === 'final' || eventKind === 'result' || eventKind === 'answer') {
      finalResult = normalizeAskResult(event.result || event.data || event);
    }
  }

  return finalResult || {
    answer: '',
    sources: [],
    memories: [],
    trace: null,
  };
}
