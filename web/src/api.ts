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

function authHeaders(): Record<string, string> {
  const profile = getStoredUserProfile();
  const userId = getStoredUserId();
  const headers: Record<string, string> = {};
  if (profile?.email) headers['X-User-Email'] = profile.email;
  if (profile?.name) headers['X-User-Name'] = profile.name;
  if (profile?.role) headers['X-User-Role'] = profile.role;
  if (profile?.accountTier) headers['X-User-Account-Tier'] = profile.accountTier;
  if (profile?.institution) headers['X-User-Institution'] = profile.institution;
  if (profile?.avatarUrl) headers['X-User-Avatar-Url'] = profile.avatarUrl;
  if (userId) headers['X-User-Id'] = userId;
  return headers;
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

export function setStoredUser(user: UserProfile) {
  localStorage.setItem(USER_ID_KEY, user.id);
  localStorage.setItem(USER_PROFILE_KEY, JSON.stringify(user));
}

export function clearStoredUser() {
  localStorage.removeItem(USER_ID_KEY);
  localStorage.removeItem(USER_PROFILE_KEY);
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(apiPath(path), {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
      ...(options?.headers || {}),
    },
  });
  if (!response.ok) {
    let message = `请求失败：${response.status}`;
    try {
      const data = await response.json();
      message = data?.detail?.message || data?.detail || data?.message || data?.error || JSON.stringify(data);
    } catch {
      message = await response.text();
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  register: async (payload: {
    name: string;
    email: string;
    phone?: string;
    password: string;
    avatarUrl?: string | null;
  }): Promise<UserProfile> => {
    const response = await request<{ user: UserProfile }>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    setStoredUser(response.user);
    return response.user;
  },

  login: async (payload: { email: string; password: string }): Promise<UserProfile> => {
    const response = await request<{ user: UserProfile }>('/auth/login', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    setStoredUser(response.user);
    return response.user;
  },

  getUser: async (userId: string): Promise<UserProfile> => {
    const user = await request<UserProfile>(`/users/${encodeURIComponent(userId)}`);
    setStoredUser(user);
    return user;
  },

  listSessions: async (): Promise<SessionSummary[]> => {
    const response = await request<{ sessions: SessionSummary[] }>('/sessions');
    return response.sessions || [];
  },

  loadSession: async (sessionId: string): Promise<ChatEvent[]> => {
    const response = await request<{ events: ChatEvent[] }>(`/sessions/${encodeURIComponent(sessionId)}`);
    return response.events || [];
  },

  renameSession: async (sessionId: string, nextSessionId: string): Promise<void> => {
    await request(`/sessions/${encodeURIComponent(sessionId)}/rename`, {
      method: 'POST',
      body: JSON.stringify({ new_session_id: nextSessionId }),
    });
  },

  refreshSummary: async (sessionId: string): Promise<void> => {
    await request(`/sessions/${encodeURIComponent(sessionId)}/summary`, {
      method: 'POST',
      body: JSON.stringify({ max_events: 20 }),
    });
  },

  deleteSession: async (sessionId: string): Promise<void> => {
    await request(`/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  },

  listRagUploads: async (): Promise<UploadedRagFile[]> => {
    const response = await request<{ files: UploadedRagFile[] }>('/rag/uploads');
    return response.files || [];
  },

  uploadRagFiles: async (files: FileList | File[]): Promise<UploadedRagFile[]> => {
    const body = new FormData();
    Array.from(files).forEach((file) => body.append('files', file));
    const response = await fetch(apiPath('/rag/uploads'), {
      method: 'POST',
      credentials: 'include',
      headers: authHeaders(),
      body,
    });
    if (!response.ok) {
      let message = `请求失败：${response.status}`;
      try {
        const data = await response.json();
        message = data?.detail?.message || data?.detail || data?.message || data?.error || JSON.stringify(data);
      } catch {
        message = await response.text();
      }
      throw new Error(message);
    }
    const data = await response.json() as { files: UploadedRagFile[] };
    return data.files || [];
  },

  deleteRagUpload: async (fileId: string): Promise<void> => {
    await request(`/rag/uploads/${encodeURIComponent(fileId)}`, { method: 'DELETE' });
  },

  ingestUploadedRagFiles: async (): Promise<{ task_id: string; upload_dir: string }> => {
    return request<{ task_id: string; upload_dir: string }>('/rag/uploads/ingest', {
      method: 'POST',
      body: JSON.stringify({}),
    });
  },

  getTask: async (taskId: string): Promise<BackgroundTask> => {
    const response = await request<{ task: BackgroundTask }>(`/tasks/${encodeURIComponent(taskId)}`);
    return response.task;
  },
};

export async function askStream(
  question: string,
  sessionId: string,
  showTrace: boolean,
  onToken: (token: string) => void,
): Promise<AskResult> {
  const response = await fetch(apiPath('/ask/stream'), {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({
      question,
      session_id: sessionId,
      write_memory: true,
      show_trace: showTrace,
    }),
  });
  if (!response.ok || !response.body) {
    let message = '请求失败';
    try {
      const data = await response.json();
      message = data?.detail?.message || data?.detail || data?.message || data?.error || message;
    } catch {
      message = await response.text();
    }
    throw new Error(message);
  }

  let finalData: AskResult | null = null;
  for await (const event of readNdjson(response.body)) {
    if (event.event === 'token') {
      onToken(event.content || '');
    } else if (event.event === 'title') {
      finalData = {
        ...(finalData || { answer: '', sources: [], memories: [] }),
        session_title: event.title,
      } as AskResult;
    } else if (event.event === 'answer') {
      finalData = event.data as AskResult;
    } else if (event.event === 'error') {
      throw new Error(event.message || '请求失败');
    }
  }
  if (!finalData) {
    throw new Error('没有收到完整回答');
  }
  return finalData;
}

async function* readNdjson(stream: ReadableStream<Uint8Array>) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed) yield JSON.parse(trimmed);
    }
  }
  if (buffer.trim()) {
    yield JSON.parse(buffer.trim());
  }
}
