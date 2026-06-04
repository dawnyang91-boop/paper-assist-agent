import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from 'react';
import {
  BookOpen,
  Bot,
  ChevronLeft,
  ChevronRight,
  Eye,
  EyeOff,
  Lock,
  LogIn,
  LogOut,
  Mail,
  MessageSquarePlus,
  Pencil,
  Phone,
  RefreshCw,
  Send,
  Trash2,
  UploadCloud,
  User,
} from 'lucide-react';
import { api, askStream, clearStoredUser, getStoredUserId, getStoredUserProfile } from './api';
import { renderMarkdown } from './markdown';
import type { BackgroundTask, MemoryContext, Message, SessionSummary, Source, UploadedRagFile, UserProfile } from './types';

type View = 'login' | 'register' | 'chat';

const APP_BASE_PATH = normalizeBasePath(import.meta.env.VITE_WEB_BASE_PATH || '/chatbot');
const TERMINAL_TASK_STATUSES = new Set(['succeeded', 'failed', 'cancelled']);
const THINKING_MESSAGES = [
  '理解需求中...',
  'RAG 检索中...',
  '工具调用中...',
  '上下文组装中...',
  'LLM 回答生成中...',
  '回答组织中...',
];

function normalizeBasePath(value: string): string {
  const trimmed = value.trim();
  if (!trimmed || trimmed === '/') return '';
  return `/${trimmed.replace(/^\/+|\/+$/g, '')}`;
}

function stripAppBase(pathname: string): string {
  if (!APP_BASE_PATH) return pathname || '/';
  if (pathname === APP_BASE_PATH) return '/';
  if (pathname.startsWith(`${APP_BASE_PATH}/`)) {
    return pathname.slice(APP_BASE_PATH.length) || '/';
  }
  return pathname || '/';
}

function appPath(path: string): string {
  return `${APP_BASE_PATH}${path === '/' ? '' : path}` || '/';
}

function routeFromPath(): View {
  const pathname = stripAppBase(window.location.pathname);
  if (pathname === '/register') return 'register';
  if (pathname === '/login') return 'login';
  return getStoredUserId() ? 'chat' : 'login';
}

function navigate(path: string) {
  window.history.pushState({}, '', path);
}

export default function App() {
  const [view, setView] = useState<View>(routeFromPath);
  const [user, setUser] = useState<UserProfile | null>(getStoredUserProfile());

  useEffect(() => {
    const onPopState = () => setView(routeFromPath());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const go = (nextView: View) => {
    const path = nextView === 'chat' ? appPath('/') : appPath(`/${nextView}`);
    navigate(path);
    setView(nextView);
  };

  const handleAuthed = (profile: UserProfile) => {
    setUser(profile);
    go('chat');
  };

  const logout = () => {
    clearStoredUser();
    setUser(null);
    go('login');
  };

  if (view === 'register') {
    return <RegisterPage onLogin={() => go('login')} onAuthed={handleAuthed} />;
  }
  if (view === 'login' || !getStoredUserId()) {
    return <LoginPage onRegister={() => go('register')} onAuthed={handleAuthed} />;
  }
  return <ChatPage user={user} onLogout={logout} />;
}

function LoginPage({ onRegister, onAuthed }: { onRegister: () => void; onAuthed: (user: UserProfile) => void }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const profile = await api.login({ email, password });
      onAuthed(profile);
    } catch (err) {
      setError(err instanceof Error ? err.message : '登录失败');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthShell
      title="欢迎回来"
      subtitle="登录后继续使用私域问答助手，账号与 Pader 论文推荐助手共享。"
      footer={(
        <span>
          还没有账号？ <button className="link-button" type="button" onClick={onRegister}>注册</button>
        </span>
      )}
    >
      <form className="auth-form" onSubmit={submit}>
        <Field label="Email" icon={<Mail size={18} />}>
          <input type="email" required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Enter your email" />
        </Field>
        <Field label="Password" icon={<Lock size={18} />} action={(
          <button className="icon-button ghost" type="button" onClick={() => setShowPassword((value) => !value)} title={showPassword ? '隐藏密码' : '显示密码'}>
            {showPassword ? <EyeOff size={17} /> : <Eye size={17} />}
          </button>
        )}>
          <input type={showPassword ? 'text' : 'password'} required value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Enter your password" />
        </Field>
        {error && <div className="auth-error">{error}</div>}
        <button className="primary-action" type="submit" disabled={submitting}>
          <span>{submitting ? '登录中...' : '登录'}</span>
          <LogIn size={18} />
        </button>
      </form>
    </AuthShell>
  );
}

function RegisterPage({ onLogin, onAuthed }: { onLogin: () => void; onAuthed: (user: UserProfile) => void }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [password, setPassword] = useState('');
  const [avatarUrl, setAvatarUrl] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError('请输入有效邮箱。');
      return;
    }
    if (!/[A-Za-z]/.test(password) || !/\d/.test(password) || password.length < 8) {
      setError('密码至少 8 位，并包含字母和数字。');
      return;
    }
    setSubmitting(true);
    try {
      const profile = await api.register({ name, email, phone, password, avatarUrl: avatarUrl || null });
      onAuthed(profile);
    } catch (err) {
      setError(err instanceof Error ? err.message : '注册失败');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <AuthShell
      title="创建账号"
      subtitle="注册后可在私域问答助手与 Pader 论文推荐助手中共享同一用户身份。"
      footer={(
        <span>
          已有账号？ <button className="link-button" type="button" onClick={onLogin}>登录</button>
        </span>
      )}
    >
      <form className="auth-form" onSubmit={submit}>
        <Field label="Username" icon={<User size={18} />}>
          <input required value={name} onChange={(event) => setName(event.target.value)} placeholder="Choose a username" />
        </Field>
        <Field label="Email" icon={<Mail size={18} />}>
          <input type="email" required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="Enter your email" />
        </Field>
        <Field label="Phone Number" icon={<Phone size={18} />}>
          <input type="tel" required value={phone} onChange={(event) => setPhone(event.target.value)} placeholder="Enter your phone number" />
        </Field>
        <Field label="Password" icon={<Lock size={18} />}>
          <input type="password" required value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Create a password" />
        </Field>
        <Field label="Avatar URL" icon={<User size={18} />}>
          <input value={avatarUrl} onChange={(event) => setAvatarUrl(event.target.value)} placeholder="Optional avatar URL" />
        </Field>
        {error && <div className="auth-error">{error}</div>}
        <button className="primary-action" type="submit" disabled={submitting}>
          <span>{submitting ? '创建中...' : '创建账号'}</span>
          <LogIn size={18} />
        </button>
      </form>
    </AuthShell>
  );
}

function AuthShell({ title, subtitle, footer, children }: { title: string; subtitle: string; footer: ReactNode; children: ReactNode }) {
  return (
    <main className="auth-page">
      <section className="auth-panel">
        <div className="auth-brand">
          <div className="brand-mark"><BookOpen size={30} /></div>
          <div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
          </div>
        </div>
        {children}
        <div className="auth-footer">{footer}</div>
      </section>
    </main>
  );
}

function Field({ label, icon, action, children }: { label: string; icon: ReactNode; action?: ReactNode; children: ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <span className="field-box">
        <span className="field-icon">{icon}</span>
        {children}
        {action}
      </span>
    </label>
  );
}

function ChatPage({ user, onLogout }: { user: UserProfile | null; onLogout: () => void }) {
  const [sessionId, setSessionId] = useState('default');
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [memories, setMemories] = useState<MemoryContext[]>([]);
  const [trace, setTrace] = useState<unknown>(null);
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [showTrace, setShowTrace] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedRagFile[]>([]);
  const [uploadTaskId, setUploadTaskId] = useState<string | null>(null);
  const [uploadTask, setUploadTask] = useState<BackgroundTask | null>(null);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [question, setQuestion] = useState('');
  const [status, setStatus] = useState('就绪');
  const [busy, setBusy] = useState(false);
  const [thinkingIndex, setThinkingIndex] = useState(0);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const uploadTaskIdRef = useRef<string | null>(null);
  const terminalTaskTimerRef = useRef<number | null>(null);
  const lastUploadAtRef = useRef(0);
  const lastIngestAtRef = useRef(0);
  const lastQuestionRef = useRef<{ text: string; ts: number }>({ text: '', ts: 0 });

  useEffect(() => {
    void refreshSessions();
    void refreshUploads();
  }, []);

  useEffect(() => {
    if (!uploadTaskId) return;
    uploadTaskIdRef.current = uploadTaskId;
    let stopped = false;
    let timer: number | undefined;
    const poll = async () => {
      const done = await refreshTask(uploadTaskId);
      if (!stopped && !done) {
        timer = window.setTimeout(poll, 1500);
      }
    };
    void poll();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [uploadTaskId]);

  useEffect(() => {
    uploadTaskIdRef.current = uploadTaskId;
  }, [uploadTaskId]);

  useEffect(() => {
    return () => {
      if (terminalTaskTimerRef.current !== null) {
        window.clearTimeout(terminalTaskTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight });
  }, [messages]);

  useEffect(() => {
    if (!busy) return undefined;
    const timer = window.setInterval(() => {
      setThinkingIndex((index) => (index + 1) % THINKING_MESSAGES.length);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [busy]);

  const refreshSessions = async () => {
    try {
      setSessions(await api.listSessions());
    } catch {
      setSessions([]);
    }
  };

  const refreshUploads = async () => {
    try {
      setUploadedFiles(await api.listRagUploads());
    } catch {
      setUploadedFiles([]);
    }
  };

  const clearTerminalTaskTimer = () => {
    if (terminalTaskTimerRef.current !== null) {
      window.clearTimeout(terminalTaskTimerRef.current);
      terminalTaskTimerRef.current = null;
    }
  };

  const hideTerminalTaskSoon = () => {
    clearTerminalTaskTimer();
    terminalTaskTimerRef.current = window.setTimeout(() => {
      if (uploadTaskIdRef.current === null) {
        setUploadTask(null);
      }
      terminalTaskTimerRef.current = null;
    }, 3500);
  };

  const refreshTask = async (taskId: string): Promise<boolean> => {
    try {
      const task = await api.getTask(taskId);
      if (uploadTaskIdRef.current !== taskId) {
        return true;
      }
      setUploadTask(task);
      if (TERMINAL_TASK_STATUSES.has(task.status || '')) {
        setUploadTaskId(null);
        uploadTaskIdRef.current = null;
        hideTerminalTaskSoon();
        await refreshUploads();
        return true;
      }
      return false;
    } catch {
      setUploadTask(null);
      setUploadTaskId(null);
      return true;
    }
  };

  const uploadFiles = async (files: FileList | null) => {
    const now = Date.now();
    if (!files?.length || uploadBusy || now - lastUploadAtRef.current < 1200) return;
    lastUploadAtRef.current = now;
    clearTerminalTaskTimer();
    setUploadTask(null);
    setUploadedFiles([]);
    setUploadBusy(true);
    setStatus('正在上传本地文档...');
    try {
      await api.uploadRagFiles(files);
      await refreshUploads();
      setStatus('上传完成，可开始向量化');
    } catch (err) {
      setStatus(err instanceof Error ? err.message : '上传失败');
    } finally {
      setUploadBusy(false);
    }
  };

  const ingestUploadedFiles = async () => {
    const now = Date.now();
    if (uploadBusy || uploadTaskIdRef.current || now - lastIngestAtRef.current < 1800) return;
    lastIngestAtRef.current = now;
    clearTerminalTaskTimer();
    setUploadBusy(true);
    setStatus('已提交向量化任务...');
    try {
      const result = await api.ingestUploadedRagFiles();
      setUploadTaskId(result.task_id);
      uploadTaskIdRef.current = result.task_id;
      setUploadTask({ status: 'pending', progress: '0', message: '任务已提交' });
    } catch (err) {
      setStatus(err instanceof Error ? err.message : '向量化任务提交失败');
    } finally {
      setUploadBusy(false);
    }
  };

  const deleteUpload = async (fileId: string) => {
    await api.deleteRagUpload(fileId);
    await refreshUploads();
  };

  const loadSession = async (nextSessionId: string) => {
    const clean = nextSessionId || 'default';
    setSessionId(clean);
    setSources([]);
    setMemories([]);
    setTrace(null);
    setSelectedMessageId(null);
    const events = await api.loadSession(clean);
    const loadedMessages = events.filter((event) => event.role === 'user' || event.role === 'assistant').map((event, index) => ({
      id: `${clean}-${index}`,
      role: event.role as 'user' | 'assistant',
      content: event.content,
      sources: event.metadata?.references?.sources || [],
      memories: event.metadata?.references?.memories || [],
      citations: event.metadata?.references?.citations || [],
    }));
    setMessages(loadedMessages);
    const lastAssistantWithReferences = [...loadedMessages].reverse().find(
      (message) => message.role === 'assistant' && ((message.sources || []).length || (message.memories || []).length),
    );
    if (lastAssistantWithReferences) {
      setSelectedMessageId(lastAssistantWithReferences.id);
      setSources(lastAssistantWithReferences.sources || []);
      setMemories(lastAssistantWithReferences.memories || []);
    }
    setStatus('会话已载入');
  };

  const newSession = () => {
    const id = `session-${new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14)}`;
    setSessionId(id);
    setMessages([]);
    setSources([]);
    setMemories([]);
    setTrace(null);
    setSelectedMessageId(null);
  };

  const renameSession = async () => {
    const next = window.prompt('新的 session id', sessionId);
    if (!next || next === sessionId) return;
    await api.renameSession(sessionId, next);
    setSessionId(next);
    await refreshSessions();
  };

  const refreshSummary = async () => {
    await api.refreshSummary(sessionId);
    await refreshSessions();
    setStatus('摘要已刷新');
  };

  const deleteSession = async () => {
    if (!window.confirm(`删除会话 ${sessionId}？`)) return;
    await api.deleteSession(sessionId);
    setSessionId('default');
    setMessages([]);
    setSelectedMessageId(null);
    await refreshSessions();
  };

  const sendQuestion = async () => {
    const clean = question.trim();
    const now = Date.now();
    if (!clean || busy) return;
    if (lastQuestionRef.current.text === clean && now - lastQuestionRef.current.ts < 1000) return;
    lastQuestionRef.current = { text: clean, ts: now };
    setBusy(true);
    setThinkingIndex(0);
    setQuestion('');
    setStatus(THINKING_MESSAGES[0]);
    const userMessage: Message = { id: crypto.randomUUID(), role: 'user', content: clean };
    const assistantId = crypto.randomUUID();
    setSelectedMessageId(assistantId);
    setMessages((items) => [...items, userMessage, { id: assistantId, role: 'assistant', content: '', isThinking: true }]);
    let streamedAnswer = '';
    try {
      const result = await askStream(clean, sessionId, showTrace, (token) => {
        streamedAnswer += token;
        setMessages((items) => items.map((item) => item.id === assistantId ? { ...item, content: streamedAnswer, isThinking: false } : item));
      });
      setMessages((items) => items.map((item) => item.id === assistantId ? {
        ...item,
        content: result.answer,
        isThinking: false,
        sources: result.sources || [],
        memories: result.memories || [],
        trace: result.trace || null,
      } : item));
      setSources(result.sources || []);
      setMemories(result.memories || []);
      setTrace(result.trace || null);
      setSelectedMessageId(assistantId);
      await refreshSessions();
      setStatus('就绪');
    } catch (err) {
      setMessages((items) => items.map((item) => item.id === assistantId ? { ...item, content: `请求失败：${err instanceof Error ? err.message : '未知错误'}`, isThinking: false } : item));
      setStatus('出错');
    } finally {
      setBusy(false);
    }
  };

  const activeSession = useMemo(() => sessions.find((item) => item.session_id === sessionId), [sessions, sessionId]);
  const selectedMessage = useMemo(
    () => messages.find((item) => item.id === selectedMessageId && item.role === 'assistant') || null,
    [messages, selectedMessageId],
  );
  const displayedSources = selectedMessage ? (selectedMessage.sources || []) : sources;
  const displayedMemories = selectedMessage ? (selectedMessage.memories || []) : memories;
  const displayedTrace = selectedMessage ? (selectedMessage.trace || null) : trace;
  const thinkingMessage = THINKING_MESSAGES[thinkingIndex];
  const shellClassName = [
    'app-shell',
    sidebarCollapsed ? 'sidebar-collapsed' : '',
    inspectorCollapsed ? 'inspector-collapsed' : '',
  ].filter(Boolean).join(' ');

  return (
    <div className={shellClassName}>
      {sidebarCollapsed && (
        <button className="edge-toggle left" title="展开会话栏" onClick={() => setSidebarCollapsed(false)}>
          <ChevronRight size={18} />
        </button>
      )}
      {inspectorCollapsed && (
        <button className="edge-toggle right" title="展开上下文栏" onClick={() => setInspectorCollapsed(false)}>
          <ChevronLeft size={18} />
        </button>
      )}

      <aside className="sidebar" aria-hidden={sidebarCollapsed}>
        <div className="brand">
          <div>
            <h1>Pader</h1>
            <p>本地私域文档将作为 RAG 知识库，优先用于回答你的问题。</p>
          </div>
          <div className="brand-actions">
            <button className="icon-button" title="新建会话" onClick={newSession}><MessageSquarePlus size={18} /></button>
            <button className="icon-button ghost" title="收起会话栏" onClick={() => setSidebarCollapsed(true)}><ChevronLeft size={18} /></button>
          </div>
        </div>

        <div className="user-chip">
          <div className="avatar">{(user?.name || 'U').slice(0, 1).toUpperCase()}</div>
          <div>
            <strong>{user?.name || 'Researcher'}</strong>
            <span>{user?.email || getStoredUserId()}</span>
          </div>
          <button className="icon-button ghost" title="退出登录" onClick={onLogout}><LogOut size={17} /></button>
        </div>

        <div className="session-actions">
          <button onClick={() => void renameSession()}><Pencil size={15} />重命名</button>
          <button onClick={() => void refreshSummary()}><RefreshCw size={15} />刷新摘要</button>
          <button onClick={() => void deleteSession()}><Trash2 size={15} />删除</button>
        </div>
        <div className="sessions">
          {sessions.length === 0 ? <div className="panel-empty">暂无历史会话</div> : sessions.map((session) => (
            <button
              key={session.session_id}
              className={`session-item ${session.session_id === sessionId ? 'active' : ''}`}
              onClick={() => void loadSession(session.session_id)}
            >
              <span className="session-title">{session.title || session.session_id}</span>
              <small>{session.last_content || ''}</small>
            </button>
          ))}
        </div>
        <LocalRagUploadPanel
          files={uploadedFiles}
          task={uploadTask}
          busy={uploadBusy}
          onUpload={uploadFiles}
          onIngest={() => void ingestUploadedFiles()}
          onRefresh={() => void refreshUploads()}
          onDelete={(fileId) => void deleteUpload(fileId)}
        />
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <h2>{activeSession?.title || activeSession?.session_id || sessionId}</h2>
            <p>{busy ? thinkingMessage : status}</p>
          </div>
          <label className="toggle">
            <input checked={showTrace} onChange={(event) => setShowTrace(event.target.checked)} type="checkbox" />
            <span>Trace</span>
          </label>
        </header>
        <section ref={messagesRef} className="messages">
          {messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              thinkingText={thinkingMessage}
              selected={message.id === selectedMessageId}
              onSelect={() => {
                if (message.role !== 'assistant' || message.isThinking) return;
                setSelectedMessageId(message.id);
                setSources(message.sources || []);
                setMemories(message.memories || []);
                setTrace(message.trace || null);
              }}
            />
          ))}
        </section>
        <section className="composer">
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                void sendQuestion();
              }
            }}
            placeholder="输入你的问题，例如：SENet 的核心贡献是什么？"
          />
          <button className="send-button" onClick={() => void sendQuestion()} disabled={busy}>
            <span>发送</span>
            <Send size={17} />
          </button>
        </section>
      </main>

      <aside className="inspector" aria-hidden={inspectorCollapsed}>
        <div className="inspector-toolbar">
          <span>上下文</span>
          <button className="icon-button ghost" title="收起上下文栏" onClick={() => setInspectorCollapsed(true)}><ChevronRight size={18} /></button>
        </div>
        <Panel title="引用来源">
          <Sources sources={displayedSources} />
        </Panel>
        <Panel title="Agent Trace">
          <pre className="trace">{displayedTrace ? JSON.stringify(displayedTrace, null, 2) : '暂无 trace'}</pre>
        </Panel>
        <Panel title="记忆召回">
          <Memories memories={displayedMemories} />
        </Panel>
      </aside>
    </div>
  );
}

function ChatMessage({
  message,
  thinkingText,
  selected,
  onSelect,
}: {
  message: Message;
  thinkingText: string;
  selected: boolean;
  onSelect: () => void;
}) {
  if (message.role === 'assistant') {
    if (message.isThinking) {
      return (
        <div className="message assistant thinking">
          <span className="thinking-dot" />
          <span>{thinkingText}</span>
        </div>
      );
    }
    return (
      <div
        className={`message assistant selectable ${selected ? 'selected' : ''}`}
        onClick={onSelect}
        dangerouslySetInnerHTML={{ __html: renderMarkdown(message.content) }}
      />
    );
  }
  return <div className="message user">{message.content}</div>;
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function Sources({ sources }: { sources: Source[] }) {
  if (!sources.length) return <div className="panel-empty">暂无引用</div>;
  return (
    <>
      {sources.map((source) => (
        <div className="source" key={`${source.doc_id}-${source.chunk_index}`}>
          <strong>[{source.doc_id}] {source.source_file || 'unknown'}</strong>
          <div>chunk: {source.chunk_index ?? 'unknown'}</div>
          <div>score: {Number(source.score || 0).toFixed(4)}</div>
          <div>{(source.heading_paths || []).join(' / ')}</div>
        </div>
      ))}
    </>
  );
}

function Memories({ memories }: { memories: MemoryContext[] }) {
  if (!memories.length) return <div className="panel-empty">暂无记忆</div>;
  return (
    <>
      {memories.slice(0, 12).map((memory, index) => (
        <div className="source" key={`${memory.type || memory.role}-${index}`}>
          <strong>[M{index + 1}] {memory.type || memory.role || 'memory'}</strong>
          {memory.score !== undefined && memory.score !== null ? <div>score: {Number(memory.score || 0).toFixed(4)}</div> : null}
          {memory.importance !== undefined && memory.importance !== null ? <div>importance: {memory.importance}</div> : null}
          <div>{memory.content || ''}</div>
        </div>
      ))}
    </>
  );
}

function LocalRagUploadPanel({
  files,
  task,
  busy,
  onUpload,
  onIngest,
  onRefresh,
  onDelete,
}: {
  files: UploadedRagFile[];
  task: BackgroundTask | null;
  busy: boolean;
  onUpload: (files: FileList | null) => void;
  onIngest: () => void;
  onRefresh: () => void;
  onDelete: (fileId: string) => void;
}) {
  const inputId = useMemo(() => `rag-upload-${crypto.randomUUID()}`, []);
  return (
    <section className="upload-panel">
      <div className="upload-header">
        <div>
          <strong>本地知识库</strong>
          <span>{files.length ? `${files.length} 个文件` : '尚未上传文件'}</span>
        </div>
        <button className="icon-button ghost" type="button" onClick={onRefresh} title="刷新文件"><RefreshCw size={15} /></button>
      </div>
      <input
        id={inputId}
        className="file-input"
        multiple
        type="file"
        onChange={(event) => {
          onUpload(event.currentTarget.files);
          event.currentTarget.value = '';
        }}
      />
      <label className={`upload-drop ${busy ? 'disabled' : ''}`} htmlFor={busy ? undefined : inputId}>
        <UploadCloud size={18} />
        <span>{busy ? '处理中...' : '上传本地文档'}</span>
      </label>
      <button className="secondary-action" type="button" disabled={busy || files.length === 0} onClick={onIngest}>
        开始向量化
      </button>
      {task ? (
        <div className={`task-status ${task.status || ''}`}>
          <strong>{task.status || 'pending'}</strong>
          <span>{task.message || ''}</span>
          {task.progress ? <small>{task.progress}%</small> : null}
          {task.error ? <small>{task.error}</small> : null}
        </div>
      ) : null}
      <div className="upload-list">
        {files.slice(0, 6).map((file) => (
          <div className="upload-item" key={file.file_id}>
            <span title={file.filename}>{file.filename}</span>
            <button className="icon-button ghost" type="button" onClick={() => onDelete(file.file_id)} title="删除文件">
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}
