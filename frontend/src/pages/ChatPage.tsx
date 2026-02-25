import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useWorkspaceStore } from '@/stores/workspace-store';
import { useAuthStore } from '@/stores/auth-store';
import { engine, parsePlotly } from '@/lib/api';
import { Button, Card, Select, Spinner, Input } from '@/components/ui';
import { Send, Bot, User, Sparkles, Trash2, Plus, Copy, Pin, Bookmark, Download, Edit2, Check } from 'lucide-react';
import Plot from 'react-plotly.js';
import toast from 'react-hot-toast';

interface ChatSummary {
  chat_id: string;
  chat_name: string;
  selected_model: string;
  session_id?: string;
  created_at: string;
  updated_at: string;
}

interface ChatMessage {
  message_id: string;
  chat_id: string;
  role: 'user' | 'assistant';
  content: string;
  charts?: Array<{ title?: string; data: any[]; layout: any; raw: any }>;
  model?: string;
  created_at: string;
  metadata?: any;
}

const MODEL_FALLBACK = ['qwen2.5:7b', 'llama3.2', 'mistral', 'phi3', 'gemma2'];

export default function ChatPage() {
  const { sessionId, addPinnedCard, addPinnedChart } = useWorkspaceStore();
  const { user } = useAuthStore();

  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [activeChatId, setActiveChatId] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [models, setModels] = useState<string[]>(MODEL_FALLBACK);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [loadingChats, setLoadingChats] = useState(false);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [renamingChatId, setRenamingChatId] = useState('');
  const [renameValue, setRenameValue] = useState('');
  const [analysisMode, setAnalysisMode] = useState<'fast' | 'balanced' | 'deep'>('deep');
  const [responseStyle, setResponseStyle] = useState<'executive' | 'technical' | 'deep'>('deep');
  const [strictMode, setStrictMode] = useState<'off' | 'on'>('off');
  const [tableScope, setTableScope] = useState<'all' | 'working'>('all');
  const [joinMode, setJoinMode] = useState<'on' | 'off'>('on');
  const [temperature, setTemperature] = useState('0.0');

  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const activeChat = useMemo(() => chats.find((c) => c.chat_id === activeChatId) || null, [chats, activeChatId]);
  const selectedModel = activeChat?.selected_model || models[0] || MODEL_FALLBACK[0];

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, sending]);

  const hydrateCharts = (arr: any[]): Array<{ title?: string; data: any[]; layout: any; raw: any }> => {
    const out: Array<{ title?: string; data: any[]; layout: any; raw: any }> = [];
    for (const item of arr || []) {
      const fig = parsePlotly(item?.plotly_json || item?.figure || item);
      if (!fig) continue;
      out.push({ title: item?.title, data: fig.data || [], layout: fig.layout || {}, raw: item });
    }
    return out;
  };

  const fetchModels = useCallback(async () => {
    try {
      const { data } = await engine.get('/chats/models');
      const list = data?.data?.models;
      if (Array.isArray(list) && list.length > 0) setModels(list);
    } catch {
      setModels(MODEL_FALLBACK);
    }
  }, []);

  const fetchChats = useCallback(async () => {
    if (!user) return;
    setLoadingChats(true);
    try {
      const { data } = await engine.get(`/chats?user_id=${user.id}`);
      const rows = (data?.data?.chats || []) as ChatSummary[];
      setChats(rows);
      if (!activeChatId && rows.length > 0) setActiveChatId(rows[0].chat_id);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to load chats');
    } finally {
      setLoadingChats(false);
    }
  }, [user, activeChatId]);

  const fetchMessages = useCallback(async (chatId: string) => {
    if (!chatId) return;
    setLoadingMessages(true);
    try {
      const { data } = await engine.get(`/chats/${chatId}/messages`);
      const rows = data?.data?.messages || [];
      const mapped: ChatMessage[] = rows.map((m: any) => ({
        message_id: m.message_id,
        chat_id: m.chat_id,
        role: m.role,
        content: m.content,
        model: m.model,
        charts: hydrateCharts(m.charts || []),
        created_at: m.created_at,
        metadata: m.metadata,
      }));
      setMessages(mapped);
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to load messages');
      setMessages([]);
    } finally {
      setLoadingMessages(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
    fetchChats();
  }, [fetchModels, fetchChats]);

  useEffect(() => {
    if (activeChatId) fetchMessages(activeChatId);
    else setMessages([]);
  }, [activeChatId, fetchMessages]);

  const createChat = async (): Promise<ChatSummary | null> => {
    if (!sessionId) {
      toast.error('Upload data first');
      return null;
    }
    try {
      const { data } = await engine.post('/chats', {
        session_id: sessionId,
        selected_model: models[0] || MODEL_FALLBACK[0],
      });
      const chat = data?.data as ChatSummary;
      setChats((prev) => [chat, ...prev]);
      setActiveChatId(chat.chat_id);
      setMessages([]);
      toast.success('New chat created');
      return chat;
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to create chat');
      return null;
    }
  };

  const renameChat = async (chatId: string, chatName: string) => {
    if (!chatName.trim()) return;
    try {
      const { data } = await engine.patch(`/chats/${chatId}`, { chat_name: chatName.trim() });
      const updated = data?.data as ChatSummary;
      setChats((prev) => prev.map((c) => (c.chat_id === chatId ? updated : c)));
      setRenamingChatId('');
      setRenameValue('');
      toast.success('Chat renamed');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Rename failed');
    }
  };

  const deleteChat = async (chatId: string) => {
    try {
      await engine.delete(`/chats/${chatId}`);
      const next = chats.filter((c) => c.chat_id !== chatId);
      setChats(next);
      if (activeChatId === chatId) setActiveChatId(next[0]?.chat_id || '');
      toast.success('Chat deleted');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Delete failed');
    }
  };

  const clearMessages = async () => {
    if (!activeChatId) return;
    try {
      await engine.delete(`/chats/${activeChatId}/messages`);
      setMessages([]);
      toast.success('Chat cleared');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Clear failed');
    }
  };

  const updateModelForActiveChat = async (model: string) => {
    if (!activeChatId) return;
    try {
      const { data } = await engine.patch(`/chats/${activeChatId}`, { selected_model: model });
      const updated = data?.data as ChatSummary;
      setChats((prev) => prev.map((c) => (c.chat_id === activeChatId ? updated : c)));
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Failed to update model');
    }
  };

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || sending) return;
    if (!sessionId) return toast.error('Upload data first');

    let targetChatId = activeChatId;
    if (!targetChatId) {
      const created = await createChat();
      targetChatId = created?.chat_id || '';
      if (!targetChatId) return;
    }

    const optimisticUser: ChatMessage = {
      message_id: `tmp-user-${Date.now()}`,
      chat_id: targetChatId,
      role: 'user',
      content: text,
      created_at: new Date().toISOString(),
      model: selectedModel,
    };

    setMessages((prev) => [...prev, optimisticUser]);
    setInput('');
    setSending(true);

    try {
      const { data } = await engine.post(`/chats/${targetChatId}/messages`, {
        session_id: sessionId,
        model: selectedModel,
        message: text,
        strict: strictMode === 'on',
        analysis_mode: analysisMode,
        response_style: responseStyle,
        use_all_tables: tableScope === 'all',
        prefer_joins: joinMode === 'on',
        temperature: Number.isFinite(Number(temperature)) ? Number(temperature) : 0.0,
      });

      const payload = data?.data || {};
      const userMsg = payload.user_message;
      const assistantMsg = payload.assistant_message;
      const nextChat = payload.chat as ChatSummary;

      const mappedUser: ChatMessage = {
        message_id: userMsg.message_id,
        chat_id: userMsg.chat_id,
        role: 'user',
        content: userMsg.content,
        created_at: userMsg.created_at,
        model: userMsg.model,
      };

      const mappedAssistant: ChatMessage = {
        message_id: assistantMsg.message_id,
        chat_id: assistantMsg.chat_id,
        role: 'assistant',
        content: assistantMsg.content,
        charts: hydrateCharts(assistantMsg.charts || []),
        created_at: assistantMsg.created_at,
        model: assistantMsg.model,
        metadata: assistantMsg.metadata,
      };

      setMessages((prev) => [...prev.slice(0, -1), mappedUser, mappedAssistant]);
      if (nextChat) {
        setChats((prev) => {
          const exists = prev.some((c) => c.chat_id === nextChat.chat_id);
          if (!exists) return [nextChat, ...prev];
          return [nextChat, ...prev.filter((c) => c.chat_id !== nextChat.chat_id)];
        });
        setActiveChatId(nextChat.chat_id);
      }
    } catch (err: any) {
      setMessages((prev) => prev.slice(0, -1));
      toast.error(err?.response?.data?.detail || 'Chat failed');
    } finally {
      setSending(false);
    }
  };

  const copyMessage = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text || '');
      toast.success('Copied');
    } catch {
      toast.error('Copy failed');
    }
  };

  const pinToReports = async (msg: ChatMessage) => {
    try {
      await engine.post('/pins', {
        type: 'report',
        chat_id: msg.chat_id,
        message_id: msg.message_id,
        payload: {
          text_markdown: msg.content,
          charts: (msg.charts || []).map((c) => ({
            title: c.title,
            plotly_json: JSON.stringify({ data: c.data || [], layout: c.layout || {} }),
          })),
          metadata: {
            chat_id: msg.chat_id,
            message_id: msg.message_id,
            timestamp: msg.created_at,
            model: msg.model || selectedModel,
          },
        },
      });
      toast.success('Pinned to Reports');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Pin failed');
    }
  };

  const pinToDashboard = async (msg: ChatMessage) => {
    try {
      const hasChart = (msg.charts || []).length > 0;
      if (hasChart) {
        const chart = msg.charts?.[0];
        addPinnedChart({
          id: `${msg.chat_id}::${msg.message_id}::chart`,
          sessionId,
          title: chart?.title || 'Chat Chart',
          plotly_json: JSON.stringify({ data: chart?.data || [], layout: chart?.layout || {} }),
          source: 'AI Chat',
          chartType: (chart?.data?.[0]?.type as string) || undefined,
        });
      } else {
        addPinnedCard({
          id: `${msg.chat_id}::${msg.message_id}::card`,
          sessionId,
          label: 'AI Chat Insight',
          value: msg.content.length > 180 ? `${msg.content.slice(0, 180)}...` : msg.content,
          source: 'AI Chat',
        });
      }

      await engine.post('/pins', {
        type: 'dashboard',
        chat_id: msg.chat_id,
        message_id: msg.message_id,
        payload: {
          text: msg.content,
          charts: (msg.charts || []).map((c) => ({
            title: c.title,
            plotly_json: JSON.stringify({ data: c.data || [], layout: c.layout || {} }),
          })),
          metadata: {
            chat_id: msg.chat_id,
            message_id: msg.message_id,
            timestamp: msg.created_at,
            model: msg.model || selectedModel,
          },
        },
      });
      toast.success('Pinned to Dashboard');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Pin failed');
    }
  };

  const saveBookmark = async (msg: ChatMessage) => {
    try {
      await engine.post('/pins', {
        type: 'bookmark',
        chat_id: msg.chat_id,
        message_id: msg.message_id,
        payload: {
          text: msg.content,
          timestamp: msg.created_at,
          model: msg.model || selectedModel,
        },
      });
      toast.success('Saved');
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Save failed');
    }
  };

  const downloadChartPng = async (divId: string, filename: string) => {
    try {
      // @ts-ignore
      const Plotly = await import('plotly.js-dist-min');
      const graph = document.getElementById(divId) as any;
      if (!graph) return toast.error('Chart not found');
      await Plotly.downloadImage(graph, { format: 'png', width: 1600, height: 900, filename });
    } catch {
      toast.error('PNG download failed');
    }
  };

  const downloadChartCsv = (chart: { data: any[]; layout: any }, filename: string) => {
    try {
      const rows: string[] = ['x,y'];
      for (const trace of chart.data || []) {
        const x = trace?.x || [];
        const y = trace?.y || [];
        const len = Math.max(x.length || 0, y.length || 0);
        for (let i = 0; i < len; i += 1) {
          rows.push(`${JSON.stringify(x[i] ?? '')},${JSON.stringify(y[i] ?? '')}`);
        }
      }
      const blob = new Blob([rows.join('\n')], { type: 'text/csv;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      toast.error('CSV download failed');
    }
  };

  const chatDisplay = (chat: ChatSummary) => (chat.chat_name || 'New Chat').trim() || 'New Chat';

  return (
    <div className="animate-fade-in h-[calc(100vh-2rem)] grid grid-cols-12 gap-4">
      <Card className="col-span-12 lg:col-span-3 h-full flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold">Chats</h2>
          <Button size="sm" icon={<Plus className="w-4 h-4" />} onClick={createChat}>New Chat</Button>
        </div>

        {loadingChats ? (
          <div className="flex items-center justify-center py-8"><Spinner /></div>
        ) : (
          <div className="space-y-2 overflow-y-auto pr-1">
            {chats.length === 0 && <p className="text-xs text-[var(--text-muted)]">No chats yet</p>}
            {chats.map((chat) => (
              <div key={chat.chat_id} className={`rounded-lg border p-2 ${activeChatId === chat.chat_id ? 'border-[var(--accent)] bg-[var(--accent-bg)]' : 'border-[var(--border)] bg-[var(--bg-secondary)]'}`}>
                {renamingChatId === chat.chat_id ? (
                  <div className="flex items-center gap-1">
                    <Input value={renameValue} onChange={setRenameValue} />
                    <Button size="sm" variant="ghost" icon={<Check className="w-4 h-4" />} onClick={() => renameChat(chat.chat_id, renameValue)} />
                  </div>
                ) : (
                  <>
                    <button className="w-full text-left" onClick={() => setActiveChatId(chat.chat_id)}>
                      <p className="text-sm font-medium truncate">{chatDisplay(chat)}</p>
                      <p className="text-[10px] text-[var(--text-muted)]">{new Date(chat.updated_at).toLocaleString()}</p>
                    </button>
                    <div className="flex items-center gap-1 mt-2">
                      <Button size="sm" variant="ghost" icon={<Edit2 className="w-3.5 h-3.5" />} onClick={() => { setRenamingChatId(chat.chat_id); setRenameValue(chatDisplay(chat)); }} />
                      <Button size="sm" variant="ghost" icon={<Trash2 className="w-3.5 h-3.5" />} onClick={() => deleteChat(chat.chat_id)} />
                    </div>
                  </>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>

      <div className="col-span-12 lg:col-span-9 h-full flex flex-col">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h1 className="text-2xl font-bold">AI Chat</h1>
            <p className="text-[var(--text-secondary)] text-sm">Persistent multi-chat assistant with model picker</p>
          </div>
          <div className="flex items-center gap-2">
            <Select value={selectedModel} onChange={updateModelForActiveChat} options={models.map((m) => ({ value: m, label: m }))} />
            <Button variant="ghost" icon={<Trash2 className="w-4 h-4" />} onClick={clearMessages}>Clear Chat</Button>
          </div>
        </div>

        <Card className="mb-3 py-3">
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-6 gap-2">
            <Select
              label="Analysis"
              value={analysisMode}
              onChange={(v) => setAnalysisMode(v as 'fast' | 'balanced' | 'deep')}
              options={[
                { value: 'fast', label: 'Fast' },
                { value: 'balanced', label: 'Balanced' },
                { value: 'deep', label: 'Deep' },
              ]}
            />
            <Select
              label="Response"
              value={responseStyle}
              onChange={(v) => setResponseStyle(v as 'executive' | 'technical' | 'deep')}
              options={[
                { value: 'executive', label: 'Executive' },
                { value: 'technical', label: 'Technical' },
                { value: 'deep', label: 'Deep Expert' },
              ]}
            />
            <Select
              label="Strict"
              value={strictMode}
              onChange={(v) => setStrictMode(v as 'off' | 'on')}
              options={[
                { value: 'off', label: 'Off' },
                { value: 'on', label: 'On (evidence)' },
              ]}
            />
            <Select
              label="Tables"
              value={tableScope}
              onChange={(v) => setTableScope(v as 'all' | 'working')}
              options={[
                { value: 'all', label: 'All Tables' },
                { value: 'working', label: 'Working Only' },
              ]}
            />
            <Select
              label="Auto Join"
              value={joinMode}
              onChange={(v) => setJoinMode(v as 'on' | 'off')}
              options={[
                { value: 'on', label: 'On' },
                { value: 'off', label: 'Off' },
              ]}
            />
            <Input
              label="Temperature"
              value={temperature}
              onChange={setTemperature}
              type="number"
              step="0.1"
              min="0"
              max="1"
              placeholder="0.0"
            />
          </div>
        </Card>

        <Card className="flex-1 min-h-0 flex flex-col">
          <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-4 pr-2 scrollbar-thin">
            {loadingMessages && <div className="flex items-center justify-center py-12"><Spinner /></div>}

            {!loadingMessages && messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center py-12">
                <div className="w-16 h-16 rounded-2xl bg-[var(--accent-bg)] flex items-center justify-center mb-4">
                  <Sparkles className="w-8 h-8 text-[var(--accent)]" />
                </div>
                <h3 className="text-lg font-semibold mb-2">Start a Conversation</h3>
                <p className="text-sm text-[var(--text-muted)] max-w-md">Create or select a chat from the left sidebar, then ask anything about your data.</p>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={msg.message_id || i} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 rounded-lg bg-[var(--accent-bg)] flex items-center justify-center shrink-0">
                    <Bot className="w-4 h-4 text-[var(--accent)]" />
                  </div>
                )}

                <div className={`max-w-[82%] ${msg.role === 'user' ? 'order-first' : ''}`}>
                  <div className={`rounded-2xl px-4 py-3 text-sm ${
                    msg.role === 'user'
                      ? 'gradient-accent text-white rounded-br-md'
                      : 'bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-bl-md'
                  }`}>
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  </div>

                  {msg.role === 'assistant' && (
                    <div className="flex items-center flex-wrap gap-1 mt-1">
                      <Button size="sm" variant="ghost" icon={<Copy className="w-3.5 h-3.5" />} onClick={() => copyMessage(msg.content)}>Copy</Button>
                      <Button size="sm" variant="ghost" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinToReports(msg)}>Pin Reports</Button>
                      <Button size="sm" variant="ghost" icon={<Pin className="w-3.5 h-3.5" />} onClick={() => pinToDashboard(msg)}>Pin Dashboard</Button>
                      <Button size="sm" variant="ghost" icon={<Bookmark className="w-3.5 h-3.5" />} onClick={() => saveBookmark(msg)}>Save</Button>
                    </div>
                  )}

                  {(msg.charts || []).map((chart, ci) => {
                    const divId = `chat-plot-${msg.message_id}-${ci}`;
                    return (
                      <Card key={divId} className="mt-2">
                        <div className="flex items-center justify-between mb-2">
                          <p className="text-xs text-[var(--text-secondary)]">{chart.title || `Chart ${ci + 1}`}</p>
                          <div className="flex items-center gap-1">
                            <Button size="sm" variant="ghost" icon={<Download className="w-3.5 h-3.5" />} onClick={() => downloadChartPng(divId, `chart_${msg.message_id}_${ci}`)}>Download PNG</Button>
                            <Button size="sm" variant="ghost" icon={<Download className="w-3.5 h-3.5" />} onClick={() => downloadChartCsv(chart, `chart_${msg.message_id}_${ci}.csv`)}>CSV</Button>
                          </div>
                        </div>
                        <div className="rounded-lg overflow-hidden" style={{ minHeight: 260 }}>
                          <Plot
                            divId={divId}
                            data={chart.data}
                            layout={{ ...(chart.layout || {}), autosize: true }}
                            config={{ responsive: true, displayModeBar: true }}
                            style={{ width: '100%', height: '100%' }}
                            useResizeHandler
                          />
                        </div>
                      </Card>
                    );
                  })}

                  <p className="text-[10px] text-[var(--text-muted)] mt-1 px-1">
                    {new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>

                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-lg bg-[var(--accent)] flex items-center justify-center shrink-0">
                    <User className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            ))}

            {sending && (
              <div className="flex gap-3">
                <div className="w-8 h-8 rounded-lg bg-[var(--accent-bg)] flex items-center justify-center">
                  <Bot className="w-4 h-4 text-[var(--accent)]" />
                </div>
                <div className="rounded-2xl px-4 py-3 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-bl-md">
                  <div className="flex items-center gap-2">
                    <Spinner size="sm" />
                    <span className="text-sm text-[var(--text-muted)]">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="mt-3 shrink-0 glass rounded-2xl p-3 flex items-end gap-3">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder="Ask about your data..."
              rows={1}
              className="flex-1 bg-transparent border-none outline-none resize-none text-sm max-h-32"
              style={{ minHeight: '40px' }}
            />
            <Button onClick={sendMessage} disabled={!input.trim() || sending} icon={<Send className="w-4 h-4" />} className="shrink-0">
              Send
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}
