import React, { useState, useRef, useEffect } from 'react';
import {
  Upload,
  Send,
  Compass,
  Eye,
  Flame,
  Layers,
  Sparkles,
  Trash2,
  ShieldAlert,
  ShieldCheck,
  Crosshair,
  MessageSquare,
  Cpu,
  Image as ImageIcon,
} from 'lucide-react';

export const API_ENDPOINT = 'https://arrival-viewless-spill.ngrok-free.dev/api/analyze';

interface ChatMessage {
  id: string;
  question: string;
  answer: string;
  timestamp: string;
  predicted_class?: string;
  confidence?: number;
  bbox?: string;
}

interface TelemetryMeta {
  predicted_class: string;
  confidence: number;
  damage_prob: number;
  nodamage_prob: number;
  bbox: string;
  resolution: string;
}

const QUICK_PROMPTS = [
  'Describe what is visible in this image',
  'Count how many houses or structures are affected',
  'Is any human, boat, or vehicle visible?',
  'Explain the Grad-CAM red anomaly zone',
];

export default function App() {
  const [image, setImage] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'split' | 'heatmap' | 'original'>('split');
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [meta, setMeta] = useState<TelemetryMeta | null>(null);

  const chatEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage(file);
      setPreview(URL.createObjectURL(file));
      setHeatmap(null);
      setMessages([]); // Reset chat history ONLY when a new image is uploaded
      setMeta(null);
    }
  };

  const handleQuery = async (customPrompt?: string) => {
    const activePrompt = (customPrompt ?? prompt).trim();
    if (!image || !activePrompt || loading) return;

    setPrompt('');
    setLoading(true);

    // Format previous Q&A history for multi-turn context
    const historyContext = messages
      .slice(-3)
      .map((m) => `User: ${m.question}\nAI: ${m.answer}`)
      .join('\n');

    const formData = new FormData();
    formData.append('image', image);
    formData.append('prompt', activePrompt);
    formData.append('history', historyContext);

    try {
      const res = await fetch(API_ENDPOINT, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();

      if (data.heatmap) {
        setHeatmap(data.heatmap);
      }

      setMeta({
        predicted_class: data.predicted_class ?? 'unknown',
        confidence: data.confidence ?? 0,
        damage_prob: data.damage_prob ?? 0,
        nodamage_prob: data.nodamage_prob ?? 0,
        bbox: data.bbox ?? 'Full Tile',
        resolution: data.resolution ?? '224×224 px',
      });

      const newMsg: ChatMessage = {
        id: Date.now().toString(),
        question: activePrompt,
        answer: data.answer || 'No response received from model.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        predicted_class: data.predicted_class,
        confidence: data.confidence,
        bbox: data.bbox,
      };

      setMessages((prev) => [...prev, newMsg]);
    } catch (err) {
      const errMsg: ChatMessage = {
        id: Date.now().toString(),
        question: activePrompt,
        answer:
          '⚠️ Connection error: Could not reach the Colab GPU backend. Verify that Cell 6 is running in Google Colab and that API_ENDPOINT matches your active ngrok URL.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  };

  const isDamage = meta?.predicted_class === 'damage';

  return (
    <div className="relative flex h-screen w-screen bg-[#040711] text-slate-100 font-sans overflow-hidden select-none">
      {/* Ambient Glowing Background Orbs */}
      <div className="pointer-events-none fixed -top-32 -left-32 w-96 h-96 rounded-full bg-emerald-500/10 blur-[130px]" />
      <div className="pointer-events-none fixed bottom-0 right-1/3 w-96 h-96 rounded-full bg-indigo-500/10 blur-[140px]" />
      <div className="pointer-events-none fixed top-1/4 right-0 w-80 h-80 rounded-full bg-cyan-500/10 blur-[120px]" />

      {/* LEFT PANEL: Satellite & Grad-CAM Interactive Viewport */}
      <div className="flex-1 flex flex-col border-r border-slate-800/80 p-6 h-full z-10">
        {/* Top Navbar */}
        <div className="flex items-center justify-between mb-5 bg-slate-900/60 backdrop-blur-md border border-slate-800/90 px-5 py-3.5 rounded-2xl shadow-lg">
          <div className="flex items-center gap-3.5">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border border-emerald-500/40 flex items-center justify-center shadow-inner">
              <Compass className="w-6 h-6 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <span className="text-lg font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-emerald-300 bg-clip-text text-transparent">
                  SatQuery AI
                </span>
                <span className="text-[11px] font-semibold uppercase tracking-wider px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  ResNet-18 CNN + VLM
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Trained on 10,000 Hurricane Harvey Satellite Tiles • 97.80% Unseen Test Accuracy
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Viewport Mode Switcher */}
            {heatmap && (
              <div className="flex items-center bg-slate-950/90 p-1 rounded-xl border border-slate-800 text-xs">
                <button
                  onClick={() => setViewMode('split')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
                    viewMode === 'split'
                      ? 'bg-indigo-600 text-white shadow'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Split View
                </button>
                <button
                  onClick={() => setViewMode('heatmap')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
                    viewMode === 'heatmap'
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Grad-CAM Only
                </button>
                <button
                  onClick={() => setViewMode('original')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
                    viewMode === 'original'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Original
                </button>
              </div>
            )}

            <label className="cursor-pointer bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 px-4 py-2.5 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all shadow-lg shadow-emerald-950/50 text-white border border-emerald-400/30">
              <Upload className="w-4 h-4" />
              Upload Aerial Tile
              <input type="file" accept="image/*" className="hidden" onChange={handleImageChange} />
            </label>
          </div>
        </div>

        {/* Main Satellite Canvas */}
        <div className="flex-1 bg-slate-900/40 backdrop-blur-sm rounded-2xl border border-slate-800/90 flex flex-col items-center justify-center overflow-hidden relative shadow-2xl p-6">
          {preview ? (
            <div className="flex items-center justify-center gap-7 w-full h-full">
              {/* Card 1: Input Satellite Tile */}
              {(viewMode === 'split' || viewMode === 'original' || !heatmap) && (
                <div className="flex flex-col items-center">
                  <div className="flex items-center gap-2 mb-2.5">
                    <ImageIcon className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                      1. Input Optical Satellite Tile
                    </span>
                  </div>
                  <div
                    className={`${
                      viewMode === 'original' || !heatmap ? 'w-[440px] h-[440px]' : 'w-[350px] h-[350px]'
                    } bg-slate-950/90 rounded-2xl border border-slate-700/80 p-2.5 flex items-center justify-center shadow-2xl transition-all duration-300`}
                  >
                    <img
                      src={preview}
                      alt="Original Satellite Tile"
                      className="w-full h-full object-contain rounded-xl"
                    />
                  </div>
                </div>
              )}

              {/* Card 2: Grad-CAM Grounding + BBox */}
              {heatmap && (viewMode === 'split' || viewMode === 'heatmap') && (
                <div className="flex flex-col items-center">
                  <div className="flex items-center gap-1.5 mb-2.5">
                    <Flame className="w-4 h-4 text-amber-400 animate-pulse" />
                    <span className="text-xs font-semibold uppercase tracking-wider text-amber-300">
                      2. ResNet-18 Grad-CAM Activation + BBox
                    </span>
                  </div>
                  <div
                    className={`${
                      viewMode === 'heatmap' ? 'w-[440px] h-[440px]' : 'w-[350px] h-[350px]'
                    } bg-slate-950/90 rounded-2xl border-2 border-amber-500/60 p-2.5 flex items-center justify-center shadow-2xl shadow-amber-950/30 transition-all duration-300`}
                  >
                    <img
                      src={heatmap}
                      alt="Grad-CAM Heatmap"
                      className="w-full h-full object-contain rounded-xl"
                    />
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center max-w-md p-8 rounded-2xl bg-slate-900/50 border border-dashed border-slate-700/80">
              <div className="w-16 h-16 rounded-2xl bg-slate-800/80 border border-slate-700 flex items-center justify-center mx-auto mb-4 shadow-inner">
                <Layers className="w-8 h-8 text-emerald-400 animate-pulse" />
              </div>
              <p className="text-slate-200 text-base font-semibold">
                Upload a Satellite or Drone Image to Start Analysis
              </p>
              <p className="text-slate-400 text-xs mt-2 leading-relaxed">
                Supports Hurricane Harvey test tiles, flood imagery, harbors, and urban sectors. Your custom
                ResNet-18 CNN generates Grad-CAM heatmaps while the Vision-Language engine answers your questions.
              </p>
            </div>
          )}

          {/* Bottom Telemetry Status Bar inside Canvas */}
          {meta && (
            <div className="mt-4 w-full max-w-3xl bg-slate-950/85 border border-slate-800 px-4 py-2.5 rounded-xl flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <Crosshair className="w-4 h-4 text-emerald-400" />
                <span className="text-slate-400">Primary Anomaly BBox:</span>
                <span className="font-mono text-emerald-300 font-semibold">{meta.bbox}</span>
              </div>
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-indigo-400" />
                <span className="text-slate-400">Tile Size:</span>
                <span className="font-mono text-slate-200">{meta.resolution}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                <span className="text-slate-300 font-medium">Colab T4 GPU Active</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* RIGHT PANEL: Persistent Multi-Turn Chat & CNN Telemetry Feed */}
      <div className="w-[480px] flex flex-col p-6 bg-slate-900/75 backdrop-blur-xl h-full z-10">
        {/* Right Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2.5">
            <Eye className="w-5 h-5 text-indigo-400" />
            <span className="text-base font-bold tracking-tight text-white">
              CNN Telemetry & Conversational Feed
            </span>
          </div>
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              className="text-xs text-slate-400 hover:text-rose-400 flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-800/70 border border-slate-700/70 transition-colors cursor-pointer"
              title="Clear conversation history"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear Chat
            </button>
          )}
        </div>

        {/* Live CNN Telemetry Cards (Fixed at top when image has been analyzed) */}
        {meta && (
          <div className="mb-4 bg-slate-950/80 border border-slate-800 rounded-2xl p-3.5 shadow-lg">
            <div className="grid grid-cols-2 gap-3 mb-3">
              <div
                className={`p-3 rounded-xl border flex items-center justify-between ${
                  isDamage
                    ? 'bg-rose-950/30 border-rose-500/40 text-rose-300'
                    : 'bg-emerald-950/30 border-emerald-500/40 text-emerald-300'
                }`}
              >
                <div>
                  <p className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">
                    ResNet-18 Verdict
                  </p>
                  <p className="text-base font-extrabold mt-0.5">
                    {meta.predicted_class.replace('_', ' ').toUpperCase()}
                  </p>
                </div>
                {isDamage ? (
                  <ShieldAlert className="w-6 h-6 text-rose-400 shrink-0" />
                ) : (
                  <ShieldCheck className="w-6 h-6 text-emerald-400 shrink-0" />
                )}
              </div>

              <div className="bg-slate-900/90 border border-slate-800 p-3 rounded-xl flex flex-col justify-center">
                <p className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">
                  CNN Confidence
                </p>
                <div className="flex items-baseline justify-between mt-0.5">
                  <p className="text-lg font-extrabold text-white">{meta.confidence}%</p>
                  <span className="text-[10px] text-indigo-400 font-medium">Softmax Prob</span>
                </div>
              </div>
            </div>

            {/* Probability Distribution Bar */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] font-medium">
                <span className="text-rose-400">Damage: {meta.damage_prob}%</span>
                <span className="text-emerald-400">No Damage: {meta.nodamage_prob}%</span>
              </div>
              <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden flex">
                <div
                  className="bg-gradient-to-r from-rose-500 to-amber-500 h-full transition-all duration-500"
                  style={{ width: `${meta.damage_prob}%` }}
                />
                <div
                  className="bg-gradient-to-r from-emerald-500 to-teal-400 h-full transition-all duration-500"
                  style={{ width: `${meta.nodamage_prob}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Scrollable Multi-Turn Conversation Feed */}
        <div className="flex-1 overflow-y-auto space-y-4 pr-1.5 select-text">
          {messages.length === 0 && !loading && (
            <div className="bg-slate-800/40 border border-slate-800 p-4 rounded-2xl text-xs text-slate-300 leading-relaxed space-y-2">
              <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
                <Sparkles className="w-4 h-4" />
                Multi-Turn Satellite Assistant Ready
              </div>
              <p className="text-slate-400">
                Upload an aerial tile and ask as many questions as you want! Previous questions and answers stay
                saved in this chat thread until you upload a new picture.
              </p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div key={msg.id} className="space-y-2.5">
              {/* User Question Bubble */}
              <div className="flex justify-end">
                <div className="max-w-[85%] bg-gradient-to-br from-indigo-600 to-violet-600 text-white px-4 py-2.5 rounded-2xl rounded-tr-xs text-xs font-medium shadow-md">
                  <div className="flex items-center justify-between gap-3 mb-1 opacity-75 text-[10px]">
                    <span>Question #{index + 1}</span>
                    <span>{msg.timestamp}</span>
                  </div>
                  {msg.question}
                </div>
              </div>

              {/* AI Answer Bubble */}
              <div className="flex justify-start">
                <div className="w-full bg-slate-800/90 border border-slate-700/80 p-4 rounded-2xl rounded-tl-xs text-xs leading-relaxed text-slate-100 shadow-lg space-y-2">
                  <div className="flex items-center justify-between border-b border-slate-700/70 pb-1.5">
                    <span className="font-bold text-emerald-400 flex items-center gap-1.5">
                      <MessageSquare className="w-3.5 h-3.5" />
                      SatQuery AI Analysis
                    </span>
                    {msg.predicted_class && (
                      <span
                        className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                          msg.predicted_class === 'damage'
                            ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                            : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                        }`}
                      >
                        CNN: {msg.predicted_class.toUpperCase()} ({msg.confidence}%)
                      </span>
                    )}
                  </div>
                  <div className="whitespace-pre-wrap text-slate-200">{msg.answer}</div>
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="bg-slate-800/60 border border-slate-700/80 p-4 rounded-2xl flex items-center gap-3">
              <div className="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin shrink-0" />
              <p className="text-slate-200 text-xs animate-pulse">
                Running ResNet-18 Grad-CAM + Vision-Language reasoning on Colab GPU...
              </p>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Quick-Prompt Suggestion Pills */}
        {image && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {QUICK_PROMPTS.map((q) => (
              <button
                key={q}
                disabled={loading}
                onClick={() => handleQuery(q)}
                className="text-[11px] bg-slate-800/90 hover:bg-slate-700/90 text-slate-300 hover:text-emerald-300 border border-slate-700/80 px-2.5 py-1 rounded-full transition-all cursor-pointer disabled:opacity-40"
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {/* Bottom Chat Input Box */}
        <div className="mt-3 flex gap-2">
          <input
            type="text"
            placeholder={
              image
                ? 'Ask a follow-up question about this satellite image...'
                : 'Upload an image first to ask questions...'
            }
            value={prompt}
            disabled={!image || loading}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
            className="flex-1 bg-slate-950/90 border border-slate-700/90 rounded-xl px-4 py-2.5 text-xs focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500 shadow-inner disabled:opacity-50"
          />
          <button
            onClick={() => handleQuery()}
            disabled={loading || !image || !prompt.trim()}
            className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-40 px-4 py-2.5 rounded-xl text-xs font-semibold flex items-center justify-center transition-all shadow-lg cursor-pointer text-white"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}