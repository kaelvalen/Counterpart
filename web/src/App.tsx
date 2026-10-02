import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ImageGeneration } from "@/components/agents/image-generation";
import { TodoList, type TodoItem } from "@/components/agents/todo-list";
import { ToolResult } from "@/components/agents/tool-result";
import { MorphingLightbox, type LightboxImage } from "@/components/motion/morphing-lightbox";

type Stage = { label: string };
type JobResult = {
  images: Record<string, string>;
  candidates: { idx: number; score: number; url: string; best: boolean }[];
  best_idx: number;
  best_score: number;
  mean_uncertainty: number;
  n: number;
  hand_mask: boolean;
};
type Job = {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  stage_label?: string;
  progress: number;
  error?: string | null;
  result?: JobResult | null;
};

const STEPS = ["Görsel", "Eksik bölge", "Üret & seç"];

function useGpu() {
  const [gpu, setGpu] = useState<string | null | undefined>(undefined);
  useEffect(() => {
    fetch("/api/meta")
      .then((r) => r.json())
      .then((m) => setGpu(m.gpu ?? null))
      .catch(() => setGpu(null));
  }, []);
  return gpu;
}

async function maskToFile(canvas: HTMLCanvasElement): Promise<File> {
  const off = document.createElement("canvas");
  off.width = canvas.width;
  off.height = canvas.height;
  const ctx = off.getContext("2d")!;
  ctx.drawImage(canvas, 0, 0);
  const data = ctx.getImageData(0, 0, off.width, off.height).data;
  const out = ctx.createImageData(off.width, off.height);
  for (let i = 0; i < data.length; i += 4) {
    const v = data[i] > 235 && data[i + 1] > 235 && data[i + 2] > 235 ? 255 : 0;
    out.data[i] = out.data[i + 1] = out.data[i + 2] = v;
    out.data[i + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
  const blob: Blob = await new Promise((res) => off.toBlob((b) => res(b!), "image/png"));
  return new File([blob], "mask.png", { type: "image/png" });
}

export default function App() {
  const gpu = useGpu();
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState("");
  const [n, setN] = useState(8);
  const [brush, setBrush] = useState(26);
  const [dirty, setDirty] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const baseRef = useRef<ImageData | null>(null);
  const painting = useRef(false);
  const last = useRef({ x: 0, y: 0 });
  const timer = useRef<number | null>(null);

  const running = job?.status === "queued" || job?.status === "running";
  const step = job?.status === "done" ? 3 : file ? (dirty ? 3 : 2) : 1;

  /* ---------- image loading ---------- */
  const loadFile = useCallback((f: File) => {
    setFile(f);
    setDirty(false);
    setJob(null);
    setError(null);
    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      baseRef.current = ctx.getImageData(0, 0, canvas.width, canvas.height);
    };
    img.src = URL.createObjectURL(f);
  }, []);

  /* ---------- mask painting ---------- */
  const pointerPos = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const r = canvas.getBoundingClientRect();
    return {
      x: ((ev.clientX - r.left) * canvas.width) / r.width,
      y: ((ev.clientY - r.top) * canvas.height) / r.height,
    };
  };
  const paint = (p: { x: number; y: number }) => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    ctx.strokeStyle = "#fff";
    ctx.lineCap = "round";
    ctx.lineWidth = (brush * canvas.width) / 380;
    ctx.beginPath();
    ctx.moveTo(last.current.x, last.current.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    setDirty(true);
  };
  const onDown = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    painting.current = true;
    last.current = pointerPos(ev);
    paint(last.current);
    ev.currentTarget.setPointerCapture(ev.pointerId);
  };
  const onMove = (ev: React.PointerEvent<HTMLCanvasElement>) => {
    if (painting.current) paint(pointerPos(ev));
  };
  const clearMask = () => {
    const canvas = canvasRef.current;
    if (!canvas || !baseRef.current) return;
    canvas.getContext("2d")!.putImageData(baseRef.current, 0, 0);
    setDirty(false);
  };

  /* ---------- job ---------- */
  const poll = useCallback((id: string) => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = window.setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${id}`);
        const data: Job = await res.json();
        setJob(data);
        if (data.status === "done" || data.status === "error") {
          if (timer.current) window.clearInterval(timer.current);
          if (data.status === "error") setError(data.error ?? "bilinmeyen hata");
        }
      } catch {
        /* transient; next tick retries */
      }
    }, 1200);
  }, []);

  useEffect(() => () => void (timer.current && window.clearInterval(timer.current)), []);

  const run = async () => {
    if (!file) return;
    setError(null);
    setJob({ id: "", status: "queued", stage: "queued", progress: 0 });
    const fd = new FormData();
    fd.append("image", file);
    if (dirty && canvasRef.current) fd.append("mask", await maskToFile(canvasRef.current));
    fd.append("prompt", prompt);
    fd.append("n", String(n));
    try {
      const res = await fetch("/api/jobs", { method: "POST", body: fd });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.detail ?? res.statusText);
      }
      const { job_id } = await res.json();
      poll(job_id);
    } catch (e) {
      setJob(null);
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const reset = () => {
    if (timer.current) window.clearInterval(timer.current);
    setFile(null);
    setJob(null);
    setError(null);
    setDirty(false);
    baseRef.current = null;
  };

  const pct = Math.round((job?.progress ?? 0) * 100);
  const result = job?.status === "done" ? job.result ?? null : null;

  const stageItems: TodoItem[] = useMemo(() => {
    const stage = job?.stage ?? "";
    const done = job?.status === "done";
    const order = ["prepare", "generate", "score", "uncertainty", "done"];
    const current = stage ? order.indexOf(stage) : -1;
    const step = (index: number, title: string, progress?: number): TodoItem => ({
      id: title,
      title,
      status: done || current > index ? "completed" : current === index ? "in-progress" : "pending",
      progress: current === index ? progress : undefined,
    });
    return [
      step(0, "Görsel hazırlandı (kırpma + maske)"),
      step(1, "Adaylar üretiliyor", job?.progress),
      step(2, "Klasik terimlerle puanlama"),
      step(3, "Belirsizlik haritası ve seçim"),
    ];
  }, [job]);

  const generationStatus = useMemo(() => {
    if (!job) return "queued" as const;
    if (job.status === "error") return "error" as const;
    if (job.status === "done") return "complete" as const;
    if (job.status === "queued") return "queued" as const;
    return job.stage === "generate" ? ("generating" as const) : ("refining" as const);
  }, [job]);

  const lightboxImages: LightboxImage[] = useMemo(
    () =>
      (result?.candidates ?? []).map((c) => ({
        id: String(c.idx),
        src: c.url,
        alt: `aday ${c.idx}`,
        width: 512,
        height: 512,
        caption: `#${c.idx} · skor ${c.score}${c.best ? " · seçilen" : ""}`,
      })),
    [result],
  );

  return (
    <>
      <header>
        <div className="logo">c</div>
        <div className="brand">
          <b>counterpart</b>
          <span>hasarlı nesne rekonstrüksiyonu — aday üret · klasik skorla seç · belirsizliği haritala</span>
        </div>
        <div className="pill">
          {gpu === undefined ? (
            "durum: bağlanıyor…"
          ) : gpu ? (
            <>
              GPU: <b>{gpu.replace("NVIDIA ", "")}</b>
            </>
          ) : (
            "CPU modu"
          )}
        </div>
      </header>

      <main>
        <section className="card">
          <h2>Girdi</h2>
          <div className="steps">
            {STEPS.map((label, i) => (
              <span key={label} className={`step ${i + 1 <= step ? "on" : ""}`}>
                {i + 1} · {label}
              </span>
            ))}
          </div>

          {!file ? (
            <div
              className={`drop ${dragOver ? "over" : ""}`}
              onClick={() => document.getElementById("file-input")?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                const f = e.dataTransfer.files?.[0];
                if (f) loadFile(f);
              }}
            >
              Görseli buraya bırak ya da tıkla
              <br />
              <small>PNG / JPG · nesne arka plandan ayrılabilir olmalı</small>
            </div>
          ) : (
            <div className="canvas-wrap">
              <canvas
                ref={canvasRef}
                onPointerDown={onDown}
                onPointerMove={onMove}
                onPointerUp={() => (painting.current = false)}
              />
              <div className="tools">
                <span>Fırça</span>
                <input
                  type="range"
                  min={6}
                  max={80}
                  value={brush}
                  onChange={(e) => setBrush(Number(e.target.value))}
                />
                <button className="ghost" onClick={clearMask}>
                  Maskeyi temizle
                </button>
              </div>
              <div className="tools muted">
                Boyamazsan bölge otomatik bulunur (ayna farkı + silüet çukurluğu).
              </div>
            </div>
          )}
          <input
            id="file-input"
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => e.target.files?.[0] && loadFile(e.target.files[0])}
          />

          <label className="field">
            Prompt (opsiyonel)
            <input
              type="text"
              value={prompt}
              placeholder="a complete white ceramic vase, product photo, white background"
              onChange={(e) => setPrompt(e.target.value)}
            />
          </label>
          <label className="field">
            Aday sayısı: <b>{n}</b>
            <input type="range" min={2} max={16} value={n} onChange={(e) => setN(Number(e.target.value))} />
          </label>

          <div className="actions">
            <button className="primary" onClick={run} disabled={!file || running}>
              {running ? "Çalışıyor…" : "Rekonstrüksiyon"}
            </button>
            <button className="ghost" onClick={reset}>
              Sıfırla
            </button>
          </div>

          {error && <div className="err">Hata: {error}</div>}

          {(running || job?.status === "done") && (
            <div className="progress">
              <TodoList
                items={stageItems}
                title={
                  running
                    ? `İşlem hattı — ${job?.stage_label ?? "hazırlanıyor"} · ${pct}%`
                    : "İşlem hattı tamamlandı"
                }
                defaultOpen
                collapseOnComplete
              />
            </div>
          )}
        </section>

        <section className="card">
          <h2>Sonuç</h2>
          {!result && (
            <div className="muted">
              Sonuçlar burada görünecek: en iyi tamamlama, belirsizlik haritası ve aday galerisi.
            </div>
          )}
          {result && (
            <div>
              <div className="shots">
                <figure>
                  <ImageGeneration
                    status={generationStatus}
                    prompt={prompt || undefined}
                    resolution="512 × 512"
                    size="fluid"
                    label="en iyi tamamlama"
                  >
                    <img src={result.images.best} alt="en iyi tamamlama" />
                  </ImageGeneration>
                  <figcaption>en iyi tamamlama (klasik birleşim, mod A)</figcaption>
                </figure>
                <figure>
                  <img src={result.images.uncertainty} alt="belirsizlik" />
                  <figcaption>piksel belirsizliği (aday ayrışması)</figcaption>
                </figure>
              </div>

              <ToolResult
                tool="counterpart · score"
                title="Klasik birleşim (mod A)"
                status="success"
                kind="custom"
                meta={`${result.n} aday · maske: ${result.hand_mask ? "elle çizildi" : "otomatik"}`}
              >
                <div className="stats">
                  <div className="stat">
                    en iyi skor: <b>{result.best_score}</b>
                  </div>
                  <div className="stat">
                    ortalama belirsizlik: <b>{result.mean_uncertainty}</b>
                  </div>
                  <div className="stat">
                    seçilen aday: <b>#{result.best_idx}</b>
                  </div>
                  <div className="stat">
                    aday: <b>{result.n}</b>
                  </div>
                </div>
              </ToolResult>

              <h2 className="sub">Adaylar — karta tıkla, büyüt</h2>
              <MorphingLightbox images={lightboxImages} label="aday tamamlamaları" />
            </div>
          )}

          <div className="how">
            <h2>Sistem ne yapıyor?</h2>
            <table>
              <tbody>
                <tr>
                  <td>1 · Segment</td>
                  <td>nesne silüetini arka plandan ayırır (eşikleme / hull)</td>
                </tr>
                <tr>
                  <td>2 · Hasar</td>
                  <td>eksik bölgeyi bulur (çizdiğin maske ya da otomatik ipucu)</td>
                </tr>
                <tr>
                  <td>3 · Üretim</td>
                  <td>dondurulmuş SD1.5-inpainting ile N aday tamamlama</td>
                </tr>
                <tr>
                  <td>4 · Skorlama</td>
                  <td>6 klasik terim: sınır · ayna · dönme · doku · frekans · kontur</td>
                </tr>
                <tr>
                  <td>5 · Seçim</td>
                  <td>görüntü içi z-normalize + eşit ağırlık; belirsizlik = aday ayrışması</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </main>

      <footer>counterpart · deneysel demo — çıktılar "makul"dur, "doğru" olduğu iddia edilmez</footer>
    </>
  );
}
