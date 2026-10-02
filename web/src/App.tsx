import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Gauge, Image as ImageIcon, Link2, Sparkles, Upload, Wand2 } from "lucide-react";
import { ImageGeneration } from "@/components/agents/image-generation";
import { TodoList, type TodoItem } from "@/components/agents/todo-list";
import { ToolResult } from "@/components/agents/tool-result";
import { MorphingLightbox, type LightboxImage } from "@/components/motion/morphing-lightbox";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ types */

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

const STEPS = ["Görsel", "Maske", "Üretim"];
const PROMPT_EXAMPLES = [
  "a complete white ceramic vase, product photo, white background",
  "a complete intact object, product photo, white background",
  "",
];

/* ------------------------------------------------------------- primitives */

function Card({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      className={cn("rounded-2xl border border-border/70 bg-card/60 p-5 shadow-sm", className)}
      {...props}
    />
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
      {children}
    </div>
  );
}

function BeforeAfter({ before, after, label }: { before: string; after: string; label?: string }) {
  const [pos, setPos] = useState(52);
  return (
    <figure className="m-0">
      <div className="relative overflow-hidden rounded-xl border border-border/70 bg-muted/30">
        <img src={before} alt="hasarlı" className="block w-full" />
        <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}>
          <img src={after} alt="tamamlama" className="block w-full" />
        </div>
        <div className="absolute inset-y-0 w-px bg-white/80 shadow-[0_0_12px_rgba(255,255,255,.5)]" style={{ left: `${pos}%` }} />
        <span className="absolute left-3 top-3 rounded-md bg-black/55 px-2 py-0.5 text-[11px] font-medium text-white/90 backdrop-blur">
          hasarlı
        </span>
        <span className="absolute right-3 top-3 rounded-md bg-black/55 px-2 py-0.5 text-[11px] font-medium text-white/90 backdrop-blur">
          tamamlanmış
        </span>
        <input
          type="range"
          min={2}
          max={98}
          value={pos}
          onChange={(e) => setPos(Number(e.target.value))}
          className="absolute bottom-3 left-1/2 w-2/3 -translate-x-1/2 accent-primary"
          aria-label="karşılaştırma"
        />
      </div>
      {label && <figcaption className="mt-2 text-xs text-muted-foreground">{label}</figcaption>}
    </figure>
  );
}

function ScoreBars({ candidates }: { candidates: JobResult["candidates"] }) {
  const max = Math.max(...candidates.map((c) => c.score), 1e-6);
  return (
    <div className="space-y-1.5">
      {candidates.slice(0, 8).map((c) => (
        <div key={c.idx} className="flex items-center gap-3 text-xs">
          <span className={cn("w-9 tabular-nums", c.best ? "font-semibold text-foreground" : "text-muted-foreground")}>
            #{c.idx}
          </span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
            <div
              className={cn("h-full rounded-full", c.best ? "bg-emerald-400" : "bg-primary/70")}
              style={{ width: `${Math.max(2, (Math.max(c.score, 0) / max) * 100)}%` }}
            />
          </div>
          <span className="w-12 text-right tabular-nums text-muted-foreground">{c.score.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

/* -------------------------------------------------------------------- app */

export default function App() {
  const [gpu, setGpu] = useState<string | null | undefined>(undefined);
  const [file, setFile] = useState<File | null>(null);
  const [prompt, setPrompt] = useState(PROMPT_EXAMPLES[0]);
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

  useEffect(() => {
    fetch("/api/meta")
      .then((r) => r.json())
      .then((m) => setGpu(m.gpu ?? null))
      .catch(() => setGpu(null));
  }, []);

  const running = job?.status === "queued" || job?.status === "running";
  const result = job?.status === "done" ? (job.result ?? null) : null;
  const pct = Math.round((job?.progress ?? 0) * 100);

  /* ----------------------------------------------------------- image load */
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

  /* --------------------------------------------------------------- canvas */
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
  const clearMask = () => {
    const canvas = canvasRef.current;
    if (!canvas || !baseRef.current) return;
    canvas.getContext("2d")!.putImageData(baseRef.current, 0, 0);
    setDirty(false);
  };

  /* ------------------------------------------------------------------ job */
  const poll = useCallback((id: string) => {
    if (timer.current) window.clearInterval(timer.current);
    timer.current = window.setInterval(async () => {
      try {
        const data: Job = await (await fetch(`/api/jobs/${id}`)).json();
        setJob(data);
        if (data.status === "done" || data.status === "error") {
          if (timer.current) window.clearInterval(timer.current);
          if (data.status === "error") setError(data.error ?? "bilinmeyen hata");
        }
      } catch {
        /* transient */
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
      poll((await res.json()).job_id);
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
  };

  /* ------------------------------------------------------------ derived UI */
  const stageItems: TodoItem[] = useMemo(() => {
    const order = ["prepare", "generate", "score", "uncertainty", "done"];
    const current = job?.stage ? order.indexOf(job.stage) : -1;
    const done = job?.status === "done";
    const mk = (i: number, title: string, detail?: string, progress?: number): TodoItem => ({
      id: title,
      title,
      detail,
      status: done || current > i ? "completed" : current === i ? "in-progress" : "pending",
      progress: current === i ? progress : undefined,
    });
    return [
      mk(0, "Görüntü hazırlandı", "kırpma + hasar maskesi"),
      mk(1, "Aday tamamlamalar", `${n} farklı hipotez`, job?.progress),
      mk(2, "Klasik skorlama", "T1–T6 tutarlılık terimleri"),
      mk(3, "Seçim ve belirsizlik", "mod A birleşimi + U_pix"),
    ];
  }, [job, n]);

  const generationStatus = !job
    ? ("queued" as const)
    : job.status === "error"
      ? ("error" as const)
      : job.status === "done"
        ? ("complete" as const)
        : job.status === "queued"
          ? ("queued" as const)
          : job.stage === "generate"
            ? ("generating" as const)
            : ("refining" as const);

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

  const step = result ? 3 : file ? (dirty ? 3 : 2) : 1;

  /* --------------------------------------------------------------- render */
  return (
    <div className="min-h-dvh">
      {/* top bar */}
      <header className="sticky top-0 z-30 border-b border-border/60 bg-background/80 backdrop-blur-md">
        <div className="mx-auto flex h-15 max-w-[1440px] items-center gap-3 px-6">
          <div className="grid size-8 place-items-center rounded-xl bg-gradient-to-br from-primary to-emerald-400 font-bold text-primary-foreground">
            c
          </div>
          <div className="leading-tight">
            <div className="text-[15px] font-semibold tracking-tight">counterpart</div>
            <div className="text-[11.5px] text-muted-foreground">
              hasarlı nesne rekonstrüksiyonu · adaylar → klasik skor → seçim
            </div>
          </div>
          <nav className="ml-auto hidden items-center gap-4 text-xs text-muted-foreground md:flex">
            <span className="inline-flex items-center gap-1.5">
              <Gauge className="size-3.5" />
              {gpu === undefined ? "bağlanıyor…" : gpu ? gpu.replace("NVIDIA ", "") : "CPU modu"}
            </span>
            <a className="inline-flex items-center gap-1.5 hover:text-foreground" href="/api/meta" target="_blank" rel="noreferrer">
              <Link2 className="size-3.5" /> API
            </a>
          </nav>
        </div>
      </header>

      <div className="mx-auto grid max-w-[1440px] grid-cols-1 gap-6 px-6 py-6 lg:grid-cols-[370px_minmax(0,1fr)]">
        {/* ------------------------------------------------------- sidebar */}
        <aside className="space-y-4 lg:sticky lg:top-21 lg:self-start">
          <Card className="space-y-4">
            <div className="flex items-center gap-2">
              {STEPS.map((label, i) => (
                <div key={label} className="flex items-center gap-2">
                  <span
                    className={cn(
                      "rounded-full border px-2.5 py-1 text-[11px] font-medium",
                      i + 1 <= step
                        ? "border-primary/60 bg-primary/10 text-foreground"
                        : "border-border/70 text-muted-foreground",
                    )}
                  >
                    {i + 1} · {label}
                  </span>
                  {i < STEPS.length - 1 && <ArrowRight className="size-3 text-muted-foreground/50" />}
                </div>
              ))}
            </div>

            {!file ? (
              <button
                type="button"
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
                className={cn(
                  "flex w-full flex-col items-center gap-2 rounded-xl border border-dashed p-8 text-center transition",
                  dragOver ? "border-primary bg-primary/5" : "border-border/80 hover:border-primary/60 hover:bg-muted/30",
                )}
              >
                <Upload className="size-5 text-muted-foreground" />
                <span className="text-sm font-medium">Görseli bırak ya da seç</span>
                <span className="text-[11.5px] text-muted-foreground">PNG / JPG · nesne arka plandan ayrılabilmeli</span>
              </button>
            ) : (
              <div className="space-y-2">
                <canvas
                  ref={canvasRef}
                  onPointerDown={(ev) => {
                    painting.current = true;
                    last.current = pointerPos(ev);
                    paint(last.current);
                    ev.currentTarget.setPointerCapture(ev.pointerId);
                  }}
                  onPointerMove={(ev) => painting.current && paint(pointerPos(ev))}
                  onPointerUp={() => (painting.current = false)}
                  className="w-full cursor-crosshair rounded-xl border border-border/70"
                />
                <div className="flex items-center gap-2 text-[11.5px] text-muted-foreground">
                  <span>fırça</span>
                  <input type="range" min={6} max={80} value={brush} onChange={(e) => setBrush(Number(e.target.value))} className="flex-1 accent-primary" />
                  <button type="button" onClick={clearMask} className="rounded-md border border-border/70 px-2 py-1 hover:bg-muted/40">
                    temizle
                  </button>
                </div>
                <p className="text-[11.5px] text-muted-foreground">Boyamazsan bölge otomatik bulunur (ayna farkı + silüet çukurluğu).</p>
              </div>
            )}
            <input
              id="file-input"
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => e.target.files?.[0] && loadFile(e.target.files[0])}
            />

            <div className="space-y-2">
              <SectionLabel>Prompt</SectionLabel>
              <input
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="a complete … product photo, white background"
                className="w-full rounded-xl border border-border/70 bg-muted/20 px-3 py-2 text-sm outline-none placeholder:text-muted-foreground/70 focus:border-primary/60"
              />
              <div className="flex flex-wrap gap-1.5">
                {PROMPT_EXAMPLES.map((p, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => setPrompt(p)}
                    className="rounded-full border border-border/70 px-2 py-0.5 text-[10.5px] text-muted-foreground hover:border-primary/50 hover:text-foreground"
                  >
                    {p === "" ? "boş" : i === 0 ? "vazo" : "genel"}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <SectionLabel>Aday sayısı</SectionLabel>
                <span className="text-xs font-semibold tabular-nums">{n}</span>
              </div>
              <input type="range" min={2} max={16} value={n} onChange={(e) => setN(Number(e.target.value))} className="w-full accent-primary" />
            </div>

            <div className="flex gap-2">
              <button
                type="button"
                onClick={run}
                disabled={!file || running}
                className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-primary px-4 py-2.5 text-sm font-semibold text-primary-foreground transition hover:opacity-90 disabled:opacity-40"
              >
                <Wand2 className="size-4" />
                {running ? "Çalışıyor…" : "Rekonstrüksiyon"}
              </button>
              {file && (
                <button type="button" onClick={reset} className="rounded-xl border border-border/70 px-3 text-sm hover:bg-muted/40">
                  Sıfırla
                </button>
              )}
            </div>

            {error && (
              <div className="rounded-xl border border-destructive/50 bg-destructive/10 px-3 py-2 text-xs text-destructive-foreground">
                {error}
              </div>
            )}
          </Card>

          {(running || job?.status === "done") && (
            <Card className="p-0">
              <TodoList
                items={stageItems}
                title={running ? `İşlem hattı · ${pct}%` : "İşlem hattı tamamlandı"}
                defaultOpen
                collapseOnComplete
                className="p-4"
              />
            </Card>
          )}
        </aside>

        {/* ---------------------------------------------------------- main */}
        <main className="space-y-6">
          {!result && (
            <>
              <Card className="overflow-hidden p-0">
                <div className="grid gap-6 p-6 lg:grid-cols-[minmax(0,1fr)_320px] lg:items-center">
                  <div className="space-y-3">
                    <div className="inline-flex items-center gap-2 rounded-full border border-border/70 px-3 py-1 text-[11px] text-muted-foreground">
                      <Sparkles className="size-3.5 text-emerald-400" />
                      dondurulmuş difüzyon + eğitimsiz klasik seçici
                    </div>
                    <h1 className="text-2xl font-semibold tracking-tight">
                      Kırık nesnenin tek fotoğrafından
                      <br />
                      çok sayıda tamamlama, tek makul seçim
                    </h1>
                    <p className="max-w-xl text-sm text-muted-foreground">
                      Model N aday üretir; 6 klasik görüntü tutarlılık terimi (sınır, simetri, doku, frekans, kontur)
                      adayları puanlar ve en tutarlısını seçer. Adayların ayrıştığı pikseller belirsizliği verir.
                    </p>
                    <div className="flex flex-wrap gap-2 pt-1">
                      {["6 klasik terim · eğitim yok", "piksel belirsizliği", "mod çeşitliliği"].map((f) => (
                        <span key={f} className="rounded-full bg-muted/40 px-3 py-1 text-[11.5px] text-muted-foreground">
                          {f}
                        </span>
                      ))}
                    </div>
                    <p className="pt-1 text-xs text-muted-foreground">
                      Başlamak için soldan bir görsel yükle — ya da sağdaki örneği incele.
                    </p>
                  </div>
                  <BeforeAfter
                    before="/app/examples/vase-before.png"
                    after="/app/examples/vase-after.png"
                    label="örnek · sentetik vazo, 12 adayın en iyisi (sürükleyerek karşılaştır)"
                  />
                </div>
              </Card>

              <Card>
                <SectionLabel>Sistem ne yapıyor?</SectionLabel>
                <div className="mt-3 divide-y divide-border/60 text-sm">
                  {[
                    ["1 · Segment", "nesne silüeti arka plandan ayrılır (eşikleme / dışbükey zarf)"],
                    ["2 · Hasar", "eksik bölge: çizdiğin maske ya da otomatik ipucu (ayna farkı + çukurluk)"],
                    ["3 · Üretim", "SD1.5-inpainting ile N aday; her biri farklı seed/prompt/guidance"],
                    ["4 · Skorlama", "her aday T1–T6 ile puanlanır; görüntü içi z-normalize"],
                    ["5 · Seçim", "eşit ağırlıklı birleşim → en tutarlı aday; belirsizlik = aday std'si"],
                  ].map(([k, v]) => (
                    <div key={k} className="grid grid-cols-[130px_minmax(0,1fr)] gap-4 py-2">
                      <span className="font-medium">{k}</span>
                      <span className="text-muted-foreground">{v}</span>
                    </div>
                  ))}
                </div>
              </Card>
            </>
          )}

          {result && (
            <>
              <Card>
                <div className="mb-4 flex items-center justify-between">
                  <SectionLabel>Sonuç</SectionLabel>
                  <span className="text-xs text-muted-foreground">
                    {result.n} aday · maske: {result.hand_mask ? "elle çizildi" : "otomatik"}
                  </span>
                </div>
                <div className="grid gap-5 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
                  <BeforeAfter
                    before={result.images.damaged}
                    after={result.images.best}
                    label={`en iyi tamamlama (aday #${result.best_idx}) — sürükleyerek hasarlı hâlle karşılaştır`}
                  />
                  <div className="space-y-3">
                    <ImageGeneration
                      status={generationStatus}
                      prompt={prompt || undefined}
                      resolution="512 × 512"
                      size="fluid"
                      label="seçilen tamamlama"
                    >
                      <img src={result.images.best} alt="en iyi tamamlama" />
                    </ImageGeneration>
                    <figure className="m-0">
                      <img src={result.images.uncertainty} alt="belirsizlik" className="w-full rounded-xl border border-border/70" />
                      <figcaption className="mt-2 text-xs text-muted-foreground">
                        piksel belirsizliği — adaylar ne kadar ayrışıyorsa o kadar parlak
                      </figcaption>
                    </figure>
                  </div>
                </div>
              </Card>

              <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
                <Card>
                  <div className="mb-4 flex items-center justify-between">
                    <SectionLabel>Aday galerisi</SectionLabel>
                    <span className="text-xs text-muted-foreground">karta tıkla → büyüt</span>
                  </div>
                  <MorphingLightbox images={lightboxImages} label="aday tamamlamaları" />
                </Card>

                <div className="space-y-6">
                  <Card>
                    <SectionLabel>Sıralama (combined_A)</SectionLabel>
                    <div className="mt-4">
                      <ScoreBars candidates={result.candidates} />
                    </div>
                  </Card>
                  <ToolResult
                    tool="counterpart · score"
                    title="Klasik birleşim (mod A)"
                    status="success"
                    kind="custom"
                    meta={`${result.n} aday`}
                  >
                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div className="rounded-xl border border-border/70 bg-muted/20 p-3">
                        <div className="text-[11px] text-muted-foreground">en iyi skor</div>
                        <div className="text-lg font-semibold tabular-nums">{result.best_score}</div>
                      </div>
                      <div className="rounded-xl border border-border/70 bg-muted/20 p-3">
                        <div className="text-[11px] text-muted-foreground">ort. belirsizlik</div>
                        <div className="text-lg font-semibold tabular-nums">{result.mean_uncertainty}</div>
                      </div>
                    </div>
                  </ToolResult>
                </div>
              </div>
            </>
          )}
        </main>
      </div>

      <footer className="border-t border-border/60 py-6 text-center text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <ImageIcon className="size-3.5" />
          counterpart · deneysel demo — çıktılar "makul"dur, "doğru" olduğu iddia edilmez
        </span>
      </footer>
    </div>
  );
}

/* ------------------------------------------------------------------ mask */

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
