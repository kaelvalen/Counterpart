# NOTES — geliştirme günlüğü

Faz sonlarında kısa girdi: ne yapıldı, ne ölçüldü, açık sorular (SPEC.md §13).

## Faz 0 — Ortam ✅ (2026-09-28)

**Kararlar (Kael, 2026-09-28):**
- Teslim tarihi projeyi bağlamıyor; kapsam = spec'in tamamı, opsiyoneller dahil
  (E9 gerçek hasar seti, Gradio demosu, PatchMatch/Criminisi, §5.2.3 stretch).
- Ağırlık modu: A ana sonuç, B ek sonuç.
- Paket adı: `counterpart-cv` (import: `counterpart`).
- Generator: SD1.5 inpainting ana; SD2-inpainting gated (401), HF hesabı açılırsa
  opsiyonel ikinci model.

**Doğrulamalar (2026-09-28):**
- `stabilityai/stable-diffusion-2-inpainting`: web 401 (gated), anonim erişim yok.
- `stable-diffusion-v1-5/stable-diffusion-inpainting`: 200 (açık).
- `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`: 200 (E7 için hazır).
- ABO: `abo-images-small.tar` 3.25 GB, `abo-listings.tar` 87 MB, `images.csv.gz` 6.4 MB — auth'suz.
  Ayrıca **tek tek orijinal çözünürlükte erişim** doğrulandı: `images/original/<path>` (256px'lik
  "small" yerine). Demo örnekleri bu yolla indirildi.
- PyPI'da `sam2`, `PyPatchMatch`, `lpips`, `open_clip_torch` mevcut.
- Makine: RTX 5060 Laptop 8151 MiB, driver 615.71.09, cc 12.0; NixOS 26.11; 634 GB boş disk.

**Ortam:**
- flake devShell: python 3.12.14, uv 0.12.17; `LD_LIBRARY_PATH` içinde `/run/opengl-driver/lib`.
- nix-direnv: `~/.config/direnv/direnvrc` eksikti, `lib/hm-nix-direnv.sh`'e bağlandı; `use flake` çalışıyor.
- torch **2.11.0+cu128** (Blackwell cc 12.0 doğrulandı, fp16 matmul çalışıyor).

**Ölçümler (check_env, 25 adım, 512×512):**
- fp16 matmul: **35.3 TFLOP/s**
- Pipeline yükleme: 130.9 s (ilk indirme dahil; ağırlık 2.02 GiB)
- Üretim: **3.38 s/görüntü** (std 0.03) — tek aday, 25 adım
- Peak VRAM: **2.61 GiB** (8 GiB'a bol; batch büyütmeye yer var)

**Faz 0 kabulü:** `demo/inputs/abo_vase_black_{damaged,mask,original}.png`
(ABO B07QD6ZV9Q, chip hasarı %11.0) → `runs/demo/abo_vase_black_damaged/` içinde
4 aday (3.3–5.9 s/aday) + `panel.png`. Görsel inceleme: doldurmalar makul,
adaylar arasında fark var (biri etiket hallucination'ı üretti — T4/T6'nın
cezalandırması gereken örnek).

**Süre tahmini (32 aday/görüntü, 3.38 s/aday):**
| Split | Görüntü | Saf üretim |
|---|---|---|
| gonogo | 100 | ~3.0 h |
| train | 300 | ~9.0 h |
| val | 100 | ~3.0 h |
| test | 300 | ~9.0 h |
| **toplam** | 800 | **~24 h GPU** |
N=16'ya düşülürse ~12 h. Ayrıca E7 (SDXL, 50 görüntü) değişken; N ve görüntü
sayısı küçültülerek planlanır.

**Açık sorular:** —
