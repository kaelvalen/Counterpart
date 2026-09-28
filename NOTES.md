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

## Faz 2 — Üretim (devam ediyor)

- `generate` resumable; gonogo üretimi arka planda sürüyor (~150 sn/örnek, 100 örnek ≈ 4 sa).
- check_env ölçümü: 3.38 s/aday (25 adım); gonogo için 100 × 32 × 3.38 ≈ 3 sa GPU.
- E0 değerlendirmesi üretim bitince koşacak (oracle/random/first/worst + karar kuralı).

## Faz 3 — Skor terimleri (T1–T6) ✅ (2026-09-28)

Tüm terimler SPEC.md §5.4 tanımlarıyla, sentetik akıl sağlığı testleriyle:

| Terim | Ölçülen skor (aday başına, ort.) | Notlar |
|---|---|---|
| T1 sınır sürekliliği | 15 ms | ΔE dikiş + gradyan + "sarkan kenar" (∂M'ye **dik** gelen kenarlar; ∂M'ye paralel hasar-dolgu kenarı filtrelenir) |
| T6 kontur sürekliliği | 42 ms | Silüetin ∂M'ye giriş/çıkış noktalarında teğet + eğrilik; hasar silüeti kesmiyorsa `applicable=False` |
| T4 doku istatistiği | 34 ms | LBP(8,1)+(16,2) χ², Gabor 4×6 enerji χ², Lab a*b* Bhattacharyya; V'nin SLIC bölgelerine **min** mesafe; OpenCV Gabor (skimage ~50× yavaştı) |
| T5 frekans profili | 3.5 ms | Hann-FFT radyal spektrum; log-log eğim + yüksek frekans oranı; ince hasarda `applicable=False` |
| T2 ayna simetrisi | 0.1–1 ms* | Eksen araması **tam silüet** üzerinden (hasarlı V değil); renk yalnızca V'de; güven eşiği altında abstain |
| T3 dönme simetrisi | 0.1–1 ms* | Sürekli polar yeniden örnekleme (bin karışması yok); k∈[2,24] katlanmış tutarlılık; Hough merkezi doğrulanır; temel k = en büyük uyumlu k |

*T2/T3 skor süreleri ölçülen örneklerde ~0 çıktı çünkü o örnekler asimetrik
sandalyeler ve terimler haklı olarak abstain ediyor (`applicable=False`);
prepare süreleri sırasıyla ~0.1 s ve ~0.8 s.

**Kalibrasyon notları:**
- T3'te kritik bug: katlanmış referans toplamı görünmeyen (hasarlı) hücrelerin
  renklerini de topluyordu → ortalama kirleniyordu; düzeltildi.
- T6'da kontakt tanımı yanlıştı (∂M'ye komşu tüm kontur pikselleri); giriş/çıkış
  noktalarına çevrildi.
- T2'de eksen araması hasarlı bölge V üzerindeyse yanlış eksene kayıyor; tam
  silüet kullanılarak düzeltildi.

## Faz 5 hazırlığı — ön sonuçlar (gonogo alt kümesi, ~25 örnek) ✅ (2026-09-28)

Tam gonogo üretimi sürerken (24/100 örnek) tüm hat uçtan uca çalıştırıldı:

**E0 (kısmi, 25 görüntü):** oracle 0.081 / random 0.109 / worst 0.160 → oracle_rel 0.75,
gap 0.027 > spread/2 0.008 → **karar: continue** (tam 100 örnekle tekrarlanacak).

**H1 ön izlemesi (25 görüntü, ortalama LPIPS):**

| Seçici | mean LPIPS | persentil | win-rate vs random |
|---|---|---|---|
| ours_eq (mod A) | **0.101** | **0.29** | **%86** |
| dino | 0.108 | 0.50 | %50 |
| first | 0.107 | 0.42 | %64 |
| clip | 0.113 | 0.49 | %59 |
| random (beklenen) | 0.109 | 0.50 | — |
| oracle | 0.081 | 0 | — |

Yani klasik birleşim, CLIP/DINO öğrenilmiş seçicilerini ve random'ı geçiyor — H1 yönünde.

**H2 ön izlemesi:** piksel Spearman(U, |best−GT|) = 0.43; AUROC(en kötü %10) = 0.83;
görüntü düzeyi Spearman(mean U, LPIPS(best)) = 0.54 → belirsizlik haritası kalibre görünüyor.

**H3 ön izlemesi:** Telea 0.376 / NS 0.373 vs generative random 0.109 / oracle 0.081 →
klasik inpainting'e karşı büyük fark.

**E3 (N ölçekleme):** N=1 → 0.111, N=8 → 0.107, N=32 → 0.098 (oracle 0.084) — seçim
N ile iyileşiyor.

**E2 ablation (kısmi):** full_combined_A 0.101; en iyi tek terim T4 0.103; LOO etkisi
T2/T3/T4'te görünür (tam tablo `runs/main/results/e2_ablation_gonogo.csv`).

**Karar:** tüm hat hazır; tam koşular için `scripts/run_all.sh` (train/val/test →
score/baselines/select/evaluate/experiments; ~20 saat GPU, resumable).

**Açık sorular:** modlar şu an çoğunlukla 1 küme (mean 1.36) — eşik val'de kalibre
edilecek; T3'ün gerçek veride ne sıklıkta applicable olduğu E2/E5'te raporlanacak.
