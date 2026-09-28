# counterpart — Proje Spesifikasyonu

> Görüntü İşleme dersi dönem projesi. Bu doküman, projeyi sıfırdan koda dökecek ajan (Lucy) için tek referanstır. Belirsiz kalan bir karar varsa varsayım yapmadan önce Kael'e sor; aşağıdaki "Açık kararlar" bölümüne bak.

---

## 0. Tek cümle

Kırık/eksik bir nesnenin görüntüsünden, dondurulmuş bir generative inpainting modeliyle **çok sayıda aday tamamlama** üreten, bu adayları **klasik görüntü işleme tutarlılık ölçütleriyle** puanlayıp seçen ve adaylar arası ayrışmadan bir **belirsizlik haritası** çıkaran, kategoriden bağımsız bir rekonstrüksiyon sistemi.

## 1. Problem tanımı ve iddialar

### 1.1 Problem
Girdi: hasarlı bir nesnenin tek RGB görüntüsü + hasar bölgesine dair kaba ipucu (sentetik deneyde tam mask).
Çıktı:
1. En olası tamamlanmış görüntü
2. Birbirinden anlamlı biçimde farklı alternatif tamamlamalar ("modlar")
3. Piksel düzeyinde belirsizlik haritası

### 1.2 Temel epistemik ayrım
Sistem **orijinali** değil, **makul (plausible) bir tamamlamayı** üretir. Ground truth yalnızca sentetik hasar deneylerinde vardır. Açık dünya örnekleri (vazo, arc reactor, Penny'nin kabuğu) nitel demodur ve sayısal iddiaya dahil edilmez. Rapor ve README bu ayrımı açıkça yapmalıdır.

### 1.3 Test edilecek hipotezler
- **H0 (go/no-go):** Aynı mask için üretilen N aday arasında kalite varyansı, seçimin anlamlı olabileceği kadar büyüktür (oracle best-of-N ≪ rastgele aday).
- **H1 (ana iddia):** Klasik görüntü tutarlılık ölçütlerinin birleşimiyle yapılan seçim, (a) rastgele seçimden ve (b) CLIP/DINO tabanlı hazır öğrenilmiş seçiciden daha düşük LPIPS verir.
- **H2:** Adaylar arası varyans, gerçek hatayla pozitif koreledir (belirsizlik haritası kalibre).
- **H3:** Generative + seçim, klasik inpainting (Telea/NS/PatchMatch) baseline'larını geçer.

H1'in tutmaması da geçerli bir sonuçtur; raporlanır. Ama bu durumda Kael'le çerçeve konuşulur (bkz. §12).

## 2. Kapsam dışı (non-goals)
- Generative modeli eğitmek veya fine-tune etmek. Dondurulmuş kalır.
- 3D rekonstrüksiyon, multi-view.
- Gerçek zamanlı çalışma.
- Tam otomatik açık-dünya hasar tespiti (stretch goal, §5.2.3).
- Web UI. CLI + statik görsel paneller yeterli. (İsteğe bağlı küçük bir Gradio demosu en sonda.)

## 3. Donanım ve ortam

### 3.1 Hedef makine
- NixOS, Hyprland
- RTX 5060 Laptop GPU, **8 GB VRAM** (Blackwell, compute capability 12.0)
- 62 GB RAM, 16 çekirdek

### 3.2 Ortam kurulumu
- `flake.nix` ile devShell: python 3.11 veya 3.12, `uv`, `git`, gerekli sistem kütüphaneleri (libGL, glib for OpenCV). `.envrc` → `use flake` (nix-direnv zaten açık).
- Python bağımlılıkları `uv` ile, `pyproject.toml` üzerinden.
- **PyTorch Blackwell için CUDA 12.8+ wheel ister.** `--index-url https://download.pytorch.org/whl/cu128` kullan. Kurulumdan sonra `torch.cuda.get_device_capability()` → `(12, 0)` ve basit bir matmul testi çalıştır.
- NixOS'ta pip wheel'lerinin `libcuda.so`'yu bulması için devShell'de `LD_LIBRARY_PATH` içine `/run/opengl-driver/lib` ekle.
- `scripts/check_env.py`: torch/CUDA, diffusers pipeline yükleme, VRAM ölçümü, tek bir inpainting çağrısı. Başka hiçbir şeye geçmeden önce bu yeşil olmalı.

### 3.3 Ana bağımlılıklar
`torch`, `torchvision`, `diffusers`, `transformers`, `accelerate`, `safetensors`, `opencv-python-headless`, `scikit-image`, `numpy`, `scipy`, `pandas`, `pyarrow`, `lpips`, `open_clip_torch` (veya transformers CLIP), `timm` veya transformers DINOv2, `scikit-learn`, `matplotlib`, `pyyaml`, `pydantic`, `typer`, `rich`, `tqdm`, `pytest`. SAM için `sam2` (bkz. §5.1).

### 3.4 8 GB VRAM profili
- Varsayılan generator: **Stable Diffusion 2 inpainting**, fp16, 512×512. Aday kaynak: `stabilityai/stable-diffusion-2-inpainting`; erişilemezse `sd-legacy/stable-diffusion-inpainting` (SD1.5 inpainting). Lucy: hangisinin HF'de erişilebilir olduğunu doğrula, config'e yaz.
- İkincil (opsiyonel) generator: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, fp16 + `enable_model_cpu_offload()` + VAE tiling. Yavaş olacak; yalnızca küçük bir alt kümede "generator'dan bağımsızlık" kontrolü için.
- Flux Fill 8 GB'a sığmaz; kullanma.
- `enable_attention_slicing()` veya SDPA; batch boyutu config'ten, ilk çalıştırmada VRAM'e göre ayarlanır.
- Throughput'u **ölç, hardcode etme**. `check_env.py` görüntü/sn değerini loglasın; deney planı bu sayıya göre ölçeklenir.
- Aynı anda generator + SAM + DINO GPU'da tutulmaz. Aşamalar ayrık çalışır, her aşama kendi modelini yükleyip bırakır.

## 4. Mimari genel bakış

```
Görüntü + (kaba hasar ipucu)
        │
 [1] Segmentasyon ─────────────► object_mask
        │
 [2] Hasar lokalizasyonu ──────► damage_mask, gen_mask
        │
 [3] Hipotez üretimi (N aday) ─► candidates/ (diske cache)
        │
 [4] Skorlama (klasik terimler) ► scores.parquet
        │
 [5] Seçim + modlar + belirsizlik
        │
 best.png · modes/ · uncertainty.png · panel.png
```

**Kritik tasarım ilkesi:** Üretim pahalı, skorlama ucuz. Adaylar bir kez üretilip diske yazılır; skorlama, seçim ve değerlendirme cache üzerinden defalarca, hızlıca yeniden çalıştırılabilir. Her aşama idempotent ve kaldığı yerden devam edebilir (resumable) olmalı.

## 5. Katmanlar

### 5.1 Katman 1 — Segmentasyon
Amaç: nesne silüeti (`object_mask`).

- **Sentetik veri (ABO, beyaz arka plan):** SAM'e gerek yok. Arka plan eşiklemesi + morfolojik kapama + en büyük bağlı bileşen. Hızlı, deterministik. Kalite kontrol: mask alanı görüntünün %5–%90'ı arasında değilse örneği at.
- **Açık dünya demo görüntüleri:** SAM 2.1 (hiera-small veya tiny), kullanıcı tarafından verilen bbox veya nokta promptuyla. `segment/sam.py`.
- Yedek: `cv2.grabCut` (bbox ile başlatma). SAM kurulamazsa demo bununla çalışır.

Çıktı: `object_mask` (uint8, 0/255), crop bilgisi.

**Crop politikası:** Tüm işlem nesne bbox'ı + %15 marj etrafında kare crop'ta yapılır, 512×512'ye yeniden boyutlanır. Değerlendirme de crop uzayında yapılır. Orijinal koordinatlara geri yapıştırma yalnızca görselleştirme için.

### 5.2 Katman 2 — Hasar lokalizasyonu

#### 5.2.1 Sentetik mod
Mask zaten bilinir (§6.2). İki mask tutulur:
- `damage_mask`: gerçekten silinen bölge. **Metrikler yalnızca bunun üzerinde** (ve sınır halkasında) hesaplanır.
- `gen_mask`: generator'a verilen mask = `damage_mask`'in `d` piksel (config, varsayılan 6) dilate edilmiş hali. Generator'ın dikiş hattını da yeniden çizebilmesi için.

#### 5.2.2 Etkileşimli mod (açık dünya demo)
- Kullanıcı kaba bir poligon veya fırça mask'i verir (`localize/annotate.py`: OpenCV penceresi; tıklama ile poligon, `s` kaydet). Alternatif: mask'i harici bir araçla PNG olarak ver.
- Rafine: kullanıcı bölgesinde GrabCut, nesne konturuna snap, küçük boşlukları morfolojik kapama ile doldur. `localize/refine.py`.
- Eksik parça senaryosunda hasar bölgesi *şu anda arka plan* görünen yeri de kapsar. Yani mask, "nesnenin olması gereken ama olmayan" alanı içermelidir. Bu yüzden kullanıcı ipucu esastır; otomatik silüet farkı buna yetmez.

#### 5.2.3 Stretch: tutarlılık tabanlı otomatik tespit
Görünür kısma koşullu, düşük strength img2img ile "sağlam" hipotezler üret. Girdinin tüm hipotezlerle sistematik olarak çeliştiği bölgeler hasar adayı olur. Generative modeller hasarı kopyalamaya meyillidir, bu yüzden güvenilmez. Yalnızca ana deneyler bittiyse ve ayrı bir deney olarak dene. Ana iddiaya dahil etme.

### 5.3 Katman 3 — Hipotez üretimi
`generate/sd_inpaint.py`, `generate/variants.py`.

- Girdi: 512×512 crop, `gen_mask`, opsiyonel kategori/prompt.
- Görüntü başına N aday (varsayılan N=32; config).
- **Çeşitlilik kaynakları** (her aday için parametreler manifest'e yazılır):
  - seed: deterministik, `hash(sample_id) + idx`
  - `guidance_scale` ∈ {4.0, 7.5} (config listesi)
  - prompt şablonları:
    - `""` (boş)
    - `"a complete intact object, product photo, white background"`
    - `"a complete intact {category}, product photo"` (ABO `product_type` metadata'sından; açık dünyada kullanıcı prompt'u)
  - negative prompt: `"broken, cracked, chipped, damaged, missing piece, hole, shattered"`
  - `num_inference_steps`: 25 (config)
- Aday dağılımı: N, (prompt × guidance) kombinasyonlarına eşit bölünür, kalan seed'lerle doldurulur.
- **Post-process:** Adayın `gen_mask` dışındaki pikselleri orijinal hasarlı görüntüyle birebir değiştirilir (VAE yeniden kodlaması görünür bölgeyi hafif bozar; bu karşılaştırmayı kirletmesin). Dikişte 3 px feather blend.
- Çıktı: `candidates/cand_{idx:03d}.png` + `candidates.jsonl` (idx, seed, prompt, guidance, steps, generator_id, süre).

### 5.4 Katman 4 — Skorlama (projenin katkısı)
`score/`. Tüm terimler **yalnızca girdi (hasarlı görüntü + mask'ler) ve adayı** kullanır; ground truth'a asla erişmez.

Ortak arayüz:
```python
class ScoreTerm(Protocol):
    name: str

    def prepare(
        self, sample: Sample
    ) -> Context: ...  # görüntü başına bir kez: halkalar, eksen, referans istatistikler
    def score(self, ctx: Context, cand: np.ndarray) -> TermResult: ...

    # TermResult: value (yüksek = iyi), applicable (bool), diagnostics (dict)
```

Tanımlar:
- `M` = `damage_mask`, `∂M` = sınırı
- `R_out` = M'nin dışında, genişliği `w` (varsayılan 8 px) olan halka ∩ object_mask ∪ arka plan
- `R_in` = M'nin içinde genişliği `w` olan halka
- `V` = görünür nesne bölgesi = object_mask \ dilate(M, w)
- Renk karşılaştırmaları CIELAB'da, ΔE2000 veya basit ΔE76 (config).

#### T1 — Sınır sürekliliği (`boundary.py`)
- **Dikiş renk farkı:** ∂M boyunca her `s` pikselde bir örnek nokta al. Normal yönünü mesafe dönüşümünün gradyanından hesapla. Normal boyunca dışarıda `k` px ve içeride `k` px ortalama Lab rengi al, ΔE ölç. Değer = −median(ΔE).
- **Gradyan sürekliliği:** Aynı noktalarda Sobel gradyan büyüklüğü ve yönünün içeri-dışarı farkı.
- **Sarkan kenar oranı:** Görünür halkadaki Canny kenarları ∂M'ye dik geliyorsa, adayda M içinde devam etmeleri beklenir. ∂M'ye ulaşan kenar bileşenlerinden adayda R_in içinde devamı olmayanların oranı.
- T1 = z-normalize edilmiş bu üç alt terimin ağırlıklı toplamı (alt ağırlıklar config'te).

#### T2 — Ayna simetrisi (`symmetry.py`)
- **prepare:** Görünür nesne (V + silüet) üzerinde ayna ekseni ara. Aday açıları θ ∈ [0,180) (2° adım) ve silüet momentlerinden başlayan ofsetler kullan, yerel iyileştir. Her eksen için görünür bölgeyi yansıt; kesişimde (V ∩ reflect(V)) silüet IoU'su ve renk tutarlılığı ölç. En iyi eksen ve güven `c` ∈ [0,1].
- `c < τ_sym` (config, ~0.6) ise `applicable=False`; term o görüntüde ağırlık almaz.
- **score:** M'nin yansıması V ile ne kadar örtüşüyorsa, orada adayın M içindeki pikselleri, V'deki yansıyan karşılıklarıyla karşılaştırılır: −c · mean ΔE. Ek olarak adayın silüeti ile yansıtılmış silüetin M bölgesindeki IoU'su. Adayın silüeti arka plan eşiklemesiyle (sentetik) ya da SAM'le (demo; box = orijinal bbox) çıkarılır.

#### T3 — Dönme simetrisi (`rotational.py`)
Arc reactor, tabak, çark gibi nesneler için.
- **prepare:** Merkezi silüet momentlerinden başlat, dairesel Hough ile rafine et. Görünür bölgenin polar dönüşümünü (`cv2.warpPolar`) al. Açısal profillerin (her yarıçap bandı için) otokorelasyonu veya FFT'si ile k-katlı simetriyi (k ∈ 2..24) tespit et. Tamamen dairesel yapılar için sürekli dönme simetrisi. Güven `c_r`.
- Eşik altında `applicable=False`.
- **score:** Adayın M içindeki polar-dönüşümlü piksellerini, aynı yarıçapta 2π/k kaydırılmış görünür karşılıklarıyla karşılaştır. Değer = −c_r · mean ΔE.

#### T4 — Doku istatistiği (`texture.py`)
- V'den ve adayın M bölgesinden (eğer M'nin adayda nesne olarak doldurulan kısmı yeterince büyükse) yama örnekleri.
- Özellikler: uniform LBP histogramları (P=8,R=1 ve P=16,R=2; gri), Gabor enerji bankası (4 ölçek × 6 yön; ortalama/std), Lab renk histogramı (a*, b* 2D).
- Mesafe: LBP ve Gabor için χ²; renk için Bhattacharyya. Değer = −ağırlıklı toplam.
- Not: V'de birden fazla doku bölgesi olabilir (vazo gövdesi + desen). Basit tut: V'yi SLIC süperpikselleriyle bölgele, adayın M-yamalarını **en yakın** V-bölgesine göre ölç (min mesafe). "Görünür bir yerde bu doku var mı?" sorusu bu.

#### T5 — Frekans profili (`frequency.py`)
Diffusion çıktıları çoğu zaman aşırı pürüzsüzdür, ya da yanlış yüksek frekans üretir.
- V'den ve M'den eşit boyutlu (ör. 32×32) yamalar, Hann pencereli FFT, radyal ortalama güç spektrumu.
- Log-log eğim ve yüksek frekans enerji oranı (f > f_c) karşılaştırması. Değer = −|farklar|.
- Yama yoksa (M çok ince, çatlak) `applicable=False`.

#### T6 — Kontur sürekliliği (`contour.py`)
Eksik parça senaryosunda en ayırt edici terim olması beklenir.
- Adayın silüet konturunu çıkar. Konturun ∂M'yi kestiği noktalarda, M dışındaki kontur parçasının yerel eğrilik ve teğet yönünü (spline veya yerel polinom fit), M içindeki parçayla karşılaştır.
- Ayrıca M içindeki kontur segmentinin eğrilik düzgünlüğü: eğrilik türevinin enerjisi. Pürüzlü, kıvrımlı uydurma sınırları cezalandırır.
- Değer = −(teğet açı sıçraması + λ·eğrilik enerjisi).

#### Birleştirme (`combine.py`)
- Her terim, **görüntü içinde adaylar arasında** z-normalize edilir (seçim görüntü içi bir sıralama problemi).
- `applicable=False` olan terimler o görüntüde 0 ağırlık alır; kalan ağırlıklar yeniden normalize edilir.
- **Ağırlık modu A (varsayılan, ana sonuç):** eşit ağırlık. Eğitim yok, sızıntı riski yok.
- **Ağırlık modu B:** train split üzerinde, GT LPIPS sıralamasına göre **pairwise lojistik regresyon** (RankNet-lite; 6 özellik, L2). Yalnızca train split'te fit edilir, val'de seçilir, test'te raporlanır. A ve B ayrı ayrı raporlanır.

#### Referans (öğrenilmiş) seçiciler — baseline, skorlayıcının parçası değil
`baselines/`:
- `clip_select.py`: aday tam görüntüsü ile `"a photo of an intact {category}"` metni arasında CLIP benzerliği.
- `dino_select.py`: aday ile hasarlı görüntünün V-maskeli halinin DINOv2 CLS benzerliği, artı M bölgesi patch token'larının V patch token'larına ortalama en yakın komşu benzerliği.

### 5.5 Katman 5 — Seçim, modlar, belirsizlik
`select/`.
- **Seçim:** en yüksek birleşik skorlu aday → `best.png`.
- **Modlar:** Her adayın M-bölgesi crop'undan DINOv2 embedding'i al (M dışı nötr gri). Kosinüs mesafesiyle aglomeratif kümeleme (average linkage, mesafe eşiği config'te). Her kümeden en yüksek skorlu aday temsilci olur. Küme boyutu bir "olasılık kütlesi" göstergesi olarak raporlanır.
- **Belirsizlik haritası:**
  - `U_pix`: Top-K (varsayılan tüm N; ayrıca top-8 varyantı) adayın Lab pikselleri üzerinde piksel başına std; M dışında 0.
  - `U_feat` (opsiyonel): adaylar arası çiftli LPIPS uzamsal haritalarının ortalaması.
- Görselleştirme: `panel.png` = [hasarlı girdi | mask | best | 3 mod temsilcisi | belirsizlik ısı haritası | (varsa) GT].

## 6. Veri

### 6.1 Kaynak
- **Birincil:** Amazon Berkeley Objects (ABO), `abo-images-small` + `listings` metadata. Beyaz arka planlı ürün görselleri + `product_type` kategorisi. Lucy: indirme URL'lerini ve boyutunu doğrula; tamamını indirmek gerekmiyorsa yalnızca gerekli alt kümeyi çek.
- Filtre: arka planı gerçekten beyaz/düz olanlar (kenar piksellerinin %95'i L* > 95), nesne alanı %15–%70, tek bağlı bileşen, minimum kısa kenar 400 px. Hasarın anlamlı olduğu kategorileri tercih et (vazo, kupa, lamba, oyuncak, mutfak eşyası, mobilya parçası, alet); kıyafet ve düz kumaş ürünleri çıkar.
- Yedek: veri seti sorun çıkarırsa Google Scanned Objects render'ları ya da Kael'in kendi çektiği fotoğraflar.

### 6.2 Sentetik hasar üreticisi (`data/masks.py`, `data/damage.py`)
Rastgele blob değil, kırılma benzeri. Seed'li ve deterministik.

Hasar tipleri (oranlar config'te):
- **Kenar kırığı (chip), %70:** Nesne konturundan rastgele bir nokta seç, içeri doğru bir yön belirle. Başlangıç poligonunu (üçgen/dörtgen) orta nokta yer değiştirmesi (fraktal, 3–5 seviye, pürüzlülük parametresi) ile pürüzlendir. object_mask ile kesiştir. Hedef alan oranı object alanının %10–%30'u (tek-tip örnekleme); tutturulamazsa yeniden dene.
- **İç delik, %15:** Aynı pürüzlü poligon, ama tamamen nesne içinde.
- **Çatlak, %15:** Konturdan içeri ilerleyen, dallanabilen rastgele yürüyüş polyline'ı, kalınlık 3–8 px. İnce yapılar T4/T5'i `applicable=False` yapabilir; bu beklenen bir durum.

Hasarlı görüntü sentezi:
- Chip ve delik: M içindeki pikseller arka plan rengiyle (beyaz) doldurulur; parça fiziksel olarak yok.
- Çatlak: M içi koyu gri tonlamayla (kırık yüzeyi), hafif gürültüyle.
- Opsiyonel gerçekçilik (config flag): kırık kenarı boyunca 1–2 px koyu "fracture rim" ve hafif gölge. Varsayılan kapalı; açık hali ayrı bir robustness deneyi olur.

Her örnek için kaydedilir: `original.png` (GT), `damaged.png`, `damage_mask.png`, `gen_mask.png`, `object_mask.png`, `meta.json` (tip, alan oranı, seed, kategori, crop).

### 6.3 Split'ler
- **Nesne (ürün) düzeyinde** ayrım: aynı ürünün farklı görselleri farklı split'lere düşmez.
- `gonogo`: 100 örnek (train'den)
- `train`: 300 (yalnızca ağırlık modu B ve eşik ayarı için)
- `val`: 100
- `test`: 300 (her şey bitene kadar dokunulmaz; tek sefer koşulur)
- Süre bütçesi yetmezse sayılar config'ten küçültülür. Ölçülen throughput'a göre Lucy bir süre tahmini tablosu üretir ve Kael'e sunar.

### 6.4 Gerçek hasar alt kümesi (stretch ama değerli)
Kael 10–20 ucuz seramik/plastik nesneyi tripod sabitken **önce sağlam, sonra kırılmış** halde, aynı poz ve ışıkta fotoğraflar. Bu, gerçek kırılma geometrisi ve gerçek GT demektir. Hizalama gerekirse ORB + homografi (`data/real_align.py`). Mask, iki görüntünün farkından + elle düzeltme ile çıkarılır. Sentetik → gerçek domain gap'ini doğrudan ölçer.

## 7. Baseline'lar

Tamamlama baseline'ları (`baselines/classical_inpaint.py`):
- `cv2.INPAINT_TELEA`, `cv2.INPAINT_NS`
- PatchMatch / exemplar-based (Criminisi): kurulabilir bir kütüphane varsa kullan (ör. `PyPatchMatch`); yoksa Criminisi'nin sade bir implementasyonu. Ders açısından değerli, ama zaman kısıtlıysa opsiyonel.

Seçim baseline'ları (aynı N aday havuzu üzerinde):
- `random`: rastgele aday (beklenen değer = aday metriklerinin ortalaması; tek örnek çekme gürültüsü yok)
- `first`: seed 0, varsayılan prompt (= "tek sample, seçim yok")
- `clip`, `dino` (§5.4)
- `ours_eq` (ağırlık modu A), `ours_learned` (modu B)
- `oracle`: GT'ye göre en iyi aday (üst sınır; yalnızca raporlama)
- `worst`: alt sınır

## 8. Metrikler (`eval/`)

### 8.1 Rekonstrüksiyon kalitesi (GT'li)
Hesaplama bölgesi: `damage_mask`'in bbox'ı + 16 px marj (LPIPS için crop), piksel metrikleri yalnızca M içinde.
- **LPIPS (AlexNet)**: birincil metrik
- PSNR, SSIM (M içi; SSIM için maskelenmiş ortalama)
- Sınır ΔE: GT ve aday arasında, R_in ∪ R_out halkasında
- Silüet IoU: adayın ve GT'nin M bölgesindeki nesne mask'leri (eksik parça senaryosunda kritik)

### 8.2 Seçim kalitesi
- **Top-1 regret** = LPIPS(seçilen) − LPIPS(oracle)
- **Persentil** = seçilen adayın, görüntünün aday havuzundaki LPIPS sıralamasındaki yeri (0 = en iyi, 1 = en kötü; random'ın beklentisi 0.5)
- Spearman ρ(skor, −LPIPS), görüntü başına hesaplanıp ortalaması alınır
- Win-rate: seçicinin random'dan daha iyi olduğu görüntü oranı

### 8.3 Belirsizlik kalibrasyonu
- Piksel düzeyi: M içindeki pikseller için Spearman(U_pix, |best − GT|_Lab)
- AUROC: U_pix'in, en kötü %10 hata piksellerini ayırt etme gücü
- Görüntü düzeyi: Spearman(mean U_pix, LPIPS(best))

### 8.4 İstatistik
- Test setinde tüm karşılaştırmalar için 1000 örnekli bootstrap %95 CI.
- Seçiciler arası eşli Wilcoxon signed-rank; çoklu karşılaştırmada Holm düzeltmesi.
- Sonuç tabloları hem ortalama ± CI, hem medyan.

## 9. Deneyler

| ID | Ad | Split | Amaç | Maliyet |
|---|---|---|---|---|
| E0 | Go/no-go | gonogo (100) | H0: aday varyansı yeterli mi | Üretim |
| E1 | Ana karşılaştırma | test (300) | H1, H3 | Üretim |
| E2 | Ablation | val → test | Her terim tek başına + leave-one-out | Sadece cache |
| E3 | N ölçekleme | test | N ∈ {1,2,4,8,16,32} alt örnekleme (20 rastgele permütasyon) | Sadece cache |
| E4 | Belirsizlik kalibrasyonu | test | H2 | Sadece cache |
| E5 | Kırılım analizi | test | Hasar tipi, alan oranı kovası, kategori bazında sonuçlar | Sadece cache |
| E6 | GT-enjeksiyon teşhisi | val | GT'yi aday havuzuna ekle: skorlayıcı GT'yi kaçıncı sıraya koyuyor? | Sadece cache |
| E7 | Generator bağımsızlığı | test'ten 50 | SDXL inpainting ile E1'in küçük tekrarı | Üretim (yavaş) |
| E8 | Açık dünya nitel | demo/ | Vazo, arc reactor, Penny'nin kabuğu + 5–10 başka örnek | Üretim |
| E9 | Gerçek hasar | real/ | §6.4 seti varsa E1 metrikleri | Üretim |

### E0 karar kuralı
- `gap = mean(LPIPS_random) − mean(LPIPS_oracle)`
- `spread = mean over images(std of candidate LPIPS)`
- **Devam:** oracle, random'ın ≥%20 altındaysa ve gap > spread/2 ise.
- **Sınırda:** çeşitliliği artır (prompt setini genişlet, guidance aralığını aç, SD2 yerine SD1.5 inpainting dene) ve E0'ı tekrarla.
- **Başarısız:** Dur. Kael'e rapor et. Olası pivot: ana iddianın belirsizlik haritası (H2) ve mod keşfi üzerine kaydırılması.

E0 çıktısı: histogramlar (görüntü başına aday LPIPS dağılımı), 10 örnek panel, sayısal özet, karar önerisi.

### E8 notları
- Penny'nin kabuğu çizgi film karesi: düz gölgeleme ve kalın kontur çizgileri T4/T5'in davranışını değiştirir, T6 ise güçlenebilir. Bu fark raporda ayrıca tartışılır.
- Arc reactor T3 (dönme simetrisi) için vitrin örneği.
- Vazo T2 (ayna simetrisi) için vitrin örneği.
- Görüntüler `demo/inputs/` altında, her birine elle mask ve opsiyonel prompt.

## 10. Repo yapısı

```
counterpart/
├── flake.nix
├── .envrc
├── pyproject.toml
├── README.md
├── SPEC.md                      # bu dosya
├── configs/
│   ├── default.yaml
│   ├── profiles/laptop_8gb.yaml
│   └── experiments/{e0,e1,...}.yaml
├── src/counterpart/
│   ├── __init__.py
│   ├── config.py                # pydantic modelleri, yaml yükleme, profile birleştirme
│   ├── types.py                 # Sample, Candidate, TermResult
│   ├── io.py                    # cache okuma/yazma, manifest
│   ├── data/
│   │   ├── abo.py               # indirme, filtreleme, crop
│   │   ├── masks.py             # chip / hole / crack üreticileri
│   │   ├── damage.py            # hasarlı görüntü sentezi
│   │   ├── splits.py
│   │   └── real_align.py        # §6.4
│   ├── segment/
│   │   ├── threshold.py
│   │   ├── sam.py
│   │   └── grabcut.py
│   ├── localize/
│   │   ├── annotate.py
│   │   ├── refine.py
│   │   └── consistency.py       # stretch
│   ├── generate/
│   │   ├── sd_inpaint.py
│   │   └── variants.py
│   ├── score/
│   │   ├── base.py
│   │   ├── geometry.py          # halkalar, normaller, mesafe dönüşümü (ortak)
│   │   ├── boundary.py          # T1
│   │   ├── symmetry.py          # T2
│   │   ├── rotational.py        # T3
│   │   ├── texture.py           # T4
│   │   ├── frequency.py         # T5
│   │   ├── contour.py           # T6
│   │   └── combine.py
│   ├── baselines/
│   │   ├── classical_inpaint.py
│   │   ├── clip_select.py
│   │   └── dino_select.py
│   ├── select/
│   │   ├── select.py
│   │   ├── modes.py
│   │   └── uncertainty.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── ranking.py
│   │   ├── calibration.py
│   │   └── stats.py
│   ├── viz/
│   │   ├── panels.py
│   │   └── figures.py           # rapor figürleri
│   └── cli.py                   # typer
├── scripts/
│   ├── check_env.py
│   ├── benchmark_throughput.py
│   └── run_e0.sh ...
├── tests/
│   ├── test_masks.py
│   ├── test_geometry.py
│   ├── test_score_sanity.py
│   └── test_metrics.py
├── demo/inputs/
├── data/                        # gitignore
└── runs/                        # gitignore
```

### Cache düzeni
```
runs/<experiment>/<split>/<sample_id>/
  original.png  damaged.png  damage_mask.png  gen_mask.png  object_mask.png  meta.json
  candidates/cand_000.png ... cand_031.png
  candidates.jsonl
  scores.parquet            # satır = aday, kolon = her terim ham + z + applicable + birleşik
  baseline_scores.parquet   # clip, dino
  selection.json            # her seçicinin seçtiği idx
  modes.json
  uncertainty.npy
  panel.png
runs/<experiment>/results/
  per_image.parquet  summary.csv  figures/
```

### CLI
```
counterpart prepare   --config ... --split test          # veri + hasar + mask'ler
counterpart generate  --config ... --split test          # adaylar (resumable, eksikleri tamamlar)
counterpart score     --config ... --split test [--terms T1,T2]
counterpart baselines --config ... --split test
counterpart select    --config ... --split test
counterpart evaluate  --config ... --split test
counterpart viz       --config ... --split test --n 20
counterpart demo      --image demo/inputs/vase.jpg --mask demo/inputs/vase_mask.png [--prompt "..."]
```
Her komut: `--limit`, `--overwrite`, `--workers` (CPU aşamaları için multiprocessing).

## 11. Uygulama aşamaları ve kabul kriterleri

**Faz 0 — Ortam (yarım gün)**
- flake + uv + torch cu128 çalışıyor; `check_env.py` yeşil; inpainting throughput ölçüldü ve loglandı.
- Kabul: tek bir ABO görüntüsünde elle mask ile 4 aday üretilip kaydedildi.

**Faz 1 — Veri + hasar (1–2 gün)**
- ABO alt kümesi indirildi, filtrelendi, crop'landı; mask üreticileri ve split'ler hazır.
- `tests/test_masks.py`: alan oranları hedef aralıkta, chip'ler konturla temas ediyor, determinizm (aynı seed → aynı mask).
- Kabul: `counterpart prepare --split gonogo` 100 örnek üretiyor; 20 örneklik görsel kontrol paneli.

**Faz 2 — Üretim + E0 (1–2 gün, çoğu GPU bekleme)**
- `generate` resumable; `evaluate` yalnızca GT metrikleri + oracle/random/worst/first ile çalışır.
- **E0 raporu Kael'e sunulur. Karar verilmeden Faz 3'e geçilmez.** Bu fazda skor terimleri yalnızca stub olarak durur.

**Faz 3 — Skor terimleri (3–5 gün)**
- Önce `geometry.py` (halkalar, normaller) ve testleri, sonra sırasıyla T1, T6, T4, T5, T2, T3 (beklenen etki sırasına göre).
- `tests/test_score_sanity.py`, her terim için sentetik akıl sağlığı testleri:
  - GT adayı, "M'yi beyazla doldur", "M'yi gürültüyle doldur", "M'yi yanlış bölgeden kopyalanmış yamayla doldur" adaylarından daha yüksek skor almalı. T2 ve T3 simetrik sentetik şekillerde test edilir.
- E6 (GT enjeksiyonu) val'de koşar; her terimin GT'yi ortalama kaçıncı persentile koyduğu raporlanır.
- Kabul: tüm terimler val'de çalışıyor, sanity testleri geçiyor, terim başına çalışma süresi ölçüldü (aday başına <200 ms hedef, CPU).

**Faz 4 — Baseline'lar + seçim + belirsizlik (2 gün)**
- Klasik inpainting, CLIP/DINO seçiciler, birleştirme modları A ve B, modlar, U_pix.

**Faz 5 — Test deneyleri (2–3 gün, çoğu GPU bekleme)**
- E1, E2, E3, E4, E5 test split'inde, tek sefer. Sonuç tabloları + figürler.

**Faz 6 — Açık dünya + opsiyoneller (2 gün)**
- E8 demo panelleri. Zaman kalırsa E7, E9, §5.2.3.

**Faz 7 — Rapor materyali (1–2 gün)**
- `viz/figures.py` tüm rapor figürlerini tek komutla yeniden üretir. README'de kurulum, tek komutluk demo ve sonuç özeti.

## 12. Riskler ve karşı önlemler

| Risk | Belirti | Önlem |
|---|---|---|
| Aday çeşitliliği düşük | E0 gap küçük | Prompt/guidance aralığını genişlet, farklı generator dene, pivot (H2 odaklı) |
| Skorlayıcı CLIP'i geçemiyor | E1'de `ours` ≈ `clip` | Ablation ile hangi terimin işe yaradığını bul; dürüstçe raporla; "klasik + öğrenilmiş hibrit" ayrı satır olarak eklenebilir, ama ana iddia değiştirilmeden önce Kael'e sorulur |
| VAE, görünür bölgeyi bozuyor | M dışı PSNR düşük | §5.3 post-process (M dışını orijinalle değiştir) |
| Sentetik ≠ gerçek kırık | E9 sonuçları çok farklı | Fracture rim flag'i ile robustness deneyi; bulguyu raporda tartış |
| Simetri yanlış tespiti | T2/T3 gürültü katıyor | Güven eşiği + `applicable`; E5'te simetrik/asimetrik kırılım |
| 8 GB VRAM yetmiyor | OOM | Model offload, batch=1, 512 çözünürlük, aşamalar ayrık |
| Throughput düşük | Test seti bütçeyi aşıyor | N=16'ya düş (E3 bunu gerekçelendirir), test=200 |
| ABO indirme/format sorunu | Faz 1 takılır | GSO render'ları veya el fotoğrafları |

## 13. Kodlama kuralları (Lucy için)

- Python 3.11+, tam type hint, `ruff` + `ruff format`. Her modülde kısa bir docstring: ne yapar, girdi/çıktı şekilleri.
- Görüntü konvansiyonu: dahili olarak `np.uint8 HxWx3 RGB`; Lab dönüşümleri `skimage.color` ile, float32. Mask'ler `bool HxW`, diskte 0/255 PNG. OpenCV çağrılarında BGR dönüşümünü tek bir yardımcıda yap, dağıtma.
- Tüm sihirli sayılar config'te. Kod içinde eşik yok.
- Her rastgelelik seed'li; seed `meta.json` ve manifest'e yazılır.
- Her aşama kendi logunu `runs/<exp>/logs/` altına yazar (`rich` + dosya).
- GT'ye yalnızca `eval/` ve `combine.py` mod B'nin **train** split fit'i erişebilir. `score/` altındaki hiçbir modül `original.png`'i okumaz. Bunu bir test ile garantile (ör. score aşamasında GT yolunu okumaya çalışınca hata fırlatan bir Sample görünümü).
- Test split'i Faz 5'ten önce hiçbir komutla işlenmez. `--split test` Faz 5 öncesinde bir onay bayrağı (`--i-know`) istesin.
- Uzun GPU işlerini kesintiye dayanıklı yaz: aday dosyası varsa atla; yarım yazılmış dosyaları geçici adla yaz, sonra rename et.
- Commit'ler faz bazında; her fazın sonunda kısa bir `NOTES.md` girdisi: ne yapıldı, ne ölçüldü, açık sorular.

## 14. Rapor iskeleti (ders için)

1. Giriş: hasarlı nesne rekonstrüksiyonu; tek cevap yerine hipotez + doğrulama
2. İlgili çalışmalar: klasik inpainting (Telea, NS, Criminisi, PatchMatch), diffusion inpainting, simetri tespiti, doku analizi (LBP, Gabor), generative çıktıların frekans imzaları, belirsizlik tahmini
3. Yöntem: 5 katman; skor terimlerinin matematiksel tanımları (bu spec'ten)
4. Deney düzeni: veri, sentetik hasar, split'ler, metrikler
5. Sonuçlar: E0, E1 tablosu, ablation, N ölçekleme eğrisi, kalibrasyon, kırılımlar
6. Nitel sonuçlar: açık dünya örnekleri; plausible ≠ faithful tartışması
7. Sınırlamalar: sentetik hasar, generator'a bağımlılık, kullanıcı mask ihtiyacı
8. Sonuç

## 15. Açık kararlar (Kael'e sorulacak)

- Teslim tarihi → faz sürelerini ve split boyutlarını buna göre sıkıştır.
- Ağırlık modu B raporda ana sonuç mu olacak, ek sonuç mu? (Varsayılan: A ana, B ek.)
- Gerçek hasar seti (§6.4) çekilecek mi?
- Gradio demosu istenecek mi?
- PyPI'da `counterpart` adı alınmış; paket adı gerekirse `counterpart-cv`.

---

### Karar kaydı (2026-09-28)

- Teslim tarihi projeyi bağlamıyor; **tam kapsam** hedeflenir (tüm opsiyoneller dahil).
- Ağırlık modu **A ana, B ek**.
- Opsiyonellerin tümü planlanır: E9 gerçek hasar seti, Gradio demosu, PatchMatch/Criminisi, §5.2.3 stretch.
- Paket adı: **`counterpart-cv`** (import: `counterpart`).
- Generator: **SD1.5 inpainting ana**; SD2-inpainting gated olduğundan HF hesabı açılırsa opsiyonel.
