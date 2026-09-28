# counterpart

Kırık/eksik bir nesnenin tek görüntüsünden, dondurulmuş bir generative inpainting
modeliyle **çok sayıda aday tamamlama** üreten; adayları **klasik görüntü işleme
tutarlılık ölçütleriyle** puanlayıp seçen; adaylar arası ayrışmadan **belirsizlik
haritası** çıkaran kategoriden bağımsız rekonstrüksiyon sistemi.

> Sistem *orijinali* değil, *makul (plausible)* bir tamamlama üretir. Ground truth
> yalnızca sentetik hasar deneylerinde vardır; açık dünya örnekleri nitel demodur.

Tam proje spec'i: [`SPEC.md`](SPEC.md) · Faz notları: [`NOTES.md`](NOTES.md)

## Kurulum

NixOS + flakes. Nix flake bütün sistemi (python 3.12, uv, CUDA userspace yolları)
sağlar; python bağımlılıkları `uv` ile `pyproject.toml`'dan gelir.

```bash
direnv allow        # .envrc -> use flake  (nix-direnv)
uv sync             # torch cu128 dahil tüm bağımlılıklar
uv run python scripts/check_env.py   # torch/CUDA/pipeline/throughput doğrulaması
```

CUDA'sız adımlar (skorlama, seçim, değerlendirme) mevcut cache üzerinden çalışır.

## Kullanım (şimdilik)

```bash
# tek görüntü + elle maske ile N aday (Faz 0 kabul testi / E8 yolu)
uv run counterpart demo --image demo/inputs/vase.jpg --mask demo/inputs/vase_mask.png --n 32
```

## Durum

| Faz | İçerik | Durum |
|---|---|---|
| 0 | Ortam + check_env + demo adayları | ✅ |
| 1 | ABO veri + sentetik hasar + split'ler | 🔄 devam |
| 2 | Üretim + E0 go/no-go | ⬜ |
| 3 | Skor terimleri (T1–T6) | ⬜ |
| 4 | Baseline'lar + seçim + belirsizlik | ⬜ |
| 5 | Test deneyleri (E1–E5) | ⬜ |
| 6 | Açık dünya + opsiyoneller (E7–E9) | ⬜ |
| 7 | Rapor materyali | ⬜ |

Demo girdileri ABO'dan (Amazon Berkeley Objects, CC BY-NC 4.0) alınmıştır;
yalnızca ders/araştırma amaçlı kullanılır.
