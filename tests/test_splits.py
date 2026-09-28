"""Split logic tests: product-level isolation, determinism, oversampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from counterpart.config import DataCfg
from counterpart.data.splits import build_split_table, filter_catalog, pick_one_image_per_product


def _catalog(
    n_items: int = 12, images_per_item: int = 2, product_type: str = "VASE"
) -> pd.DataFrame:
    rows = []
    for i in range(n_items):
        item = f"B{i:04d}"
        for j in range(images_per_item):
            rows.append(
                {
                    "image_id": f"{item}_img{j}",
                    "path": f"aa/{item}_{j}.jpg",
                    "width": 800,
                    "height": 900,
                    "item_id": item,
                    "product_type": product_type,
                    "item_name": f"nice {product_type.lower()} {i}",
                    "is_main": j == 0,
                }
            )
    return pd.DataFrame(rows)


def test_one_image_per_product_prefers_main() -> None:
    catalog = _catalog()
    picked = pick_one_image_per_product(catalog)
    assert len(picked) == catalog.item_id.nunique()
    assert picked[picked.item_id == "B0000"].image_id.iloc[0] == "B0000_img0"


def test_split_isolation_and_determinism() -> None:
    catalog = _catalog(n_items=30)
    sizes = {"gonogo": 5, "train": 10, "val": 5, "test": 10}
    a = build_split_table(catalog, sizes, seed=7)
    b = build_split_table(catalog, sizes, seed=7)
    assert a.equals(b), "same seed must produce identical splits"

    assert len(a) == sum(sizes.values())
    for split, size in sizes.items():
        assert (a.split == split).sum() == size
    # product-level isolation
    products = a.groupby("split").item_id.apply(set)
    all_ids = [pid for s in products for pid in s]
    assert len(all_ids) == len(set(all_ids)), "a product must not span splits"


def test_oversample_marks_quota() -> None:
    catalog = _catalog(n_items=70)
    sizes = {"gonogo": 5, "train": 10, "val": 5, "test": 10}
    table = build_split_table(catalog, sizes, seed=1, oversample_factor=2.0)
    for split, size in sizes.items():
        part = table[table.split == split]
        assert part.within_quota.sum() == size
        assert len(part) == 2 * size


def test_filter_catalog_drops_keywords_and_small_images() -> None:
    catalog = _catalog(n_items=10)
    # item-level exclusions: rename/retype every row of the product
    catalog.loc[catalog.item_id == "B0000", "item_name"] = "nice t-shirt cotton"
    catalog.loc[catalog.item_id == "B0001", "product_type"] = "SHOES"
    # image-level exclusion: one small image of B0002; the other one stays
    row = (catalog.item_id == "B0002") & ~catalog.is_main
    catalog.loc[row, "height"] = 200

    filtered = filter_catalog(catalog, DataCfg())
    assert filtered.item_id.nunique() == 8  # B0000, B0001 dropped
    assert len(filtered) == 1 + 7 * 2  # B0002 keeps only its large image


def test_insufficient_products_raises() -> None:
    catalog = _catalog(n_items=3)
    with np.testing.assert_raises(ValueError):
        build_split_table(catalog, {"train": 10}, seed=0)
