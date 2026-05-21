"""Crawl Wikimedia Commons images for a curated subset of kb_ids.

Usage:
    python scripts/data_collection/crawl_selected.py [--limit 12] [--group all|2|3|8|9|11]

Reuses WikimediaCrawler from crawler.py. Queries are tuned per kb_id to avoid
generic matches (e.g. "Pottery" -> "Bat Trang pottery").
"""
import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import config
from crawler import WikimediaCrawler


GROUPS = {
    "2_bat_trang": {
        "bat_trang_communal_house": "Bat Trang communal house",
        "bat_trang_gate": "Bat Trang village gate",
        "bat_trang_market": "Bat Trang market",
        "bat_trang_mother_temple": "Bat Trang mother temple",
        "bat_trang_pagoda": "Bat Trang pagoda",
        "bat_trang_pottery_museum": "Bat Trang pottery museum",
        "bat_trang_pottery_shop": "Bat Trang pottery shop",
        "bat_trang_pottery_village": "Bat Trang pottery village",
        "ceramic_kiln": "Bat Trang ceramic kiln",
        "pottery": "Bat Trang pottery Hanoi",
    },
    "3_van_phuc": {
        "van_phuc_communal_house": "Van Phuc communal house Ha Dong",
        "van_phuc_pagoda": "Van Phuc pagoda Ha Dong",
        "van_phuc_silk_craft_founder_temple": "Van Phuc silk craft founder temple",
        "van_phuc_silk_village": "Van Phuc silk village Hanoi",
        "van_phuc_silk_village_cultural_house": "Van Phuc silk village cultural house",
        "van_phuc_silk_village_gate": "Van Phuc silk village gate",
        "van_phuc_silk_village_stage": "Van Phuc silk village stage",
        "flat_knitting_machine": "Van Phuc silk flat knitting machine",
        "loom": "Van Phuc silk loom Hanoi",
        "silk_village_market": "Van Phuc silk market",
        "warping_machine": "Van Phuc silk warping machine",
        "weaving_machine": "Van Phuc silk weaving machine",
        "memorial_temple": "Van Phuc memorial temple Ha Dong",
    },
    "8_van_mieu": {
        "dai_thanh_courtyard": "Temple of Literature Dai Thanh courtyard Hanoi",
        "thanh_dat_section": "Temple of Literature Thanh Dat courtyard Hanoi",
        "nhap_dao_section": "Temple of Literature Nhap Dao Hanoi",
    },
    "9_phu_tay_ho": {
        "ancient_malayan_banyan_tree": "Phu Tay Ho temple Hanoi banyan",
        "main_shrine": "Phu Tay Ho temple Hanoi shrine",
        "son_trang_palace": "Son Trang palace Hanoi",
        "tam_quan_gate": "Tam Quan gate Hanoi temple",
    },
    "11_hoan_kiem": {
        "hanoi_cathedral": "Hanoi Saint Joseph Cathedral",
        "ly_thai_to_park": "Ly Thai To park Hanoi",
        "maria_lady_statue": "Saint Joseph Cathedral Hanoi Mary statue",
        "the_ready_to_die_for_the_fatherland_to_live_monument": "Cam Tu Cho To Quoc Quyet Sinh monument Hanoi",
        "walking_street": "Hoan Kiem walking street Hanoi",
    },
}


def crawl(kb_id: str, query: str, limit: int) -> int:
    target_dir = os.path.join(config.DATASET_DIR, kb_id)
    os.makedirs(target_dir, exist_ok=True)
    crawler = WikimediaCrawler(download_dir=target_dir)
    print(f"\n[{kb_id}] query={query!r}")
    images = crawler.search_images(query, limit=limit)
    if not images:
        print(f"  -> no results")
        return 0
    crawler.download_images(images)
    return len(images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument(
        "--group",
        choices=["all", *GROUPS.keys(), "2", "3", "8", "9", "11"],
        default="all",
    )
    args = ap.parse_args()

    alias = {
        "2": "2_bat_trang",
        "3": "3_van_phuc",
        "8": "8_van_mieu",
        "9": "9_phu_tay_ho",
        "11": "11_hoan_kiem",
    }
    selected = (
        list(GROUPS.keys())
        if args.group == "all"
        else [alias.get(args.group, args.group)]
    )

    totals = {}
    for g in selected:
        print(f"\n========== Group: {g} ==========")
        items = GROUPS[g]
        n = 0
        for kb_id, query in items.items():
            n += crawl(kb_id, query, args.limit)
        totals[g] = n

    print("\n========== Summary ==========")
    for g, n in totals.items():
        print(f"  {g}: {n} image references downloaded")


if __name__ == "__main__":
    main()
