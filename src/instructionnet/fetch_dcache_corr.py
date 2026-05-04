"""Analyze correlation between recent dcache miss and fetch_latency."""
import numpy as np
import sys

RECORD_DTYPE = np.dtype([
    ('pc',                    'u8'),
    ('type',                  'u1'),
    ('int_reg',               'u4'),
    ('fp_reg',                'u4'),
    ('branch_hist',           'u4'),
    ('isMispredicted',        'u1'),
    ('branch_dir_wrong',      'u1'),
    ('branch_target_wrong',   'u1'),
    ('isControl',             'u1'),
    ('isCondCtrl',            'u1'),
    ('isMemRef',              'u1'),
    ('same_icache_line_hist', 'u8'),
    ('same_dcache_line_hist', 'u8'),
    ('same_page_hist',        'u8'),
    ('fetch_latency',         'u2'),
    ('exec_latency',          'u2'),
    ('dcache_hit_level',      'u1'),
    ('icache_hit_level',      'u1'),
    ('icache_hit',            'u1'),
    ('dcache_hit',            'u1'),
])


if __name__ == "__main__":
    path = sys.argv[1]
    data = np.memmap(path, dtype=RECORD_DTYPE, mode="r", offset=8)
    n = len(data)

    fetch = data['fetch_latency'].astype(np.float64)
    dcache = data['dcache_hit_level']
    is_mem = data['isMemRef']

    print(f"Total records: {n}")

    # Overall stats
    print(f"\n=== Fetch Latency Overall ===")
    print(f"  mean: {fetch.mean():.2f}  median: {np.median(fetch):.1f}  "
          f"p95: {np.percentile(fetch, 95):.1f}  p99: {np.percentile(fetch, 99):.1f}  max: {fetch.max()}")

    # Group by current instruction's dcache_hit_level (mem instructions only)
    print(f"\n=== Fetch Latency by Current Instruction's DCache Level (mem only) ===")
    for v in range(3):
        mask = is_mem & (dcache == v)
        if mask.sum() > 0:
            f = fetch[mask]
            print(f"  dcache={v}: count={mask.sum():>8d}  mean={f.mean():.2f}  "
                  f"median={np.median(f):.1f}  p99={np.percentile(f, 99):.1f}")

    # Group by: does any instruction within the past K instructions have dcache_hit_level=2?
    for K in [4, 8, 16, 32]:
        # sliding window: for each position i, check if any of [i-K, i) has dcache=2
        # dcache=2 for mem instructions, and 255 for non-mem (skip non-mem)
        dcache_miss = (is_mem & (dcache == 2)).astype(np.int32)

        # cumulative sum approach
        cum = np.cumsum(dcache_miss)
        # miss count in [i-K, i) = cum[i] - cum[i-K]
        padded = np.concatenate([[0], cum])
        recent_miss = padded[K + 1:] - padded[1:-K] if K < n else np.zeros(n)

        # Ensure same length
        min_len = min(len(recent_miss), n)
        recent_miss_full = np.zeros(n, dtype=np.int32)
        recent_miss_full[:min_len] = recent_miss[:min_len]

        has_miss = recent_miss_full > 0
        no_miss = ~has_miss
        # Skip first K positions (incomplete window)
        valid = np.zeros(n, dtype=bool)
        valid[K:] = True

        print(f"\n=== Fetch Latency: Recent {K} instrs have dcache miss to memory ===")
        for label, mask in [("has_miss", has_miss & valid), ("no_miss ", no_miss & valid)]:
            cnt = mask.sum()
            if cnt > 0:
                f = fetch[mask]
                print(f"  {label}: count={cnt:>8d}  mean={f.mean():.2f}  "
                      f"median={np.median(f):.1f}  p95={np.percentile(f, 95):.1f}  "
                      f"p99={np.percentile(f, 99):.1f}")

        # Also: fetch latency conditional on distance to last dcache miss
        print(f"  --- Distance from last dcache=2 (mem miss) ---")
        last_miss_pos = -999999
        distances = []
        fetch_at_dist = {}
        for i in range(n):
            if dcache_miss[i]:
                last_miss_pos = i
            dist = i - last_miss_pos
            if dist > 0 and valid[i]:
                if dist not in fetch_at_dist:
                    fetch_at_dist[dist] = []
                fetch_at_dist[dist].append(fetch[i])

        # Show mean fetch_latency at various distances after a dcache miss
        for d in [1, 2, 3, 4, 5, 6, 7, 8, 10, 16, 24, 32]:
            if d in fetch_at_dist:
                arr = np.array(fetch_at_dist[d])
                print(f"    dist={d:>3d}: count={len(arr):>6d}  mean={arr.mean():.2f}")
