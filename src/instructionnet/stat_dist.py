"""Print class distribution statistics for a binary dataset."""
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

    print(f"Total records: {len(data)}")
    print(f"  isMemRef:  {data['isMemRef'].sum()} / {len(data)}")
    print(f"  isControl: {data['isControl'].sum()} / {len(data)}")

    # ICache hit level (all instructions)
    print("\n=== ICache Hit Level (all) ===")
    icache = data['icache_hit_level']
    for v in range(3):
        cnt = (icache == v).sum()
        print(f"  level {v}: {cnt:>12d}  ({cnt / len(data):.2%})")

    # DCache hit level (memory instructions only)
    mem_mask = data['isMemRef'] == 1
    mem_count = mem_mask.sum()
    print(f"\n=== DCache Hit Level (isMemRef=1, count={mem_count}) ===")
    dcache = data['dcache_hit_level'][mem_mask]
    for v in range(3):
        cnt = (dcache == v).sum()
        print(f"  level {v}: {cnt:>12d}  ({cnt / mem_count:.2%})" if mem_count > 0 else f"  level {v}: N/A")
    # Show raw values >= 3 (non-memory markers)
    if not mem_mask.all():
        dcache_nonmem = data['dcache_hit_level'][~mem_mask]
        vals, counts = np.unique(dcache_nonmem, return_counts=True)
        print(f"  non-mem values: {dict(zip(vals.tolist(), counts.tolist()))}")

    # Branch prediction (control instructions only)
    ctrl_mask = data['isControl'] == 1
    ctrl_count = ctrl_mask.sum()
    print(f"\n=== Branch Prediction (isControl=1, count={ctrl_count}) ===")
    isMis = data['isMispredicted'][ctrl_mask]
    dir_w = data['branch_dir_wrong'][ctrl_mask]
    tgt_w = data['branch_target_wrong'][ctrl_mask]
    bp = np.where(isMis == 0, 0, np.where(tgt_w == 1, 2, 1))
    for v, name in enumerate(["correct", "dir_wrong", "tgt_wrong"]):
        cnt = (bp == v).sum()
        print(f"  {name} ({v}): {cnt:>12d}  ({cnt / ctrl_count:.2%})" if ctrl_count > 0 else f"  {name}: N/A")
