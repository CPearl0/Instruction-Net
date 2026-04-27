import numpy as np
import argparse
import os
import glob

header_dtype = np.dtype([
    ("seq_length", "u4"),
    ("reserved", "u4"),
])

record_dtype = np.dtype([
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


def _inst_info(r):
    parts = []
    if r['isCondCtrl']:
        parts.append('cctl')
    elif r['isControl']:
        parts.append('ctrl')
    if r['isMemRef']:
        parts.append('mem')
    return '+'.join(parts) if parts else '-'


def _branch_str(r):
    if not r['isControl']:
        return '-'
    if r['isMispredicted'] == 0:
        return 'ok'
    if r['branch_dir_wrong']:
        return 'dir'
    if r['branch_target_wrong']:
        return 'tgt'
    return 'mis'


def _label_strs(r):
    dhl = str(r['dcache_hit_level']) if r['isMemRef'] else '-'
    br = _branch_str(r) if r['isControl'] else '-'
    return (str(r['fetch_latency']), str(r['exec_latency']),
            str(r['icache_hit_level']), dhl, br)


def _fmt_bits(value, n_bits):
    bits = format(int(value), f'0{n_bits}b')
    return ' '.join(bits[i:i+8] for i in range(0, n_bits, 8))


def print_verbose(file_path, start, count):
    header = np.fromfile(file_path, dtype=header_dtype, count=1)
    seq_length = int(header[0]["seq_length"])
    data = np.memmap(file_path, dtype=record_dtype, mode="r", offset=8, shape=(seq_length,))

    end = min(start + count, seq_length)
    print(f"File: {file_path}  Total instructions: {seq_length}  Showing: [{start}, {end})")

    for i in range(start, end):
        r = data[i]
        flags = _inst_info(r)
        print(f"\n--- [{i}] PC=0x{r['pc']:016x} Type={r['type']} Flags={flags} ---")
        print(f"  IntReg:  {_fmt_bits(r['int_reg'], 32)}")
        print(f"  FPReg:   {_fmt_bits(r['fp_reg'], 32)}")
        print(f"  BrHist:  {_fmt_bits(r['branch_hist'], 32)}")
        print(f"  ICacheLineHist: {_fmt_bits(r['same_icache_line_hist'], 64)}")
        print(f"  fLat={r['fetch_latency']}  eLat={r['exec_latency']}  iHL={r['icache_hit_level']}")

        if r['isControl']:
            print(f"  BrPred: {_branch_str(r)}")

        if r['isMemRef']:
            print(f"  DCacheLineHist: {_fmt_bits(r['same_dcache_line_hist'], 64)}")
            print(f"  DPageHist:      {_fmt_bits(r['same_page_hist'], 64)}")
            print(f"  dHL={r['dcache_hit_level']}")

    if end < seq_length:
        print(f"\n... {seq_length - end} more instructions not shown")


def print_records(file_path, start, count):
    header = np.fromfile(file_path, dtype=header_dtype, count=1)
    seq_length = int(header[0]["seq_length"])
    data = np.memmap(file_path, dtype=record_dtype, mode="r", offset=8, shape=(seq_length,))

    end = min(start + count, seq_length)
    print(f"File: {file_path}  Total instructions: {seq_length}  Showing: [{start}, {end})")
    print(f"{'Idx':>7}  {'PC':>18}  {'Type':>4}  {'Class':>9}  {'fLat':>5}  {'eLat':>5}  {'iHL':>3}  {'dHL':>3}  {'BrPred':>5}")
    print("-" * 82)

    for i in range(start, end):
        r = data[i]
        fl, el, ihl, dhl, br = _label_strs(r)
        print(
            f"{i:>7}  "
            f"0x{r['pc']:016x}  "
            f"{r['type']:>4}  "
            f"{_inst_info(r):>9}  "
            f"{fl:>5}  {el:>5}  {ihl:>3}  {dhl:>3}  {br:>5}"
        )

    if end < seq_length:
        print(f"\n... {seq_length - end} more instructions not shown")


def print_comparison(gt_path, pred_path, start, count):
    gt_header = np.fromfile(gt_path, dtype=header_dtype, count=1)
    pred_header = np.fromfile(pred_path, dtype=header_dtype, count=1)
    gt_len = int(gt_header[0]["seq_length"])
    pred_len = int(pred_header[0]["seq_length"])

    if gt_len != pred_len:
        print(f"Warning: length mismatch GT={gt_len} vs Pred={pred_len}, using min")

    seq_length = min(gt_len, pred_len)
    end = min(start + count, seq_length)

    gt = np.memmap(gt_path, dtype=record_dtype, mode="r", offset=8, shape=(gt_len,))
    pred = np.memmap(pred_path, dtype=record_dtype, mode="r", offset=8, shape=(pred_len,))

    print(f"GT:    {gt_path}")
    print(f"Pred:  {pred_path}")
    print(f"Total instructions: {gt_len}  Showing: [{start}, {end})")
    header = (f"{'Idx':>7}  {'PC':>18}  {'Type':>4}  {'Class':>9}  | "
              f"{'fLat':>5} {'eLat':>5} {'iHL':>4} {'dHL':>4} {'BrPr':>4}  | "
              f"{'fLat':>5} {'eLat':>5} {'iHL':>4} {'dHL':>4} {'BrPr':>4}")
    print(header)
    print("-" * len(header))

    for i in range(start, end):
        g, p = gt[i], pred[i]
        gfl, gel, gihl, gdhl, gbr = _label_strs(g)
        pfl, pel, pihl, pdhl, pbr = _label_strs(p)
        print(
            f"{i:>7}  "
            f"0x{g['pc']:016x}  "
            f"{g['type']:>4}  "
            f"{_inst_info(g):>9}  | "
            f"{gfl:>5} {gel:>5} {gihl:>4} {gdhl:>4} {gbr:>4}  | "
            f"{pfl:>5} {pel:>5} {pihl:>4} {pdhl:>4} {pbr:>4}"
        )

    if end < seq_length:
        print(f"\n... {seq_length - end} more instructions not shown")


def _resolve_path(filename, data_dir):
    if os.path.isfile(filename):
        return filename
    path = os.path.join(data_dir, filename)
    if os.path.isfile(path):
        return path
    pattern = os.path.join(data_dir, f"{filename}.*.bin")
    matches = glob.glob(pattern)
    if len(matches) == 1:
        return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Inspect binary instruction dataset")
    parser.add_argument("file", nargs="?", default=None, help="Path to .bin dataset file. If omitted, lists available datasets in data dir.")
    parser.add_argument("--pred", default=None, help="Prediction .bin file for side-by-side comparison with groundtruth")
    parser.add_argument("-d", "--data-dir", default="./data", help="Directory containing .bin files (default: ./data)")
    parser.add_argument("-n", "--count", type=int, default=200, help="Number of instructions to display (default: 200)")
    parser.add_argument("-s", "--start", type=int, default=0, help="Starting index (default: 0)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all fields per instruction in detail")
    args = parser.parse_args()

    if args.file is None:
        bins = sorted(glob.glob(os.path.join(args.data_dir, "*.bin")))
        if not bins:
            print(f"No .bin files found in {args.data_dir}/")
            return
        print("Available datasets (pass filename as argument):")
        for b in bins:
            header = np.fromfile(b, dtype=header_dtype, count=1)
            length = int(header[0]["seq_length"])
            print(f"  {os.path.basename(b):30s}  {length:>12,} instructions")
        return

    gt_path = _resolve_path(args.file, args.data_dir)
    if gt_path is None:
        print(f"File not found: {args.file}")
        return

    if args.pred:
        pred_path = _resolve_path(args.pred, args.data_dir)
        if pred_path is None:
            print(f"Prediction file not found: {args.pred}")
            return
        print_comparison(gt_path, pred_path, args.start, args.count)
    elif args.verbose:
        print_verbose(gt_path, args.start, args.count)
    else:
        print_records(gt_path, args.start, args.count)


if __name__ == "__main__":
    main()
