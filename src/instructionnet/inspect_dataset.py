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


def print_records(file_path, start, count):
    header = np.fromfile(file_path, dtype=header_dtype, count=1)
    seq_length = int(header[0]["seq_length"])
    data = np.memmap(file_path, dtype=record_dtype, mode="r", offset=8, shape=(seq_length,))

    end = min(start + count, seq_length)
    print(f"File: {file_path}  Total instructions: {seq_length}  Showing: [{start}, {end})")
    print(f"{'Idx':>7}  {'PC':>18}  {'Type':>4}  {'Mis':>3}  {'Class':>9}  {'fLat':>5}  {'eLat':>5}  {'iHL':>3}  {'dHL':>3}")
    print("-" * 75)

    for i in range(start, end):
        r = data[i]
        parts = []
        if r['isCondCtrl']:
            parts.append('cctl')
        elif r['isControl']:
            parts.append('ctrl')
        if r['isMemRef']:
            parts.append('mem')
        inst_class = '+'.join(parts) if parts else '-'
        dhl = str(r['dcache_hit_level']) if r['isMemRef'] else '-'
        print(
            f"{i:>7}  "
            f"0x{r['pc']:016x}  "
            f"{r['type']:>4}  "
            f"{r['isMispredicted']:>3}  "
            f"{inst_class:>9}  "
            f"{r['fetch_latency']:>5}  "
            f"{r['exec_latency']:>5}  "
            f"{r['icache_hit_level']:>3}  "
            f"{dhl:>3}"
        )

    if end < seq_length:
        print(f"\n... {seq_length - end} more instructions not shown")


def main():
    parser = argparse.ArgumentParser(description="Inspect binary instruction dataset")
    parser.add_argument("file", nargs="?", default=None, help="Path to .bin dataset file. If omitted, lists available datasets in data dir.")
    parser.add_argument("-d", "--data-dir", default="./data", help="Directory containing .bin files (default: ./data)")
    parser.add_argument("-n", "--count", type=int, default=200, help="Number of instructions to display (default: 200)")
    parser.add_argument("-s", "--start", type=int, default=0, help="Starting index (default: 0)")
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

    file_path = args.file
    if not os.path.isfile(file_path):
        file_path = os.path.join(args.data_dir, file_path)
    if not os.path.isfile(file_path):
        # Try matching by first 3 digits of benchmark name (e.g. "507" -> "507.cactuBSSN.bin")
        pattern = os.path.join(args.data_dir, f"{args.file}.*.bin")
        matches = glob.glob(pattern)
        if len(matches) == 1:
            file_path = matches[0]
        else:
            print(f"File not found: {args.file}")
            return

    print_records(file_path, args.start, args.count)


if __name__ == "__main__":
    main()
