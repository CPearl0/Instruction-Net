#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <string>
#include <sys/stat.h>

#pragma pack(push, 1)
struct inst_record {
    uint64_t pc;
    uint8_t type;
    uint32_t int_reg;
    uint32_t fp_reg;
    uint32_t branch_hist;
    uint8_t isMispredicted;
    uint8_t branch_dir_wrong;
    uint8_t branch_target_wrong;
    uint8_t isControl;
    uint8_t isCondCtrl;
    uint8_t isMemRef;
    uint64_t same_icache_line_hist;
    uint64_t same_dcache_line_hist;
    uint64_t same_page_hist;
    uint16_t fetch_latency;
    uint16_t exec_latency;
    uint8_t dcache_hit_level;
    uint8_t icache_hit_level;
    uint8_t icache_hit;
    uint8_t dcache_hit;
};
#pragma pack(pop)

struct sequence_header {
    uint32_t seq_length;
    uint32_t reserved;
};

static_assert(sizeof(inst_record) == 59, "inst_record size mismatch");
static_assert(sizeof(sequence_header) == 8, "sequence_header size mismatch");

static std::vector<std::string> read_dataset_list(const char* path) {
    std::vector<std::string> files;
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Warning: cannot open %s, reading from stdin\n", path);
        fp = stdin;
    }
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (!*p || *p == '\n' || *p == '\r') continue;
        char* start = p;
        while (*p && *p != '\n' && *p != '\r') p++;
        *p = '\0';
        // Trim trailing whitespace
        while (p > start && (*(p-1) == ' ' || *(p-1) == '\t')) { *--p = '\0'; }
        if (*start) files.push_back(start);
    }
    fclose(fp);
    return files;
}

int main(int argc, char* argv[]) {
    const char* dataset_list = argc > 1 ? argv[1] : "datasets.txt";
    const char* output_path = argc > 2 ? argv[2] : "fine_data/merged.bin";
    uint64_t target = argc > 3 ? strtoull(argv[3], nullptr, 10) : 200000000ULL;
    uint32_t block_size = argc > 4 ? (uint32_t)atoi(argv[4]) : 512;
    int skip = argc > 5 ? atoi(argv[5]) : 4;

    auto all_files = read_dataset_list(dataset_list);
    if ((int)all_files.size() <= skip) {
        fprintf(stderr, "Not enough files (need >%d, have %zu)\n", skip, all_files.size());
        return 1;
    }
    std::vector<std::string> train_files(all_files.begin() + skip, all_files.end());
    fprintf(stderr, "Total files: %zu, training: %zu (skipping first %d)\n",
            all_files.size(), train_files.size(), skip);

    // Pass 1: compute per-block unique PC count for each training file
    // Block diversity = number of unique PCs in a block of block_size instructions
    std::vector<std::vector<uint32_t>> per_file_diversity;
    uint64_t total_blocks = 0;
    uint64_t total_insts = 0;

    for (auto& path : train_files) {
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "Cannot open: %s\n", path.c_str()); continue; }
        fprintf(stderr, "Pass 1: %s\n", path.c_str());

        std::vector<uint32_t> block_divs;
        sequence_header hdr;
        while (fread(&hdr, sizeof(hdr), 1, fp) == 1) {
            uint32_t n = hdr.seq_length;
            uint32_t processed = 0;
            while (processed < n) {
                uint32_t bsz = std::min(block_size, n - processed);
                std::unordered_set<uint64_t> pcs;
                bool ok = true;
                for (uint32_t i = 0; i < bsz; i++) {
                    inst_record rec;
                    if (fread(&rec, sizeof(rec), 1, fp) != 1) { ok = false; break; }
                    pcs.insert(rec.pc);
                }
                if (!ok) goto done1;
                block_divs.push_back((uint32_t)pcs.size());
                total_blocks++;
                total_insts += bsz;
                processed += bsz;
            }
        }
        done1:
        fclose(fp);
        per_file_diversity.push_back(std::move(block_divs));
    }
    fprintf(stderr, "Total: %lu instructions in %lu blocks (block_size=%u)\n",
            (unsigned long)total_insts, (unsigned long)total_blocks, block_size);

    // Determine diversity threshold via binary search
    // Find min threshold T such that sum(block_size for blocks with unique_pcs >= T) >= target
    std::vector<uint32_t> sorted_divs;
    sorted_divs.reserve(total_blocks);
    for (auto& dv : per_file_diversity)
        sorted_divs.insert(sorted_divs.end(), dv.begin(), dv.end());
    std::sort(sorted_divs.begin(), sorted_divs.end());

    // Count cumulative instructions from highest diversity downward
    uint64_t cum = 0;
    uint32_t div_threshold = 0;
    for (int64_t i = (int64_t)sorted_divs.size() - 1; i >= 0; i--) {
        cum += block_size;
        if (cum >= target) {
            div_threshold = sorted_divs[i];
            break;
        }
    }
    fprintf(stderr, "Diversity threshold: %u unique PCs / %u block size (%.1f%%)\n",
            div_threshold, block_size, 100.0 * div_threshold / block_size);

    // Diversity distribution
    printf("\n--- Block diversity distribution ---\n");
    int dbins[] = {1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500};
    int ndbins = sizeof(dbins) / sizeof(dbins[0]);
    for (int i = 0; i < ndbins && dbins[i] <= (int)block_size; i++) {
        uint64_t cnt = 0;
        for (auto d : sorted_divs) if (d <= (uint32_t)dbins[i]) { cnt++; } // approximate
        // recount precisely
        cnt = std::upper_bound(sorted_divs.begin(), sorted_divs.end(), (uint32_t)dbins[i])
              - sorted_divs.begin();
        printf("  unique_pcs <= %4d: %lu blocks (%.1f%%)\n",
               dbins[i], (unsigned long)cnt, 100.0 * cnt / total_blocks);
    }

    // Pass 2: keep blocks with diversity >= threshold, write as one big sequence
    mkdir("fine_data", 0755);
    FILE* out = fopen(output_path, "wb");
    if (!out) { fprintf(stderr, "Cannot create %s\n", output_path); return 1; }

    // Write placeholder header, will update seq_length after writing all data
    sequence_header hdr_out{0, 0};
    fwrite(&hdr_out, sizeof(hdr_out), 1, out);

    uint64_t total_kept = 0;
    uint64_t kept_blocks = 0;

    for (size_t fi = 0; fi < train_files.size(); fi++) {
        FILE* fp = fopen(train_files[fi].c_str(), "rb");
        if (!fp) continue;
        fprintf(stderr, "Pass 2: %s\n", train_files[fi].c_str());

        auto& block_divs = per_file_diversity[fi];
        uint64_t block_idx = 0;

        sequence_header hdr;
        while (fread(&hdr, sizeof(hdr), 1, fp) == 1) {
            uint32_t n = hdr.seq_length;
            uint32_t processed = 0;
            while (processed < n) {
                uint32_t bsz = std::min(block_size, n - processed);
                std::vector<inst_record> block(bsz);
                if (fread(block.data(), sizeof(inst_record), bsz, fp) != bsz) {
                    goto done2;
                }
                if (block_idx < block_divs.size() && block_divs[block_idx] >= div_threshold) {
                    fwrite(block.data(), sizeof(inst_record), bsz, out);
                    total_kept += bsz;
                    kept_blocks++;
                }
                block_idx++;
                processed += bsz;
            }
        }
        done2:
        fclose(fp);
    }

    // Update header with actual seq_length
    fseek(out, 0, SEEK_SET);
    hdr_out.seq_length = (uint32_t)total_kept;
    fwrite(&hdr_out, sizeof(hdr_out), 1, out);
    fclose(out);

    fprintf(stderr, "\n--- Result ---\n");
    fprintf(stderr, "Kept: %lu blocks, %lu instructions (%.1f%% of %lu)\n",
            (unsigned long)kept_blocks, (unsigned long)total_kept,
            100.0 * total_kept / total_insts, (unsigned long)total_insts);
    fprintf(stderr, "Output file: %s\n", output_path);
    return 0;
}
