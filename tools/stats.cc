#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <string>

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

struct PCStats {
    uint64_t count = 0;
    uint32_t type = 0;
    double exec_lat_sum = 0;
    uint64_t exec_lat_min = UINT64_MAX;
    uint64_t exec_lat_max = 0;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_file1> [data_file2] ...\n", argv[0]);
        return 1;
    }

    std::unordered_map<uint64_t, PCStats> pc_map;
    uint64_t total_insts = 0;
    uint64_t total_seqs = 0;

    for (int f = 1; f < argc; f++) {
        const char* path = argv[f];
        FILE* fp = fopen(path, "rb");
        if (!fp) {
            fprintf(stderr, "Cannot open: %s\n", path);
            continue;
        }
        fprintf(stderr, "Processing: %s\n", path);

        sequence_header hdr;
        while (fread(&hdr, sizeof(hdr), 1, fp) == 1) {
            total_seqs++;
            uint32_t n = hdr.seq_length;
            for (uint32_t i = 0; i < n; i++) {
                inst_record rec;
                if (fread(&rec, sizeof(rec), 1, fp) != 1) {
                    fprintf(stderr, "Unexpected EOF in sequence\n");
                    goto next_file;
                }
                total_insts++;
                auto& s = pc_map[rec.pc];
                s.count++;
                s.type = rec.type;
                s.exec_lat_sum += rec.exec_latency;
                if (rec.exec_latency < s.exec_lat_min) s.exec_lat_min = rec.exec_latency;
                if (rec.exec_latency > s.exec_lat_max) s.exec_lat_max = rec.exec_latency;
            }
        }
        next_file:
        fclose(fp);
    }

    fprintf(stderr, "\n--- Summary ---\n");
    printf("Files processed: %d\n", argc - 1);
    printf("Total sequences: %lu\n", total_seqs);
    printf("Total instructions: %lu\n", total_insts);
    printf("Unique PCs: %lu\n", (unsigned long)pc_map.size());

    // Sort PCs by count descending
    std::vector<std::pair<uint64_t, PCStats*>> sorted;
    sorted.reserve(pc_map.size());
    for (auto& [pc, s] : pc_map) {
        sorted.push_back({pc, &s});
    }
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second->count > b.second->count; });

    // Cumulative distribution
    printf("\n--- Top 20 PCs by sample count ---\n");
    printf("%-18s %-10s %-6s %-10s %-12s %-12s %-12s\n",
           "PC", "Count", "Type", "AvgLat", "MinLat", "MaxLat", "CumPct%");
    uint64_t cum = 0;
    for (int i = 0; i < 20 && i < (int)sorted.size(); i++) {
        auto& [pc, s] = sorted[i];
        cum += s->count;
        printf("0x%016lx %-10lu %-6u %-10.1f %-12lu %-12lu %.2f%%\n",
               (unsigned long)pc, (unsigned long)s->count, s->type,
               s->exec_lat_sum / s->count,
               (unsigned long)s->exec_lat_min,
               (unsigned long)s->exec_lat_max,
               100.0 * cum / total_insts);
    }

    // Distribution: what % of samples come from top N% of PCs
    printf("\n--- Coverage analysis ---\n");
    int pcts[] = {1, 5, 10, 20, 50, 80};
    for (int p : pcts) {
        int n = std::max(1, (int)(pc_map.size() * p / 100.0));
        uint64_t sum = 0;
        for (int i = 0; i < n && i < (int)sorted.size(); i++) {
            sum += sorted[i].second->count;
        }
        printf("Top %2d%% PCs (%6d) cover %6.2f%% of samples\n",
               p, n, 100.0 * sum / total_insts);
    }

    // Count distribution histogram
    printf("\n--- Sample count distribution across PCs ---\n");
    int bins[] = {1, 2, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 1000000};
    int nbins = sizeof(bins) / sizeof(bins[0]);
    std::vector<uint64_t> bin_counts(nbins + 1, 0);
    for (auto& [pc, s] : pc_map) {
        int placed = 0;
        for (int i = 0; i < nbins; i++) {
            if (s.count <= (uint64_t)bins[i]) {
                bin_counts[i]++;
                placed = 1;
                break;
            }
        }
        if (!placed) bin_counts[nbins]++;
    }
    printf("%-15s %-10s %-10s\n", "Count range", "#PCs", "CumPct%");
    uint64_t cum_pc = 0;
    for (int i = 0; i < nbins; i++) {
        cum_pc += bin_counts[i];
        printf("<= %-10d   %-10lu %.2f%%\n", bins[i], (unsigned long)bin_counts[i],
               100.0 * cum_pc / pc_map.size());
    }
    cum_pc += bin_counts[nbins];
    printf(">  %-10d   %-10lu %.2f%%\n", bins[nbins - 1], (unsigned long)bin_counts[nbins],
           100.0 * cum_pc / pc_map.size());

    return 0;
}
