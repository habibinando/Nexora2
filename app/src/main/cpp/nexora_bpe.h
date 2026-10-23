#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <algorithm>
#include <sstream>

namespace nexora {

struct BPEMergeRule {
    uint32_t left;
    uint32_t right;
    uint32_t target_id;
    uint32_t rank;
};

class BPETokenizerEngine {
public:
    static constexpr uint32_t PAD_TOKEN = 0;
    static constexpr uint32_t EOS_TOKEN = 1;
    static constexpr uint32_t BOS_TOKEN = 2;
    static constexpr uint32_t UNK_TOKEN = 3;
    static constexpr uint32_t BYTE_OFFSET = 4;
    static constexpr uint32_t SUBWORD_START = 260; // 4 + 256
    static constexpr uint32_t MAX_VOCAB_SIZE = 16384;

    BPETokenizerEngine() : next_vocab_index(SUBWORD_START) {
        initBase();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mtx);
        initBaseUnlocked();
    }

    uint32_t getVocabSize() const {
        std::lock_guard<std::mutex> lock(mtx);
        return next_vocab_index;
    }

    /**
     * Daftarkan subword baru jika belum ada.
     */
    uint32_t registerWord(const std::string& word) {
        std::lock_guard<std::mutex> lock(mtx);
        return registerWordUnlocked(word);
    }

    /**
     * BPE Encode: Mengubah UTF-8 bytes mentah menjadi sequence token ID.
     * Menerapkan aturan merge BPE secara iteratif berbasis rank terkecil.
     */
    std::vector<uint32_t> encode(const uint8_t* data, size_t length, bool addBos = false, bool addEos = false) {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<uint32_t> tokens;
        if (!data || length == 0) {
            if (addBos) tokens.push_back(BOS_TOKEN);
            if (addEos) tokens.push_back(EOS_TOKEN);
            return tokens;
        }

        tokens.reserve(length + (addBos ? 1 : 0) + (addEos ? 1 : 0));
        if (addBos) tokens.push_back(BOS_TOKEN);

        // Langkah 1: Tokenisasi byte awal (Byte-Level UTF-8, 100% coverage tanpa mojibake)
        std::vector<uint32_t> current;
        current.reserve(length);
        for (size_t i = 0; i < length; ++i) {
            current.push_back(BYTE_OFFSET + static_cast<uint32_t>(data[i]));
        }

        // Langkah 2: Iterative BPE Merge berdasarkan rank aturan
        while (current.size() >= 2) {
            int best_pair_idx = -1;
            uint32_t best_rank = UINT32_MAX;
            uint32_t best_target = 0;

            for (size_t i = 0; i + 1 < current.size(); ++i) {
                uint64_t key = (static_cast<uint64_t>(current[i]) << 32) | static_cast<uint64_t>(current[i + 1]);
                auto it = pair_to_rule.find(key);
                if (it != pair_to_rule.end()) {
                    if (it->second.rank < best_rank) {
                        best_rank = it->second.rank;
                        best_target = it->second.target_id;
                        best_pair_idx = static_cast<int>(i);
                    }
                }
            }

            if (best_pair_idx == -1) {
                break; // Tidak ada lagi pasangan yang bisa dimerge
            }

            // Lakukan merge untuk semua kemunculan pasangan terbaik
            uint32_t left = current[best_pair_idx];
            uint32_t right = current[best_pair_idx + 1];
            std::vector<uint32_t> next_tokens;
            next_tokens.reserve(current.size());

            for (size_t i = 0; i < current.size(); ++i) {
                if (i + 1 < current.size() && current[i] == left && current[i + 1] == right) {
                    next_tokens.push_back(best_target);
                    ++i; // Lewati elemen kanan yang sudah dimerge
                } else {
                    next_tokens.push_back(current[i]);
                }
            }
            current = std::move(next_tokens);
        }

        tokens.insert(tokens.end(), current.begin(), current.end());
        if (addEos) tokens.push_back(EOS_TOKEN);

        return tokens;
    }

    /**
     * BPE Decode: Mengubah sequence token ID kembali ke byte sequence UTF-8 mentah.
     * Mengumpulkan seluruh byte secara utuh sehingga multi-byte UTF-8 (aksen, Cyrillic, CJK, Emoji)
     * tidak pernah terpotong di tengah jalan.
     */
    std::vector<uint8_t> decode(const uint32_t* tokens, size_t count) {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<uint8_t> output;
        output.reserve(count * 2);

        for (size_t i = 0; i < count; ++i) {
            uint32_t tid = tokens[i];
            if (tid == PAD_TOKEN || tid == BOS_TOKEN) {
                continue;
            }
            if (tid == EOS_TOKEN) {
                break;
            }

            auto it = id_to_bytes.find(tid);
            if (it != id_to_bytes.end()) {
                output.insert(output.end(), it->second.begin(), it->second.end());
            } else if (tid >= BYTE_OFFSET && tid < SUBWORD_START) {
                output.push_back(static_cast<uint8_t>(tid - BYTE_OFFSET));
            }
        }

        return output;
    }

    /**
     * BPE Training: Belajar pasangan byte / subword paling sering dari korpus teks.
     */
    int trainFromBytes(const uint8_t* data, size_t length, int maxNewTokens) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!data || length == 0 || maxNewTokens <= 0) return 0;

        // Pecah teks menjadi segmen kata berdasarkan spasi/pemisah untuk menjaga batas semantik
        std::vector<std::vector<uint32_t>> words;
        std::vector<uint32_t> current_word;

        for (size_t i = 0; i < length; ++i) {
            uint8_t b = data[i];
            uint32_t tid = BYTE_OFFSET + static_cast<uint32_t>(b);
            current_word.push_back(tid);

            // Batas pemisah kata pada whitespace atau tanda baca dasar
            if (b == ' ' || b == '\n' || b == '\t' || b == ',' || b == '.' || b == '!' || b == '?') {
                if (!current_word.empty()) {
                    words.push_back(std::move(current_word));
                    current_word.clear();
                }
            }
        }
        if (!current_word.empty()) {
            words.push_back(std::move(current_word));
        }

        int added = 0;
        while (added < maxNewTokens && next_vocab_index < MAX_VOCAB_SIZE) {
            // Hitung frekuensi pasangan adjacent di seluruh kata
            std::unordered_map<uint64_t, int> pair_counts;
            for (const auto& w : words) {
                if (w.size() < 2) continue;
                for (size_t i = 0; i + 1 < w.size(); ++i) {
                    uint64_t key = (static_cast<uint64_t>(w[i]) << 32) | static_cast<uint64_t>(w[i + 1]);
                    pair_counts[key]++;
                }
            }

            if (pair_counts.empty()) break;

            // Cari pasangan dengan frekuensi tertinggi
            uint64_t best_pair = 0;
            int max_count = 0;
            for (const auto& kv : pair_counts) {
                if (kv.second > max_count) {
                    max_count = kv.second;
                    best_pair = kv.first;
                }
            }

            if (max_count < 2) {
                // Tidak ada pasangan yang muncul berulang
                break;
            }

            uint32_t left = static_cast<uint32_t>(best_pair >> 32);
            uint32_t right = static_cast<uint32_t>(best_pair & 0xFFFFFFFF);

            // Bentuk byte representation gabungan
            std::vector<uint8_t> merged_bytes;
            auto itL = id_to_bytes.find(left);
            if (itL != id_to_bytes.end()) merged_bytes.insert(merged_bytes.end(), itL->second.begin(), itL->second.end());
            else if (left >= BYTE_OFFSET && left < SUBWORD_START) merged_bytes.push_back(static_cast<uint8_t>(left - BYTE_OFFSET));

            auto itR = id_to_bytes.find(right);
            if (itR != id_to_bytes.end()) merged_bytes.insert(merged_bytes.end(), itR->second.begin(), itR->second.end());
            else if (right >= BYTE_OFFSET && right < SUBWORD_START) merged_bytes.push_back(static_cast<uint8_t>(right - BYTE_OFFSET));

            uint32_t new_id = next_vocab_index++;
            uint32_t new_rank = static_cast<uint32_t>(all_merges.size());

            BPEMergeRule rule{left, right, new_id, new_rank};
            all_merges.push_back(rule);
            pair_to_rule[best_pair] = rule;
            id_to_bytes[new_id] = merged_bytes;
            std::string str_repr(reinterpret_cast<const char*>(merged_bytes.data()), merged_bytes.size());
            id_to_str[new_id] = str_repr;
            str_to_id[str_repr] = new_id;

            // Terapkan merge ke dataset words
            for (auto& w : words) {
                if (w.size() < 2) continue;
                std::vector<uint32_t> next_w;
                next_w.reserve(w.size());
                for (size_t i = 0; i < w.size(); ++i) {
                    if (i + 1 < w.size() && w[i] == left && w[i + 1] == right) {
                        next_w.push_back(new_id);
                        ++i;
                    } else {
                        next_w.push_back(w[i]);
                    }
                }
                w = std::move(next_w);
            }

            added++;
        }

        return added;
    }

    /**
     * Ekspor seluruh subwords yang terdaftar untuk disimpan di Checkpoint / sesi berikutnya.
     */
    std::vector<std::string> exportVocabulary() {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<std::string> vocab;
        vocab.reserve(next_vocab_index - SUBWORD_START);
        for (uint32_t id = SUBWORD_START; id < next_vocab_index; ++id) {
            auto it = id_to_str.find(id);
            if (it != id_to_str.end()) {
                vocab.push_back(it->second);
            } else {
                vocab.push_back("");
            }
        }
        return vocab;
    }

    /**
     * Impor subwords dari Checkpoint untuk memulihkan kosa kata dan merge table secara deterministik.
     */
    void importVocabulary(const std::vector<std::string>& vocabList) {
        std::lock_guard<std::mutex> lock(mtx);
        initBaseUnlocked();
        for (const auto& w : vocabList) {
            if (!w.empty()) {
                registerWordUnlocked(w);
            }
        }
    }

private:
    mutable std::mutex mtx;
    uint32_t next_vocab_index;

    std::unordered_map<uint64_t, BPEMergeRule> pair_to_rule;
    std::vector<BPEMergeRule> all_merges;
    std::unordered_map<uint32_t, std::vector<uint8_t>> id_to_bytes;
    std::unordered_map<std::string, uint32_t> str_to_id;
    std::unordered_map<uint32_t, std::string> id_to_str;

    void initBase() {
        std::lock_guard<std::mutex> lock(mtx);
        initBaseUnlocked();
    }

    void initBaseUnlocked() {
        next_vocab_index = SUBWORD_START;
        pair_to_rule.clear();
        all_merges.clear();
        id_to_bytes.clear();
        str_to_id.clear();
        id_to_str.clear();

        // 1. Inisialisasi token byte mentah (0..255)
        for (uint32_t b = 0; b < 256; ++b) {
            uint32_t tid = BYTE_OFFSET + b;
            std::vector<uint8_t> bytes = { static_cast<uint8_t>(b) };
            id_to_bytes[tid] = bytes;
            std::string s(1, static_cast<char>(b));
            str_to_id[s] = tid;
            id_to_str[tid] = s;
        }

        // 2. Kosa kata dasar Bahasa Indonesia & istilah teknis umum
        static const std::vector<std::string> baseWords = {
            " ", "\n", "\t", "yang", "dan", "di", "ini", "itu", "untuk", "dengan", "dari",
            "pada", "adalah", "ke", "tidak", "akan", "dalam", "bisa", "ada", "saya",
            "kamu", "dia", "mereka", "kita", "kami", "oleh", "juga", "sebagai", "lebih",
            "sudah", "hanya", "dapat", "kata", "bila", "atau", "jika", "karena", "tentang",
            "seperti", "setelah", "ia", "saat", "bahwa", "harus", "sampai", "antara",
            "kemudian", "banyak", "orang", "hari", "tahun", "waktu", "dunia", "negara",
            "masyarakat", "indonesia", "ai", "model", "training", "data", "nexora", "lora",
            "belajar", "proses", "hasil", "sistem", "bahasa", "teknologi", "komputer",
            "cerdas", "jaringan", "bobot", "akurasi", "cepat", "kinerja", "memori",
            "apa", "siapa", "mengapa", "bagaimana", "kapan", "dimana", "apakah",
            "halo", "hai", "selamat", "pagi", "siang", "sore", "malam", "terima", "kasih",
            "sama", "baik", "bagus", "hebat", "benar", "salah", "coba", "tanya", "jawab",
            "the", "of", "and", "to", "in", "is", "that", "it", "was", "for", "as", "with",
            "text", "loss", "step", "epoch", "neural", "tensor", "gradient", "vector",
            "optimasi", "adamw", "lion", "parameter", "dimensi", "matriks", "skalar"
        };

        for (const auto& w : baseWords) {
            registerWordUnlocked(w);
        }
    }

    uint32_t registerWordUnlocked(const std::string& word) {
        if (word.empty()) return UNK_TOKEN;
        auto it = str_to_id.find(word);
        if (it != str_to_id.end()) return it->second;

        if (next_vocab_index >= MAX_VOCAB_SIZE) return UNK_TOKEN;

        // Bangun BPE merge bertahap untuk word ini dari byte-level tokens
        std::vector<uint8_t> word_bytes(word.begin(), word.end());
        std::vector<uint32_t> current_tokens;
        current_tokens.reserve(word_bytes.size());
        for (uint8_t b : word_bytes) {
            current_tokens.push_back(BYTE_OFFSET + static_cast<uint32_t>(b));
        }

        // Lakukan merge berurutan hingga terbentuk satu token utuh untuk kata ini
        while (current_tokens.size() > 1 && next_vocab_index < MAX_VOCAB_SIZE) {
            uint32_t left = current_tokens[0];
            uint32_t right = current_tokens[1];
            uint64_t key = (static_cast<uint64_t>(left) << 32) | static_cast<uint64_t>(right);

            uint32_t merged_id;
            auto rule_it = pair_to_rule.find(key);
            if (rule_it != pair_to_rule.end()) {
                merged_id = rule_it->second.target_id;
            } else {
                merged_id = next_vocab_index++;
                uint32_t rank = static_cast<uint32_t>(all_merges.size());
                BPEMergeRule rule{left, right, merged_id, rank};
                all_merges.push_back(rule);
                pair_to_rule[key] = rule;

                std::vector<uint8_t> m_bytes;
                auto itL = id_to_bytes.find(left);
                if (itL != id_to_bytes.end()) m_bytes.insert(m_bytes.end(), itL->second.begin(), itL->second.end());
                else if (left >= BYTE_OFFSET && left < SUBWORD_START) m_bytes.push_back(static_cast<uint8_t>(left - BYTE_OFFSET));

                auto itR = id_to_bytes.find(right);
                if (itR != id_to_bytes.end()) m_bytes.insert(m_bytes.end(), itR->second.begin(), itR->second.end());
                else if (right >= BYTE_OFFSET && right < SUBWORD_START) m_bytes.push_back(static_cast<uint8_t>(right - BYTE_OFFSET));

                id_to_bytes[merged_id] = m_bytes;
                std::string s(reinterpret_cast<const char*>(m_bytes.data()), m_bytes.size());
                id_to_str[merged_id] = s;
                str_to_id[s] = merged_id;
            }

            current_tokens.erase(current_tokens.begin());
            current_tokens[0] = merged_id;
        }

        uint32_t final_id = current_tokens.empty() ? UNK_TOKEN : current_tokens[0];
        str_to_id[word] = final_id;
        id_to_str[final_id] = word;
        id_to_bytes[final_id] = word_bytes;
        return final_id;
    }
};

} // namespace nexora
