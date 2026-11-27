// Minimal read-only GGUF (v2/v3) parser over an mmap'ed file.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace ink {

enum GGType : uint32_t {
    GG_F32 = 0, GG_F16 = 1, GG_Q4_0 = 2, GG_Q5_0 = 6, GG_Q8_0 = 8, GG_Q4_K = 12, GG_Q5_K = 13, GG_Q6_K = 14,
    GG_IQ2_XXS = 16, GG_IQ2_XS = 17, GG_IQ3_XXS = 18, GG_IQ1_S = 19, GG_IQ2_S = 22, GG_IQ4_XS = 23, GG_MXFP4 = 39, GG_Q2_0 = 42, GG_I8 = 24, GG_I16 = 25, GG_I32 = 26, GG_I64 = 27, GG_F64 = 28,
    GG_BF16 = 30, GG_NVFP4 = 40,
};

struct GGTensor {
    std::string name;
    std::vector<uint64_t> dims;   // GGUF order: dims[0] is the innermost (contiguous) dimension
    uint32_t type = 0;
    uint64_t offset = 0;          // relative to the data section
    const uint8_t * data = nullptr;
    uint64_t nbytes = 0;
    uint64_t nelements() const { uint64_t n = 1; for (auto d : dims) n *= d; return n; }
};

struct GGValue {
    // scalars are widened; arrays are kept by element kind
    std::variant<std::monostate, uint64_t, int64_t, double, bool, std::string,
                 std::vector<uint64_t>, std::vector<int64_t>, std::vector<double>, std::vector<bool>, std::vector<std::string>> v;
};

class GGUF {
public:
    explicit GGUF(const std::string & path);   // throws std::runtime_error
    ~GGUF();
    GGUF(const GGUF &) = delete;
    GGUF & operator=(const GGUF &) = delete;

    bool has(const std::string & key) const { return kv_.count(key) != 0; }
    uint64_t u64(const std::string & key) const;
    uint64_t u64(const std::string & key, uint64_t def) const { return has(key) ? u64(key) : def; }
    double f64(const std::string & key) const;
    double f64(const std::string & key, double def) const { return has(key) ? f64(key) : def; }
    bool boolean(const std::string & key, bool def) const;
    const std::string & str(const std::string & key) const;
    std::string str(const std::string & key, const std::string & def) const { return has(key) ? str(key) : def; }
    const std::vector<std::string> & str_array(const std::string & key) const;
    std::vector<int64_t> int_array(const std::string & key) const;   // any integer/bool array, widened
    size_t array_size(const std::string & key) const;

    const GGTensor * find(const std::string & name) const;
    const GGTensor & get(const std::string & name) const;            // throws if missing
    const std::vector<GGTensor> & tensors() const { return tensors_; }

    const uint8_t * base() const { return base_; }
    uint64_t size() const { return size_; }
    const std::string & path() const { return path_; }

private:
    std::string path_;
    int fd_ = -1;
    const uint8_t * base_ = nullptr;
    uint64_t size_ = 0;
    std::map<std::string, GGValue> kv_;
    std::vector<GGTensor> tensors_;
    std::map<std::string, size_t> index_;
};

}  // namespace ink
