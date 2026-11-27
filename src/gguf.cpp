#include "gguf.h"

#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ink {

namespace {

struct Reader {
    const uint8_t * p;
    const uint8_t * end;
    template <typename T> T rd() {
        if (p + sizeof(T) > end) throw std::runtime_error("gguf: truncated file");
        T v;
        memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        uint64_t n = rd<uint64_t>();
        if (p + n > end) throw std::runtime_error("gguf: truncated string");
        std::string s((const char *) p, n);
        p += n;
        return s;
    }
};

enum : uint32_t { T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

uint64_t type_row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
        case GG_F32: case GG_I32: return n * 4;
        case GG_F16: case GG_BF16: case GG_I16: return n * 2;
        case GG_I8: return n;
        case GG_I64: case GG_F64: return n * 8;
        case GG_Q8_0: return n / 32 * 34;
        case GG_Q5_K: return n / 256 * 176;
        case GG_Q4_0: return n / 32 * 18;
        case GG_Q5_0: return n / 32 * 22;
        case GG_Q4_K: return n / 256 * 144;
        case GG_Q6_K: return n / 256 * 210;
        case GG_IQ2_XXS: return n / 256 * 66;
        case GG_IQ2_XS: return n / 256 * 74;
        case GG_IQ2_S: return n / 256 * 82;
        case GG_IQ3_XXS: return n / 256 * 98;
        case GG_IQ1_S: return n / 256 * 50;
        case GG_IQ4_XS: return n / 256 * 136;
        case GG_MXFP4: return n / 32 * 17;
        case GG_Q2_0: return n / 64 * 18;
        case GG_NVFP4: return n / 64 * 36;
        default: throw std::runtime_error("gguf: unsupported tensor type " + std::to_string(type));
    }
}

}  // namespace

GGUF::GGUF(const std::string & path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st;
    fstat(fd_, &st);
    size_ = (uint64_t) st.st_size;
    void * m = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) throw std::runtime_error("gguf: mmap failed for " + path);
    base_ = (const uint8_t *) m;

    Reader r{base_, base_ + size_};
    if (r.rd<uint32_t>() != 0x46554747u) throw std::runtime_error("gguf: bad magic");
    uint32_t version = r.rd<uint32_t>();
    if (version < 2 || version > 3) throw std::runtime_error("gguf: unsupported version");
    uint64_t n_tensors = r.rd<uint64_t>();
    uint64_t n_kv = r.rd<uint64_t>();

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = r.str();
        uint32_t t = r.rd<uint32_t>();
        GGValue val;
        auto scalar = [&](uint32_t ty, GGValue & out) {
            switch (ty) {
                case T_U8: out.v = (uint64_t) r.rd<uint8_t>(); break;
                case T_I8: out.v = (int64_t) r.rd<int8_t>(); break;
                case T_U16: out.v = (uint64_t) r.rd<uint16_t>(); break;
                case T_I16: out.v = (int64_t) r.rd<int16_t>(); break;
                case T_U32: out.v = (uint64_t) r.rd<uint32_t>(); break;
                case T_I32: out.v = (int64_t) r.rd<int32_t>(); break;
                case T_U64: out.v = (uint64_t) r.rd<uint64_t>(); break;
                case T_I64: out.v = (int64_t) r.rd<int64_t>(); break;
                case T_F32: out.v = (double) r.rd<float>(); break;
                case T_F64: out.v = (double) r.rd<double>(); break;
                case T_BOOL: out.v = (bool) (r.rd<uint8_t>() != 0); break;
                case T_STR: out.v = r.str(); break;
                default: throw std::runtime_error("gguf: bad value type");
            }
        };
        if (t == T_ARR) {
            uint32_t et = r.rd<uint32_t>();
            uint64_t n = r.rd<uint64_t>();
            if (et == T_STR) {
                std::vector<std::string> a;
                a.reserve(n);
                for (uint64_t j = 0; j < n; ++j) a.push_back(r.str());
                val.v = std::move(a);
            } else if (et == T_F32 || et == T_F64) {
                std::vector<double> a(n);
                for (uint64_t j = 0; j < n; ++j) a[j] = et == T_F32 ? (double) r.rd<float>() : r.rd<double>();
                val.v = std::move(a);
            } else if (et == T_BOOL) {
                std::vector<bool> a(n);
                for (uint64_t j = 0; j < n; ++j) a[j] = r.rd<uint8_t>() != 0;
                val.v = std::move(a);
            } else {
                std::vector<int64_t> a(n);
                for (uint64_t j = 0; j < n; ++j) {
                    GGValue e;
                    scalar(et, e);
                    a[j] = std::holds_alternative<uint64_t>(e.v) ? (int64_t) std::get<uint64_t>(e.v) : std::get<int64_t>(e.v);
                }
                val.v = std::move(a);
            }
        } else {
            scalar(t, val);
        }
        kv_[key] = std::move(val);
    }

    tensors_.resize(n_tensors);
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GGTensor & t = tensors_[i];
        t.name = r.str();
        uint32_t nd = r.rd<uint32_t>();
        t.dims.resize(nd);
        for (uint32_t d = 0; d < nd; ++d) t.dims[d] = r.rd<uint64_t>();
        t.type = r.rd<uint32_t>();
        t.offset = r.rd<uint64_t>();
        index_[t.name] = i;
    }
    uint64_t align = u64("general.alignment", 32);
    uint64_t data_off = (uint64_t) (r.p - base_);
    data_off = (data_off + align - 1) / align * align;
    for (auto & t : tensors_) {
        t.data = base_ + data_off + t.offset;
        t.nbytes = type_row_bytes(t.type, t.nelements());
        if (t.data + t.nbytes > base_ + size_) throw std::runtime_error("gguf: tensor data out of range: " + t.name);
    }
}

GGUF::~GGUF() {
    if (base_) munmap((void *) base_, size_);
    if (fd_ >= 0) close(fd_);
}

static const GGValue & need(const std::map<std::string, GGValue> & kv, const std::string & key) {
    auto it = kv.find(key);
    if (it == kv.end()) throw std::runtime_error("gguf: missing key " + key);
    return it->second;
}

uint64_t GGUF::u64(const std::string & key) const {
    const auto & v = need(kv_, key).v;
    if (auto p = std::get_if<uint64_t>(&v)) return *p;
    if (auto p = std::get_if<int64_t>(&v)) return (uint64_t) *p;
    if (auto p = std::get_if<bool>(&v)) return *p;
    if (auto p = std::get_if<double>(&v)) return (uint64_t) *p;
    throw std::runtime_error("gguf: key is not an integer: " + key);
}

double GGUF::f64(const std::string & key) const {
    const auto & v = need(kv_, key).v;
    if (auto p = std::get_if<double>(&v)) return *p;
    if (auto p = std::get_if<uint64_t>(&v)) return (double) *p;
    if (auto p = std::get_if<int64_t>(&v)) return (double) *p;
    throw std::runtime_error("gguf: key is not a number: " + key);
}

bool GGUF::boolean(const std::string & key, bool def) const {
    if (!has(key)) return def;
    const auto & v = need(kv_, key).v;
    if (auto p = std::get_if<bool>(&v)) return *p;
    return u64(key) != 0;
}

const std::string & GGUF::str(const std::string & key) const {
    const auto & v = need(kv_, key).v;
    if (auto p = std::get_if<std::string>(&v)) return *p;
    throw std::runtime_error("gguf: key is not a string: " + key);
}

const std::vector<std::string> & GGUF::str_array(const std::string & key) const {
    const auto & v = need(kv_, key).v;
    if (auto p = std::get_if<std::vector<std::string>>(&v)) return *p;
    throw std::runtime_error("gguf: key is not a string array: " + key);
}

std::vector<int64_t> GGUF::int_array(const std::string & key) const {
    const auto & v = need(kv_, key).v;
    if (auto p = std::get_if<std::vector<int64_t>>(&v)) return *p;
    if (auto p = std::get_if<std::vector<bool>>(&v)) return std::vector<int64_t>(p->begin(), p->end());
    throw std::runtime_error("gguf: key is not an integer array: " + key);
}

size_t GGUF::array_size(const std::string & key) const {
    const auto & v = need(kv_, key).v;
    return std::visit([](const auto & a) -> size_t {
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, std::vector<int64_t>> || std::is_same_v<T, std::vector<double>> ||
                      std::is_same_v<T, std::vector<bool>> || std::is_same_v<T, std::vector<std::string>> ||
                      std::is_same_v<T, std::vector<uint64_t>>) return a.size();
        else return 0;
    }, v);
}

const GGTensor * GGUF::find(const std::string & name) const {
    auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const GGTensor & GGUF::get(const std::string & name) const {
    const GGTensor * t = find(name);
    if (!t) throw std::runtime_error("gguf: missing tensor " + name);
    return *t;
}

}  // namespace ink
