#pragma once

#include "llama.h"

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

using llama_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer_t>;

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

const char * llama_file_version_name(llama_fver version);

struct llama_model_loader {
    // Holds information on a model weight
    // 这是一个内部结构体，用于描述模型中的一个权重（Tensor）在原始文件中的位置和状态。
    // 当模型被加载时，加载器会先扫描文件头（GGUF 元数据），为每个张量创建一个这种“索引”，以便后续真正读取数据时知道去文件的哪里找。
    struct llama_tensor_weight {
        // 源文件索引：现在的模型往往很大，经常会被切分成多个文件（如 model-00001-of-00005.gguf）。
        // 这个 idx 就代表这个张量存放在第几个分片文件中。
        uint16_t  idx; // source file index
        // 数据在文件中的偏移量：它记录了该张量的原始二进制数据从文件的第几个字节开始。
        size_t   offs; // tensor data offset in the original file

        // 指向内存中张量对象的指针：这是在 GGML 计算图中代表该权重的对象。
        // 虽然这个结构体被创建时，数据可能还没读进内存，但这个指针已经指向了描述该权重形状（Shape）、类型（Type）的 ggml_tensor 结构。
        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            // 根据张量的名字（如 "token_embd.weight"），在 GGUF 文件的索引区查找它的序号 tensor_idx。
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            // 计算偏移量：
            // gguf_get_data_offset(gguf_ctx)：获取 GGUF 文件中所有张量数据块的起始位置（跳过文件头和 KV 键值对后的位置）。
            // gguf_get_tensor_offset(gguf_ctx, tensor_idx)：获取该特定张量在数据块中的相对偏移。
            // 两者相加，就是该张量在整个 .gguf 文件中的绝对起始偏移量。
            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }
    };

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            // 提取层号：
            // 尝试从权重名中提取层号（例如，从 "blk.10.attn_q.weight" 中提取 10）。
            // sscanf 函数在这里被用来解析字符串。
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            // 按层号排序：
            // 如果两个权重属于不同的层，就按照层号的大小来排序。
            // 这样就能确保 blk.0 的所有权重都在 blk.1 的所有权重之前。
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    // 非必需张量。如果文件中缺少该张量，加载器不会报错。
    static const int TENSOR_NOT_REQUIRED    = 1 << 0;
    // 重复张量。标记该张量的数据可能被多个对象共享，在内存分配时需要特殊处理。
    static const int TENSOR_DUPLICATED      = 1 << 1;
    // 跳过张量。该张量不应该被加载（例如，由于内存限制或用户配置）。
    static const int TENSOR_SKIP            = 1 << 2;
    static const int TENSOR_SKIP_IF_VIRTUAL = 1 << 3;
    static const int TENSOR_ALLOW_RESHAPE   = 1 << 4;
    static const int TENSOR_READ_LAZY       = 1 << 5; // read rows on demand instead of loading whole tensor; requires mmap for now

    // 模型的 KV 键值对数量。
    int n_kv      = 0;
    // 模型中的张量总数。
    int n_tensors = 0;
    // 已经在内存中成功创建的张量数量。
    int n_created = 0;

    // 模型中所有张量（参数）的总元素数量。
    uint64_t n_elements = 0;
    // 模型中所有张量的总字节数。
    size_t   n_bytes    = 0;

    // 是否使用内存映射（mmap）来加载模型。
    bool use_mmap = false;
    // 是否使用直接 I/O（Direct I/O）来读取文件。
    bool use_direct_io = false;
    // 是否检查张量。
    bool check_tensors;
    // 是否不分配内存。
    bool no_alloc;
    bool load_mtp;

    // handle TENSOR_READ_LAZY
    // use case: keep PLE / engrams embd tensors on disk, read them on demand
    struct lazy_read {
        // set by the caller before the create_tensor() calls
        enum llama_lazy_mode mode = LLAMA_LAZY_MODE_OFF;

        // decide whether this tensor is read lazily
        // pass w to also record it, or nullptr to only ask
        bool add(const std::string & name, const ggml_tensor * t, const llama_tensor_weight * w);

        bool any() const {
            return !ranges.empty();
        }

        bool has(const ggml_tensor * t) const {
            return tensors.count(ggml_get_name(t)) > 0;
        }

        const llama_mmap::ranges & for_file(uint32_t idx) const {
            static const llama_mmap::ranges none;

            const auto it = ranges.find(idx);
            return it == ranges.end() ? none : it->second;
        }

        // lazy tensors are gathered on the host, so no offload setting applies to them
        static ggml_backend_buffer_type_t buft();

    private:
        std::map<uint32_t, llama_mmap::ranges> ranges;
        std::set<std::string>                  tensors;
    } lazy;

    // 模型文件列表，管理由于过大而被切分成多个分片的 .gguf 文件。
    llama_files files;
    // 模型文件类型，例如 F16, Q4_K_M 等，代表了模型的压缩/量化级别。
    llama_ftype ftype;
    // 模型文件版本，GGUF 的版本号，如 v3。
    llama_fver  fver;

    // 一个维护所有 mmap 映射关系的列表，用于在模型销毁时释放资源。
    llama_mmaps mappings;

    // 权重名称到 llama_tensor_weight 映射，使用自定义比较器按层排序。
    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;
    // KV 键值对覆盖。用户可以在启动时手动指定一些参数，这些手动设置的值会存放在这里，加载时会优先使用它们，而不是文件里自带的值
    std::unordered_map<std::string, llama_model_kv_override> kv_overrides;
    // 张量缓冲区覆盖。允许用户在启动时指定某些张量使用特定的缓冲区类型（例如，强制使用 GPU 缓冲区而不是 CPU 缓冲区）。
    const llama_model_tensor_buft_override * tensor_buft_overrides;

    // GGUF 文件的元数据上下文。
    gguf_context_ptr metadata_ptr;
    struct gguf_context * metadata; // either metadata_ptr.get() or externally set
    llama_model_set_tensor_data_t set_tensor_data;
    void * set_tensor_data_ud;
    // GGUF 文件中的张量数据上下文。
    std::vector<ggml_context_ptr> contexts;

    // 模型架构名称。
    std::string arch_name;
    // 一个辅助对象，用于根据模型的架构快速读取对应的元数据键值。
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    // size_data 是总共要加载的数据量
    // size_done 是当前已经加载完成的数据量。它们配合起来用于显示加载进度条（xx%）。
    // 已使用的内存映射区间，记录了哪些文件片段已经被映射到了内存地址空间中。
    size_t size_done = 0;
    size_t size_data = 0;
    std::vector<std::pair<size_t, size_t>> mmaps_used;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    // lazy tensors need dedicated context
    struct ctx_key {
        ggml_backend_buffer_type_t buft;
        bool lazy;
    };

    struct ctx_key_comparator {
        bool operator()(const ctx_key & lhs, const ctx_key & rhs) const {
            if (lhs.lazy != rhs.lazy) {
                return lhs.lazy < rhs.lazy;
            }
            return strcmp(ggml_backend_buft_name(lhs.buft), ggml_backend_buft_name(rhs.buft)) < 0;
        }
    };

    std::map<ctx_key, ggml_context_ptr, ctx_key_comparator> ctx_map;

    // track tensors that had to be moved for debugging:
    size_t n_tensors_moved = 0;
    std::string first_tensor_moved_name;
    std::string first_tensor_moved_type_name;
    ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
    ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

    llama_model_loader(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits, // optional, only need if the split does not follow naming scheme
        FILE * file,
        llama_load_mode load_mode,
        bool check_tensors,
        bool no_alloc,
        bool load_mtp,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required = true);

    template<typename T>
    bool get_arr(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, bool required = true);

    template<typename T>
    bool get_key(enum llm_kv kid, T & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required = true);

    template<typename T>
    bool get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required = true);

    bool get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required = true);

    std::string get_arch_name() const;

    enum llm_arch get_arch() const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const std::string & name) const;

    const struct ggml_tensor * check_tensor_dims(
            const std::string & name,
            const std::vector<int64_t> & ne,
            bool required,
            bool allow_reshape) const;

    struct ggml_tensor * create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    void done_getting_tensors(bool partial = false) const;

    // mlock_mmaps：把映射锁死在 RAM 上，绝不允许被交换到硬盘分页里（旧版 llama_model_params.use_mlock 的继任机制）。
    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr);

    void get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const;

    // release a weight's mmap pages
    void unmap_weight(const llama_tensor_weight & w) const;

    // read a byte range of a weight's data
    // with mmap, returns a pointer into the mapping, otherwise reads into buf and returns buf
    const void * load_data_range(const llama_tensor_weight & w, size_t offs, size_t size, void * buf) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    std::string ftype_name() const;

    void print_info() const;
};
