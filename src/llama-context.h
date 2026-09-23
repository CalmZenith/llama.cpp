#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <array>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

// 核心数据结构
struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    // 作用：强制等待所有正在显卡上运行的计算任务全部完成，再继续往下走。强制同步
    void synchronize();

    // 返回指向“大脑（Model）”的引用
    const llama_model   & get_model()   const;
    // 返回指向“参数（Params）”的引用
    const llama_cparams & get_cparams() const;

    // 获取调度器
    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;  // 返回当前上下文的总容量（能记住多少个 Token）。
    uint32_t n_ctx_seq() const;  // 返回当前上下文的序列长度（实际用了多少 Token）。
    uint32_t n_batch()   const;  // 返回当前单次推理能处理的最大逻辑批次大小（一次最多能处理多少个 Token）。
    uint32_t n_ubatch()  const;  // 返回当前硬件底层执行的物理 批次大小（一次能处理多少个 Token）。
    uint32_t n_seq_max() const;  // 返回当前会话支持的最大并行序列数。

    uint32_t n_threads()       const;  // 返回生成单个 Token 时使用的 CPU 线程数。
    uint32_t n_threads_batch() const;  // 返回批量处理 Prompt 时使用的 CPU 线程数。

    // 返回一个包含内存细分数据的结构体,可以知道：为了运行这个会话，模型占了多少内存、上下文占了多少内存、临时计算又占了多少内存。
    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    // 查询当前使用的是哪种池化算法
    enum llama_pooling_type pooling_type() const;

    float * get_logits();  // 获取每个词可能出现的“原始分数”
    float * get_logits_ith(int32_t i);  // 获取第 i 个位置的 Logits（原始概率分布）。

    float * get_embeddings();  // 获取文本被模型高度浓缩后的坐标向量。
    float * get_embeddings_ith(int32_t i);  // 获取第 i 个位置的 Embedding 向量。
    float * get_embeddings_seq(llama_seq_id seq_id);  // 获取指定序列 ID 的 Embedding 向量。

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;  // 获取采样后的 Token 序列。
    llama_token   get_sampled_token_ith(int32_t idx);  // 获取采样后的第 i 个 Token。

    float * get_sampled_logits_ith(int32_t idx);  // 获取采样后第 i 个位置的 Logits。
    size_t  get_sampled_logits_count(int32_t idx);  // 获取采样后第 i 个位置的 Logits 数量。

    float * get_sampled_probs_ith(int32_t idx);  // 获取采样后第 i 个位置的概率分布。
    size_t  get_sampled_probs_count(int32_t idx);  // 获取采样后第 i 个位置的概率分布数量。

    const llama_token * get_sampled_candidates_ith(int32_t idx);  // 获取采样后第 i 个位置的候选 Token。
    size_t get_sampled_candidates_count(int32_t idx);  // 获取采样后第 i 个位置的候选 Token 数量。

    // 把一个外部的“线程池”挂载到当前会话上。
    // 通俗解释：如果同时开了好几个 AI 窗口（Context），可以让它们共用一套 CPU 核心，
    // 防止 CPU 因频繁切换任务而卡顿。
    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    // 卸载线程池
    void detach_threadpool();

    // 实时修改生成和批处理时使用的线程数，不需要重启模型
    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    // 设置中断回调函数，用于在生成过程中中断
    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    // 开启/关闭 Embedding 向量输出
    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    // 开启/关闭因果注意力机制
    void set_causal_attn(bool value);
    // 开启/关闭预热
    void set_warmup(bool value);

    // 加载 LoRA 适配器
    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    // 应用适配器向量（旧版函数名为 apply_adapter_cvec，参数不变）
    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    // 编码：把文字转成 Token ID
    int encode(const llama_batch & batch_inp);
    // 解码：把 Token ID 转成文字，它接收一个 llama_batch（你要给模型看的所有新词），然后启动整个推理流程。
    // 做了什么：它会自动把大的 Batch 拆成刚才说的 ubatch。它会协调显卡把这些词过一遍神经网络。
    // 最关键的：它会把这些词的信息存入 KV Cache（短期记忆）。这样模型在算下一个词的时候，就能记得刚才说了什么。
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    // 获取当前上下文的状态大小
    size_t state_get_size();
    // 获取当前上下文的状态数据
    size_t state_get_data(      uint8_t * dst, size_t size);
    // 设置当前上下文的状态数据
    size_t state_set_data(const uint8_t * src, size_t size);

    // 获取指定序列的状态大小
    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    // 获取指定序列的状态数据
    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    // 设置指定序列的状态数据
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    // 从文件中加载状态
    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    // 把当前上下文的状态保存到文件中
    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    // 从文件中加载指定序列的状态
    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    // 把指定序列的状态保存到文件中
    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    // 获取性能数据
    llama_perf_context_data perf_get_data() const;
    // 重置性能数据
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;  // 获取内存使用情况

    //
    // training
    //

    // 这部分代码涉及 llama.cpp 中一个比较高级且不常被普通用户看到的功能：模型优化与微调（Training/Optimization）。
    // 虽然我们平时主要用 llama.cpp 来 运行（推理） 模型，但它其实也内置了训练模型的能力（比如支持 Adam 或 L-BFGS 优化器）。
    // 初始化优化器的环境。
    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    //
    // output
    //
    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    // 确保有足够的空间来存放输出。
    uint32_t output_reserve(int32_t n_outputs);

    // 重新排列输出。
    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    // 计算图的最大节点数。
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    // 设置采样器。
    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_result * get_gf_res_prev();

    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    // 获取图的回调。
    llm_graph_cb graph_get_cb() const;

    // disable auto fused ops (Flash Attention, Gated Delta Net) whose op lands on a device
    // that differs from the layer it belongs to (usually due to missing backend support)
    void resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs);

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    // 读取状态数据。
    size_t state_read_data (llama_io_read_i  & io);

    // 写入指定序列的状态数据。
    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    // 读取指定序列的状态数据。
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    // 模型的引用。
    const llama_model & model;

    // 上下文参数。
    llama_cparams cparams;

    // 适配器参数。
    llama_adapter_cvec_ptr  cvec;
    // LoRA 适配器参数。
    llama_adapter_loras_ptr loras;

    // 跨注意力（Cross-Attention）的临时处理结构。
    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    // 内存管理。
    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    // 词汇表大小（float 类型）的指针。
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        // 采样器。
        // 这是一个映射表（Map），记录了每一个对话序列（seq_id）对应使用哪套采样规则（采样器）。
        std::map<llama_seq_id, llama_sampler *> samplers;

        // logits: 存放经过处理后的“原始得分”。
        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};  // 这是一个缓冲区，专门用来存放最终被选中（胜出）的那个 Token ID，这就是你最后在屏幕上看到的那个词。
        buffer_view<float>       probs      = {nullptr, 0};  // probs: 存放转化后的“百分比概率”。
        buffer_view<llama_token> candidates = {nullptr, 0};  // 存放所有进入“决赛圈”的候选词。如果你设置了 Top-K 为 10，那么这里就存着那 10 个最有希望的备选词。

        // 记录每个位置分别产生了多少个 Logits、Probs 或候选词。因为并不是每个位置都会产生相同数量的备选方案。
        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        // 存放完整词汇表 ID 的容器。
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_input_tensors = 0; // number of tensors marked as input during the last graph reserve
    // 实际输出数量，记录在当前这一轮计算中，真正产生了结果（Logits 或 Embeddings）的词有多少个。
    // 你可能一次喂给模型 10 个词，但你设置成只需要模型算出最后一个词的概率。那么 n_outputs 就是 1。
    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    // 映射批处理位置到 logits 和 embd 缓冲区的 ID。
    // 它是一个整数数组，记录了批次中的第 i 个 Token，对应输出缓冲区里的第几个结果。
    // 如果你只要求模型算出第 3 个和第 10 个词的概率，这个数组就会记录下这个对应关系。让程序能从一大堆计算结果中挑出需要的那几个。
    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    // 交换信息，用于交换两个位置的输出。
    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    // 在并行计算中，尤其是当一部分算在 CPU，一部分算在 GPU 时，最后出来的结果可能在内存里是乱序的。
    // output_swaps 记录了需要进行的 “位置交换”（比如把第 5 个位置和第 8 个位置的数据换一下），
    // 从而保证最后给用户的 Logits 数组是按正确单词顺序排列的。
    std::vector<swap_info> output_swaps;

    // 调度器指针，用于管理计算任务的调度。
    // 它负责根据你电脑的硬件（你有几个 CPU 核心、几块 GPU），智能地把神经网络计算任务拆开，分别派发给最合适的芯片去执行。
    ggml_backend_sched_ptr sched;

    // 如果你刚才通过代码改了大模型的一些关键参数（比如变长了上下文，或者加载了新的 LoRA 插件），
    // 这个标记就会 true。它提醒系统：原有的计算图空间可能已经不够用了，待会儿需要重新计算并预留一次物理内存。
    bool sched_need_reserve = true;

    // CPU 后端。
    // 专门指向 CPU 处理器的指针。不管你有多少块显卡，CPU 总是作为最后的“保底”存在（保底工）。
    ggml_backend_t backend_cpu = nullptr;
    // 这是一个容器，里面装着所有可用的“加速器”。
    // 如果你电脑有 NVIDIA 显卡，这里面就有一个 CUDA 指针；如果你是苹果 M1/M2/M3，这里面就有一个 Metal 指针。
    // 程序会根据这个列表，智能地把计算任务分发给它们。
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    // 线程池，用于并行计算。
    // 代表了用于“逐字蹦词”的 CPU 线程池
    ggml_threadpool_t threadpool       = nullptr;
    // 代表了用于“批量阅读”（Batch processing）的 CPU 线程池
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    // 中断回调函数的数据指针。
    void *              abort_callback_data = nullptr;

    // 背景：不同的硬件（比如苹果 M2 芯片、NVIDIA 显卡、Intel CPU）修改线程数的方法都不一样。
    // 功能：这个列表里存了一张“联系表”。上面记着每个计算设备对应的“修改线程数”的函数指针。
    // 好处：当你调用 set_n_threads 时，模型不需要知道这些硬件的具体细节，它只需要按着这张“指令单”，挨个打电话通知它们改速度就行了。
    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    // Separate arenas give batches with and without outputs distinct CUDA graph cache keys.
    std::array<llm_graph_result_ptr, 2> gf_res_prev;
    // 作用：为了提速。
    // Graph Reuse：构建一个计算图（告诉显卡该按什么顺序算哪一层）是很费时间的。如果这一次算的 Prompt 和上一次非常像，模型会去查 gf_res_prev。
    // 如果能复用之前的“解题思路（Graph）”，模型就不用重新画图，直接开算，大大减少了 CPU 的准备耗时。
    llm_graph_result_ptr gf_res_reserve;

    llm_graph_result * gf_res_prev_active = nullptr;

    // host buffer for the model output (logits and embeddings)
    // 作用：存放模型算出来的结果（Logits 和 Embeddings）。
    // 解释：模型每算完一个 Token，都会产生一堆数字（Logits），代表下一个词的概率。这些数字需要一个地方存着，等填满一个缓冲区后，再通过后端传给 CPU 或 GPU 显示出来。
    // 简单说：它就是模型算完账后，把“账单”暂时放在这里的地方。
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    // 记录当前会话是否已经进行过至少一次计算。
    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;  // 整个会话启动的初始时刻。
    mutable int64_t t_load_us   = 0;  // 加载模型、初始化后端花了多久。
    mutable int64_t t_p_eval_us = 0;  // 预处理 Prompt（Tokenization + Graph 构建）花了多久。
    mutable int64_t t_eval_us   = 0;  // 实际生成 Token 的总耗时。

    mutable int64_t t_compute_start_us = 0;  // 当前这一次具体计算开始的瞬间。
    mutable int64_t n_queued_tokens    = 0;  // 当前这一次具体计算中，一共有多少个 Token 在排队等待处理。

    // 在 Prompt 阶段一共“读过”多少个词。
    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    // 在生成阶段一共“写出”多少个词。
    mutable int32_t n_eval   = 0; // number of eval calls

    // 上一次的计算图被复用了多少次。
    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
