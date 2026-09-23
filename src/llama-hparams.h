#pragma once

#include "llama.h"

#include <array>
#include <bitset>
#include <cassert>
#include <cmath>

// bump if necessary
#define LLAMA_MAX_LAYERS  512
#define LLAMA_MAX_EXPERTS 1024 // Kimi K3
#define LLAMA_MAX_PLE_NGRAM 8  // qwen4exp
#define LLAMA_MAX_PLE_HEADS 64 // qwen4exp

enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_NONE           = 0,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX        = 1,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID        = 2,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = 3, // applied to the router weights instead of the logits
    LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS  = 4,
};

enum llama_swa_type {
    LLAMA_SWA_TYPE_NONE      = 0,  // 不开启
    LLAMA_SWA_TYPE_STANDARD  = 1,  // 标准
    LLAMA_SWA_TYPE_CHUNKED   = 2,  // 分块
    LLAMA_SWA_TYPE_SYMMETRIC = 3,  // 对称，前后全看
};

// how the non-causal mask should be constructed with llama_set_causal_attn(ctx, false)
// (e.g. mtmd decoding image tokens)
enum llama_non_causal_type {
    LLAMA_NON_CAUSAL_TYPE_ALL      = 0, // all layers non-causal, SWA still applied (gemma 3, qwen-vl, ...)
    LLAMA_NON_CAUSAL_TYPE_SWA_ONLY = 1, // SWA layers non-causal, dense layers stay causal (gemma 4)
    LLAMA_NON_CAUSAL_TYPE_SWA_FULL = 2, // all layers non-causal, SWA not applied between tokens of the current ubatch (deepseek 4)
};

// forward declaration; full definition in llama-graph.h
enum llm_ffn_op_type : int;

struct llama_hparams_posnet {
    uint32_t n_embd;
    uint32_t n_layer;  // 模型的层数
};

struct llama_hparams_convnext {
    uint32_t n_embd;
    uint32_t n_layer;
};

struct llama_hparams {
    // note: use the `_impl` suffix to avoid name conflict between members and getters
    //       for example: n_embd_out() vs n_embd_out_impl

    // 与模型架构相关
    bool vocab_only;  // 只加载词表
    bool no_alloc;  // 不分配内存，加载时会解析出模型所有张量的结构、形状、类型，但不真正向系统申请内存块。
    // 主要用于“估算”。比如程序想先看看这个模型如果全加载需要多少 GB 显存，如果发现显存不够，就提前报错，而不是等申请内存失败再崩溃。
    bool rope_finetuned;  // RoPE 微调
    bool use_par_res;  // 并行残差
    bool swin_norm;  // Swin 归一化，决定归一化的顺序。
    bool norm_before_residual = false;
    bool norm_before_fc       = false;

    // 基础维度相关
    uint32_t n_ctx_train;  // context size the model was trained on 训练上下文
    uint32_t n_embd;  // 嵌入向量维度
    uint32_t n_layer_all;
    uint32_t n_layer_nextn = 0;  // 预测后几个词（MTP / nextN，旧版字段名 nextn_predict_layers）

    // granite-switch: index of the single-head "router" KV layer that encodes
    // per-token adapter selection. -1 when the model has no such layer.
    int32_t  router_layer = -1;
    uint32_t n_expert = 0;  // MoE 的专家总数
    // 主要用于 T5 等模型。它不是用 RoPE 来标位置，而是把词与词之间的距离放进不同的“桶”里来给分。这个值定义了有多少个距离区间。
    uint32_t n_rel_attn_bkts = 0;

    // TODO: this needs to be reworked
    // 缓存与位置相关
    // 一个优化参数。在一些新型模型中，并不是每一层都有 KV 缓存。如果这个值 >= 0，只有前 N 层会占用显存来存 KVCache，
    // 后面的层可能共用或不存
    int32_t  n_layer_kv_from_start = -1; // if non-negative, the first n_layer_kv_from_start layers have KV cache

    // different head size for full_attention and SWA layers
    uint32_t n_embd_head_k_full; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v_full; // dimension of values (d_v) aka n_embd_head
    uint32_t n_embd_head_k_swa;
    uint32_t n_embd_head_v_swa;

    // different RoPE dimensions for full_attention and SWA layers
    // 旋转维度：RoPE 只作用于向量的一部分。比如向量是 128 维，可能只有前 64 维在“旋转”以标记位置。
    // 新版按注意力类型拆成两份：n_rot_full（全注意力层）与 n_rot_swa（滑动窗口层）。
    uint32_t n_rot_full;
    uint32_t n_rot_swa;

    // note: deepseek2 using MLA converts into MQA with larger heads, then decompresses to MHA
    uint32_t n_embd_head_k_mla_impl = 0;
    uint32_t n_embd_head_v_mla_impl = 0;

    // for WavTokenizer
    struct llama_hparams_posnet   posnet;
    struct llama_hparams_convnext convnext;

    // 专门为 Mamba 或 RWKV-6/7 等非 Transformer 架构准备的参数
    uint32_t n_shortconv_l_cache  = 0;

    // 存储每一层的头数和键值头数
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    // per-layer expert feed-forward size
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_exp_arr;
    // per-layer top-k expert routing count
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_expert_used_arr;

    // 在一些 MoE（混合专家）模型中，并不是所有层都是专家层。
    // 有些模型前几层是普通的稠密层。这个值定义了模型最开始有多少层是纯稠密的。
    uint32_t n_layer_dense_lead = 0;
    // deepseek 专用，MLA 将 Q, K, V 压缩进一个低秩的潜在空间。
    // 这两个值分别代表了 Q 和 KV 压缩后的秩（维度）。通过这种低秩分解技术，可以用小代价还原出全量的注意力。
    uint32_t n_lora_q           = 0;
    uint32_t n_lora_kv          = 0;
    // 专家系统进阶，DeepSeek 的专家系统比一般的 Mixtral 复杂得多。
    // 不同的专家角色维度。exp 是常规路由专家，
    // shexp 是 Shared Expert（共享专家），chexp 则是 Coupled/Condensed Expert（耦合专家）。
    uint32_t n_ff_shexp         = 0;  // 常规路由专家隐层维度
    uint32_t n_ff_chexp         = 0;  // 耦合专家隐层维度
    // 共享专家（Shared Experts）：
    //    普通的 MoE 每层都有自己独立的专家组。而 DeepSeek 引入了“共享专家”的概念。
    //    这几个“共享专家”是所有 Token 都要经过的，相当于一个基础的“公共知识库”。
    //    而其他专家是“班级专用”的。这能大幅减少参数量。
    uint32_t n_expert_shared    = 0;  // 共享专家数
    // 用于 GroupNorm 或 RMSNorm 的分组计算。有些模型不是对整个向量做归一化，而是切成几块（Group）分别归一化。
    uint32_t n_norm_groups      = 0;
    // 分层专家组。为了提高效率，DeepSeek 把成百上千个专家分成了若干个“小组”。
    // 路由时，先选组，再在组里选专家。这三个值分别定义了：总共有多少组、每次激活多少个组、每组里有多少个专家。
    uint32_t n_expert_groups    = 0;
    uint32_t n_group_used       = 0;
    uint32_t n_group_experts    = 0;

    // MLA + SWA (i.e. dots3note)
    uint32_t n_lora_kv_swa           = 0;
    uint32_t n_embd_head_k_mla_swa   = 0;
    uint32_t n_embd_head_v_mla_swa   = 0;

    // MoE 路由缩放相关
    float    expert_group_scale   = 0.05f;  // 专家组缩放
    float    expert_weights_scale = 0.0f;  // 专家权重缩放
    bool     expert_weights_norm  = false;  // 专家权重归一化开关
    // 门控函数相关
    // 一个枚举值，定义了用哪种数学公式来“找专家”，常见的有：Softmax、Sigmoid 或者 NONE。
    uint32_t expert_gating_func   = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;  // 专家门控函数类型
    uint32_t moe_every_n_layers   = 0;  // 每隔多少层使用 MoE，如果这个值是 2，意味着：一层 Dense，一层 MoE
    uint32_t moe_latent_size      = 0;

    // 归一化的防止除 0 加的小常数
    float f_norm_eps;  // 用于标准的 LayerNorm
    float f_norm_rms_eps;  // 用于 RMSNorm
    float f_norm_group_eps;  // 用于 GroupNorm

    // 软截断 Soft-Capping，在模型计算中，Logits（逻辑值） 是经过矩阵乘法算出来的“生原始分”。
    // 如果这个分值太大（比如 1000），经过 Softmax 后会变成极其悬殊的概率分布，导致计算不稳定。
    // 以前大家用 Hard Clipping（强行把超过 50 的数变成 50），但这样太暴力，会让导数断掉。
    // 现在用一个平滑的函数（通常是 tanh）把值限制在一定范围内。
    float f_attn_logit_softcapping   = 50.0f;  // 注意力的软截断，发生在计算 Q·K 之后，做 Softmax 之前。
    float f_router_logit_softcapping = 30.0f;  // MoE 路由得分的软截断，发生在 MoE 选专家的时候。
    float f_final_logit_softcapping  = 30.0f;  // 最终输出 Logits 的软截断，模型最后一层输出最终词表概率之前。

    // for RWKV
    uint32_t rescale_every_n_layers = 0;
    uint32_t time_mix_extra_dim     = 0;
    uint32_t time_decay_extra_dim   = 0;
    uint32_t wkv_head_size          = 0;
    uint32_t token_shift_count      = 2;
    uint32_t n_lora_decay           = 0;
    uint32_t n_lora_iclr            = 0;
    uint32_t n_lora_value_res_mix   = 0;
    uint32_t n_lora_gate            = 0;

    // RoPE 缩放参数
    float    rope_attn_factor = 1.0f;  // 训练时的注意力缩放因子
    float    rope_freq_base_train;  // 训练时的基础频率
    float    rope_freq_base_train_swa  = 10000.0f;  // SWA 时的基础频率，SWA 代表滑动窗口注意力
    float    rope_freq_scale_train;  // 训练时的频率缩放因子，线性
    float    rope_freq_scale_train_swa = 1.0f;  // SWA 时的频率缩放因子，线性
    float    rope_scaling_alpha        = 0.0f;  // NTK-aware alpha for XDRoPE

    // YARN 缩放参数
    uint32_t n_ctx_orig_yarn;  // 原始训练长度
    float    rope_yarn_log_mul = 0.0f;  // 对数乘法因子，用于对旋转频率进行对数空间下的微调，进一步平滑位置感

    float    yarn_ext_factor  = -1.0f;  // 外推因子，如果设为 -1，通常代表由程序根据缩放比例自动计算。
    float    yarn_attn_factor =  1.0f;  // 注意力缩放因子
    float    yarn_beta_fast   = 32.0f;  // 快速衰减因子，对于变化极快的特征维度，直接进行外推。
    float    yarn_beta_slow   =  1.0f;  // 慢速衰减因子，对于变化很慢的维度，进行内插。

    std::array<int, 4> rope_sections;

    // Per-layer RoPE enable flags (1 = use RoPE, 0 = NoPE)
    // by default, all layers use RoPE (controlled by rope_finetuned)
    std::array<uint32_t, LLAMA_MAX_LAYERS> rope_pattern;

    // Sliding Window Attention (SWA)
    llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;
    // the size of the sliding window (0 - no SWA)
    uint32_t n_swa = 0;

    // see llama_non_causal_type
    // note: for SWA_FULL, older tokens (outside the current ubatch) are still window-clipped
    llama_non_causal_type non_causal_type = LLAMA_NON_CAUSAL_TYPE_ALL;

    // if is_swa_impl[il] == 1, then layer il is SWA
    // if is_swa_impl[il] == 0, then layer il is dense (i.e. non-SWA)
    // by default, all layers are dense
    // note: using uint32_t type for compatibility reason
    std::array<uint32_t, LLAMA_MAX_LAYERS> is_swa_impl;

    // for hybrid state space models
    std::array<uint32_t, LLAMA_MAX_LAYERS> is_recr_impl;

    // for State Space Models
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    uint32_t ssm_n_group = 0;

    // for MiniMax-Text-01 linear attention
    uint32_t n_embd_head_la = 0;

    // for Kimi Linear KDA
    uint32_t n_embd_head_kda = 0;
    bool     kda_safe_gate = false;

    // kimi-k3
    uint32_t n_expert_latent      = 0;      // routed_expert_hidden_size (0 = experts run at n_embd)
    uint32_t attn_res_block_size  = 0;      // 0 = no cross-layer attention residuals
    float    kda_gate_lower_bound = -INFINITY;
    float    situ_beta            = 1.0f;
    float    situ_linear_beta     = 0.0f;   // 0 = no linear-beta transform on the up branch

    // hrm-text (looped H/L stacks)
    uint32_t n_hrm_layers_per_stack = 0;
    uint32_t n_hrm_h_cycles = 0;
    uint32_t n_hrm_l_cycles = 0;
    bool     hrm_prefix_lm = false;

    // 为 Mamba 准备，SSM 归一化开关
    bool ssm_dt_b_c_rms = false;

    // 为 Mamba 准备
    float f_clamp_kqv      = 0.0f;  // 硬截断，KQV 强行截断
    float f_max_alibi_bias = 0.0f;  // ALiBi 偏置上限
    float f_logit_scale    = 0.0f;  // Logits 缩放因子

    // Additional scale factors (Granite/Granite MoE)
    float f_residual_scale  = 0.0f;  // 残差缩放因子
    float f_embedding_scale = 0.0f;  // Embedding 缩放因子
    float f_attention_scale = 0.0f;  // 注意力缩放因子

    // grok-2
    float    f_attn_out_scale = 0.0f;  // 注意力输出缩放
    uint32_t attn_temp_length = 0;  // 注意力温度长度

    float    f_attn_value_scale = 0.0f;

    bool causal_attn   = true;  // 因果注意力开关
    bool use_alibi     = false;  // ALiBi 偏置开关
    bool attn_soft_cap = false;  // 注意力软截断开关
    bool use_kq_norm   = false;  // KQ 归一化开关

    // for Classifiers
    uint32_t n_cls_out = 1;

    // input embedding dimension (0 = use n_embd)
    uint32_t n_embd_inp_impl = 0;

    // encoder input embedding dimension (0 = use n_embd_inp())
    // e.g. the eagle3 encoder fuses target_layers * target_hidden features
    uint32_t n_embd_inp_enc_impl = 0;

    // output embedding dimension (0 = use n_embd)
    // output embedding dimension (0 = use n_embd，即中间是多少维，输出就是多少维)
    // 模型的中间向量是 4096 维，输出到词表预测下一个词时，直接用 4096 维数组去乘词表矩阵。
    // 有些模型在最后输出前，会做一个 “瓶颈压缩”或“扩张”（比如把 4096 压缩成 1024 维，再送给词表）。
    uint32_t n_embd_out_impl = 0;

    uint32_t dflash_block_size       = 0;
    uint32_t dflash_conv_kernel_size = 0;
    uint32_t dflash_conv_group_size  = 0;
    uint32_t dflash_selector_rank    = 0;
    uint32_t dflash_selector_top_k   = 0;

    // llama4 smallthinker
    uint32_t n_moe_layer_step        = 0;  // 每隔多少层使用 MoE
    uint32_t n_no_rope_layer_step    = 4;  // 有多少层在计算时可以“关掉”位置编码
    uint32_t n_attn_temp_floor_scale = 0;  // 缩放系数的下限
    float    f_attn_temp_scale       = 0.0f;  // 缩放系数
    float    f_attn_temp_offset      = 0.0f;  // 偏移量

    // gemma3n altup
    uint32_t n_altup      = 4; // altup_num_inputs
    uint32_t i_altup_act  = 0; // altup_active_idx
    uint32_t laurel_rank  = 64;
    uint32_t n_embd_altup = 256;

    // needed for sentence-transformers dense layers
    uint32_t dense_2_feat_in  = 0;  // in_features of the 2_Dense
    uint32_t dense_2_feat_out = 0;  // out_features of the 2_Dense
    uint32_t dense_3_feat_in  = 0;  // in_features of the 3_Dense
    uint32_t dense_3_feat_out = 0;  // out_features of the 3_Dense

    // xIELU
    std::array<float, LLAMA_MAX_LAYERS> xielu_alpha_n;  // 负向 Alpha，控制输入值为负数时的曲线斜率
    std::array<float, LLAMA_MAX_LAYERS> xielu_alpha_p;  // 正向 Alpha，控制输入值为正数时的曲线斜率
    std::array<float, LLAMA_MAX_LAYERS> xielu_beta;  // 贝塔系数，总体的缩放偏移量，用来调整激活函数的零点位置
    std::array<float, LLAMA_MAX_LAYERS> xielu_eps;  // 数值稳定性常数，防指数计算精度丢失（旧版为标量，现按层拆分）

    // DSA (deepseek sparse attention)
    uint32_t indexer_n_head    = 0;
    uint32_t indexer_head_size = 0;
    uint32_t indexer_top_k     = 0;
    // MSA
    uint32_t indexer_block_size  = 0;
    uint32_t indexer_local_blocks = 0;

    // Indexer is "full" (1) or "shared" (0)
    // Shared indexers reuse top-k from previous full layer
    std::array<uint32_t, LLAMA_MAX_LAYERS> is_indexer_full_impl;

    // DeepSeek-V4
    uint32_t dsv4_o_group_count        = 0;
    uint32_t dsv4_o_lora_rank          = 0;
    uint32_t dsv4_hc_mult              = 0;
    uint32_t dsv4_hc_sinkhorn_iters    = 0;
    uint32_t dsv4_hash_layer_count     = 0;
    float    dsv4_compress_rope_base   = 0.0f;
    float    dsv4_hc_eps               = 0.0f;
    std::array<uint32_t, LLAMA_MAX_LAYERS> dsv4_compress_ratios;

    // 0 = full rank (DeepSeek-V4)
    uint32_t hc_low_rank = 0;

    // scale of the hyper-connection post gate (DeepSeek-V4 hardcodes 2.0)
    float    hc_magnitude = 0.0f;

    uint32_t ple_ngram_size      = 0;
    uint32_t ple_heads_per_ngram = 0;
    uint32_t ple_conv_kernel     = 0;
    uint32_t ple_n_heads         = 0;   // (ngram_size - 1) * heads_per_ngram
    uint32_t ple_head_dim        = 0;
    uint32_t ple_eos_token_id    = 0;
    // the id the PLE hash stands in at image positions; 0 makes the loader fall back to EOS
    uint32_t ple_image_token_id  = 0;
    // the file lists PLE layer indices, so this is never a per-layer gguf array and can hold one bit per layer
    std::bitset<LLAMA_MAX_LAYERS> is_ple_impl;
    // the hash multipliers reach ~2e13 and have to stay 64-bit
    std::array<uint64_t, LLAMA_MAX_PLE_NGRAM>  ple_layer_multipliers;
    // head offsets and vocab sizes are token-space indices; the gather truncates them to int32 anyway
    std::array<uint32_t, LLAMA_MAX_PLE_HEADS>  ple_head_offsets;
    std::array<uint32_t, LLAMA_MAX_PLE_HEADS>  ple_head_vocab_sizes;

    bool is_ple(uint32_t il) const;

    // PLE conv history rows: (kernel - 1) * ngram_size; 0 without a PLE module
    uint32_t ple_conv_state() const;

    // qwen3vl deepstack
    // When parsed from GGUF, this implies the first N layers consume the first
    // N deepstack embeddings. Use deepstack_mapping_arr if you need a more
    // complex mapping. If using deepstack_mapping_arr, also make sure to set
    // n_deepstack_layers to the number of unique deepstack layers so that
    // n_embd_imp is accurate (see granite.cpp).
    // TODO: can be expressed via the `new n_embd_inp_impl` and remove this param
    uint32_t n_deepstack_layers = 0;

    // deepstack layer array (Granite4 Vision)
    // -1  => no deepstack
    // >=0 => input embedding index for deepstack injection
    std::array<int32_t, LLAMA_MAX_LAYERS> deepstack_mapping_arr;

    // gemma4 per-layer embedding
    uint32_t n_embd_per_layer = 0;

    // needed by encoder-decoder models (e.g. T5, FLAN-T5)
    // ref: https://github.com/ggml-org/llama.cpp/pull/8141
    llama_token dec_start_token_id = LLAMA_TOKEN_NULL;  // 解码起始 token
    uint32_t    dec_n_layer        = 0;  // 解码器层数

    enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;  // 池化类型
    enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;  // RoPE 类型
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;  // RoPE 缩放类型


    // Resolved FFN gated activation flavor for archs that read
    // `<arch>.hidden_activation` from the GGUF (e.g. ModernBert derivatives).
    // Defaults to LLM_FFN_NONE (sentinel = 0); the mapping from the GGUF
    // string to a real op is done at hparam-load time via
    // llm_ffn_op_type_from_string() in llama-model.cpp, mirroring how
    // rope_scaling_type_train is handled.
    enum llm_ffn_op_type llm_ffn_op;

    // Step35: optional per-layer clamps for (Swi)GLU
    std::array<float, LLAMA_MAX_LAYERS> swiglu_clamp_exp; // clamping for expert FFN
    std::array<float, LLAMA_MAX_LAYERS> swiglu_clamp_shexp; // shared expert

    // this value n_pattern means that every nth layer is dense (i.e. non-SWA)
    // dense_first means whether the pattern is start with a dense layer
    // note that if n_pattern == 0, all layers are SWA
    //           if n_pattern == 1, all layers are dense
    // example 1: n_pattern = 3, dense_first = false
    //   il == 0: swa
    //   il == 1: swa
    //   il == 2: dense
    //   il == 3: swa
    //   il == 4: swa
    //   il == 5: dense
    //   il == 6: swa
    //   etc ...
    // example 2: n_pattern = 2, dense_first = true
    //   il == 0: dense
    //   il == 1: swa
    //   il == 2: dense
    //   il == 3: swa
    //   etc ...
    void set_swa_pattern(uint32_t n_pattern, bool dense_first = false);

    // return true if one of the layers is SWA
    bool is_swa_any() const;

    bool is_swa(uint32_t il) const;

    bool is_indexer_full(uint32_t il) const;

    void set_recr_pattern(uint32_t n_pattern, bool dense_first = false);

    // whether or not the given layer is recurrent (for hybrid models)
    bool is_recr(uint32_t il) const;

    // il = Index of Layer（层索引）
    uint32_t n_head(uint32_t il = 0) const;

    uint32_t n_head_kv(uint32_t il = 0) const;

    uint32_t n_ff(uint32_t il = 0) const;

    uint32_t n_ff_exp(uint32_t il = 0) const;

    uint32_t n_expert_used(uint32_t il = 0) const;

    // return the maximum n_expert_used across all layers
    uint32_t n_expert_used_max() const;

    // GQA (Grouped-Query Attention，分组查询注意力)
    uint32_t n_gqa(uint32_t il = 0) const;

    uint32_t n_rot(uint32_t il = 0) const;

    // dimension of main + auxiliary input embeddings
    uint32_t n_embd_inp() const;

    // dimension of the encoder input embeddings
    uint32_t n_embd_inp_enc() const;

    // dimension of output embeddings
    uint32_t n_embd_out() const;

    // dimension of key/value embeddings for each head (per layer)
    uint32_t n_embd_head_k(uint32_t il = 0) const;
    uint32_t n_embd_head_v(uint32_t il = 0) const;

    // dimension of key embeddings across all k-v heads
    uint32_t n_embd_k_gqa(uint32_t il = 0) const;

    // dimension of value embeddings across all k-v heads
    uint32_t n_embd_v_gqa(uint32_t il = 0) const;

    // true if any layer has a different n_embd_k_gqa/n_embd_v_gqa
    bool is_n_embd_k_gqa_variable() const;
    bool is_n_embd_v_gqa_variable() const;

    // return the maximum n_embd_k_gqa/n_embd_v_gqa across all layers
    uint32_t n_embd_k_gqa_max() const;
    uint32_t n_embd_v_gqa_max() const;

    // dimension of the rolling state embeddings
    // corresponds to Mamba's conv_states size or RWKV's token_shift states size
    uint32_t n_embd_r() const;

    // dimension of the recurrent state embeddings
    uint32_t n_embd_s() const;

    uint32_t n_pos_per_embd() const;

    // note: currently only support if either all or none of the layers are MLA
    bool is_mla() const;

    uint32_t n_embd_head_k_mla() const;
    uint32_t n_embd_head_v_mla() const;

    bool has_kv(uint32_t il) const;

    bool has_rope(uint32_t il) const;

    // number of effective layers (excludes nextn layers)
    uint32_t n_layer() const;

    // note that this function uses different SWA parameters from those in the hparams
    // note: inlined on purpose for performance reasons
    // TODO: think of a better place for this function
    // TODO: pack the SWA params in a struct?
    static bool is_masked_swa(uint32_t n_swa, llama_swa_type swa_type, llama_pos p0, llama_pos p1) {
        assert(p0 >= 0 && p1 >= 0);

        switch (swa_type) {
            case LLAMA_SWA_TYPE_NONE:
                {
                } break;
            case LLAMA_SWA_TYPE_STANDARD:
                {
                    if (p1 - p0 >= (int32_t) n_swa) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_CHUNKED:
                {
                    const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;

                    if (p0 < pos_chunk_start) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_SYMMETRIC:
                {
                    const int32_t half_n_swa = (int32_t) n_swa / 2;
                    const int32_t pos_diff = p1 - p0;

                    // Mask if outside the symmetric window
                    if (pos_diff < -half_n_swa || pos_diff > half_n_swa) {
                        return true;
                    }
                } break;
        }

        return false;
    }


    bool use_mrope() const;
};

static_assert(std::is_trivially_copyable<llama_hparams>::value, "llama_hparams must be trivially copyable");
