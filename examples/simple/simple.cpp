#include "llama.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

// 测试
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // path to the model gguf file
    std::string model_path;
    // prompt to generate text from
    // 初始提示词
    std::string prompt = "Hello my name is";
    // number of layers to offload to the GPU
    // n_gpu_layers 的缩写，允许把模型中的多少层放到显卡上
    int ngl = 99;
    // number of tokens to predict
    // 预测词数，即限制大模型生成多长
    int n_predict = 32;

    // parse command line arguments

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        n_predict = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // prompt starts here
                break;
            }
        }
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            prompt = argv[i++];
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // load dynamic backends

    // 解析命令行参数
    // 在 llama.cpp 和它底层的 ggml 张量计算库中，backends（后端）指的是“专门用于执行矩阵计算的底层硬件加速引擎”。
    // 简单来说：不管大模型多复杂，本质上都是海量的矩阵乘法。
    // 由于每个人的电脑硬件不同，有的有 NVIDIA 独立显卡，有的是苹果 M 系列芯片，有的是纯靠 CPU 算，
    // 所以程序需要不同的“底层驱动”去指挥这些不同的硬件干活，这些驱动在这里统称为 backends。
    ggml_backend_load_all();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // 这两行代码的核心作用是：获取模型的词表，并计算出一段文本（prompt）会被拆解成多少个 Token。
    const llama_vocab * vocab = llama_model_get_vocab(model);
    // tokenize the prompt

    // find the number of tokens in the prompt
    // tokenize the prompt
    // find the number of tokens in the prompt
    // llama_tokenize 函数的作用：将字符串转换为数字 ID 数组。
    // 传入 NULL, 0, true, true，表示不传入数组，只计算 token 数量
    // 告诉函数：我现在手头没有任何地方（NULL）来存这些 Token，并且我提供的空位数量是 0。
    // 函数内部发现你没给地方存，它就不会执行“写入”操作。但它会完成所有的分词逻辑，计算出如果真的要存，需要多少个空位
    // 如果提供的缓冲区不够大（比如你给了 0 个位置），会返回所需 Token 数量的负值
    // 比如这句话分词后有 10 个 Token，由于你只给了 0 个位置，它就返回 -10。
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    // 再次调用 llama_tokenize，这次把容器的地址（.data()）和大小传进去。
    // 函数现在有地方存了，它会把分词后的 Token ID 一个个填进这个数组里。
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    // n_ctx is the context size
    // 动态分配上下文窗口大小
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    // 动态分配批处理大小
    // n_batch 决定了模型一次能处理多少个字。把这个值设为 Prompt 的总长度，
    // 意味着我们在初次“预习” Prompt 时，不用分批，直接一个 Batch 全部算完。
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    // enable performance counters
    // 启用性能计数器
    ctx_params.no_perf = false;

    // 核心：基于加载好的模型和参数，创建一个具体的“运行环境”（Context）。
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token

    for (auto id : prompt_tokens) {
        char buf[128];
        // 在模型内部，文字全都被变成了数字（Token ID，比如“苹果”可能是 1234），因此需要把这些数字翻译回人类能看懂的文字（或文字碎片）。
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }

    // prepare a batch for the prompt

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // 大部分我们熟悉的模型（比如 Llama, Qwen, GPT）都是 “只有解码器（Decoder-only）” 的架构。它们直接从你的第一句话往后接龙。
    // 但有些模型（比如 T5, Whisper, BART）有两个大脑：
    // 编码器（Encoder）：负责“理解”你输入的整段话。
    // 解码器（Decoder）：负责根据理解的内容，从零开始“写”出回答。
    if (llama_model_has_encoder(model)) {
        // 程序把你的 Prompt 整块丢给 Encoder。这个大脑会把你的文字转化成一组复杂的数学特征（隐藏状态），存放在模型内部。
        if (llama_encode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return 1;
        }

        // 找到 Decoder 的“启动信号”。
        // 对于 Llama 这种模型，这个信号通常就是 BOS（Beginning of Sequence）Token。
        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        //  如果模型没有启动信号，程序就默认用 BOS (Beginning of Sentence，句子开头) 标记（通常是 <s>）。
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        // 准备一个只包含启动信号的新 Batch。这是 Decoder 开始“说话”的起点。
        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    // main loop

    const auto t_main_start = ggml_time_us();  // 记录开始时间
    int n_decode = 0;  // 记录生成的 token 数量
    llama_token new_token_id;  // 存储新生成的 token ID

    // 循环条件：只要当前处理的位置加上 Batch 大小，还没达到“预估长度”（Prompt长度 + 想要生成的长度），就继续。
    // n_pos: 当前处理到的位置（从 0 开始计数）。
    // batch.n_tokens：当前这批要处理的单词数。第一次循环：它是你输入的整段 Prompt 的长度。之后每次循环：它只有 1（即刚生成的那个新词）。
    // n_pos + batch.n_tokens: 加上这批数据后，总共处理到了哪里。
    // n_prompt + n_predict: 预估的总长度（Prompt长度 + 想要生成的长度）。
    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
        // evaluate the current batch with the transformer model
        // evaluate the current batch with the transformer model
        // 把 batch 里的词喂给模型，模型会计算出几万个候选词的概率分布（Logits）。
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        // 更新当前处理到的位置
        n_pos += batch.n_tokens;

        // sample the next token
        {
            // 从概率分布中采样下一个 token
            // llama_sampler_sample 是最终的“决策官”。它根据你设定的采样规则（比如“只选概率最高的”），在几万个词里选定一个词。
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // 将 token 转换回字符并打印
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    // 记录结束时间
    const auto t_main_end = ggml_time_us();

    // 打印生成速度
    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    // 打印采样环节（即从概率中选词的过程）用了多久。
    llama_perf_sampler_print(smpl);
    // 打印最核心的两个耗时：Prompt eval time：读你的问题用了多久（预习时间）。Eval time：一个字一个字蹦出来用了多久（生成时间）。
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);  // 销毁采样器
    llama_free(ctx);  // 释放推理上下文（包括显存里的 KV 缓存）
    llama_model_free(model);  // 从显存/内存中彻底卸载大模型（核心权重）

    return 0;
}
