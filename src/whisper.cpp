#include "whisper.h"

#include <fstream>

using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

static buft_list_t make_buft_list(whisper_context_params& params)
{
    buft_list_t buft_list;

    // CPU Extra
    auto *cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    auto *cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);
    auto get_extra_bufts_fn =
        (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
            cpu_reg, "ggml_backend_dev_get_extra_bufts");
    if (get_extra_bufts_fn) {
        ggml_backend_buffer_type_t *extra_bufts = get_extra_bufts_fn(cpu_dev);
        while (extra_bufts && *extra_bufts) {
            buft_list.emplace_back(cpu_dev, *extra_bufts);
            ++extra_bufts;
        }
    }

    // CPU
    buft_list.emplace_back(cpu_dev, ggml_backend_cpu_buffer_type());
    return buft_list;
}

template <typename T>
static void read_safe(whisper_model_loader* loader, T& dest)
{
    loader->read(loader->context, &dest, sizeof(T));
}

const char* whisper_lang_str(int id)
{
    for (const auto &kv : g_lang) {
        if (kv.second.first == id) {
            return kv.first.c_str();
        }
    }
    fprintf(stderr, "%s: unknown language id %d\n", __func__, id);
    return nullptr;
}

static bool whisper_model_load(struct whisper_model_loader* loader, whisper_context& wctx)
{
    fprintf(stdout, "%s: loading model\n", __func__);

    const int64_t t_start_us = ggml_time_us();

    wctx.t_start_us = t_start_us;

    auto& model = wctx.model;
    auto& vocab = wctx.vocab;

    // verify magic
    {
        uint32_t magic;
        read_safe(loader, magic);
        if (magic != GGML_FILE_MAGIC)
        {
            fprintf(stderr, "%s: invalid model data (bad magic)\n", __func__);
            return false;
        }
    }

    // load hparams
    {
        auto& hparams = model.hparams;

        read_safe(loader, hparams.n_vocab);
        read_safe(loader, hparams.n_audio_ctx);
        read_safe(loader, hparams.n_audio_state);
        read_safe(loader, hparams.n_audio_head);
        read_safe(loader, hparams.n_audio_layer);
        read_safe(loader, hparams.n_text_ctx);
        read_safe(loader, hparams.n_text_state);
        read_safe(loader, hparams.n_text_head);
        read_safe(loader, hparams.n_text_layer);
        read_safe(loader, hparams.n_mels);
        read_safe(loader, hparams.ftype);

        std::string mver = "";

        if (hparams.n_audio_layer == 4)
            model.type = e_model::MODEL_TINY;

        if (hparams.n_audio_layer == 6)
            model.type = e_model::MODEL_BASE;

        if (hparams.n_audio_layer == 12)
            model.type = e_model::MODEL_SMALL;

        if (hparams.n_audio_layer == 24)
            model.type = e_model::MODEL_MEDIUM;

        if (hparams.n_audio_layer == 32)
        {
            model.type = e_model::MODEL_LARGE;

            if (hparams.n_vocab == 51866)
                mver = " v3";
        }

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;
        hparams.ftype %= GGML_QNT_VERSION_FACTOR;

        wctx.wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
        if (wctx.wtype == GGML_TYPE_COUNT) {
            fprintf(stderr, "%s: invalid model (bad ftype value %d)\n", __func__, model.hparams.ftype);
            return false;
        }

        fprintf(stdout, "%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        fprintf(stdout, "%s: n_audio_ctx   = %d\n", __func__, hparams.n_audio_ctx);
        fprintf(stdout, "%s: n_audio_state = %d\n", __func__, hparams.n_audio_state);
        fprintf(stdout, "%s: n_audio_head  = %d\n", __func__, hparams.n_audio_head);
        fprintf(stdout, "%s: n_audio_layer = %d\n", __func__, hparams.n_audio_layer);
        fprintf(stdout, "%s: n_text_ctx    = %d\n", __func__, hparams.n_text_ctx);
        fprintf(stdout, "%s: n_text_state  = %d\n", __func__, hparams.n_text_state);
        fprintf(stdout, "%s: n_text_head   = %d\n", __func__, hparams.n_text_head);
        fprintf(stdout, "%s: n_text_layer  = %d\n", __func__, hparams.n_text_layer);
        fprintf(stdout, "%s: n_mels        = %d\n", __func__, hparams.n_mels);
        fprintf(stdout, "%s: ftype         = %d\n", __func__, model.hparams.ftype);
        fprintf(stdout, "%s: qntvr         = %d\n", __func__, qntvr);
        fprintf(stdout, "%s: type          = %d (%s%s)\n", __func__, model.type, g_model_name.at(model.type).c_str(), mver.c_str());
    }

    // load mel filters
    {
        auto& filters = wctx.model.filters;

        read_safe(loader, filters.n_mel);
        read_safe(loader, filters.n_fft);

        filters.data.resize(filters.n_mel * filters.n_fft);
        loader->read(loader->context, filters.data.data(), filters.data.size() * sizeof(float));
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        read_safe(loader, n_vocab);

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int i = 0; i < n_vocab; i++)
        {
            uint32_t len;
            read_safe(loader, len);
            
            if (len > 0)
            {
                tmp.resize(len);
                loader->read(loader->context, &tmp[0], tmp.size());
            }
            else
            {
                word = "";
            }
            
            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }

        vocab.n_vocab = model.hparams.n_vocab;

        if (vocab.is_multilingual())
        {
            fprintf(stderr, "%s: Our whisper_lite.cpp supports only English-based models for simplicity.\n", __func__);
            return false;
        }
        
        if (n_vocab < model.hparams.n_vocab)
        {
            fprintf(stdout, "%s: adding %d extra tokens\n", __func__, model.hparams.n_vocab - n_vocab);
            for (int i = n_vocab; i < model.hparams.n_vocab; i++)
            {
                if (i > vocab.token_beg) {
                    word = "[_TT_" + std::to_string(i - vocab.token_beg) + "]";
                } else if (i == vocab.token_eot) {
                    word = "[_EOT_]";
                } else if (i == vocab.token_sot) {
                    word = "[_SOT_]";
                } else if (i == vocab.token_translate) {
                    word = "[_TRANSLATE_]";
                } else if (i == vocab.token_transcribe) {
                    word = "[_TRANSCRIBE_]";
                } else if (i == vocab.token_solm) {
                    word = "[_SOLM_]";
                } else if (i == vocab.token_prev) {
                    word = "[_PREV_]";
                } else if (i == vocab.token_nosp) {
                    word = "[_NOSP_]";
                } else if (i == vocab.token_not) {
                    word = "[_NOT_]";
                } else if (i == vocab.token_beg) {
                    word = "[_BEG_]";
                } else if (i > vocab.token_sot &&
                           i <= vocab.token_sot + vocab.num_languages()) {
                    word =
                        "[_LANG_" +
                        std::string(whisper_lang_str(i - vocab.token_sot - 1)) +
                        "]";
                } else {
                    word = "[_extra_token_" + std::to_string(i) + "]";
                }
                vocab.token_to_id[word] = i;
                vocab.id_to_token[i] = word;
            }
        }
    }

    const ggml_type wtype = wctx.wtype;
    const ggml_type vtype = wctx.wtype == GGML_TYPE_F32 ? GGML_TYPE_F32 : GGML_TYPE_F16;

    const auto& hparams = model.hparams;
    const int n_audio_layer = hparams.n_audio_layer;
    const int n_text_layer = hparams.n_text_layer;

    const size_t n_tensors =
        10 /* input */ + 15 + 15 * n_audio_layer + 24 * n_text_layer;
    
    std::map<ggml_backend_buffer_t, ggml_context*> ctx_map;
    auto get_ctx = [&](ggml_backend_buffer_type_t buft) -> ggml_context* {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/n_tensors * ggml_tensor_overhead(),
                /*.mem_buffer =*/nullptr,
                /*.no_alloc   =*/true,
            };

            ggml_context *ctx = ggml_init(params);
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context");
            }

            ctx_map[buft] = ctx;
            model.ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    // Create a list of available bufts in priority order
    buft_list_t buft_list = make_buft_list(wctx.params);

    return true;
}

struct whisper_context* whisper_init_with_params_no_state(struct whisper_model_loader* loader)
{
    ggml_time_init();

    fprintf(stdout, "%s: device = %zu\n", __func__, ggml_backend_dev_count());
    fprintf(stdout, "%s: backends = %zu\n", __func__, ggml_backend_reg_count());

    whisper_context* ctx = new whisper_context;
    if (!whisper_model_load(loader, *ctx))
    {
        loader->close(loader->context);
        fprintf(stderr, "%s: failed to load model\n", __func__);
        delete ctx;
        return nullptr;
    }
    loader->close(loader->context);
    return ctx;
}

struct whisper_context* whisper_init_from_file_with_params_no_state(const char* path_model)
{
    fprintf(stdout, "%s: loading model from '%s'\n", __func__, path_model);
    auto fin = std::ifstream(path_model, std::ios::binary);

    if (!fin)
    {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_model);
        return nullptr;
    }

    whisper_model_loader loader = {};

    loader.context = &fin;

    loader.read = [](void* ctx, void* output, size_t read_size) {
        std::ifstream* fin = (std::ifstream*)ctx;
        fin->read((char *)output, read_size);
        return read_size;
    };

    loader.eof = [](void *ctx) {
        std::ifstream *fin = (std::ifstream *)ctx;
        return fin->eof();
    };

    loader.close = [](void *ctx) {
        std::ifstream *fin = (std::ifstream *)ctx;
        fin->close();
    };

    auto ctx = whisper_init_with_params_no_state(&loader);

    if (ctx)
    {
        ctx->path_model = path_model;
    }
    return ctx;
}


struct whisper_context* whisper_init_from_file_with_params(const char* path_model)
{
    whisper_context* ctx = whisper_init_from_file_with_params_no_state(path_model);
    if (!ctx)
        return nullptr;
    return ctx;
}
