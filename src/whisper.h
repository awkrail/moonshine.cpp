#ifndef WHISPER_H
#define WHISPER_H

#include "ggml.h"
#include "ggml-cpu.h"

#include <string>
#include <vector>
#include <map>

struct whisper_model_loader;
struct whisper_context;
struct whisper_model;
struct whisper_vocab;
struct whisper_state;
struct whisper_hparams;
struct whisper_filters;
struct whisper_layer_encoder;
struct whisper_layer_decoder;

enum e_model
{
    MODEL_UNKNOWN,
    MODEL_TINY,
    MODEL_BASE,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
};

static const std::map<e_model, std::string> g_model_name = {
    {MODEL_UNKNOWN, "unknown"},
    {MODEL_TINY, "tiny"},
    {MODEL_BASE, "base"},
    {MODEL_SMALL, "small"},
    {MODEL_MEDIUM, "medium"},
    {MODEL_LARGE, "large"},
};

static const std::map<std::string, std::pair<int, std::string>> g_lang = {
    {"en",
     {
         0,
         "english",
     }},
    {"zh",
     {
         1,
         "chinese",
     }},
    {"de",
     {
         2,
         "german",
     }},
    {"es",
     {
         3,
         "spanish",
     }},
    {"ru",
     {
         4,
         "russian",
     }},
    {"ko",
     {
         5,
         "korean",
     }},
    {"fr",
     {
         6,
         "french",
     }},
    {"ja",
     {
         7,
         "japanese",
     }},
    {"pt",
     {
         8,
         "portuguese",
     }},
    {"tr",
     {
         9,
         "turkish",
     }},
    {"pl",
     {
         10,
         "polish",
     }},
    {"ca",
     {
         11,
         "catalan",
     }},
    {"nl",
     {
         12,
         "dutch",
     }},
    {"ar",
     {
         13,
         "arabic",
     }},
    {"sv",
     {
         14,
         "swedish",
     }},
    {"it",
     {
         15,
         "italian",
     }},
    {"id",
     {
         16,
         "indonesian",
     }},
    {"hi",
     {
         17,
         "hindi",
     }},
    {"fi",
     {
         18,
         "finnish",
     }},
    {"vi",
     {
         19,
         "vietnamese",
     }},
    {"he",
     {
         20,
         "hebrew",
     }},
    {"uk",
     {
         21,
         "ukrainian",
     }},
    {"el",
     {
         22,
         "greek",
     }},
    {"ms",
     {
         23,
         "malay",
     }},
    {"cs",
     {
         24,
         "czech",
     }},
    {"ro",
     {
         25,
         "romanian",
     }},
    {"da",
     {
         26,
         "danish",
     }},
    {"hu",
     {
         27,
         "hungarian",
     }},
    {"ta",
     {
         28,
         "tamil",
     }},
    {"no",
     {
         29,
         "norwegian",
     }},
    {"th",
     {
         30,
         "thai",
     }},
    {"ur",
     {
         31,
         "urdu",
     }},
    {"hr",
     {
         32,
         "croatian",
     }},
    {"bg",
     {
         33,
         "bulgarian",
     }},
    {"lt",
     {
         34,
         "lithuanian",
     }},
    {"la",
     {
         35,
         "latin",
     }},
    {"mi",
     {
         36,
         "maori",
     }},
    {"ml",
     {
         37,
         "malayalam",
     }},
    {"cy",
     {
         38,
         "welsh",
     }},
    {"sk",
     {
         39,
         "slovak",
     }},
    {"te",
     {
         40,
         "telugu",
     }},
    {"fa",
     {
         41,
         "persian",
     }},
    {"lv",
     {
         42,
         "latvian",
     }},
    {"bn",
     {
         43,
         "bengali",
     }},
    {"sr",
     {
         44,
         "serbian",
     }},
    {"az",
     {
         45,
         "azerbaijani",
     }},
    {"sl",
     {
         46,
         "slovenian",
     }},
    {"kn",
     {
         47,
         "kannada",
     }},
    {"et",
     {
         48,
         "estonian",
     }},
    {"mk",
     {
         49,
         "macedonian",
     }},
    {"br",
     {
         50,
         "breton",
     }},
    {"eu",
     {
         51,
         "basque",
     }},
    {"is",
     {
         52,
         "icelandic",
     }},
    {"hy",
     {
         53,
         "armenian",
     }},
    {"ne",
     {
         54,
         "nepali",
     }},
    {"mn",
     {
         55,
         "mongolian",
     }},
    {"bs",
     {
         56,
         "bosnian",
     }},
    {"kk",
     {
         57,
         "kazakh",
     }},
    {"sq",
     {
         58,
         "albanian",
     }},
    {"sw",
     {
         59,
         "swahili",
     }},
    {"gl",
     {
         60,
         "galician",
     }},
    {"mr",
     {
         61,
         "marathi",
     }},
    {"pa",
     {
         62,
         "punjabi",
     }},
    {"si",
     {
         63,
         "sinhala",
     }},
    {"km",
     {
         64,
         "khmer",
     }},
    {"sn",
     {
         65,
         "shona",
     }},
    {"yo",
     {
         66,
         "yoruba",
     }},
    {"so",
     {
         67,
         "somali",
     }},
    {"af",
     {
         68,
         "afrikaans",
     }},
    {"oc",
     {
         69,
         "occitan",
     }},
    {"ka",
     {
         70,
         "georgian",
     }},
    {"be",
     {
         71,
         "belarusian",
     }},
    {"tg",
     {
         72,
         "tajik",
     }},
    {"sd",
     {
         73,
         "sindhi",
     }},
    {"gu",
     {
         74,
         "gujarati",
     }},
    {"am",
     {
         75,
         "amharic",
     }},
    {"yi",
     {
         76,
         "yiddish",
     }},
    {"lo",
     {
         77,
         "lao",
     }},
    {"uz",
     {
         78,
         "uzbek",
     }},
    {"fo",
     {
         79,
         "faroese",
     }},
    {"ht",
     {
         80,
         "haitian creole",
     }},
    {"ps",
     {
         81,
         "pashto",
     }},
    {"tk",
     {
         82,
         "turkmen",
     }},
    {"nn",
     {
         83,
         "nynorsk",
     }},
    {"mt",
     {
         84,
         "maltese",
     }},
    {"sa",
     {
         85,
         "sanskrit",
     }},
    {"lb",
     {
         86,
         "luxembourgish",
     }},
    {"my",
     {
         87,
         "myanmar",
     }},
    {"bo",
     {
         88,
         "tibetan",
     }},
    {"tl",
     {
         89,
         "tagalog",
     }},
    {"mg",
     {
         90,
         "malagasy",
     }},
    {"as",
     {
         91,
         "assamese",
     }},
    {"tt",
     {
         92,
         "tatar",
     }},
    {"haw",
     {
         93,
         "hawaiian",
     }},
    {"ln",
     {
         94,
         "lingala",
     }},
    {"ha",
     {
         95,
         "hausa",
     }},
    {"ba",
     {
         96,
         "bashkir",
     }},
    {"jw",
     {
         97,
         "javanese",
     }},
    {"su",
     {
         98,
         "sundanese",
     }},
    {"yue",
     {
         99,
         "cantonese",
     }},
};

struct whisper_hparams
{
    int32_t n_vocab = 51864;
    int32_t n_audio_ctx = 1500;
    int32_t n_audio_state = 384;
    int32_t n_audio_head = 6;
    int32_t n_audio_layer = 4;
    int32_t n_text_ctx = 448;
    int32_t n_text_state = 384;
    int32_t n_text_head = 6;
    int32_t n_text_layer = 4;
    int32_t n_mels = 80;
    int32_t ftype = 1;
    float eps = 1e-5f;
};

struct whisper_filters
{
    int32_t n_mel;
    int32_t n_fft;
    std::vector<float> data;
};

struct whisper_layer_encoder
{
    // encoder.blocks.*.attn_ln
    struct ggml_tensor* attn_ln_0_w;
    struct ggml_tensor* attn_ln_0_b;

    // encoder.blocks.*.attn.out
    struct ggml_tensor* attn_ln_1_w;
    struct ggml_tensor* attn_ln_1_b;

    // encoder.blocks.*.attn.query
    struct ggml_tensor* attn_q_w;
    struct ggml_tensor* attn_q_b;

    // encoder.blocks.*.attn.key
    struct ggml_tensor* attn_k_w;

    // encoder.blocks.*.attn.value
    struct ggml_tensor* attn_v_w;
    struct ggml_tensor* attn_v_b;

    // encoder.blocks.*.mlp_ln
    struct ggml_tensor* mlp_ln_w;
    struct ggml_tensor* mlp_ln_b;

    // encoder.blocks.*.mlp.0
    struct ggml_tensor* mlp_0_w;
    struct ggml_tensor* mlp_0_b;

    // encoder.blocks.*.mlp.2
    struct ggml_tensor* mlp_1_w;
    struct ggml_tensor* mlp_1_b;
};

struct whisper_layer_decoder
{
    // decoder.blocks.*.attn_ln
    struct ggml_tensor *attn_ln_0_w;
    struct ggml_tensor *attn_ln_0_b;

    // decoder.blocks.*.attn.out
    struct ggml_tensor *attn_ln_1_w;
    struct ggml_tensor *attn_ln_1_b;

    // decoder.blocks.*.attn.query
    struct ggml_tensor *attn_q_w;
    struct ggml_tensor *attn_q_b;

    // decoder.blocks.*.attn.key
    struct ggml_tensor *attn_k_w;

    // decoder.blocks.*.attn.value
    struct ggml_tensor *attn_v_w;
    struct ggml_tensor *attn_v_b;

    // decoder.blocks.*.cross_attn_ln
    struct ggml_tensor *cross_attn_ln_0_w;
    struct ggml_tensor *cross_attn_ln_0_b;

    // decoder.blocks.*.cross_attn.out
    struct ggml_tensor *cross_attn_ln_1_w;
    struct ggml_tensor *cross_attn_ln_1_b;

    // decoder.blocks.*.cross_attn.query
    struct ggml_tensor *cross_attn_q_w;
    struct ggml_tensor *cross_attn_q_b;

    // decoder.blocks.*.cross_attn.key
    struct ggml_tensor *cross_attn_k_w;

    // decoder.blocks.*.cross_attn.value
    struct ggml_tensor *cross_attn_v_w;
    struct ggml_tensor *cross_attn_v_b;

    // decoder.blocks.*.mlp_ln
    struct ggml_tensor *mlp_ln_w;
    struct ggml_tensor *mlp_ln_b;

    // decoder.blocks.*.mlp.0
    struct ggml_tensor *mlp_0_w;
    struct ggml_tensor *mlp_0_b;

    // decoder.blocks.*.mlp.2
    struct ggml_tensor *mlp_1_w;
    struct ggml_tensor *mlp_1_b;
};

struct whisper_model
{
    e_model type = MODEL_UNKNOWN;

    whisper_hparams hparams;
    whisper_filters filters;
    
    // encoder.positional_embedding
    struct ggml_tensor* e_pe;

    // encoder.conv1
    struct ggml_tensor *e_conv_1_w;
    struct ggml_tensor *e_conv_1_b;

    // encoder.conv2
    struct ggml_tensor *e_conv_2_w;
    struct ggml_tensor *e_conv_2_b;

    // encoder.ln_post
    struct ggml_tensor *e_ln_w;
    struct ggml_tensor *e_ln_b;

    // decoder.positional_embedding
    struct ggml_tensor *d_pe;

    // decoder.token_embedding
    struct ggml_tensor *d_te;

    // decoder.ln
    struct ggml_tensor *d_ln_w;
    struct ggml_tensor *d_ln_b;

    std::vector<whisper_layer_encoder> layers_encoder;
    std::vector<whisper_layer_decoder> layers_decoder;

    // ggml context that contains all the meta information
    std::vector<ggml_context*> ctxs;
    
    // the model backend data is read-only and can be shared between processors
    std::vector<ggml_backend_buffer_t> buffers;

    // tensors
    int n_loaded;
    std::map<std::string, struct ggml_tensor*> tensors;
};

struct whisper_vocab
{
    using id = int32_t;
    using token = std::string;

    int n_vocab = 51864;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    id token_eot = 50256;
    id token_sot = 50257; // task tokens (used only for multilingual models)
    id token_translate = 50357;
    id token_transcribe = 50358; // other special tokens
    id token_solm = 50359; // [TDRZ] used by tinydiarize models to indicate speaker turn
    id token_prev = 50360;
    id token_nosp = 50361;
    id token_not = 50362; // no timestamps
    id token_beg = 50363; // begin timestamps
    
    bool is_multilingual() const { return n_vocab >= 51865; }
    
    int num_languages() const { return n_vocab - 51765 - (is_multilingual() ? 1 : 0); }
};

struct whisper_model_loader
{
    void* context;
    size_t (*read)(void* ctx, void* output, size_t read_size);
    bool   (*eof)(void* ctx);
    void  (*close)(void* ctx);
};

struct whisper_context
{
    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    ggml_type wtype = ggml_type::GGML_TYPE_F16;
    ggml_type itype = ggml_type::GGML_TYPE_F16;

    whisper_model model;
    whisper_vocab vocab;

    whisper_state* state = nullptr;
    std::string path_model;
};

struct whisper_context* whisper_init_from_file_with_params(const char* path_model);

#endif
