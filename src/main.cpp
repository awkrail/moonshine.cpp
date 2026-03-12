#include "whisper.h"

#include <cstdio>
#include <string>

struct whisper_params
{
    int32_t n_threads = 4;
    int32_t offset_t_ms = 0;
    int32_t duration_ms = 0;

    float temperature = 0.0f;
    float no_speech_thold = 0.6f;

    std::string language = "en";
    std::string prompt;
    std::string model = "models/ggml-base.en.bin";
    std::string fname_inp;
};



int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        fprintf(stderr, "failed to parse args. Usage: ./build/whisper_lite jfk.wav\n");
        return 1;
    }

    whisper_params params;
    params.fname_inp = std::string(argv[1]);

    // init context
    whisper_context* ctx = whisper_init_from_file_with_params(params.model.c_str());
    if (!ctx)
    {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 1;
    }




}
