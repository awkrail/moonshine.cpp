# Convert Hugging Face moonshine models to ggml format.
# Code is based on https://github.com/ggml-org/whisper.cpp/blob/master/models/convert-h5-to-ggml.py
#
# Usage:
#
#   cd repos
#   git clone https://huggingface.co/UsefulSensors/moonshine-tiny
#   python python/convert-h5-to-ggml.py ./repos/moonshine-tiny models

import io
import os
import sys
import struct
import json
import code
import torch
import numpy as np
from pathlib import Path

from transformers import MoonshineForConditionalGeneration

conv_map = {
        'self_attn.k_proj'              : 'attn.key',
        'self_attn.q_proj'              : 'attn.query',
        'self_attn.v_proj'              : 'attn.value',
        'self_attn.out_proj'            : 'attn.out',
        'self_attn_layer_norm'          : 'attn_ln',
        'encoder_attn.q_proj'           : 'cross_attn.query',
        'encoder_attn.v_proj'           : 'cross_attn.value',
        'encoder_attn.out_proj'         : 'cross_attn.out',
        'encoder_attn_layer_norm'       : 'cross_attn_ln',
        'fc1'                           : 'mlp.0',
        'fc2'                           : 'mlp.2',
        'final_layer_norm'              : 'mlp_ln',
        'encoder.layer_norm.bias'       : 'encoder.ln_post.bias',
        'encoder.layer_norm.weight'     : 'encoder.ln_post.weight',
        'encoder.embed_positions.weight': 'encoder.positional_embedding',
        'decoder.layer_norm.bias'       : 'decoder.ln.bias',
        'decoder.layer_norm.weight'     : 'decoder.ln.weight',
        'decoder.embed_positions.weight': 'decoder.positional_embedding',
        'decoder.embed_tokens.weight'   : 'decoder.token_embedding.weight',
        'proj_out.weight'               : 'decoder.proj.weight',
        }

if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir_model dir-output [use-f32]\n")
    sys.exit(1)

dir_model = Path(sys.argv[1])
dir_out = Path(sys.argv[2])

hparams = json.load((dir_model / "config.json").open("r", encoding="utf8"))
generation_hparams = json.load((dir_model / "generation_config.json").open("r", encoding="utf8"))
tokenizer = json.load((dir_model / "tokenizer.json").open("r", encoding="utf8"))

model = MoonshineForConditionalGeneration.from_pretrained(dir_model)

fname_out = dir_out / "ggml-moonshine.bin"
tokens = tokenizer['model']['vocab']

# use 16-bit or 32-bit floats
use_f16 = True
if len(sys.argv) > 3:
    use_f16 = False
    fname_out = dir_out / "ggml-model-f32.bin"

# params
fout = open(fname_out, "wb")
fout.write(struct.pack("i", 0x67676d6c)) # magic: ggml in hex
fout.write(struct.pack("i", hparams["vocab_size"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["encoder_num_attention_heads"]))
fout.write(struct.pack("i", hparams["encoder_num_hidden_layers"]))
fout.write(struct.pack("i", generation_hparams["max_length"]))
fout.write(struct.pack("i", hparams["hidden_size"]))
fout.write(struct.pack("i", hparams["decoder_num_attention_heads"]))
fout.write(struct.pack("i", hparams["decoder_num_hidden_layers"]))
fout.write(struct.pack("i", use_f16))

# tokens
fout.write(struct.pack("i", len(tokens)))
tokens = sorted(tokens.items(), key=lambda x: x[1])
for key in tokens:
    text = key[0].encode('utf-8')
    fout.write(struct.pack("i", len(text)))
    fout.write(text)

# save model params
list_vars = model.state_dict()
for name in list_vars.keys():
    # this seems to not be used
    # ref: https://github.com/huggingface/transformers/blob/9a5b84a0076a04fe9596da72e8668069d4f09ea0/src/transformers/models/whisper/modeling_whisper.py#L1099-L1106
    if name == "proj_out.weight":
        print('Skipping', name)
        continue

    src = name

    nn = name
    if name != "proj_out.weight":
        nn = nn.split(".")[1:]
    else:
        nn = nn.split(".")

    if nn[1] == "layers":
        nn[1] = "blocks"
        if ".".join(nn[3:-1]) == "encoder_attn.k_proj":
            mapped = "attn.key" if nn[0] == "encoder" else "cross_attn.key"
        else:
            try:
                mapped = conv_map[".".join(nn[3:-1])]
            except:
                import ipdb; ipdb.set_trace()
        name = ".".join(nn[:3] + [mapped] + nn[-1:])
    else:
        name = ".".join(nn)
        name = conv_map[name] if name in conv_map else name

    print(src, ' -> ', name)
    data = list_vars[src].squeeze().numpy()
    data = data.astype(np.float16)

    # reshape conv bias from [n] to [n, 1]
    if name in ["encoder.conv1.bias", "encoder.conv2.bias"]:
        data = data.reshape(data.shape[0], 1)
        print("  Reshaped variable: " , name , " to shape: ", data.shape)

    n_dims = len(data.shape)
    print(name, n_dims, data.shape)

    # looks like the whisper models are in f16 by default
    # so we need to convert the small tensors to f32 until we fully support f16 in ggml
    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype = 1
    if use_f16:
        if n_dims < 2 or \
                name == "encoder.conv1.bias"   or \
                name == "encoder.conv2.bias"   or \
                name == "encoder.positional_embedding" or \
                name == "decoder.positional_embedding":
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype = 0
    else:
        data = data.astype(np.float32)
        ftype = 0

    # header
    str_ = name.encode('utf-8')
    fout.write(struct.pack("iii", n_dims, len(str_), ftype))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str_)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " , fname_out)
print("")
