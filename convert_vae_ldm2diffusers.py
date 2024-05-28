import torch
import diffusers
import re
import sys
sys.path.append('/home/jiangzhupeng/workspace/latent-diffusion/ldm/models')
from autoencoder import AutoencoderKL

model_cfg = {
    #"ckpt_path":"/home/jiangzhupeng/workspace/test_vae_ft/vae-ft-ema-560000-ema-pruned.ckpt",
    "ckpt_path": "/home/jiangzhupeng/workspace/latent-diffusion/logs/2024-05-21T19-34-36_autoencoder_kl_f8_ft/checkpoints/epoch=000006.ckpt",
    # "base_learning_rate": 4.5e-6,
    # "target": "ldm.models.autoencoder.AutoencoderKL",
    "monitor": "val/rec_loss",
    "embed_dim": 4,
    "lossconfig": {
        "target": "ldm.modules.losses.LPIPSWithDiscriminator",
        "params": {
            "disc_start":50001,
            "kl_weight": 0.000001,
            "disc_weight": 0.5,
        },
    },
    "ddconfig": {
        "double_z": True,
        "z_channels": 4,
        "resolution": 256,
        "in_channels": 3,
        "out_ch":3,
        "ch":128,
        "ch_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [ ],
        "dropout": 0.0,
    },
}

def reverse_ldm_decoder_up_0_3_to_3_0(state_dict)->dict:
    new_state_dict = {}
    count = 0
    for k, v in state_dict.items():
        if 'loss' in k:
            continue
        if "decoder.up.0" in k:
            new_k = k.replace("decoder.up.0", "decoder.up.3")
            new_state_dict[new_k] = v
        elif "decoder.up.1" in k:
            new_k = k.replace("decoder.up.1", "decoder.up.2")
            new_state_dict[new_k] = v
        elif "decoder.up.2" in k:
            new_k = k.replace("decoder.up.2", "decoder.up.1")
            new_state_dict[new_k] = v
        elif "decoder.up.3" in k:
            new_k = k.replace("decoder.up.3", "decoder.up.0")
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
        count += 1
    #print('count', count)
    return new_state_dict

def find_indexs(s, pattern):
    indexs = []
    for i in range(len(s)):
        if s[i] == pattern[0]:
            if s[i:i+len(pattern)] == pattern:
                indexs.append(i)
    return indexs

def convert_vae_ldm2diffusers(state_dict_ldm: dict, state_dict_diffusers: dict):

    # reverse ldm decoder.up.0-3 -> 3-0 first
    state_dict_ldm = reverse_ldm_decoder_up_0_3_to_3_0(state_dict_ldm)

    ldm2diffusers_dict = {
        'encoder.down.*.block.*.nin_shortcut' : 'encoder.down_blocks.*.resnets.*.conv_shortcut',
        'encoder.down.*.block' : 'encoder.down_blocks.*.resnets',
        'encoder.down.*.downsample' : 'encoder.down_blocks.*.downsamplers.0',
        'encoder.mid.block_1': 'encoder.mid_block.resnets.0', # single
        'encoder.mid.attn_1.norm': 'encoder.mid_block.attentions.0.group_norm', # single
        'encoder.mid.attn_1.q': 'encoder.mid_block.attentions.0.to_q', # single
        'encoder.mid.attn_1.k': 'encoder.mid_block.attentions.0.to_k', # single
        'encoder.mid.attn_1.v': 'encoder.mid_block.attentions.0.to_v', # single
        'encoder.mid.attn_1.proj_out': 'encoder.mid_block.attentions.0.to_out.0', # single
        'encoder.mid.block_2': 'encoder.mid_block.resnets.1', # single
        'encoder.norm_out': 'encoder.conv_norm_out',

        'decoder.mid.block_1': 'decoder.mid_block.resnets.0', # single
        'decoder.mid.attn_1.norm': 'decoder.mid_block.attentions.0.group_norm', # single
        'decoder.mid.attn_1.q': 'decoder.mid_block.attentions.0.to_q', # single
        'decoder.mid.attn_1.k': 'decoder.mid_block.attentions.0.to_k', # single
        'decoder.mid.attn_1.v': 'decoder.mid_block.attentions.0.to_v', # single
        'decoder.mid.attn_1.proj_out': 'decoder.mid_block.attentions.0.to_out.0', # single
        'decoder.mid.block_2': 'decoder.mid_block.resnets.1', # single

        'decoder.up.*.block.*.nin_shortcut': 'decoder.up_blocks.*.resnets.*.conv_shortcut',
        'decoder.up.*.block': 'decoder.up_blocks.*.resnets',
        'decoder.up.*.upsample': 'decoder.up_blocks.*.upsamplers.0',
        'decoder.norm_out': 'decoder.conv_norm_out',

    }
    
    new_state_dict = {}

    # convert ldm state_dict to diffusers state_dict
    for k, v in state_dict_ldm.items():
        patten_flag = True
        for k_diff, _ in state_dict_diffusers.items():
            if k_diff == k:
                patten_flag = False
                new_state_dict[k_diff] = v
                break
        if patten_flag:
            for k_ldm, k_diffusers in ldm2diffusers_dict.items():
                patten_in = k_ldm.replace('*', r'\d')
                patten_res = re.findall(patten_in, k)
                
                if len(patten_res) > 0:
                    patten_res = patten_res[0]
                    patten_indexs = find_indexs(k_ldm, '*')
                    final_res = k_diffusers
                    for i in range(len(patten_indexs)):
                        num_replace = patten_res[patten_indexs[i]]
                        final_res = final_res.replace('*', num_replace, 1)
                    
                    new_k = k.replace(patten_res, final_res)
                    diff_v = state_dict_diffusers[new_k]
                    # print('diff_v', diff_v)
                    # print('new_k', new_k)
                    if v.squeeze().shape == diff_v.shape:
                        v = v.squeeze()
                    new_state_dict[new_k] = v
                    patten_flag = False
                    break
        if patten_flag:
            print('patten_flag', k)
    print('len(new_state_dict)', len(new_state_dict))
    assert len(new_state_dict) == len(state_dict_diffusers), 'new: {} != ori {}'.format(len(new_state_dict),len(state_dict_diffusers))
    return new_state_dict

if __name__=="__main__":
    diff_model_path = "/home/jiangzhupeng/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-ema/snapshots/f04b2c4b98319346dad8c65879f680b1997b204a"
    diff_vae_model = diffusers.AutoencoderKL.from_pretrained(diff_model_path)

    ldm_vae_model = AutoencoderKL(**model_cfg)

    diff_vae_model_dict = diff_vae_model.state_dict()
    ldm_vae_model_dict = ldm_vae_model.state_dict()

    new_state_dict = convert_vae_ldm2diffusers(ldm_vae_model_dict, diff_vae_model_dict)

    diff_vae_model.load_state_dict(new_state_dict, strict=True)
    diffusers.AutoencoderKL.save_pretrained(diff_vae_model, 'my_self_vae')

