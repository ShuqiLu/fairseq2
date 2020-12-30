import torch
import os
roberta=torch.load('/home/dihe/Projects/data/roberta.base/model.pt',map_location=lambda storage, loc: storage)
roberta_decode=torch.load('/home/dihe/Projects/fairseq/checkpoints_init/checkpoint_best.pt',map_location=lambda storage, loc: storage)
state_dict=roberta['model']
prefix=''
for k in list(state_dict.keys()):
    if k.startswith(prefix + 'decoder'):
        new_k = prefix + 'encoder' + k[len(prefix + 'decoder'):]
        state_dict[new_k] = state_dict[k]
        del state_dict[k]

items_to_add = {}
keys_to_remove = []
for k in state_dict.keys():
    prefix = '.'.join(k.split('.')[:-1])+'.'
    
    #print('???',k,prefix)
    # if k.endswith("in_proj_weight"):
    #     print('???',k,'    ',prefix + "in_proj_weight")
    #     assert 1==0
    if k.endswith(prefix + "in_proj_weight"):
        print('???')
        dim = int(state_dict[k].shape[0] / 3)
        items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
        items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
        items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]
        keys_to_remove.append(k)
        k_bias = prefix + "in_proj_bias"
        if k_bias in state_dict.keys():
            dim = int(state_dict[k].shape[0] / 3)
            items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
            items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                dim : 2 * dim
            ]
            items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]
            keys_to_remove.append(prefix + "in_proj_bias")
for k in keys_to_remove:
    del state_dict[k]
for key, value in items_to_add.items():
    state_dict[key] = value
state_dict['decoder.embed_tokens.weight']=state_dict['encoder.sentence_encoder.embed_tokens.weight']
state_dict['decoder.output_projection.weight']=state_dict['encoder.sentence_encoder.embed_tokens.weight']
# for item in state_dict:
#     if item not in roberta_decode['model'].keys():
#         print(item)

for item in roberta_decode['model'].keys():
    if item not in state_dict:
        print(item)

print('len: ',len(roberta_decode['model']))
roberta_decode['model'].update(state_dict)
print('len: ',len(roberta_decode['model']))
del roberta_decode['last_optimizer_state']
roberta_decode['extra_state']=roberta['extra_state']

# prefix=''
# for k in list(roberta_decode['model'].keys()):
#     if k.startswith(prefix + 'decoder'):
#         print('!!!')
#         new_k = prefix + 'encoder.' + k
#         roberta_decode['model'][new_k] = roberta_decode['model'][k]
#         del roberta_decode['model'][k]

torch.save(roberta_decode,os.path.join('/home/dihe/Projects/fairseq/checkpoints_init/', 'roberta_decode3.pkl'))
#print('???',roberta_decode['model'].keys())
#b=torch.load(os.path.join('/home/dihe/Projects/fairseq/checkpoints_init/', 'roberta_decode.pkl'),map_location=lambda storage, loc: storage)
