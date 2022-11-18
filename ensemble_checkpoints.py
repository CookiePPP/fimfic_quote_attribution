import torch

checkpoints = [
    "checkpoints/run_04_256context/lm-epoch=00-valid_loss=1.05.ckpt",
    "checkpoints/run_04_256context/lm-epoch=00-valid_loss=1.06.ckpt",
    "checkpoints/run_04_256context/lm-epoch=00-valid_loss=1.06-v1.ckpt",
    "checkpoints/run_04_256context/lm-epoch=00-valid_loss=1.06-v2.ckpt",
    "checkpoints/run_04_256context/lm-epoch=00-valid_loss=1.07.ckpt",
    "checkpoints/run_04_256context/lm-epoch=00-valid_loss=1.07-v1.ckpt",
    "checkpoints/run_04_256context/lm-epoch=00-valid_loss=1.08-v1.ckpt",
]

state_dict = {}
for checkpoint in checkpoints:
    sd = torch.load(checkpoint)['state_dict']
    for k, v in sd.items():
        if k not in state_dict:
            state_dict[k] = v
        else:
            state_dict[k] += v

for k, v in state_dict.items():
    state_dict[k] = v / len(checkpoints)

sd['state_dict'] = state_dict
torch.save(sd, "checkpoints/run_04_256context/ensemble.ckpt")