import torch
import sys
sys.path.append("/home/samirdar/projects/OAReactDiff")
ckpt_orig = torch.load("/home/samirdar/projects/react-ot/reactdiff-pretrained.ckpt","cpu")
ckpt_new = torch.load("/home/samirdar/projects/OAReactDiff/oa_reactdiff/trainer/checkpoint/OAReactDiff/leftnet-pretrained_no_en_decoder-95c4e836e498/last.ckpt","cpu")

key = "ddpm.dynamics.model.embedding.weight"

value_orig = ckpt_orig["state_dict"][key]
value_new = ckpt_new["state_dict"][key]

# Check if the tensors are the same
diff = torch.abs(value_orig-value_new)
print(f"Model weights are equal? {torch.all(diff<1e-5)}; max diff: {torch.max(diff)}")

key = "ddpm.dynamics.encoders.0.mlp.0.linear.weight"

value_orig = ckpt_orig["state_dict"][key]
value_new = ckpt_new["state_dict"][key]

# Check if the tensors are the same
diff = torch.abs(value_orig-value_new)
print(f"Model weights are equal? {torch.all(diff<1e-5)}; max diff: {torch.max(diff)}")
