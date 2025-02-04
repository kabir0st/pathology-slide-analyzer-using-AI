import torch
from transformers import BioGptConfig
from histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

histogpt = HistoGPTForCausalLM(BioGptConfig(), PerceiverResamplerConfig())
histogpt = histogpt.to(device)

text = torch.randint(0, 42384, (1, 256)).to(device)
image = torch.rand(1, 1024, 768).to(device)

print(histogpt(text, image).logits.size())
