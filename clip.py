import torch
from transformers import pipeline

clip = pipeline(
   task="zero-shot-image-classification",
   model="openai/clip-vit-base-patch32",
   dtype=torch.bfloat16,
   device=0
)
labels = ["a logo of a utah credit union", "a logo of a bank of america", "a logo of a citibank"]
clip("/uufs/chpc.utah.edu/common/home/u1475870/logo_detection/outputs/utah_credit_2.png", candidate_labels=labels)