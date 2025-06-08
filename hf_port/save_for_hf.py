# UPDATED save_for_hf.py
import torch
import os
from transformers import AutoTokenizer
from configuration_joint_causal import JointCausalConfig
from modeling_joint_causal import JointCausalModel

# 1. Create a config object.
config = JointCausalConfig()

# 2. Instantiate the model using the config.
model = JointCausalModel(config)

# 3. Load your trained weights.
model_weights_path = r"src/jointlearning/expert_bert_softmax/expert_bert_softmax_model.pt"
model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))

# 4. As per the tutorial, register the classes for AutoModel support.
# This will automatically create the 'auto_map' in your config.json.
JointCausalConfig.register_for_auto_class()
JointCausalModel.register_for_auto_class("AutoModel")

# 5. Save the model. The save_pretrained from PreTrainedModel is powerful
# and will handle everything correctly.
save_directory = "hf_port/joint_causal_model_for_hf"
model.save_pretrained(save_directory)

# 6. Save the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
tokenizer.save_pretrained(save_directory)

print("Model saved correctly with the new, robust structure!")
