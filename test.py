# %%
from transformers import AutoTokenizer, AutoModel 
# %%
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)
# %%
inputs = tokenizer("Hello, my doggizalation is cute", return_tensors="pt")
# %%
print(inputs)
# %%
# print word ids with word_ids method
print(inputs.word_ids())
# %%
# print tokens 
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
# %%
input = tokenizer("Hello, my doggizalation is cute", return_tensors="pt", add_special_tokens=False)
# %%
input.word_ids()
# %%
