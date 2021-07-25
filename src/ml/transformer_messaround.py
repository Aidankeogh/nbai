import torch
from transformers import GPT2Model, GPT2Tokenizer
model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "bert is better than gpt2"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input["input_ids"].shape)
print(encoded_input["attention_mask"].shape)
def custom_forward(self, x):
    for block in self.h:
        outputs = block(x)
        x = outputs[0]
    return x
model.forward = custom_forward
random_input = torch.rand(1, 5, 768)
output = model(model, random_input) 
print(output.shape)
#print(model.h)