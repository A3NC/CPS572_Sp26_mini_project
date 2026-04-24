
from tinker import SamplingClient, SamplingParams, ServiceClient, ModelInput
from tinker_cookbook.tokenizer_utils import get_tokenizer
# print(help(SamplingClient))

from tinker import ModelID
model_path = "tinker://90d3e25e-75de-5316-a53c-da7c0de7fbe3:train:0/sampler_weights/demo"

# params = SamplingParams(
#     max_new_tokens=300,
#     temperature=0.7
# )
service_client = ServiceClient()

sampling_client = service_client.create_sampling_client(
          model_path="tinker://90d3e25e-75de-5316-a53c-da7c0de7fbe3:train:0/sampler_weights/demo"
      )
textprompt = '''	
from typing import List


def below_zero(operations: List[int]) -> bool:
""" You're given a list of deposit and withdrawal operations on a bank account that starts with
zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
at that point function should return True. Otherwise it should return False.
>>> below_zero([1, 2, 3])
False
>>> below_zero([1, 2, -4, 5])
True
"""
'''


MODEL = "meta-llama/Llama-3.1-8B"
tokenizer = get_tokenizer(MODEL)
sampling_client = service_client.create_sampling_client(base_model=MODEL)
prompt = ModelInput.from_ints(tokenizer.encode(textprompt))
params = SamplingParams(max_tokens=2000, temperature=0.7)
future = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
result = future.result()

sequence = result.sequences[0]
tokens = sequence.tokens

text = tokenizer.decode(tokens, skip_special_tokens=True)

print(text)
