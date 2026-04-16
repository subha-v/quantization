
QVLA
- 

- Find some characteristics of long tail thinking in the alpamayo paper
- Even in the weights, have separate bit widths for those
- Have a finer grained situation where for harder situations then 
- We can try long tail aware bit weight allocation things during runtime with a smart estimator 
- Even for the same weights, then the activations would change significantly. Read DP LLM to see this 


1. Find some main differences between long tail and general situations since their quantization sensitivity should be different as well (DPLLM found this)
2. Would be helpful to go into more fine grained (e.g. which layers are most sensitive)
3. We can quantize layer 1 for long tailed situations 

- If we quantize the 

- keep fp16 KV cache for the long tailed situations

Two types of dataset
- long tailed/hard situation
- easy situation dataset

- Clear advantage of benefit from our specifi cworkload
- in our case it could be long tail vs small task and lightweight estimator 

- For the initial steps we can run in FP16 and then for the remaining run in lower precision

- Applying block wise quantization to VLAs helpful 
- That block is aligned with certain action sensitivity 
- Work on block wise quantization of VLA weight matrices and figure out which blocks correspond to which action outputs would also be relevant, since QVLA operated at a channel level

- People might be worried about safety

- Similar situation to autonomous driving that security precision may be affordable 
- For long horizon tasks, if we can well specify the weights/activations that are relevant to that long context, etc.
- Start from some layer wise sensitivity metrics to see if quantizing long horizon vs short horizon tasks creates a big difference in the final accuract, etc.
- https://huggingface.co/docs/lerobot/libero
- https://huggingface.co/docs/lerobot/groot

https://github.com/NVlabs/alpamayo1.5
