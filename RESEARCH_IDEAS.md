
## Research Direction to Explore
Current KV cache quantization methods (KIVI, TurboQuant, KVQuant) assign uniform precision to all cached key and value vectors — every attention head at every layer gets the same bitwidth. However, attention heads are not equally important: some heads carry critical information for the model's output while others contribute minimally, and this sensitivity likely varies across layers and even across decoding steps. We propose a lightweight, calibration-free online estimator that, as each new key/value activation vector is produced during autoregressive decoding, examines simple statistical properties of the vector (such as magnitude range, variance, or kurtosis) to predict its quantization sensitivity, and then assigns it a precision level from a formatbook of number representations — extending BlockDialect's per-block dialect selection from general activations to the streaming KV cache setting. By assigning lower precision to insensitive heads and preserving higher precision for critical ones, the method aims to achieve better model quality than uniform KV cache quantization at the same average compression ratio, or equivalently, achieve the same quality at a lower average bitwidth. The approach is inherently online since KV cache vectors arrive one at a time during generation and cannot be calibrated offline, making this a natural application for the kind of input-adaptive sensitivity estimation that existing weight quantization methods do not require.


## General notes
- Estimator should be ideally very simple
- Look at activation profiling across simple vs complicated steps


## Wonsuk's ideas:

Also, following up on our last meeting, I think it would be helpful to do some background research on prior mixed-precision methods for VLA. If possible, it may also be worth looking into whether there are meaningful activation distribution differences depending on task complexity you mentioned.


## High Level ideas
At a high level, the main goal is calibration-free, online activation/(weights: can be profiled offline, but their bit-width may still need to be allocated online depending on the input.) sensitivity profiling. For weights, there are already many works that estimate each weight’s end-to-end impact using second-order information such as the Hessian, often approximated through the Fisher information matrix. These analyses are then used to identify weights that have relatively small impact on the final output and assign them lower precision. In our case, however, activation sensitivity is much harder to analyze online. Existing approaches often focus on only weights or either
(1) rely on calibration data to estimate average sensitivity offline, or
(2) iteratively search for an appropriate precision assignment by running inference multiple times and measuring the effect.

It would be interesting if we could find a lightweight estimator, possibly supported by hardware, that can assign mixed precision to activations online, somewhat in the spirit of DP-LLM paper (NeurIPS 2025) - It seems to focus only on weights, but it assigns different bit-widths depending on the input and the decoding step.

I was thinking about two possible directions to extend this idea:

First, we could apply it to a formatbook-based approach such as BlockDialect with a large number of non-standard formats. In this setting, low-sensitivity activations could be assigned even lower precision than 4-bit. For example, instead of using 8 representable values per format (requiring a 3-bit index), we could use only 4 representable values for low-sensitivity activations (requiring a 2-bit index), which would save 1 bit. Related to this, it may also be interesting to consider online optimal formatbook construction, since the existing approach uses an empirical global formatbook and is not necessarily optimal.

Second, we could consider selecting among multiple standard formats, such as MXFP4, MXFP6, MXFP8 (as in MicroMix), and possibly other standard or slightly modified standard formats such as FP16. The idea would again be to assign lower-precision formats to low-sensitivity activations, but in this case by choosing among a set of standard formats rather than relying on a formatbook.

These are still rough directions, so please feel free to suggest other ideas as well. If you think these directions are promising, it would also be great to look into related papers such as DP-LLM, MicroMix, and related work notated in these two papers and develop the ideas further. I would be very happy to discuss this more freely at the meeting on the 30th.
