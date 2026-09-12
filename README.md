# Provable Attention Ranges for Encrypted Language Models

This is the reference implementation of our research on Provable Ranges for Language Models under Approximate Homomorphic Encryption with and no need for external network access except the first download of the public GPT-2 weights and the WikiText-2 test split.

## Files and contents

```
hesoftmax/
  ckks/          RNS-CKKS: number theoretic transform, canonical embedding encoder,
                 key generation, encryption, multiplication with relinearization,
                 rescaling with level dependent scales, rotation, hybrid key switching
  polyapprox/    Chebyshev fitting, minimum degree search, baby step giant step evaluation
  softmax/       the six homomorphic softmax methods, in ciphertext and plaintext form
  bounds/        provable range bounds for a pre normalization transformer, GPT-2 forward pass,
                 symbolic error bound, noise model and flooding parameters
  attack/        decryption oracle key recovery and the average case flooding sweep
  experiments/   all measurement drivers, and result presentations
```

The six methods are `HETAL`, `NEXUS`, `2Quad`, `THOR`, `Cho24` and `ACS`, the last being the
method introduced in the paper. Each has a ciphertext implementation in `softmax/ours.py` and
`softmax/baselines.py`, a plaintext twin in `softmax/plaintext.py` used for accuracy studies, and
a causally masked twin in `softmax/masked.py` used inside the GPT-2 forward pass.

## Running
The requirements are Python 3.10 or later, numpy>=1.24, scipy>=1.10, matplotlib>=3.6, psutil>=5.9, torch>=2.0, transformers>=4.30, datasets>=2.12, safetensors>=0.3, huggingface_hub>=0.16. The accuracy experiment downloads `openai-community/gpt2` and `Salesforce/wikitext`
(WikiText-2, raw, test split) from the Hugging Face hub on first use. Both are public; WikiText-2
is distributed under CC BY-SA 4.0 and GPT-2 under the modified MIT license of OpenAI.
```
python -m hesoftmax.experiments.run_all
```
This runs, in order, the provable range bounds, the complexity counts, the decryption oracle attack,
the plaintext accuracy sweep over the half spread,
the GPT-2 perplexity study, the homomorphic microbenchmark. Results are written to `results/` as JSON, figures to `../final_paper/figures` as EPS. 
```
python -m hesoftmax.experiments.range_bounds --windows 8
python -m hesoftmax.experiments.complexity --spreads 16 105 1103.5
python -m hesoftmax.attack.indcpad
python -m hesoftmax.experiments.accuracy_sweep
python -m hesoftmax.experiments.gpt2_accuracy --windows 24 --calib 8 --s-meas 105
python -m hesoftmax.experiments.microbench --n 128 --spreads 105 1103.5 --out results/mb_all.json
python -m hesoftmax.experiments.figures
python -m hesoftmax.experiments.make_tables
```
The microbenchmark is the long step. It evaluates every method on real ciphertexts at ring degree
2^16 with a 30 level modulus chain, which is roughly three hours on one core and needs about
1 GB of memory. It writes its results incrementally and skips the pairs already present in the
output file, so it can be interrupted and resumed, and it can be split across cores by giving
each invocation a subset of `--methods` and its own `--out`. Pass `--param H128s` for a fast
functional check at ring degree 2^14, which is not a secure parameter set and is meant only for
smoke testing. 

## Parameter set
Ring degree 2^16 with 2^15 slots, scaling factor 2^40, one base prime of 50 bits and 30 primes of
40 bits chosen adaptively so that every level scale stays within 2^-21 of the nominal one, a
special modulus of 11 primes of 45 bits, hybrid key switching with 3 digits, ternary secret and
error standard deviation 3.2. This gives log2(PQ) = 1745, which is 128 bit security by the
homomorphic encryption standard. Evaluation keys store only the b component and a seed for the
uniform a component, which halves the key material. The softmax dimension is 128 with 256 rows
packed per ciphertext, the accuracy target is 16 bits, and the two score half-spreads are 105,
measured on WikiText-2, and 1104, proven by the range theorem.
