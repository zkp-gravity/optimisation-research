![](https://i.imgur.com/v9lyGdV.jpg)

It has been close to three months since we first implemented zero-knowledge proofs for Weightless Neural Networks (WNNs), [winning first prize](https://twitter.com/__zkhack__/status/1643571993046810627) at the ZK Hack Lisbon event. Our team member Ben has written an [excellent post](https://hackmd.io/@benjaminwilson/zero-gravity) on this, which we highly recommend reading before this post!

To summarize our original effort, as the result of our two sleepless nights, we produced some ["high-quality" hackathon code](https://github.com/zkp-gravity/0g/blob/main/scripts/generate_aleo_code.py), essentially consisting of a Python script that generates [Aleo instructions](https://developer.aleo.org/aleo/language/). We proved that WNNs can be implemented quite efficiently in a zero-knowledge framework, however - of course - a hackathon codebase is no good for actually verifying performance nor for testing out new ideas and driving the project forward.

So, part of our original team, in particular Artem and Georg, have sat down and re-implemented the ideas using Halo2 and further check the efficiency of the model. As a result, we have a set of command-line tools that lets you train WNN models, prove inference of these models, and have these proofs verified on the Ethereum (or any EVM-compatible) blockchain.

Let's dive in!

## Features

This work comprises two repositories:
- [`zkp-gravity/BTHOWeN-0g`](https://github.com/zkp-gravity/BTHOWeN-0g) which is used to train WNN models. Unless you want to play around with something else than [our MNIST models](https://github.com/zkp-gravity/0g-halo2/tree/main/models), you can safely ignore it.
- [`zkp-gravity/0g-halo2`](https://github.com/zkp-gravity/0g-halo2) which contains the core contribution: A command-line interface (CLI) that lets you generate proofs of inference for WNNs. It also exposes the core components as Halo2 gadgets, so that you can use them in your own circuits.

### Training a model

The particular flavor of WNN we use is called BTHOWeN (see [Susskind et al.](https://arxiv.org/abs/2203.01479)), so we forked their codebase, switched out their Bloom filter hash function to a ZKP-friendly one, described in the original Ben's blog post, and added some utility scripts. Head to [`zkp-gravity/BTHOWeN-0g`](https://github.com/zkp-gravity/BTHOWeN-0g) and follow the setup instructions to get started.

Training a model on the MNIST dataset involves the following steps:
```bash
# Train the model by providing the dataset name and BTHOWeN parameters.
# This example uses the same parameters as the "MNIST-Medium" model in the
# BTHOWeN paper, which achieves ~94% test set accuracy.
./train_swept_models.py \
    MNIST \
    --filter_inputs 28 \
    --filter_entries 2048 \
    --filter_hashes 2 \
    --bits_per_input 3
    
# The output of the training step will be a ".pickle.lzma" file, which is easy
# to work with in Python. To be able to use it in our Rust codebase, we'll export
# it to the HDF5 format.
./convert_to_hdf5.py models/MNIST/model_28input_2048entry_2hash_3bpi.pickle.lzma

# Our Rust codebase expects image files as input. The following small script
# exports the MNIST test set to a dictionary of PNG files.
# This step is optional, but it allows us to validate the test set accuracy.
./export_mnist.py
```

### Prove inference of a private image

In the [`zkp-gravity/0g-halo2`](https://github.com/zkp-gravity/0g-halo2) repository, you'll find instructions on how to install the `zero_g` command-line tool.

The `help` command shows you which commands are available:
```
zk-SNARKs for weightless neural networks (WNNs).

Usage: zero_g <COMMAND>

Commands:
  predict               Predict inference of a particular image (no proving)
  compute-accuracy      Compute the accuracy on the test set
  mock-proof            Step 0: Mock proof inference of a particular image. This can be helpful to figure out the right value of `k` and to test the correctness of the circuit
  generate-srs          Step 1: Generate the SRS
  generate-keys         Step 2: Generate the proving and verifying keys
  dry-run-evm-verifier  Step 2.1: Generate the EVM verifier and run a test proof
  deploy-evm-verifier   Step 2.2: Generate and deploy the EVM verifier
  proof                 Step 3: Proof inference of a particular image
  verify                Step 4: Verify the proof
  submit-proof          Step 4.1: Submit the proof to the (deployed) EVM verifier
  help                  Print this message or the help of the given subcommand(s)
```

As you can see, you'll be able to go all the way from doing a simple prediction to having a proof verified on an EVM-compatible chain! For example, have a look at our [CLI test script](https://github.com/zkp-gravity/0g-halo2/blob/main/test_cli.sh). It deploys the verifier contract to a local Anvil instance, but deploying to an actual chain is supported as well!

### Use `zero_g` in your own circuit

To be frank, the proofs generated by the command-line tool are by themselves not very useful: It just shows that you know some pixel values that lead to some (public) prediction. In real-world use cases, you might want to also publish the hash of the image, or validate its signature (e.g. if it comes from an attested sensor), or you could also want to keep the prediction private and add some additional circuit that takes the prediction as input.

Luckily, this can be achieved easily by using our Halo2 gadget in your own circuit. [This integration test](https://github.com/zkp-gravity/0g-halo2/blob/main/tests/using_zero_g_as_a_library.rs) illustrates an example.

## Implementation

We decided to use the [Halo2 fork](https://github.com/privacy-scaling-explorations/halo2) of the Privacy and Scaling Explorations (PSE) group from the Ethereum Foundation. It provides table lookups and a [library](https://github.com/privacy-scaling-explorations/snark-verifier) to generate the EVM verifier contract for any Halo2 circuit.

The circuit takes as (private) input a list of raw pixel values (8-bit unsigned integers) and (publicly) outputs a score for each class (10 in the case of MNIST). The model parameters are assumed to be public and committed to in the key generation phase.

In essence, a BTHOWeN-style WNN boils down to encoding its input into a bit string and interpreting random subsets of it as indices into a [Bloom filter](https://en.wikipedia.org/wiki/Bloom_filter). We'll leave it to [Ben's post](https://hackmd.io/@benjaminwilson/zero-gravity) to explain the inner workings in more detail but want to highlight an optimization that proved essential for achieving low proving time.

<!-- This is probably too detailed

The BTHOWeN models essentially work as follows:
- Input is a list of raw pixel values: an 8-bit unsigned integer encoding the brightness of the pixel. These are private inputs to our circuit.
- These values are encoded as a bit string, using a technique called "Thermometer Encoding".
- The bits are pseudo-randomly shuffled, where the permutation is fixed at training time.
- Chunks of the resulting bit string are interpreted as unsigned integers.
- These integers are used to perform a lookup into a bloom filter, which involves:
  - Hashing the input multiple times (or, equivalently, hashing it once and chunking the resulting hash).
  - Performing an array lookup. These bit arrays are learned during the training of the network and are public in our implementation.
  - The response of the bloom filter is positive if and only if all of the looked-up bits are 1.
- Results of the bloom filters are summed up per class, which yields the scores for each class.

The figure shows an example:

![](https://hackmd.io/_uploads/SkkHIs6w3.png)

-->

### Optimization: A 3-level array lookup

A core operation of a BTHOWeN model is an array lookup into an array of bits. Naively, this can be proven via a simple [lookup argument](https://zcash.github.io/halo2/design/proving-system/lookup.html): Given $n$ arrays of size $s$, we enumerate all tuples  $\{(a_j, i_j, b_j)\}_{j \in [1..n \cdot s]}$ such that $b_j = arrays[a_j][i_j]$ and commit to them at key generation time. At proving time, a lookup argument convinces the verifier that a tuple $(a, i, b)$ is included in the table. This is the approach we took in our hackathon submission.

The problem with this approach is that the number of rows of such a table can become large. For example, the "MNIST-Medium" model trained above contains 840 arrays of length 2048, which is just below $2^{21}$ rows. In Halo2, the tables are basically part of the circuit, so the instance we're proving will have a size of at least $2^{21}$.

The key insight is that we can pack multiple bits (say, 32) into one "word". This reduces the number of rows by a factor of 32, to $2^{16}$! Given an 11-bit index to address one of the 2048 bits, we can re-interpret it as 3 indices as follows:

<img src="https://hackmd.io/_uploads/ByJYACpvh.png" style="width: 70%;
                                                           display: block;
                                                           margin-left: auto;
                                                           margin-right: auto;" />


Looking up the correct bit now involves three steps:
1. Using the word index, we perform a table lookup to select the 32-bit word - similar to how we looked up single bits in the naive approach.
2. A [custom gadget](https://github.com/zkp-gravity/0g-halo2/blob/f5d7facb77ea9fa40c769de61e064720a7e7ae26/src/gadgets/bloom_filter/byte_selector.rs#L42-L64) selects the correct byte in the 32-bit word. This involves listing all the bytes, which means this gadget uses a number of advice rows that is linear in the number of bytes (in this example 4). Put differently, the number of bytes per word becomes a parameter that should be set such that the number of advice rows and table rows is balanced well.
3. We perform another table lookup to select the correct bit. This can be done by precomputing the result for all possible inputs: There are $2^8$ possible bytes and $8 = 2^3$ possible indices, so this table will be $2^{11}$ entries large.

As a result, we obtain a Halo2 instance that is smaller by a factor of 32, at the cost of introducing some complexity. Because of the higher complexity, the proving time speed-up is "only" about 15x, but we'll take it!

### Caulk, maybe?

We want to stress that the only reason this is necessary is that Halo2 does not yet implement more efficient lookup arguments, like [Caulk+](https://eprint.iacr.org/2022/957), [cq](https://eprint.iacr.org/2022/1763), or related approaches. These types of arguments would allow us to view the lookup table as being external to the circuit, with no influence on proving time. The day this becomes available in Halo2 will be a day of celebration for Zero Gravity; we can undo this optimization and obtain Halo2 instances that are *even smaller*, while also reducing the complexity of the circuit!

## Evaluation

So how does the weightless neural network (WNN) approach compare to deep neural network (DNN) implementations?

To test this, we took MNIST image classification as a benchmark and looked at the trade-off between test set accuracy and proving time (for one single inference). We compared our approach to [EZKL](https://github.com/zkonduit/ezkl) and [Daniel Kang's `zkml` library](https://github.com/ddkang/zkml).

The results are as follows:
<!--
Source for this chart:
https://docs.google.com/spreadsheets/d/1R-jxOnfU-FgiWcPe7X8Piegj0H-OTTTx8e3vwWGbMCQ/edit?usp=sharing
-->
![](https://hackmd.io/_uploads/BywEOgiOh.png)

As can be seen in this chart, while WNNs perform comparably in the fast proving time / low accuracy regime, they do not offer a significant advantage and are clearly outperformed by DNNs in the high-accuracy regime.

The WNN we used is largely based on a [model invented in the 80s](https://www.emerald.com/insight/content/doi/10.1108/eb007637/full/html) and has since received little attention from the machine learning research community (in contrast to DNNs!). While it is impressive that it can achieve an accuracy of ~95% on the character recognition task, DNNs with only a few thousand parameters can already achieve an accuracy of ~99%. Even if 95% are acceptable, DNNs achieve a comparable accuracy/proving time tradeoff.

That being said, WNNs might still have their place in situations where extremely fast proving time is essential and the task is simple, perhaps simpler than image recognition.

## Conclusion

After testing the general feasibility of applying WNNs in zero-knowledge proof systems during the ZK Hack Lisbon, we have re-implemented it in Halo2 which can generate proofs for arbitrary BTHOWeN models, validate them on the Ethereum blockchain, and includes some first performance optimizations.

Our evaluation shows that the achieved proving time has been more than 5-fold quicker than the original hackathon version and is comparable to existing DNN-based approaches in the low-accuracy regime. The high-accuracy regime seems unreachable, at least without significant improvements to the WNN architecture.

As mentioned above, the most obvious opportunity for future improvement would be to adopt a more advanced lookup argument. Other possible directions include exploring folding, feature engineering, and improving on the WNN architecture.

---

## Appendix: Details on the evaluation
**Zero Gravity**: We used the model configurations "MNIST-Small", "MNIST-Medium" and "MNIST-Large" as detailed in Table III of [Susskind et al.](https://arxiv.org/abs/2203.01479). Note that our accuracy is comparable to theirs but not exactly the same because we re-trained the models with a different hash function. To measure the numbers, we ran `cargo bench` in our [main repository](https://github.com/zkp-gravity/BTHOWeN-0g).

For EZKL and `ddkang/zkml`, we publish our evaluation scripts at [zkp-gravity/zkml-benchmark](https://github.com/zkp-gravity/zkml-benchmark).

**`ddkang/zkml`**: The repository contains a checked-in MNIST model with a proving runtime comparable to ours, so we just measured its test accuracy (97.81%). The network architecture of this model consists of 5 convolutional layers and two average pooling layers, as can be seen in [this image](https://hackmd.io/_uploads/H1as0esuh.png) generated from the checked-in `model.tflite` using [Netron](https://netron.app/).

**EZKL**: In the case of EZKL, we trained several models from scratch and explored different quantization parameters:
- In all networks, we use three hidden layers: Two 5x5 convolutions followed by one fully-connected layer. The result is fed into a final output layer that computes scores for each of the 10 classes.
- We varied the number of features for each layer:
  - `2-4-8`: This tiny configuration already achieves a 97.27% test accuracy on MNIST if no quantization is performed.
  - `4-8-32`: This configuration achieves a 98.78% test accuracy without any extra quantization.
- Finally, we tried different scale parameters (4 and 5). A lower scale corresponds to more aggressive quantization, which leads to lower proving time but also worse accuracy.

All numbers have been measured on a [Hetzner CX51](https://www.hetzner.com/cloud) virtual server with 8 Intel CPUs 32GB of RAM. The number that went into the chart above are detailed in [this sheet](https://docs.google.com/spreadsheets/d/1R-jxOnfU-FgiWcPe7X8Piegj0H-OTTTx8e3vwWGbMCQ/edit?usp=sharing).