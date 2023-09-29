Zero Gravity (The Weight is Over)

Neural nets perform exceptionally well at a variety of classification tasks - tasks like determining if an image contains an airplace, or recognising handwritten digits. These models use millions or even billions of floating point parameters to compute their classifications using multiple layers of matrix multiplications and non-linearities. While these calculations can be carried out very efficiently, it remains challenging to efficiently prove that the calculations were carried out correctly. Overcoming this challenge would allow slow computers (e.g. blockchains, or edge devices such as smartphones) to delegate neural network inference tasks to untrusted parties, enabling applications such as trustless biometric identification and smart contracts that are truly very smart.

The problem is that the primitives of ZK and ML are difficult to reconcile. ZK operates at a fundamental level with modular arithmetic (i.e. with discrete values over a finite field) whereas neural nets and most machine learning models perform “smooth” operations on floating point numbers, called “weights”. Existing approaches have attempted to bridge this divide by quantizing the weights of a neural net, so that they can be represented as elements of the finite field. Care must be taken to avoid a “wrap-around” occurring in the (now, modular!) arithmetic of the quantized network, and weight quantization can only decrease model accuracy. But more than anything, it does feel a little like trying to force a square peg into a round hole.

We propose a different approach: let’s go back to a time before the NN paradigm was settled, to a time when a greater variety of neural nets roamed the earth, and let’s find a machine learning model more amenable to ZKP. One such model is the “Weightless Neural Network”. It’s claimed to be the first ever neural net to be commercialized! Wow. But wow again, it is a very dusty dinosaur. Over the decades, it has received very little attention compared to familiar NNs. We set out to develop a system for proving the inferences of this weightless wonder … and we call it … Zero Gravity (The Weight is Over).

Weightless means no weights, no floating point arithmetic, and no expensive linear algebra, let alone non-linearities - so none of the challenges mentioned above. Will there be different challenges, and will they be worse? This is what we set out to discover at ZKHack hackathon (Lisbon, 2023).

What’s a Weightless Neural Network?

A Weightless Neural Network (WNN) is entirely combinatorial. Its input is a bitstring (e.g. encoding an image), and their output is one of several predefined classes, e.g. corresponding to the ten digits. It learns from a dataset of (input, output) pairs by remembering observed bitstring patterns in a bunch of devices called RAM cells, grouped into “discriminators” that correspond to each output class. RAM cells are so called since they are really just big lookup tables, indexed by bitstring patterns, and storing a 1 when that pattern has been observed in an input string that is labeled with the class of this discriminator.

(Figures from the BTHOWeN paper, see below)

Recalling patterns with RAM cells

Each RAM cell is connected to only a small number of inputs i.e. bits of the input bitstring. This is necessary since the size of its “random-access” lookup table will grow exponentially in the number of inputs (addressed using Bloom filters, see below). The wiring from the bits of the input bitstring to the RAM cells is typically randomized using a fixed permutation…

Another important thing to note is that there is only one layer of RAM cells. Consequently a WNN may excel in learning combinatorial, superficial patterns in the input bits, but can’t hope to learn the composite, semantically rich features that can be learnt by a deep neural network. Why just one layer? Remember that WNNs come from the time before the back-prop and deep nets won out. And there has been comparatively very little work done on them since. To make WNNs deep, you’ll need to invent an analog of back-prop (task for another hackathon?). Despite their simplicity, WNNs perform impressively on datasets such as MNIST - the BTHOWeN model, discussed below, achieves a test set accuracy exceeding 95%.

Bloom filters make RAM cells scalable

RAM cells are terribly space inefficient, but they are very sparse, since most bit patterns are never observed. Bloom filters offer a space efficient method of representing the data of a RAM cell, allowing the RAM cells to receive many more inputs. What’s a Bloom filter? Bloom filters are space-efficient data structures for probabilistically testing set membership. False positives are possible, false negatives are not - that is, Bloom filters will (efficiently) answer with “x is definitely not in the set” or “x is in the set with high probability”.

Under the hood, a Bloom filter consists of a fixed-length bit array and a number of functions that map potential set elements to positions in the array. These “hash functions” are chosen so as to hit each index of the bit array with uniform probability - though they may not always have cryptographic properties.

What does Zero Gravity (The Weight is Over) do?

Zero Gravity is a system for proving an inference run (i.e. a classification) for a pre-trained, public WNN and a private input. In Zero Gravity, the prover claims to know an input bitstring 
x
such that the public model classifies it as class 
y
. The input 
x
 can be treated as a private input, in which case the system is zero-knowledge: although inference does reveal something about 
x
to the verifier (namely its corresponding output class 
y
), this information is already contained in the statement being proved.

Zero Gravity builds upon the recent BTHOWeN model by Susskind et al (2022), in which the authors improve upon earlier WNN models in a number of interesting ways. Most importantly for this hackathon project, they helpfully provide an implementation complete with pre-trained models and reproducible benchmarks.

The interesting problem of proving that a WNN has been correctly trained or updated is left for another hackathon!

A challenge overcome: the choice of hash function

The hash functions in the WNN consume a short substring of the permuted input bitstring, outputting an index in a Bloom filter. The BTHOWeN authors chose their hash function to match their target domain: edge devices, and FPGAs in particular. Our application domain is entirely different and imposes different constraints. We want the hash function to be appropriate for a zero knowledge proof system. What sort of hash function should we use?

A cryptographic hash function is not an appropriate choice, since they are expensive to implement in a ZK proof system and as the hash functions only consume a short bitstring (e.g. of length 49 for MNIST) they are brute-force invertible in any case. Hash functions involving bit decompositions are also too expensive. We want a hash function 
H
 defined using arithmetic operations.

Linear functions are not a good choice

How about the hash function 
H
 defined by 
z
↦
a
z
+
b
 
(
mod
 
2
L
)
 where 
a
,
b
 are the integers, and 
2
L
L
 is the length of the Bloom filter? Choosing 
a
 to be a small odd prime ensures that 
H
maps uniformly onto the range 
0
,
…
,
2
L
−
1
, assuming that its inputs are chosen with uniform probability over the domain. This theoretically minimizes the probability of hash collisions. However, bitstrings encoding real-world datasets (e.g. MNIST) are not uniformly distributed. Early experiments confirmed that hash collisions indeed occur more often than desired. This led us to the choice of hash function defined below.

Small powers as permutations

The problem with a linear 
H
 (as above) is that it fails to sufficiently “scramble” the structure of real-world datasets. We consider instead functions of the form
H
:
z
↦
(
z
3
 
mod
 
p
)
 
mod
 
2
L
,

where 
p
>
2
L
 is prime, chosen such that 
3
 is not a divisor of 
p
−
1
. If 
p
 is chosen in this way, then 
z
↦
z
3
 is a permutation of 
Z
p
. This means that it maps uniformly onto the Bloom filter indices (as in the linear case). Unlike that case, however, hash collisions do not occur with undue frequency, and with this choice of hash function we are able to match the performance of the original BTHOWeN model!

Suitable primes 
p
 are easy to find. For example, if 
L
=
20
 (so 
2
L
=
1048576
), then
p
=
2097143
=
2
21
−
9
 satisfies these requirements.

We called our new, non-cryptographic hash function the “'MishMash” after its discoverer, team member HaMish.

Implementation

We wrote our proving system in Aleo with some metaprogramming in Python, and modified the BTHOWeN implementation to use our choice of hash function in order to re-train the models. Messy, hackathon quality code has been shamelessly made available.