# Entering The Unknown

Feature: Alternative Hash Function

2. Feature: Preprocessing and WNN
Specification
Create the Model Training Python branch to explore preprocessing in WNN. Instead of providing raw inputs, we will use standard feature extractors and test if this improves accuracy. This work will also be represented in the LaTeX writeup in a dedicated section
“Preprocessing and WNN”.
Create the Rust Application branch to explore folding in WNN. This work will also be represented in the LaTeX writeup in a dedicated section “Folding WNN”. If the implementation will not be finished, the writeup will include the eﬀiciency assessment of incorporating the folding scheme into WNN.
In the LaTeX writeup, have a section on scaling WNN, exploring the possibility of running WNN on ImageNet. This section should provide a concrete result or suggest future work.
 3. Writeup: Folding WNN
4. Writeup: Scaling WNN
8

 Number Deliverable
5. Writeup: Translating DNN into WNN
6. Writeup: Multilayer WNN
7. Writeup: Training WNN
## Optimisations of the Rust Prover 

### 1. Removing useless proves

There are several layers of the WNN that we currently prove, yet proving them does not add anything to the security of the system.

In particular, all these layers apply an easily invertable function $f$ to the input value $in$, obtain result $out=f(in)$ and then never again reference $in$ in the WNN. 

Let us imagine two constraint systems, one with the $out==f(in)$ check and one without, and prove that they are equivelent. By equivelent we imply that the set of accepted assigments to $out$ is the same in both cases. 

To see that, let us consider that the system without the $out==f(in)$ check accepts an extra assigment $out_x$. However, as $f$ is easily invertable, we can compute $in_x$, such that $f(in_x)=out_x$. As all other checks are identical, it implies that $out_x$ is also accepted assigment for the constraint system with extra $out==f(in)$ check.

We must note that this is only true if there are no constraints on the $in$ value. For example, when we want to show that we know such an input $in$, that it is recognised as a Cat by the WNN model. 

However, we might have additinal constraints on the input, and then we would need to prove all function transitions. For example, we took part in a contest who first would find an input that will be recognised as a Cat. The first person commits to there input, and then once the competition is over they reveal the input, prove that WNN output is indeed a Cat, and also that hash of input is equal to the commited value. In that case, the optimisation would not be applicable. Thus it should be enabled by the user settings.

As for the exact layers that the reasoning is applicable to:
1. Thermometer Encoding layer
2. Permutation layer

*Strictly speaking, as we are not using cryptographycally secure hash functions, the whole WNN model is invertable. Though it might be too computationally expensive to do it, thus we see this as not an issue.*

### 2. Encoding multiple bits/numbers into native field elements

We operate in a field, which size is about $2^{264}$, where as the biggest numbers we actually use are around $2^{13}$. So for lookups, for instance, we can , meaning to optimise the encoding efficientcy, we can encode such nu

## Lookup Optimisations

### Origami - Lookup focused Sangria

https://hackmd.io/@aardvark/rkHqa3NZ2


### Lasso

### Batching 

Lookup batching for the same permutations??

### Data Compression

We have a circuit with a fixed mapping between an index i, from 0 to N, and the corresponding bit flag ${0,1}$. The circuit needs to prove that the index i indeed corresponds to flag {0,1}. Circuit is implemented in a field of size ~ M bits.

#### Solution 1

We create a lookup that contains 2 columns $(index_i, flag_i)$.

This requires space 2*N.

To prove an $index_i$ corresponds to $flag_i$, we just run a lookup.  


#### Solution 2 

We notice that so far we store a single binary flag in a field element, which can accomodate up to M bits of information.

So we decide to encode multiple indexes flags into a single field element. Iitally, we use straight forward concatenation for that. 

As a result, we merge n indeces together, where $n < M$. Now, when we do lookups, we first lookup an compression index $comp_index = index_i / n$ which corresponds to a compression of n flags. For now the compression is concatenation $f_1..f_n$, a number between 0 and $2^n$. 

We then need to prove value of the i-th flag, stored in the compression.

There are several strategies we can do:

1. Decomposition: 
    - $pre_part + flag_i * 2^i + post_part = f_1..f_n$
    - range check 0 <= pre_part < 2^i
    - range check 2^i < post_part < 2^n 
    Requires 2^n lookup rows. 2 Columns
2. Index Combination Lookup: 
    We create a lookup table of all possible combinations between n bit values and a bit index, which allows us to directly extract the i-th bit. 
    Requires 2^n*n lookup rows. 3 Columns.
3. Chunk Decomposition: 
    We change the problem to selecting a correct chunk of k bits from n bits.
    This is a generalisation of approach one and can be used recursively.
    - $chunk_0 = acc_0$
    - $chunk_i + acc_i-1 = acc_i$
    - $select_chunk_j = select_chunk_j-1 + chunk_j * (j == select_i)$
    Requires 2^(n-k) rows. About 7 columns.
    
#### Solution 3

Binary encoding do not seem most optimal for decompression in cuirtuits. They were designed for regular transistors, not for field arithmetics. Perhaps, we can compress n flags in a more efficient way?

**Attempt 1**

We will correspond to each index two prime numbers. We once again do lookups, and for index i first lookup a generalized index $gen_index = index_i / n$ which corresponds to an encoded value x, that stores information on the n flags.

To extract information about the i-th flag, we first lookup two corresponding primes to index i, $p_{i_0}, p_{i_1}$. We then check if $x=quotient * p_(i_0)$, then i-th flag is 1, otherwise if $x=quotient * p_(i_1)$, then i-th flag is 0. Note that both cases should be mutually exclusive. 

To encode a value into such x, one just looks at flag values, and then computes $$\prod_{i=0}^{n} (p_{i_0} * flag_i + p_{i_1} * (1 - flag_i)$$.

Unfortunatelly, the adversary can always compute such $quotient=x* p_(i_0)^{-1}$, such that even if $flag_i==0$, $x=quotient * p_(i_0)$. One can fix this by rangechecking $quotient < \prod_{i=0}^{n} p_{i_1}$, which for $n=32$ would require lookup of up to $2^200$, which is too much.

**Attempt 2**

We again correspond to each index two number, though this time these numbers does not have a square root $sqrt$. Simular as before, we arrive to x, that stores information on the n flags.

To extract information about the i-th flag, we first lookup two corresponding number to index i, $t_{i_0}, t_{i_1}$. We then check if $x * t_(i_0)$ has a square root. If the square root exists, then i-th flag is 1, otherwise if $x * t_(i_1)$ has a square root exists, then i-th flag is 0. Note that both cases should be mutually exclusive. 

We require two numbers as it is easy to prove that the square root exists by presenting it. Though it is hard to prove the oposite. 

This solution would require a lot of precomputation, possibly exponensial amount, though it only needs $2n$ lookups of the numbers and additional 1 hint of the square root.

We conjecture there should be even more efficient solutions that encode more efficiently.


## Reviewed

https://www.researchgate.net/publication/284697754_Weightless_neural_models_A_review_of_current_and_past_works

https://ieeexplore.ieee.org/abstract/document/8850525

https://hackmd.io/@aardvark/rkHqa3NZ2

 https://github.com/ingonyama-zk/papers/blob/main/lookups.pdf
 
 https://zeroknowledge.fm/280-2/