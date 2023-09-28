# Efficiently Provable Bit Decoding in a Finite Field

Below, we provide an implementation of a protocol that can generate proofs for the value of the $i^{th}$ bit of $n$ bits encoded in a single field element.

This protocol operates within finite field arithmetic, leveraging the property that some numbers have square roots while others don't, as a mechanism to encode bits into a field element.

## Overview
The protocol's intent is to take $n$ bits, encode them into a single field element, and subsequently allow for a proof that a given $i^{th}$ bit of the encoded $n$ bits was indeed either a 1 or a 0. Impressively, our protocol requires only a single constraint to provide this proof for the $i^{th}$ bit.

## Implementation

### Requirements

To run the code, you will need to install the following:

- [Python 3.7](https://www.python.org/downloads/release/python-370/)
- [Jupyter Notebook](https://jupyter.org/install)
- [Numpy](https://numpy.org/install/)
- [Sage Math](https://www.sagemath.org/download.html)

Be sure to set the kernel of the notebook to the SageMath kernel.

### Running the Code

To run the code, simply open the notebook in Jupyter Notebook and run the cells in order. The notebook will walk you through the protocol, and provide a proof for the $i^{th}$ bit of the encoded $n$ bits.