# General Grant Proposal: EfficientZKML

* **Project:** EfficientZKML

## Project Overview :page_facing_up: 

### Overview

Zero-Knowledge Machine Learning (ZKML) research currently largely follows the paradigm of porting Deep Neural Networks (DNNs) to zero-knowledge (ZK) proving systems. Unfortunately, DNNs and current ZK proving systems are not a great match: While floating point arithmetic is pervasive in DNNs, ZK proving systems work on top of finite field arithmetic. This causes inherent losses in accuracy and efficiency when porting DNNs. 

This project aims to follow up on the [ZKHack 2023 Winning solution]((https://twitter.com/__zkhack__/status/1643571993046810627?s=20)) and implement provers for Weightless Neural Networks (WNNs), such as [BTHOWeN](https://arxiv.org/pdf/2203.01479.pdf). In contrast to DNNs, WNNs process bitstrings using (non-cryptographic) hash functions, array lookups, and simple counting.

### Project Details

The main goal of this project is to validate that using WNNs in ZKML use cases is a promising direction to pursue. To achieve that, we will:
- Implement an inference prover in Halo2
- Implement a Solidity verifier
- Benchmark our system, and compare to related work, such as [EZKL](https://github.com/zkonduit/ezkl) and [ZKML by Daniel Kang](https://github.com/ddkang/zkml)
- Investigate further research directions, such as proving training, applying folding techniques, improving the model itself

During the Hackathon, we implemented the inference of the BTHOWeN WNN using the Aleo proving system. We successfully ran it on the MNIST dataset in a reasonable time, comparable to EZKL. We want to continue our work and to answer the question if WNNs could be the right way for proving ZKML.

The outcome of our hackathon project is documented here:

- [Main Blog Post](https://hackmd.io/nCoxJCMlTqOr41_r1W4S9g?view)
- [Prover Technical Details](https://hackmd.io/6KGgnvv3RQujUnUCIbnEZA?view)
- [Github repository](https://github.com/zkp-gravity/0g)


## Team :busts_in_silhouette:

### Team members
1. 
    * Artem Grigor
    * 4rtem.grigor@gmail.com
    * @ArtemGrig
2. 
    * Georg
    * georgwiese@gmail.com
    * @clsdb

### Team's experience

Our team consists of two members from the Zero Gravity team that has developed the original WNN solution in Aleo and won the [ZK Hack 2023](https://twitter.com/__zkhack__/status/1643571993046810627?s=20). Both team members are experienced in Machine Learning, Cryptography, and Zero Knowledge Proofs. In particular:

- [Artem Grigor](https://www.linkedin.com/in/artem-grigor/) is currently working as a Research Engineer at Aragon ZK Team, where he has been focused on developing solutions to provide DAOs with privacy and security. He has recently been a co-author of 2 papers on Inpossibility of Sussinct and Anonymous Off-chain Voting accepted to [5-th DLT workshop](https://dltgroup.dmi.unipg.it/DLTWorkshop/dlt2023.html) and Multi-Input Non-Interactive Functional Encryption accepted to [C2SI conference](http://www.c2si-conference.org). Artem has also built multiple privacy preserving PoCs for various industries, including a [Anonymous Voting prototype in Rust](https://research.aragon.org/batravot_imp.html) and multiple solutions in Java during his work at R3. Artem has done Summer Semester in Data Science at Stanford and is a winner of Stanford class ST202 Forecasting Competition. He has also implemented [PLONK proving system from scratch](https://github.com/GurakG/PlonkImplementation) in Python Sage.
- [Georg Wiese](https://www.linkedin.com/in/georgwiese) worked on [image analysis with deep neural networks](https://scalableminds.com/voxelytics) until 2022. Since then, he has focused on studying cryptography, particularly zero-knowledge. He implemented a [post-quantum signature scheme in Rust](https://github.com/georgwiese/hash-based-signatures), implemented an [anonymous voting protocol](https://github.com/georgwiese/zk-vote-hackathon), worked on integrating PlonK into the [ZoKrates compiler](https://github.com/Zokrates/ZoKrates), and won the prize for the best write-up for the [third ZK HACK III puzzle](https://zkhack.dev/zkhackIII/solutionT3.html).

### Team Code Repos
* https://github.com/zkp-gravity/0g
* Initial Halo2 attempt: https://github.com/zkp-gravity/0g-halo2 


## Development Roadmap :nut_and_bolt: 

### Overview
* **Total Estimated Duration:** 3 month
* **Full-time equivalent (FTE):**  1.0
* **Total Costs:** $30,000
* **Start Date:** May 1 2023

### Milestone 1 - Application to prove WNN Inference 
* **Estimated Duration:** 1 month
* **FTE:**  1
* **Costs:** $10,000
* **Estimated delivery date**: June 1 2023

| Number | Deliverable | Specification |
| ------------- | ------------- | ------------- |
| 0a. | Documentation | We will provide both inline documentation of the code and a basic tutorial that explains how a user can spin up and use the application. With the tutorial, we will have an example file with WNN and example input, as well as how to run the Rust application to obtain the output.  |
| 0b. | Testing Guide | The code will have proper unit-test coverage to ensure functionality and robustness. In the guide, we will describe how to run these tests. |
| 0c. | WNN Serialisation Specification | We will produce a specification on how to represent the serialised WNN models. We will take Tensorflow's HDF5-based format as an inspiration on how to store the learned bloom filters and the hyperparameters of the model.  | 
| 1. | Model Training Setup | We will fork (MIT-licensed) [ZSusskind/BTHOWeN](https://github.com/ZSusskind/BTHOWeN) and incorporate the code from the [hackathon fork](https://github.com/zkp-gravity/BTHOWeN) for training models with SNARK friendly hash function.  | 
| 2. | Functionality: Model Serialisation | We will write a Python function to serialize the model based on the serialization specification.  | 
| 3. | Functionality: Model Deserialisation | We will write a Rust library component that will deserialize the WNN model from the file | 
| 4. | Functionality: Inference | We will write a Rust library component that will perform inference based on the provided WNN and input and provide the output of the inference | 
| 5. | Functionality: Proof of Inference | We will write a Rust library component that will generate a proof of inference using Halo2 and output it. | 
| 6. | Functionality: Rust Verifier | We will write a Rust library component to verify the Rust Application WNN inference proof. | 

### Milestone 2 - On-Chain Verifier 

* **Estimated Duration:** 2 weeks
* **FTE:**  1
* **Costs:** $5,000
* **Estimated delivery date**: June 15 2023

| Number | Deliverable | Specification |
| ------------- | ------------- | ------------- |
| 0a. | Documentation | We will provide both inline documentation of the code and a basic tutorial that explains how a user can spin up and use the application. We will provide an example of how a user can submit the proof of inference to the on-chain verifier.  |
| 0b. | Testing Guide | The code will have proper unit-test coverage to ensure functionality and robustness. In the guide we, will describe how to run these tests. |
| 1. | Functionality: On-chain Verifier | We will write a Solidity Smart Contract to verify the Rust Application WNN inference proof. | 
| 2. | Functionality: On-chain Interaction App  | We will write a Rust Application that will use the `ethereum` crate to convert the inference proof to Solidity-friendly format and submit it to the on-chain verifier. |

### Milestone 3 - Benchmarking of our system + related work

* **Estimated Duration:** 2 weeks
* **FTE:**  1
* **Costs:** $5,000
* **Estimated delivery date**: July 1 2023

| Number | Deliverable | Specification |
| ------------- | ------------- | ------------- |
| 0a. | Documentation | We will provide both inline documentation of the code and a basic tutorial that explains how a user can spin up and use the application. We will provide an example of how a user can submit the proof of inference to the on-chain verifier.  |
| 0b. | Testing Guide | The code will have proper unit-test coverage to ensure functionality and robustness. In the guide we, will describe how to run these tests. |
| 0c. | ZKML Benchmarking Specification | We will provide a hackmd document that will suggest a standard for running ZKML Benchmarking, describing the metrics, including relative prover time, relative verifier time, and relative proof size as well as model accuracy as well as the datasets to be used for benchmarking.  | 
| 0d. | ZKML Benchmarking List | We will provide a hackmd with the list of benchmarked ZKML solutions following our Benchmarking Specification. This will allow us to compare our solution to the current industry standards. | 
| 1. | Functionality: Benchmarking Scripts | We will write a Python script used to benchmark our Rust application, as well as provide PRs to add scripts to benchmark current industry standards.  | 

### Milestone 4 - Improvement Research 

* **Estimated Duration:** 1 month
* **FTE:**  1
* **Costs:** $10,000
* **Estimated delivery date**: August 1 2023

| Number | Deliverable | Specification |
| ------------- | ------------- | ------------- |
| 0a. | Documentation | We will provide both inline documentation of the code and a basic tutorial that explains how a user can spin up and use the application. We will provide an example of howa user can submit the proof of inference to the on-chain verifier.  |
| 0b. | LaTeX writeup | The writeup will aim to answer the question if the WNN could be the best paradigm for doing ZKML. The writeup will consist of several parts, described bellow.   | 
| 0c. | Blog Post | The blog post will summarise the key aspects of the LaTeX writeup on a high enough level. It will highlight the significance of this project, explain the technologies involved and will provide an evaluation of the WNN paradigm, and assess its potential.  | 
| 1. | Feature: Alternative Hash Function | Create the Rust Application branch to explore alternative hash functions, such as inverse, to be used in WNN. This work will also be represented in the LaTeX writeup in a dedicated section "Selecting Hash Function".  | 
| 2. | Feature: Preprocessing and WNN | Create the Model Training Python branch to explore preprocessing in WNN. Instead of providing raw inputs, we will use standard feature extractors and test if this improves accuracy. This work will also be represented in the LaTeX writeup in a dedicated section "Preprocessing and WNN".  | 
| 3. | Writeup: Folding WNN | Create the Rust Application branch to explore folding in WNN. This work will also be represented in the LaTeX writeup in a dedicated section "Folding WNN". If the implementation will not be finished, the writeup will include the efficiency assessment of incorporating the folding scheme into WNN.  | 
| 4. | Writeup: Scaling WNN | In the LaTeX writeup, have a section on scaling WNN, exploring the possibility of running WNN on ImageNet. This section should provide a concrete result or suggest future work. | 
| 5. | Writeup: Translating DNN into WNN | In the LaTeX writeup, have a section on "Translating DNN into WNN", talking about the possibility of translating the DNN into WNN, without needing to retrain WNN. This section should provide a concrete result or suggest future work. | 
| 6. | Writeup: Multilayer WNN | In the LaTeX writeup, have a section on "Multilayer WNN" describing techniques to achieve multilayer WNN and the impact of this on efficiency and accuracy. This section should provide a concrete result or suggest future work. | 
| 7. | Writeup: Training WNN | In the LaTeX writeup, have a section on "Verifiable WNN Training" describing the possibility of verifiable WNN training. We would also explore the susceptibility of the model to attacks, similar to the "Planting Undetectable Backdoors in Machine Learning Models" paper. This section should provide a concrete result or suggest future work. | 
