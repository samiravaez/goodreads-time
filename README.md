## Description

This repository contains the reproduction code for "User and Recommender Behavior Over Time" paper at FairUMAP 2025

> Samira Vaez Barenji, Sushobhan Parajuli, Michael D. Ekstrand. 2025. User and Recommender Behavior Over Time: Contextualizing Activity, Effectiveness, Diversity, and Fairness in Book Recommendation. In <cite>Proceedings of the 33rd ACM Conference on User Modeling, Adaptation and Personalization</cite> (UMAP Adjunct ’25). ACM. DOI: [10.1145/3708319.3733710](https://dl.acm.org/doi/10.1145/3708319.3733710). arXiv: [2505.04518](https://arxiv.org/abs/2505.04518) [cs.IR]. 

## Environment Setup

We recommend using [pixi](https://pixi.sh/latest/) to reproduce the environment.

1. **Install pixi** (if you haven’t already):
   ```console
   curl -fsSL https://pixi.sh/install.sh | sh
   ```

2. **Run a shell in the pixi environment**:
   ```console
   pixi shell
   ```

This will activate the environment and install all the required dependencies listed in `pixi.toml`.

## Instructions
1. **Clone the required data tools**:

   Check out the [PIReT/INERTIAL Book Data Tools](https://bookdata.inertial.science) into a directory named `bookdata`, placed alongside this repository:

   ```
   parent-directory/
   ├── bookdata/          
   └── goodreads-time/   
   ```
   Follow the instructions in the PIReT/INERTIAL Book Data Tools repository to generate your copy of the UCSD Book Graph.

2. **Run the experiment pipeline**:

   From inside the `goodreads-time` directory, run:

   ```console
   dvc repro
   ```

   This will run the recommendation and evaluation pipeline.

3. **Generate figures**:

   Use the provided Jupyter notebooks to create the charts and figures from the experiment results.
