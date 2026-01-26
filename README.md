# Beyond the Training Domain: Robust Generative Transition State Models for Unseen Chemistry

[![arXiv](https://img.shields.io/badge/arXiv-2601.16469-b31b1b.svg)](https://arxiv.org/abs/2601.16469)

**Beyond the Training Domain: Robust Generative Transition State Models for Unseen Chemistry**  
Samir Darouich, Jacob W. Toney, Weiliang Luo, Johannes Kästner, Mathias Niepert, and Heather J. Kulik

---

## 🧪 Overview

Accurate prediction of transition states (TSs) is essential for understanding chemical reaction mechanisms and kinetics, but remains a major challenge in computational chemistry. Recent machine-learning models achieve high accuracy for small organic reactions, yet their ability to generalize to chemically novel systems—such as reactions involving unseen elements or transition-metal complexes—has not been systematically evaluated.

This repository provides benchmarks, models, and training pipelines for studying generative TS prediction beyond the training domain. Building on the Transition1x dataset, we introduce curated benchmark extensions that probe chemical and structural novelty through controlled elemental substitutions and diverse transition-metal complexes (TMCs). These benchmarks reveal fundamental limitations of existing generative TS models, which often produce unphysical geometries and large energetic errors when applied to previously unseen chemistry.

To address this challenge, we implement a self-supervised pretraining strategy based on equilibrium conformers. By constructing pseudo-reactions from abundant equilibrium structures, the model is exposed to diverse chemical environments prior to supervised fine-tuning on TS data. This pretraining substantially improves generalization to unseen systems, reducing TS geometry errors and significantly lowering the amount of fine-tuning data required—enabling reliable performance even in low-data regimes.

Overall, this work establishes a scalable and chemically robust framework for generative TS prediction beyond small organic molecules, providing a foundation for exploring complex and catalytically relevant reaction landscapes.

## 📦 Installation

The original React-OT repository explains how to install the environment needed for OAReactDiff and React-OT (https://github.com/deepprinciple/react-ot). These local versions of both repositories include the new embedding style.

To install AEFM it is advised to refer to the original repository (https://github.com/samirdarouich/AEFM)

## ⚙️ Usage

Scripts to curate the data (reaction optimization as well as conformer generation with CREST) are provided in "data_curation". To train OAReactDiff / ReactOT scripts are provided in the corresponding "trainer" folder. The evaluation script for React-OT inference is provided. For AEFM, the used config files are provided, as well as a script to train and evaluate.

## Reproduction

Pretrained models and the datasets are available at: https://doi.org/10.5281/zenodo.18338077

## 📕 Citation 

@article{
    darouich2026trainingdomainrobustgenerative,
    title={Beyond the Training Domain: Robust Generative Transition State Models for Unseen Chemistry}, 
    author={Samir Darouich and Jacob W. Toney and Weiliang Luo and Johannes Kästner and Mathias Niepert and Heather J. Kulik},
    year={2026},
    journal={arXiv preprint arXiv:2601.16469},
}
