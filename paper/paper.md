---
title: 'NORDic: a Network-Oriented package for the Repurposing of Drugs'
tags:
  - Python
  - network analysis
  - boolean network
  - network inference
  - biomarker identification
  - drug repurposing
authors:
  - name: Clémence Réda
    orcid: 0000-0003-3238-0258
    affiliation: 1 
  - name: Andrée Delahaye-Duriez
    orcid:  0000-0003-4324-7372
    affiliation: "1, 2, 3"
affiliations:
 - name: Université Paris Cité, Neurodiderot, Inserm, F-75019 Paris, France
   index: 1
 - name: Université Sorbonne Paris Nord, UFR de santé, médecine et biologie humaine, F-93000 Bobigny, France
   index: 2
 - name: Unité fonctionnelle de médecine génomique et génétique clinique, Hôpital Jean Verdier, AP-HP, F-93140 Bondy, France
   index: 3
date: 5 October 2023
bibliography: paper.bib
---

# Introduction

Genes, proteins and messenger RNAs are shown to interact on each other in order to modulate gene activity. Gene 
regulatory networks, which are graphs connecting biological entities according to their known regulatory interactions, are useful models 
that enable a better understanding of those regulatory mechanisms [@Karlebach2008]. 

In particular, one type of gene regulatory networks, called Boolean networks, allows the definition of so-called regulatory functions 
[@Thomas1973; @Kauffman1969]. Those functions are specific to each node in the graph, and determine the activity of this node 
according to its regulators. Those functions are defined on the Boolean domain (**True** or **False**), meaning that we only consider binary 
gene activities. Subsequently, studying this type of networks as a dynamical system 
remains rather tractable [@Moon2022]. The potential applications are numerous. Taking into account network dynamics should improve tools originally developed using non Boolean networks for the identification of interesting 
biomarkers [@Nicolle2015] or drug repurposing. 

However, the construction and analysis of Boolean networks become extremely tedious and time-consuming in the absence of experimental data or 
when considering the activity of a large number of genes at a time [@Collombet2017]. Moreover, the identification of interesting drug targets, via the detection of master regulators suffers 
from not exploiting the full network topology, which might account 
for toxic unexpected side effects [@Bolouri2003; @Huang2019]. Finally, regulatory mechanisms at transcriptomic level are inherently stochastic [@Raj2008]. As a consequence, naive algorithms for Boolean network-based *in silico* drug repurposing rely on testing a given drug a large number of times, in order to get a good estimate of its effect on gene activity. Such methods might resort to the simulation of drug treatment on Boolean network in either a patient-specific approach [@Montagud2022], or by ignoring the stochastic part of gene regulation. 

# Statement of need

The development of **NORDic** relies on avoiding *ad hoc* solutions, by implementation of approaches which are relevant to all kinds of 
diseases regardless of the level of knowledge present in the literature. Please refer to Figure 1 for an overview of the package. Solutions proposed in this package emphasize on, first, the modularity of the methods, by providing functions which can tackle different 
types of regulatory dynamics for instance; second, on the transparency of the approaches, by allowing the finetuning of each method through parameters with a clearly 
defined impact on the result.

![Overview of the different modules in NORDic.](overview.png)

## Automated identification of disease-related Boolean networks

Most prior works about building Boolean networks assume the existence of a preselected set of known regulatory 
interactions and/or a set of perturbation experiments, where the gene activity of a subset of genes is measured after a single gene perturbation, for a group of genes of interest. However, for rare diseases for instance, pinpointing a 
subset of genes of interest is already a difficult task by itself.

Moreover, there exist two hurdles to building Boolean networks which are specific to the Boolean framework. First, gene activity data must be binarized, meaning that 
one has to decide when a given gene is considered active or inactive in each sample. Such a process leads to an unavoidable loss of information. In order to avoid bias in the inference process, this step should be 
data-driven and user-controlled. For instance, when using PROFILE [@Beal2021], a majority of genes might end up with an undetermined status --meaning that they are considered 
neither significantly strongly nor weakly active-- which considerably undermines the input from experimental constraints. 

Second, the problem of identification of a Boolean 
network is usually underdetermined, as there is too few of experiments and measurements in practice, compared to the size of the considered gene set.

Module **NORDic Network Identification (NI)** addresses these issues in an automated and user-controllable manner, by performing information extraction from large online 
sources of biological data, and data quality filtering according to user-selected parameters, which control every step of the process. As such, the hope is that **NORDic** 
makes the generation of disease-specific Boolean networks easier, reproducible, even in the absence of previously curated experiments, prior knowledge networks, or even a set of disease-associated genes. The 
pipeline implemented in **NORDic** was applied to epilepsy in a preliminary work [@Reda22022].

## Prioritization of master regulators in Boolean networks

The identification of master regulators might relate to the disease onset or affected biological pathways of interest. Most prior works [@Wu2018] either only leverage topological information about the network, and do not take into account gene activity data related to the disease; or do not take into account regulatory effects which trickle down the network, beyond the targets directly regulated by the gene [@Zerrouk2020]. That is, the gene activity context does not impact the genewise values computed on the network.

Module **NORDic PMR** detects master regulators in a Boolean network, given examples of gene activity profiles from patients. In contrast to prior works, the score assigned to 
(groups of) master regulators takes into account the network topology as well as its dynamics with respect to the diseased profiles. The approach, based on a machine learning 
algorithm solving the influence maximization problem [@Kempe2003], is described in @Reda22022.

## Novel approaches for scoring drug effects & repurposing drugs

**NORDic** also proposes to tackle two problems related to drug repurposing: first, drug scoring, based on its ability to reverse the diseased gene activity profile 
(**NORDic DS**); second, the computation of an online sampling procedure which determines which drugs to test during drug screening for repurposing, in order to guarantee a bound on the error in recommendation, while 
remaining as sample-efficient as possible (**NORDic DR**).

There exist other approaches performing signature reversion, as mentioned in introduction. However, module **NORDic DS** (since version 2.0 of **NORDic**) is the first package to 
implement drug scoring based on Boolean networks, which can apply to any disease --for instance, it does not need the definition of specific biological phenotypes that should 
be observed after exposure to treatment [@Montagud2022]. The method implemented in **NORDic DS** is described in @Reda2022.

Similarly, module **NORDic DR** is the first approach that aims at solving the lack of guarantees in recommendation error. **NORDic DR** relies on bandit algorithms, which are sequential 
reinforcement learning algorithms that enable the recommendation of most efficient drugs. Based on Boolean network simulations performed on the fly, those algorithms can adaptively select the next drug to test in order to perform recommendations with as few samples 
as possible. Algorithms implemented in **NORDic DR** are described and theoretically analyzed in @Reda22021 (for the *m-LinGapE* algorithm), and in @Reda2021 (*MisLid* algorithm).

## Extraction of information from large public data sets & simulation module

In all four present modules in **NORDic**, helper functions in module **NORDic UTILS** are implemented in order to extract and curate data in a transparent way from the LINCS 
L1000 [@Subramanian2017], OmniPath [@Turei2016], DisGeNet [@Pinero2016] and STRING [@Szklarczyk2021] databases. **NORDic** also proposes a simulation module, which allows to test 
the accuracy of the predictions made by the network compared to known measurements. This module also enables the study and the visualization of the behaviour of the network under various perturbations and types of regulatory dynamics.

# Summary

Building a representation of gene interactions and their influences on gene activity, in an automated and reproducible way, helps to model more complex diseases and 
biological phenomena on a larger set of genes. These models might speed up the understanding of the gene regulation hierarchy by bioinformaticians and biologists; and allow to 
predict novel drugs or gene targets which might be investigated later for healthcare purposes. In particular, the network-oriented approach might be able to predict off-targets. The
**NORDic** Python package aims at tackling those problems, with a focus on reproducibility and modularity. It primarily relies on popular formats for network description files, such 
as the .bnet format. Moreover, **NORDic** enables further study of the network in Cytoscape, by providing a direct conversion to .sif formats, along with a dedicated style file. The different pipelines present in **NORDic** produce intermediary files, which might be checked by the user, and can be fed again to the pipeline in order to reproduce the results.

To get started with the different modules proposed in **NORDic**, please check out the tutorials (Jupyter notebooks) on the GitHub repository [@Reda2023], which provides an application to a 
disease called Congenital Central Hypoventilation Syndrome (CCHS).

# Acknowledgements

This work was supported by Université Paris Cité, Université Sorbonne Paris Nord, the French National
Research Agency (#ANR-18-CE17-0009-01) (A.D.-D., C.R.), (#ANR-18-CE37-0002-03) (A.D.-D.), (#ANR-21-RHUS-009) (A.D.-D., C.R.). The 
implementation of the bandit algorithm *MisLid* in **NORDic DR** was achieved with the help of Andrea Tirinzoni and of Rémy Degenne at Inria, 
UMR 9198-CRIStAL, F-59000 Lille, France. The funders had no role in review design, decision to publish, or preparation of the manuscript.

# References