# iSCAT Partikelspårning / iSCAT Particle Tracking

> **IMPORTANT DISCLAIMER:** 
> This project heavily relies on code from:
> - The MAGIK implementation in [DeepTrack](https://github.com/DeepTrackAI/DeepTrack2)
> - The LodeSTAR implementation in [Deeplay](https://github.com/DeepTrackAI/deeplay)
> 
> We want to emphasize that these components were not developed by us, but we built upon their excellent work.

In the spring of 2024, we (the collaborators of this repository) did a joint bachelor thesis project in Physics at Chalmers University of Technology. The project title was *Nanoparticle tracking with iSCAT and machine learning* or *Spårning av nanopartiklar med iSCAT och maskininlärning* in Swedish. The project is published and available on Chalmers Open Digital Repository [here](https://odr.chalmers.se/items/c62d19ca-2549-4139-b840-9ccaa8460a30).

This repository contains Jupyter notebooks with examples of how to recreate the results presented in the thesis.

## Project Overview
Our study focused on tracking 100 nm particles using iSCAT microscopy in both two and three dimensions, utilizing machine learning and algorithm-based tracking methods. The key aspects of our research include:

1. Comparison of particle position predictions in 2D between:
   - [LodeSTAR method](https://www.nature.com/articles/s41467-022-35004-y) (a self-supervised particle localization learning method).
   - [Radial Variance Transform algorithm (RVT)](https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-7-11070&id=449504) (an algorithm for particle detection through radial symmetry detection).

2. Analysis of how different detection methods affect:
   - Creation of particle trajectories with [MAGIK](https://www.nature.com/articles/s42256-022-00595-0).
   - Particle diffusivity calculations.

3. Extension to 3D tracking:
   - By modifications to the LodeSTAR method allowing training on synthesized images of particles created with [DeepTrack](https://github.com/DeepTrackAI/DeepTrack2) at various depths.
   - Evaluation through comparison of experimental and theoretical diffusivity values for trajectories.
   - Analysis of trajectories and covariances in the z-dimension.

## Key Findings
### 2D Tracking
- RVT better showed expected evidence of Brownian motion in particle detections than LodeSTAR.
- LodeSTAR demonstrated better performance in:
  - Number of correctly predicted particles.
  - Tracking trajectories over extended time period.

### 3D Tracking
- LodeSTAR extracted depth information to some degree.
- Current accuracy has limitations for useful 3D detections.
- Potential for improvements of the method through enhanced synthesized training data and higher temporal resolution of microscopy videos.

## Usage
Directly under ```train_and_trace3d/``` and ```train_and_trace2d/``` lays Jupyter notebooks with examples of how the training, tracking and analysis was done for 3D-, respectively, 2D-tracking.

## Special dependencies
- [deeplay](https://github.com/DeepTrackAI/deeplay)
- [DeepTrack](https://github.com/DeepTrackAI/DeepTrack2)
- [PyTorch](https://github.com/pytorch/pytorch)

## Acknowledgments
- Daniel Midtvedt at Gothenburg University for being our supervisor.
- Erik Olsén at University of British Columbia for providing iSCAT microscopy footage.
- The team behind DeepTrack and Deeplay for their implementations.
