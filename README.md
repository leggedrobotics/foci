# FOCI: Trajectory Optimization on Gaussian Splats  


- **Project page:** [https://rffr.leggedrobotics.com/works/foci/](https://rffr.leggedrobotics.com/works/foci/)

## Authors

- Mario Gomez Andreu\*¹
- Maximum Wilder-Smith\*¹
- Victor Klemm¹
- Vaishakh Patil¹
- Jesus Tordesillas²
- Marco Hutter¹

\*Equal contribution  
¹ Robotic Systems Lab, ETH Zurich  
² Comillas Pontifical University

## Overview

![Stonehenge paths](./docs/stonerobot.gif)

FOCI is a novel method to compute orientation aware trajectories for robots using 3D Gaussian Splats to model both the robot and the environment. 


## Installation
1. Obtain an HSL Licence for a linear solver (`ma27`/ `ma57`), download the corresponding files, rename the extracted folder to `coinhsl` and move it to the root directory in this repository. These are require
2. `docker build .` to build the provided docker container.
3. `` to attach to the container.
4. `` to run the demo script. You should see a visualisation similar to the one in this `README.md`


## Citing
If you find this work useful, please consider citing our paper:

```bibtex
@article{andreuwildersmith2025foci,
        author        = {Mario Gomez Andreu and Maximum Wilder-Smith and Victor Klemm and Vaishakh Patil and Jesus Tordesillas and Marco Hutter},
        title         = {FOCI: Trajectory Optimization on Gaussian Splats},
        year          = {2025},
        eprint        = {2505.08510},
        archivePrefix = {arXiv},
        primaryClass  = {cs.RO},
        url           = {https://arxiv.org/abs/2505.08510}
}
```