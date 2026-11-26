# FOCI: Trajectory Optimization on Gaussian Splats for Gaussian Robots

### Journal submission: [https://arxiv.org/abs/2611.33380](https://arxiv.org/abs/2611.33380)
### IROS 2025 paper: [https://arxiv.org/abs/2505.08510](https://arxiv.org/abs/2505.08510)

<table align="center" border="1" style="border-collapse: collapse;">
  <tr>
    <th align="center" style="border: 1px solid; padding: 6px;">Static Environment</th>
  </tr>
  <tr>
    <td style="border: 1px solid; padding: 6px;"><img src="./docs/stonerobot.gif" width="480" height="240" alt="Stonehenge paths" /></td>
  </tr>
  <tr>
    <th align="center" style="border: 1px solid; padding: 6px;">Dynamic Environment</th>
  </tr>
  <tr>
    <td style="border: 1px solid; padding: 6px;"><img src="./docs/barrel_rock_epitrochoid.gif" width="480" height="240" alt="Rock and barrels course" /></td>
  </tr>
</table>

FOCI is a novel method to compute orientation aware trajectories for robots using 3D Gaussian Splats to model both the robot and the environment. 


## Authors

- Mario Gomez Andreu\*<sup>1</sup>
- Carlos Martínez\*<sup>2</sup>
- Maximum Wilder-Smith\*<sup>1</sup>
- Victor Klemm<sup>1</sup>
- Vaishakh Patil<sup>1</sup>
- Jesus Tordesillas<sup>2</sup>
- Marco Hutter<sup>1</sup>

\*Equal contribution
<br><sup>1</sup> Robotic Systems Lab, ETH Zurich
<br><sup>2</sup> Trustworthy Robotics and AI Lab (TRAIL), ICAI-Comillas

## Citation

When using FOCI, please cite:

```bibtex
@misc{andreu2026focitrajectoryoptimizationgaussian,
      title={FOCI: Trajectory Optimization on Gaussian Splats for Gaussian Robots}, 
      author={Mario Gomez Andreu and Carlos Martinez Cuenca and Maximum Wilder-Smith and Victor Klemm and Vaishakh Patil and Marco Hutter and Jesus Tordesillas},
      year={2026},
      eprint={2611.33380},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2611.33380}, 
}

@inproceedings{andreu2025foci,
  title={Foci: Trajectory optimization on gaussian splats},
  author={Andreu, Mario Gomez and Klemm, Victor and Patil, Vaishakh and Tordesillas, Jesus and Hutter, Marco},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1884--1891},
  year={2025},
  organization={IEEE}
}
```


## Installation
1. Clone the repo with `git clone git@github.com:leggedrobotics/foci.git`
2. Obtain a licence for the `MA27` solver from [https://licences.stfc.ac.uk/product/coin-hsl](https://licences.stfc.ac.uk/product/coin-hsl), download the corresponding ZIP file, rename the extracted folder to `coinhsl` and move it to the root directory of this repository.
3. Create the virtual environment with `python3 -m venv venv` and install dependencies with `pip install -r requirements.txt`.
4. Download the Gaussian splat scenes listed under [Data Attribution](#data-attribution) and place them in the `data/` directory (these `.ply` files are not tracked by git, see `.gitignore`).
5. Launch any demo file with `python3 -m demos.<demo_name>`.


## Data Attribution
The Gaussian splat file (`data/stonehenge.ply`) used by this repository was generated using processed data from the [Splat-Nav project](https://github.com/chengine/splatnav), which is licensed under the MIT License. 
We acknowledge the authors of Splat-Nav for making their resources available.

The following scenes were obtained from the [Polycam](https://poly.cam) community gallery and are licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):

| File | Original capture |
|------|------------------|
| `data/Barrel.ply` | [Barrel](https://poly.cam/capture/a4b03e8a-f574-47a9-9c6b-4885f1f948cf) | 
| `data/Rock on forest path.ply` | [Rock on forest path](https://poly.cam/capture/2ebe580f-f0c8-401b-ab91-f1d20d3e35e9) | 
| `data/Bonsai.ply` | [new bonsai](https://poly.cam/capture/af057c60-4d80-4bf4-80a0-e623e5e6b257) | 
| `data/Coffee_Table.ply` | [Coffee Table](https://poly.cam/capture/f258d3c3-5f0f-4e8f-ba77-4cc4662cbdf6) | 
| `data/Forest.ply` | [Forest Scan](https://poly.cam/capture/ea94947f-ac7f-4029-b652-95a38a6c1a81) | 

These scenes are third-party assets and are **not** covered by the license of this repository's source code.
We thank the Polycam community members for sharing their captures.

For the manipulator experiments, the [UR5](https://www.universal-robots.com/es/blog/universal-robots-ur5/}{UR5) is used and the [ARDrone](https://github.com/mmyself/ros_ardrone/tree/master) for the aerial ones.
