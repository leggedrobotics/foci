# FOCI: Trajectory Optimization on Gaussian Splats for Gaussian Robots

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

## Installation
1. Clone the repo
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
