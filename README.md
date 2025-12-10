# reaction-diffusion-advection
Simulator for reaction-diffusion-advection systems for ML.
We use OpenFOAM as the underlying simulator.


## Installation

First, we need to install openfoam, usually using a package manager or precompiled binaries. For example, on Ubuntu, you can use:

```bash
wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | sudo bash
sudo apt-get install ca-certificates
sudo apt-get update
sudo apt-get install openfoam2412-default
```


## Prompt

I want you to create a openfoam simulation for a reaction-diffusion-advection system. My goal is to simulate many combinations of parameters to create a dataset for machine learning. The system should include:

- 2D square domain
- periodic boundary conditions
- advection term with a constant velocity field
- gray-scott reaction diffusion


I already installed the latest version of openfoam. Create the all necessary files and folders to run the simulation.
Ideally, I would like to have a script that can generate the simulation files for different parameter combinations (diffusion coefficients, feed rates, kill rates, advection velocities, etc.).

Check out these repositories for inspiration:
- gray-scott reaction diffusion in matlab: https://github.com/danfortunato/spectral-gray-scott
- openfoam reaction diffusion: https://github.com/shor-ty/GrayScottModel
- reaction diffusion advection: https://github.com/georginaalbadri/reaction-advection-diffusion