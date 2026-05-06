# Quadruped Universal Docker Environment

This folder contains the Docker configuration to run the Quadruped software identically across your laptop (Ubuntu 24), your VDI (Ubuntu 22), and the physical robot (Ubuntu 20).

## What is this `docker/` folder?
A common point of confusion is thinking that the project code needs to be moved *inside* the `docker/` folder. **This is not the case.** 

The `docker/` folder simply acts as a recipe book. It holds the `Dockerfile` (the recipe) that tells your computer how to build the isolated Linux environment. 

Your `Quadruped` code stays exactly where it is in the root directory. When you run the Docker container, we use a "Volume Mount" (`-v $(pwd):/app`). This creates a live portal between your computer's hard drive and the container's virtual hard drive. If you edit a Python script on your laptop, the Docker container sees the change instantly.

## 1. Building the Docker Image
You only need to do this once per machine, or if we add new `pip` or `apt` dependencies to the `Dockerfile`.

Run this command from the **root of your `Quadruped` project** (not inside the docker folder):
```bash
sudo docker build -t quadruped_env -f docker/Dockerfile .
```
*(Note: Building takes a few minutes because it compiles the Unitree SDK communication layer from source).*

## 2. Running the Docker Container
Run this from the **root of your project**:

```bash
sudo docker run -it --rm \
  --name quadruped_container \
  --network host \
  --privileged \
  -v $(pwd):/app \
  quadruped_env
```

### What do these flags mean?
*   `-it`: Starts an interactive terminal so you can type commands inside the container.
*   `--rm`: Automatically deletes the container when you exit it (keeps your system clean).
*   `--network host`: Gives the container direct access to your machine's network card (essential for ROS 2 and communicating with the robot).
*   `--privileged`: Gives the container permission to access hardware devices (like USB cameras or IMUs if plugged into your laptop).
*   `-v $(pwd):/app`: The "Live Portal". It maps your current working directory (the root of the project) to the `/app` folder inside the container.

## 3. Working inside the Container
Once you run the command above, your terminal will change to look something like `(Quadruped-Docker) root@docker:/app#`.

You are now inside the isolated ROS 2 Humble environment! You can now run any of your scripts (like the `Unitree` hardware drivers or the `IsaacSim` launchers) safely.

## 4. Creating a Shortcut (Alias)
To avoid typing the long `docker run` command every time, you can create a shortcut alias on your host machine (laptop/VDI/robot).

Run this single command in your **normal terminal** (outside the docker container):
```bash
echo "alias quaddocker='xhost +local:docker && sudo docker run -it --rm --name quadruped_container --network host --privileged --device /dev/dri:/dev/dri -e DISPLAY=\$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v \$(pwd):/app quadruped_env'" >> ~/.bashrc
source ~/.bashrc
```

From now on, you just need to navigate to your `Quadruped` folder and type:
```bash
quaddocker
```
This will instantly launch the container and drop you into the `/app` folder!

## 5. Opening a Second Terminal
If the container is already running (e.g., you started it with `quaddocker`) and you want to open another terminal inside it, open a new terminal on your host machine and run:
```bash
sudo docker exec -it quadruped_container bash
```

To make this easier, you can add a second alias:
```bash
echo "alias quadattach='sudo docker exec -it quadruped_container bash'" >> ~/.bashrc
source ~/.bashrc
```

## 6. Known Issues & Limitations

### Low FPS in MuJoCo (VDI Environment)
When running GUI applications like the MuJoCo Viewer inside the Docker container on the VDI, you may experience extremely low framerates. This is because Docker defaults to software (CPU) rendering for OpenGL applications.

**Attempted Solutions (For Future Reference):**
1. **NVIDIA Container Toolkit (`--gpus all`)**: We attempted to pass the NVIDIA GPU directly into the container using `--gpus all` and various `-e NVIDIA_DRIVER_CAPABILITIES` flags. This failed with an OCI runtime crash (`open /usr/bin/nvidia-cuda-mps-control: no such file or directory`) because the NVIDIA Docker daemon configuration on the VDI is misconfigured/broken at the system level.
2. **Dummy MPS File Workaround**: We attempted to trick the NVIDIA runtime by creating a dummy `nvidia-cuda-mps-control` file on the host. This failed, indicating the issue is deeply rooted in the Docker daemon's interaction with the VDI's vGPU.
3. **Direct Rendering Infrastructure (`--device /dev/dri:/dev/dri`)**: We bypassed the NVIDIA runtime entirely and mounted the raw Linux graphics devices. While this successfully allowed the container to boot and render the GUI, it did not resolve the low FPS, likely because the VDI's virtualized display pipeline does not expose native hardware acceleration through standard DRI.

*Current Status*: MuJoCo is fully functional for simulation, but the graphical viewer will run at a low framerate on the VDI until the host machine's NVIDIA Docker Runtime is repaired by system administrators.
