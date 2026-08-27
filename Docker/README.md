# Quadruped Universal Docker Environment

This folder contains the Docker configuration to run the Quadruped software identically across your laptop (Ubuntu 24), your VDI (Ubuntu 22), and the physical robot (Ubuntu 20).

## What is this `docker/` folder?

A common point of confusion is thinking that the project code needs to be moved *inside* the `docker/` folder. **This is not the case.**

The `docker/` folder simply acts as a recipe book. It holds the `Dockerfile` (the recipe) that tells your computer how to build the isolated Linux environment.

Your `Quadruped` code stays exactly where it is in the root directory. When you run the Docker container, we use a "Volume Mount" (`-v $(pwd):/app`). This creates a live portal between your computer's hard drive and the container's virtual hard drive. If you edit a Python script on your laptop, the Docker container sees the change instantly.

## 1. Getting the Image (Recommended: just pull it)

Prebuilt images live in the GitHub Container Registry under a single package,
`ghcr.io/felipelenschow/quadruped_env`, with **one tag per architecture**:

| Tag | Machine | Reports as (`uname -m`) |
| --- | --- | --- |
| `amd64` | Laptop / VDI | `x86_64` |
| `arm64` | Physical robot (Jetson Orin) | `aarch64` |

There is **no `latest` tag**, and the two tags are *not* joined into a multi-arch
manifest — so Docker will not pick for you. You have to name the tag matching the
machine you are on. Pulling the wrong one gives you an image that dies with
`exec format error` the moment you run it.

Pick the tag explicitly:

```bash
# Laptop / VDI (x86_64)
sudo docker pull ghcr.io/felipelenschow/quadruped_env:amd64
sudo docker tag  ghcr.io/felipelenschow/quadruped_env:amd64 quadruped_env

# Physical robot (aarch64)
sudo docker pull ghcr.io/felipelenschow/quadruped_env:arm64
sudo docker tag  ghcr.io/felipelenschow/quadruped_env:arm64 quadruped_env
```

...or let the shell fill it in, which makes it the same line on every machine:

```bash
ARCH=$(dpkg --print-architecture)   # prints exactly "amd64" or "arm64"
sudo docker pull ghcr.io/felipelenschow/quadruped_env:$ARCH && \
sudo docker tag ghcr.io/felipelenschow/quadruped_env:$ARCH quadruped_env
```

The `docker tag` line re-tags whichever image you pulled as plain `quadruped_env`,
which is why every command and alias further down this file is
architecture-independent and works unchanged on all three machines.

*On the physical robot:* if the CMOS battery is dead and the clock is wrong, the
pull will fail with a TLS certificate error. Fix the clock first
(`sudo date -s "$(date -u)"` from a machine with the right time, or `sudo ntpdate pool.ntp.org`).

## 1b. Building the Image Yourself (fallback)

Only needed if you're offline, are testing local `Dockerfile` changes before
pushing them, or the registry is unreachable.

Run this command from the **root of your `Quadruped` project** (not inside the docker folder):

```bash
sudo docker build --network host -t quadruped_env -f Docker/Dockerfile .
```

This builds **only for the architecture of the machine you are building on** — an
image built on the laptop will not run on the robot. To refresh both published
tags, see the next section.

*(Note: Building takes a few minutes because it compiles the Unitree SDK communication layer from source).*

## 1c. Publishing New Images to GHCR

The two tags are pushed by hand — there is no CI job doing it, so after changing
the `Dockerfile` you must republish both or the machines will drift apart.

One-time setup on the laptop/VDI:

```bash
# Log in to GHCR with a Personal Access Token that has the write:packages scope
echo <YOUR_GITHUB_PAT> | sudo docker login ghcr.io -u FelipeLenschow --password-stdin

# Register QEMU so an x86 machine can also produce arm64 images
sudo docker run --privileged --rm tonistiigi/binfmt --install all
```

Then build and push one tag per architecture, from the **root of the project**:

```bash
sudo docker buildx build --network host --platform linux/amd64 \
  -t ghcr.io/felipelenschow/quadruped_env:amd64 \
  -f Docker/Dockerfile --push .

sudo docker buildx build --network host --platform linux/arm64 \
  -t ghcr.io/felipelenschow/quadruped_env:arm64 \
  -f Docker/Dockerfile --push .
```

The `arm64` build runs under QEMU emulation on an x86 host, so expect it to be
several times slower than the native one (it compiles CycloneDDS from source).
Building it natively on the robot and pushing from there is the faster route if
the robot has a decent network connection.

The `LABEL org.opencontainers.image.source` in the `Dockerfile` is what attaches
the package to this GitHub repo. The package itself only has to be switched to
**Public** once, in the repo's *Packages* settings — otherwise every `docker pull`
demands a login.

## 2. Running the Docker Container

Run this from the **root of your project**. This form needs no display and is
safe on every machine, screen or not:

```bash
sudo docker run -it --rm \
  --name quadruped_container \
  --network host \
  --privileged \
  -v $(pwd):/app \
  quadruped_env
```

### What do these flags mean?

* `-it`: Starts an interactive terminal so you can type commands inside the container.
* `--rm`: Automatically deletes the container when you exit it (keeps your system clean).
* `--network host`: Gives the container direct access to your machine's network card (essential for ROS 2 and communicating with the robot).
* `--privileged`: Gives the container permission to access hardware devices (like USB cameras or IMUs if plugged into your laptop).
* `-v $(pwd):/app`: The "Live Portal". It maps your current working directory (the root of the project) to the `/app` folder inside the container.

## 3. Working inside the Container

Once you run the command above, your terminal will change to look something like `(Quadruped-Docker) root@docker:/app#`.

You are now inside the isolated ROS 2 Humble environment! You can now run any of your scripts (like the `Unitree` hardware drivers or the `IsaacSim` launchers) safely.

## 4. Creating a Shortcut (Alias)

To avoid typing the long `docker run` command every time, create a `quaddocker`
alias on the host machine.

The GUI machine (laptop/VDI) and the headless machine (the robot) are never the
same box, so **the alias keeps the same name everywhere** — install the variant
that matches the machine you are setting up and then always type `quaddocker`.

### On a machine with a screen (laptop / VDI)

```bash
echo "alias quaddocker='xhost +local:docker && sudo docker run -it --rm --name quadruped_container --network host --privileged --gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility -e DISPLAY=\$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v \$(pwd):/app quadruped_env'" >> ~/.bashrc
source ~/.bashrc
```

`--gpus all` is the flag that makes the MuJoCo viewer usable — without it the
container has no NVIDIA GL driver and silently falls back to `llvmpipe` CPU
rasterization at under 10 fps. Section 7 has the measurements and a one-line
check to confirm it took effect.

There is deliberately **no** `--device /dev/dri:/dev/dri`: on an NVIDIA machine
that flag does nothing useful (it only makes Mesa try, and fail, to load
`nouveau`), and `--gpus all` supersedes it. Keep it only if the host GPU is
Intel or AMD, where Mesa is the correct driver.

### On the robot (or any machine with no display)

```bash
echo "alias quaddocker='sudo docker run -it --rm --name quadruped_container --network host --privileged -v \$(pwd):/app quadruped_env'" >> ~/.bashrc
source ~/.bashrc
```

No `xhost`, no `DISPLAY`, no X11 socket, no `--gpus all` — section 6 explains how
each of those fails on a screenless box.

### Using it

From your `Quadruped` folder, on any machine:

```bash
quaddocker
```

> **After editing the alias, open a new terminal.** Bash caches aliases at shell
> startup, so a terminal that was already open keeps the old definition and will
> silently keep launching containers with the old flags. `source ~/.bashrc`
> updates the current shell but will not remove an alias you have since renamed
> or deleted.

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

## 6. Running on a Machine With No Screen (Headless)

The physical robot, a box reached over plain `ssh`, and any server without an X
server all fall in this category. Nothing about the *image* changes — only the
`docker run` flags you use, and which parts of the launcher are actually usable.

### Use the headless alias

The GUI variant of `quaddocker` from section 4 fails here in four different ways,
which is why the headless machine gets the stripped-down variant instead:

| Flag | What goes wrong without a display |
| --- | --- |
| `xhost +local:docker` | `xhost: unable to open display ""` — there is no X server to grant access to, and the `&&` means the container never starts. |
| `-e DISPLAY=$DISPLAY` | `DISPLAY` is empty, so anything graphical inside dies with `cannot open display`. |
| `-v /tmp/.X11-unix:/tmp/.X11-unix` | That path does not exist on the host, so Docker silently creates an empty directory instead. |
| `--device /dev/dri:/dev/dri` | Hard failure (`error gathering device information ... no such file or directory`) on any box without a render node. Check with `ls /dev/dri` and drop the flag if it is missing. |

The minimal command that works anywhere is the one from section 2:

```bash
sudo docker run -it --rm \
  --name quadruped_container \
  --network host \
  --privileged \
  -v $(pwd):/app \
  quadruped_env
```

### Inside the container

`launcher.py` already handles this: on ARM64 it sets headless mode
unconditionally (`IS_ROBOT`), and elsewhere it prompts `Headless Mode? [y/N]` —
answer `y`. Menu entries that exist only to open a window (RViz, PlotJuggler,
rqt, the interactive MCAP replayer) will start but never draw anything.

If you drive MuJoCo directly rather than through the launcher, either pass
`--headless` (`Mujoco/eval_mujoco.py`) or force software rendering so the
viewer's GL context does not need a display:

```bash
export MUJOCO_GL=osmesa   # the image ships libosmesa6-dev for exactly this
```

Stepping physics with no viewer at all needs no GL whatsoever, and is by far the
fastest path on the robot.

### Getting a GUI anyway

Three options, in rough order of preference:

1. **Foxglove (best for the robot).** Start the bridge inside the container
   (launcher → Foxglove, or `ros2 launch foxglove_bridge foxglove_bridge_launch.xml`),
   then open `app.foxglove.dev` in a browser on your laptop and connect to
   `ws://<robot-ip>:8765`. `--network host` means the port is already reachable;
   nothing needs to be rendered on the robot itself.
2. **Record now, look later.** Use the MCAP recording actions in the launcher,
   then copy the `.mcap` file back to a machine with a screen and replay it there.
3. **SSH X11 forwarding.** Works, but is slow over wireless. Connect with
   `ssh -X`, then start the container against the *forwarded* display instead of
   the local socket:

   ```bash
   sudo docker run -it --rm --name quadruped_container \
     --network host --privileged \
     -e DISPLAY=$DISPLAY \
     -v $HOME/.Xauthority:/root/.Xauthority:ro \
     -v $(pwd):/app quadruped_env
   ```

   `ssh -X` sets `DISPLAY` to something like `localhost:10.0`, i.e. a TCP
   connection rather than the `/tmp/.X11-unix` socket — which is why this variant
   mounts `.Xauthority` and *not* the X11 socket. `--network host` is what lets
   the container reach the forwarded port sitting on the host.

## 7. Known Issues & Limitations

### Low FPS in the MuJoCo viewer — SOLVED (add `--gpus all`)

**Symptom:** the MuJoCo viewer (launcher → Visualizers → MuJoCo Twin, or Play MuJoCo)
crawls at well under 10 fps inside the container.

**Cause:** the container has no NVIDIA GL driver. The `Dockerfile` installs only
Mesa (`libgl1-mesa-glx`, `libosmesa6-dev`, `libglfw3-dev`), so the image ships
`libGLX_mesa.so.0` and no `libGLX_nvidia.so.0`. Without `--gpus all`, Mesa sees
the card through `/dev/dri`, tries to load the open-source **nouveau** driver,
fails, and silently falls back to `llvmpipe` — pure CPU rasterization:

```
libGL error: glx: failed to create dri3 screen
libGL error: failed to load driver: nouveau
```

**Fix:** add `--gpus all` (already in the GUI `quaddocker` alias in section 4).
NVIDIA's OpenGL never travels through `/dev/dri` + Mesa; it needs
`libGLX_nvidia.so.0` and `/dev/nvidia*` injected by the container runtime, which
is exactly what that flag does.

**Measured** (offscreen render of `unitree_go2/scene.xml`, 640×480, identical image):

| Flags | `GL_RENDERER` | fps |
| --- | --- | --- |
| `--device /dev/dri`, no `--gpus` | `llvmpipe (LLVM 15.0.7)` | **9.9** |
| no `/dev/dri`, no `--gpus` | `llvmpipe` | 9.6 |
| `--gpus all` | `NVIDIA RTX 2000 Ada` | **683.7** |
| `--gpus all` + `/dev/dri` | `NVIDIA RTX 2000 Ada` | 669.6 |

On-screen at 1200×900 the fixed path holds a solid 60.0 fps vsync-locked
(1170 fps with vsync off), including the twin's transparent ghost robot.

Note `--device /dev/dri:/dev/dri` buys nothing on an NVIDIA host — it only
changes *how* Mesa fails. Keep it only for Intel/AMD GPUs, where Mesa is the
correct driver.

**Verify it actually took effect.** The failure is silent — GL just gets slow,
with no error unless you look. Run this inside the container:

```bash
python -c "import ctypes,mujoco; c=mujoco.GLContext(64,64); c.make_current(); \
g=ctypes.CDLL('libGL.so.1'); g.glGetString.restype=ctypes.c_char_p; \
print(g.glGetString(0x1F01).decode())"
```

It must print your NVIDIA card. If it prints `llvmpipe`, the GPU flag did not
reach this container. From the host you can check the same thing with
`sudo docker inspect quadruped_container --format '{{.HostConfig.DeviceRequests}}'`
— an empty `[]` means no GPU.

> **Two traps that make the fix look like it didn't work.**
>
> 1. *An alias only applies to newly started containers.* If `quadruped_container`
>    is already running it keeps the flags it was launched with. Check `sudo docker ps`.
> 2. *Bash caches aliases at shell startup.* Editing `~/.bashrc` does nothing to a
>    terminal that is already open — it will keep using the old definition and go on
>    launching containers with the old flags. Open a **new terminal**; `source
>    ~/.bashrc` updates the current shell but will not unbind an alias you have
>    since renamed or deleted.

*Historical note:* this section previously blamed a broken host NVIDIA runtime
(`open /usr/bin/nvidia-cuda-mps-control: no such file or directory`). That host
issue has since been repaired — `nvidia-container-toolkit` 1.20.0 is installed,
the `nvidia` runtime is registered in `/etc/docker/daemon.json`, and the MPS
binary exists. `--gpus all` no longer crashes.

### Physical Robot Deployment Quirks

When deploying this Docker image directly onto the physical Unitree robot, several embedded hardware issues were identified and resolved to make the image truly universal:

1. **Architecture Mismatch (`exec format error`)**: The image is based on `ros:humble-ros-base` instead of `desktop` because the robot utilizes an ARM64 processor, whereas laptops use AMD64. The `ros-base` image natively supports both. Note that the published images are still one tag *per* architecture (`:amd64` / `:arm64`, see section 1) — pulling the wrong tag reproduces this exact error.
2. **DNS Resolution Failure**: The robot's network configuration often drops Docker's internal DNS requests during the build phase. This is resolved by passing `--network host` to the `docker build` command so it inherits the robot's working internet connection.
3. **Dead CMOS Battery (Clock Drift)**: Physical robots often lose track of real-world time when powered off, resetting to 1970 or falling hours behind.
   * This causes `git pull` to fail due to SSL verification. Workaround: `env GIT_SSL_NO_VERIFY=true git pull`.
   * This causes `apt-get update` to fail because repository signatures appear to be from the "future". The `Dockerfile` permanently resolves this by passing `-o Acquire::Check-Valid-Until=false` to bypass time-checks.
4. **ROS 2 telemetry arriving in bursts (bad DDS return path)**: the Go2's Jetson ships on Unitree's internal `192.168.123.0/24`. Plugged into a lab switch on a *different* subnet, DDS discovery still finds peers via multicast, but the unicast return path has no valid source address or route. Reliable-QoS readers then stall on each lost packet and receive the retransmission batch all at once — telemetry lands in **bursts**, which looks like a laggy visualizer even though the renderer is at a solid 60 fps. Give `eth0` an address on the lab subnet and a default route:

   ```bash
   sudo ip addr add 10.13.3.18/24 dev eth0
   sudo ip route add default via 10.13.3.254
   sudo rm -f /etc/resolv.conf
   echo "nameserver 192.146.156.1" | sudo tee /etc/resolv.conf
   echo "10.13.3.190 gitlab.perro.tru.ca" | sudo tee -a /etc/hosts
   ```

   Diagnose before you reach for graphics flags: `ros2 topic hz /sensors/joint_states` should be a flat 50 Hz with a std dev in the tens of microseconds. Large `max` values or a jumpy rate mean transport, not rendering.

   > **These do not survive a reboot.** `ip addr add` / `ip route add` are runtime-only, and `/etc/resolv.conf` gets regenerated by whatever manages the robot's network. Make them permanent in the robot's own network config (netplan / NetworkManager / systemd-networkd) once the addresses are settled.
