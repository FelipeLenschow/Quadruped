# 1. Use the official ROS 2 Humble base image (Ubuntu 22.04)
# We use 'ros-base' instead of 'desktop' because it officially supports both AMD64 (Laptop) and ARM64 (Robot)!
FROM ros:humble-ros-base

# 2. Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# 3. Enable the Universe repository and Gazebo Harmonic Repository
RUN apt-get update && apt-get install -y curl gnupg lsb-release && \
    curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-venv \
    wget \
    libglfw3-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    gz-harmonic \
    python3-gz-transport13 \
    python3-gz-msgs10 \
    && rm -rf /var/lib/apt/lists/*

# 3.5 Install PlotJuggler and Teleop tools
RUN apt-get update -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false && \
    apt-get install -y \
    ros-humble-plotjuggler-ros \
    ros-humble-teleop-twist-keyboard \
    || echo "Some ROS tools unavailable on this architecture. Continuing."

# 4. Create a workspace
WORKDIR /app

# 5. Pre-compile CycloneDDS (0.10.x) for Unitree SDK compatibility
RUN git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x /tmp/cyclonedds && \
    mkdir -p /tmp/cyclonedds/build && cd /tmp/cyclonedds/build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build . --target install && \
    rm -rf /tmp/cyclonedds

# 6. Setup Python Virtual Environment (Allow access to system site-packages for Gazebo bindings)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv --system-site-packages $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 6.5 Define paths for CycloneDDS (required for pip install cyclonedds)
ENV CYCLONEDDS_HOME=/usr/local
ENV LD_LIBRARY_PATH="/usr/local/lib"

# 7. Install Python dependencies (Robot + Sim2Sim)
# Upgrading pip and installing wheel first is good practice
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir \
    protobuf==3.20.3 \
    cyclonedds==0.10.2 \
    numpy \
    pyyaml \
    scipy \
    mujoco \
    torch \
    torchvision \
    torchaudio \
    onnxruntime \
    skrl \
    tensorboard

# 8. Set environment variables for Domain Isolation and Protobuf compatibility
ENV ROS_DOMAIN_ID=1
ENV ROS_LOCALHOST_ONLY=0
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 9. Source ROS 2 automatically when entering the container
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc

# 10. Customize the bash prompt so you know you are inside the Docker container
RUN echo 'export PS1="(Quadruped-Docker) \[\033[01;32m\]\u@docker\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]# "' >> ~/.bashrc

# Default command
CMD ["bash"]
