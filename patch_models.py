import os

def patch_sdf(filepath, robot_name):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    with open(filepath, 'r') as f:
        content = f.read()

    plugins = f"""
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>FL_hip_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>FR_hip_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>RL_hip_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>RR_hip_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>FL_thigh_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>FR_thigh_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>RL_thigh_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>RR_thigh_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>FL_calf_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>FR_calf_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>RL_calf_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce">
      <joint_name>RR_calf_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher">
        <topic>/model/{robot_name}/joint_state</topic>
        <update_frequency>1000</update_frequency>
        <joint_name>FL_hip_joint</joint_name>
        <joint_name>FR_hip_joint</joint_name>
        <joint_name>RL_hip_joint</joint_name>
        <joint_name>RR_hip_joint</joint_name>
        <joint_name>FL_thigh_joint</joint_name>
        <joint_name>FR_thigh_joint</joint_name>
        <joint_name>RL_thigh_joint</joint_name>
        <joint_name>RR_thigh_joint</joint_name>
        <joint_name>FL_calf_joint</joint_name>
        <joint_name>FR_calf_joint</joint_name>
        <joint_name>RL_calf_joint</joint_name>
        <joint_name>RR_calf_joint</joint_name>
    </plugin>
    <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
      <odom_frame>odom</odom_frame>
      <robot_base_frame>base</robot_base_frame>
      <odom_publish_frequency>1000</odom_publish_frequency>
      <dimensions>3</dimensions>
    </plugin>
    <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
      <publish_link_pose>false</publish_link_pose>
      <use_sensor_names>true</use_sensor_names>
      <publish_collision_pose>false</publish_collision_pose>
      <publish_visual_pose>false</publish_visual_pose>
      <publish_nested_model_pose>true</publish_nested_model_pose>
      <update_frequency>50</update_frequency>
    </plugin>"""

    import re
    # Strip ALL existing plugins (since they are Gazebo 11/Classic plugins and will break Gazebo Harmonic)
    content = re.sub(r'<plugin[^>]*/>', '', content)
    content = re.sub(r'<plugin[^>]*>(?:(?!</plugin>).)*?</plugin>', '', content, flags=re.DOTALL)
    
    # Insert new plugins before </model>
    content = content.replace("  </model>", plugins + "\n  </model>")
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Patched {filepath}")

def create_config(filepath, robot_name):
    config_content = f"""<?xml version="1.0"?>
<model>
  <name>{robot_name}_description</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>

  <author>
    <name>Unitree</name>
    <email>info@unitree.com</email>
  </author>

  <description>
    Unitree {robot_name.capitalize()} Quadruped Robot
  </description>
</model>
"""
    with open(filepath, 'w') as f:
        f.write(config_content)
    print(f"Created {filepath}")

if __name__ == "__main__":
    patch_sdf("Unitree_Go1/models/go1_description/model.sdf", "go1")
    create_config("Unitree_Go1/models/go1_description/model.config", "go1")
    
    patch_sdf("Unitree_A1/models/a1_description/model.sdf", "a1")
    create_config("Unitree_A1/models/a1_description/model.config", "a1")
