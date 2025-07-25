from lerobot.teleoperators.koch_leader import KochLeaderConfig, KochLeader
from lerobot.robots.koch_follower import KochFollowerConfig, KochFollower
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

# Camera configuration matching the command line
camera_config = {
    "front": OpenCVCameraConfig(
        index_or_path=0,
        width=640,
        height=480,
        fps=30
    ),
    "side": OpenCVCameraConfig(
        index_or_path=1,
        width=640,
        height=480,
        fps=30
    )
}

robot_config = KochFollowerConfig(
    port="COM7",
    id="follower",
    cameras=camera_config
)

teleop_config = KochLeaderConfig(
    port="COM8",
    id="leader",
)

# Set display_data to True as per command line
display_data = False

robot = KochFollower(robot_config)
teleop_device = KochLeader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    if display_data:
        print("Observation:", observation)
    action = teleop_device.get_action()
    robot.send_action(action)
