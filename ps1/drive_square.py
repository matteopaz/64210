"""Drive the 2D finger around SQUARE with a waypoint follower.

Edit the tunables below and run:

    python drive_square.py

A Meshcat window opens with a playback recording of the motion.
"""

from cartesian_2d_robot import (
    FINGER_START,
    SQUARE,
    create_2d_robot_diagram_with_waypoints,
    finger_s0,
)
from dtsystems import (
    get_meshcat,
    get_positions,
    plot_log,
    plot_path,
    show_meshcat,
    simulate,
)
from utils.plotting import plt
from utils.viz import keep_meshcat_open

######################################################################
## Tunables
######################################################################

GAIN = 100
EPSILON = 0.1
T = 5.0

# JOINT_DAMPING lives in cartesian_2d_robot.py (it is baked into the SDF at
# import time).  Gradescope 8.3 wants that set to 20.

######################################################################


def main() -> None:
    meshcat = get_meshcat()
    diagram, plant = create_2d_robot_diagram_with_waypoints(
        controller_gain=GAIN,
        waypoints=SQUARE,
        epsilon=EPSILON,
        meshcat=meshcat,
    )
    simulator = simulate(diagram, finger_s0(FINGER_START), T)
    q_final = get_positions(plant, simulator)

    print(f"   Gain:                    {GAIN}")
    print(f"   Epsilon:                 {EPSILON}")
    print(f"   T:                       {T} s")
    print(f"   Initial finger position: {FINGER_START}")
    print(f"   Final finger position:   {q_final}")

    plt.figure(figsize=(6, 6))
    plot_path(diagram, simulator, "log_plant")

    plt.figure(figsize=(8, 4))
    plot_log(diagram, simulator, "log_plant", "log_follower")

    if meshcat is not None:
        print(f"   Meshcat URL:             {meshcat.web_url()}")
        print("   Use the Meshcat playback slider to replay the recording.")
    show_meshcat()
    keep_meshcat_open()


if __name__ == "__main__":
    main()
