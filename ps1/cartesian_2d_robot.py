######################################################################
## Code for nobody but Python to read
######################################################################

import numpy as np
from dtsystems import (
    Constant,
    HEADLESS,
    Observer,
    PController,
    PController2,
    SimpleTrajectoryFollower,
    get_meshcat,
    get_positions,
    plot_log,
    plot_path,
    series_composition,
    show_meshcat,
    simulate,
)
from numpy.typing import ArrayLike
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Diagram,
    DiagramBuilder,
    LogVectorOutput,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    plot_system_graphviz,
)

from utils.plotting import plt

######################################################################
## Code for students to be aware of
######################################################################

# --- define Robot and table --------------------------------------------------------

FINGER_RADIUS = 0.05
FINGER_HEIGHT = 0.3
FINGER_MASS = 1.0

# Table top is at z = 0.05.  Stand the cylinder on it with a small clearance so
# the finger slides freely rather than dragging on the surface.
TABLE_TOP_Z = 0.05
FINGER_CLEARANCE = 0.001
FINGER_Z = TABLE_TOP_Z + FINGER_HEIGHT / 2 + FINGER_CLEARANCE

JOINT_DAMPING = 20.0

# Solid-cylinder inertia about its center of mass.
_IXX = FINGER_MASS * (3 * FINGER_RADIUS**2 + FINGER_HEIGHT**2) / 12
_IZZ = FINGER_MASS * FINGER_RADIUS**2 / 2

table_sdf = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="table">
    <link name="table_link">
      <visual name="visual">
        <geometry>
          <box>
            <size>2 2 0.1</size>
          </box>
        </geometry>
        <material>
          <diffuse>0.7 0.6 0.5 1.0</diffuse>
        </material>
      </visual>
    </link>
    <joint name="table_weld" type="fixed">
      <parent>world</parent>
      <child>table_link</child>
    </joint>
  </model>
</sdf>
"""

# The finger hangs off the world on two prismatic joints (x then y), so it has
# exactly two degrees of freedom and is structurally incapable of tipping over.
# "carrier" is a small link that carries the x slide.
finger_sdf = f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="finger">
    <link name="carrier">
      <pose>0 0 {FINGER_Z} 0 0 0</pose>
      <inertial>
        <mass>0.01</mass>
        <inertia>
          <ixx>1e-5</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>1e-5</iyy><iyz>0</iyz><izz>1e-5</izz>
        </inertia>
      </inertial>
    </link>
    <link name="finger">
      <pose>0 0 {FINGER_Z} 0 0 0</pose>
      <inertial>
        <mass>{FINGER_MASS}</mass>
        <inertia>
          <ixx>{_IXX}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{_IXX}</iyy><iyz>0</iyz><izz>{_IZZ}</izz>
        </inertia>
      </inertial>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{FINGER_RADIUS}</radius>
            <length>{FINGER_HEIGHT}</length>
          </cylinder>
        </geometry>
        <material>
          <diffuse>0.9 0.3 0.2 1.0</diffuse>
        </material>
      </visual>
    </link>
    <joint name="finger_x" type="prismatic">
      <parent>world</parent>
      <child>carrier</child>
      <axis>
        <xyz>1 0 0</xyz>
        <dynamics><damping>{JOINT_DAMPING}</damping></dynamics>
      </axis>
    </joint>
    <joint name="finger_y" type="prismatic">
      <parent>carrier</parent>
      <child>finger</child>
      <axis>
        <xyz>0 1 0</xyz>
        <dynamics><damping>{JOINT_DAMPING}</damping></dynamics>
      </axis>
    </joint>
  </model>
</sdf>
"""
# --------------------------------------------------------------------------

# A square, closed by repeating the first corner so the finger returns home.
SQUARE = [
    np.array([0.3, 0.3]),
    np.array([-0.3, 0.3]),
    np.array([-0.3, -0.3]),
    np.array([0.3, -0.3]),
    np.array([0.3, 0.3]),
]

######################################################################
## Code for students to study
######################################################################

TIME_STEP = 1e-3


def create_2d_robot_diagram(
    force: ArrayLike = (0.0, 0.0), meshcat: Meshcat | None = None
) -> tuple[Diagram, MultibodyPlant]:
    """
    Build a diagram holding the table-and-finger MultibodyPlant, with a
    Constant source applying the given force at the actuation input, a state
    logger named "log_plant", and a MeshcatVisualizer when meshcat is given.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=TIME_STEP)
    parser = Parser(plant)
    parser.AddModelsFromString(table_sdf, "sdf")
    parser.AddModelsFromString(finger_sdf, "sdf")
    plant.Finalize()

    source = builder.AddSystem(Constant(force))
    builder.Connect(source.get_output_port(), plant.get_actuation_input_port())

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    logger.set_name("log_plant")

    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")
    return diagram, plant


def finger_s0(q0: np.ndarray) -> list[np.ndarray]:
    """
    Initial state list for simulate(): group 0 is the plant, whose discrete
    state is [x, y, xdot, ydot].
    """
    return [np.hstack([q0, np.zeros(2)])]


FINGER_START = np.array([0.3, 0.3])


def test_const_input(force: ArrayLike = [2.0, 0.0]) -> None:
    """
    No controller: shove the finger with a constant force.
    """
    meshcat = get_meshcat()
    diagram, plant = create_2d_robot_diagram(force=np.array(force), meshcat=meshcat)
    if not HEADLESS:
        plt.figure(figsize=(12, 6))
        plot_system_graphviz(diagram)
        plt.show()
    simulator = simulate(diagram, finger_s0(FINGER_START), 5.0)
    q_final = get_positions(plant, simulator)
    print(f"   Initial finger position: {FINGER_START}")
    print(f"   Final finger position:   {q_final}")
    plot_log(diagram, simulator, "log_plant")
    show_meshcat()


######################################################################
##  Code for students to write
######################################################################

# Some of the code for this question lives in dtsystems.py, because it is
# useful both for this simple robot and for the iiwa.


def create_2d_robot_diagram_with_controller(
    controller_gain: float, q_desired: np.ndarray, meshcat: Meshcat | None = None
) -> tuple[Diagram, MultibodyPlant]:
    """
    Like create_2d_robot_diagram, but the finger is driven by a proportional
    position controller instead of a constant force: compose an Observer that
    keeps the positions out of the plant state with a PController with the
    given gain and target q_desired, wired from the plant's state output back
    into its actuation input.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=TIME_STEP)
    parser = Parser(plant)
    parser.AddModelsFromString(table_sdf, "sdf")
    parser.AddModelsFromString(finger_sdf, "sdf")
    plant.Finalize()

    # Plant state is [x, y, xdot, ydot]; the P controller only wants [x, y].
    observer = Observer(4, [0, 1])
    controller = PController(2, q_desired, controller_gain)
    control = builder.AddSystem(series_composition(observer, controller))
    builder.Connect(plant.get_state_output_port(), control.get_input_port())
    builder.Connect(control.get_output_port(), plant.get_actuation_input_port())

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    logger.set_name("log_plant")

    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")
    return diagram, plant


def create_2d_robot_diagram_with_waypoints(
    controller_gain: float,
    waypoints: list[np.ndarray],
    epsilon: float = 0.02,
    meshcat: Meshcat | None = None,
) -> tuple[Diagram, MultibodyPlant]:
    """
    Drive the finger along a coarse trajectory: an Observer feeds the current
    position to a SimpleTrajectoryFollower over the given waypoints, whose
    output is the target for a PController2 with the given gain, whose output
    drives the plant's actuation input.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=TIME_STEP)
    parser = Parser(plant)
    parser.AddModelsFromString(table_sdf, "sdf")
    parser.AddModelsFromString(finger_sdf, "sdf")
    plant.Finalize()

    observer = builder.AddSystem(Observer(4, [0, 1]))
    follower = builder.AddSystem(SimpleTrajectoryFollower(waypoints, epsilon))
    controller = builder.AddSystem(PController2(2, controller_gain))

    builder.Connect(plant.get_state_output_port(), observer.get_input_port())
    builder.Connect(observer.get_output_port(), follower.get_input_port())
    builder.Connect(follower.get_output_port(), controller.GetInputPort("target"))
    builder.Connect(observer.get_output_port(), controller.GetInputPort("actual"))
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    logger.set_name("log_plant")
    log_follower = LogVectorOutput(follower.get_output_port(), builder)
    log_follower.set_name("log_follower")

    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")
    return diagram, plant


######################################################################
##  Test rigs
######################################################################


def test_position_controller(
    q_desired: ArrayLike = (0.9, 0.6), gain: float = 100, T: float = 5.0
) -> None:
    """
    Drive the finger from FINGER_START to q_desired with a P controller.
    """
    meshcat = get_meshcat()
    q_desired = np.asarray(q_desired, dtype=float)
    diagram, plant = create_2d_robot_diagram_with_controller(
        controller_gain=gain, q_desired=q_desired, meshcat=meshcat
    )
    simulator = simulate(diagram, finger_s0(FINGER_START), T)
    q_final = get_positions(plant, simulator)
    print(f"   Initial finger position: {FINGER_START}")
    print(f"   Target finger position:  {q_desired}")
    print(f"   Final finger position:   {q_final}")

    log_plant = diagram.GetSubsystemByName("log_plant")
    log = log_plant.FindLog(simulator.get_context())
    times = log.sample_times()
    data = log.data()
    pos_err = np.linalg.norm(data[:2] - q_desired[:, None], axis=0)
    speed = np.linalg.norm(data[2:4], axis=0)
    at_rest = (pos_err < 1e-3) & (speed < 1e-3)
    if np.any(at_rest):
        # First time after which the finger stays at the target.
        still_moving = np.where(~at_rest)[0]
        t_rest = times[0] if still_moving.size == 0 else times[still_moving[-1] + 1]
        print(f"   Time to rest at target:  {t_rest:.4f} s")
    else:
        print("   Finger did not come to rest at the target within T.")

    plt.figure(figsize=(8, 4))
    plt.plot(times, data[0], label="x")
    plt.plot(times, data[1], label="y")
    plt.axhline(q_desired[0], color="C0", linestyle="--", linewidth=1, label="x desired")
    plt.axhline(q_desired[1], color="C1", linestyle="--", linewidth=1, label="y desired")
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.title("finger position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    png_path = __file__.replace("cartesian_2d_robot.py", "finger_position.png")
    plt.savefig(png_path, dpi=150)
    print(f"   Saved position plot:     {png_path}")
    plt.close()
    if meshcat is not None:
        html_path = __file__.replace("cartesian_2d_robot.py", "finger_position_meshcat.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(meshcat.StaticHtml())
        print(f"   Saved meshcat recording: {html_path}")
        print(f"   Meshcat URL:             {meshcat.web_url()}")
    show_meshcat()


if __name__ == "__main__":
    test_position_controller()
