# iiwa problems for pset 1

import numpy as np
from dtsystems import (
    Constant,
    HEADLESS,
    Observer,
    PController,
    PDController,
    PDController2,
    SimpleTrajectoryFollower,
    get_meshcat,
    get_positions,
    series_composition,
    show_meshcat,
    simulate,
    plot_log,
)
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

from utils.drake_models import explain_model_download_error
from utils.plotting import plt

######################################################################
## Code for students to be aware of
######################################################################

IIWA14_URL = (
    "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)

Q_START = np.array([0, 1.0, 0.3, 0.7, 0, 0, 0])

# The joint-space step used in the PD-controller question: drive the arm from
# Q_START to Q_START + Q_STEP (a step on joints 1, 2 and 4).
Q_STEP = np.array([0.5, -0.3, 0, 0.4, 0, 0, 0])


def iiwa_s0(q0: np.ndarray) -> list[np.ndarray]:
    """
    Initial state list for simulate(): group 0 is the plant, whose discrete
    state is [q, v].
    """
    return [np.hstack([q0, np.zeros(7)])]


# Joint configurations that put the end effector at the corners of a 0.4 m
# square in the vertical plane x = 0.5, with the end-effector frame held
# axis-aligned, found (offline) by inverse kinematics -- a topic we will
# study properly later in the term.  Closed by repeating the first corner,
# like SQUARE in cartesian_2d_robot.py.  Drive the arm through them in order
# and the end effector draws the square.
SQUARE_IIWA = [
    np.array([-0.3743, 1.2271, 0.0031, 0.861, 0.0169, -0.3627, 0.3545]),
    np.array([0.2311, 1.894, 0.8932, 1.6624, -1.2178, -0.9016, 1.1868]),
    np.array([0.9917, 1.894, 0.8958, 1.6696, -1.2453, -0.9026, 0.4584]),
    np.array([0.7547, 1.1085, 0.6942, 0.8787, -1.4182, -0.6161, 0.2746]),
]
SQUARE_IIWA.append(SQUARE_IIWA[0])

######################################################################
## Code for students to study
######################################################################


def create_IIWA14_diagram(
    torques: np.ndarray = np.zeros(7), meshcat: Meshcat | None = None
) -> tuple[Diagram, MultibodyPlant]:
    """
    Build a diagram holding the welded-base iiwa14 MultibodyPlant, with a
    Constant source applying the given joint torques at the actuation input,
    a state logger named "log_plant", and a MeshcatVisualizer when meshcat is
    given.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    parser = Parser(plant, scene_graph)
    try:
        parser.AddModelsFromUrl(IIWA14_URL)
    except RuntimeError as e:
        # The first load downloads the models; this explains the one common
        # way that fails (a space or such in the venv's path) before re-raising.
        explain_model_download_error(e)
        raise
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
    plant.Finalize()

    source = builder.AddSystem(Constant(torques))
    builder.Connect(source.get_output_port(), plant.get_actuation_input_port())

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    logger.set_name("log_plant")

    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")
    return diagram, plant


def test_const_torque(q_initial: np.ndarray, torques: np.ndarray) -> None:
    """
    Simulate the arm from q_initial under a constant joint-torque vector.
    """
    meshcat = get_meshcat()
    diagram, plant = create_IIWA14_diagram(torques=torques, meshcat=meshcat)
    if not HEADLESS:
        plt.figure(figsize=(12, 6))
        plot_system_graphviz(diagram)
        plt.show()
    simulator = simulate(diagram, iiwa_s0(q_initial), 5.0)
    plot_log(diagram, simulator, "log_plant")
    q_final = get_positions(plant, simulator)
    print(f"   Initial joint positions: {q_initial}")
    print(f"   Final joint positions:   {q_final}")
    show_meshcat()


######################################################################
##  Code for students to write
######################################################################


def _iiwa_plant(builder: DiagramBuilder):
    """
    Welded-base iiwa14 plant plus its scene graph, matching create_IIWA14_diagram.
    """
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    parser = Parser(plant, scene_graph)
    try:
        parser.AddModelsFromUrl(IIWA14_URL)
    except RuntimeError as e:
        # The first load downloads the models; this explains the one common
        # way that fails (a space or such in the venv's path) before re-raising.
        explain_model_download_error(e)
        raise
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
    plant.Finalize()
    return plant, scene_graph


def create_IIWA14_diagram_with_pcontroller(
    controller_gain: float, q_desired: np.ndarray, meshcat: Meshcat | None = None
) -> tuple[Diagram, MultibodyPlant]:
    """
    Like create_IIWA14_diagram, but the joints are driven by a proportional
    position controller: compose an Observer that keeps the positions out of
    the 14-D plant state [q, v] with a PController with the given gain and
    target q_desired, wired from the plant's state output back into its
    actuation input.
    """
    builder = DiagramBuilder()
    plant, scene_graph = _iiwa_plant(builder)

    observer = Observer(14, list(range(plant.num_positions())))
    controller = PController(7, q_desired, controller_gain)
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


def create_IIWA14_diagram_with_pd_controller(
    controller_gain: float,
    damping_gain: float,
    q_desired: np.ndarray,
    dt: float = 0.01,
    meshcat: Meshcat | None = None,
) -> tuple[Diagram, MultibodyPlant]:
    """
    Like create_IIWA14_diagram_with_pcontroller, but with the PDController
    defined in dtsystems.py: controller_gain, damping_gain, and dt go through
    to it.
    """
    builder = DiagramBuilder()
    plant, scene_graph = _iiwa_plant(builder)

    observer = Observer(14, list(range(plant.num_positions())))
    controller = PDController(7, q_desired, controller_gain, damping_gain, dt)
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


def test_pdcontroller(
    q_desired: np.ndarray,
    controller_gain: float,
    damping_gain: float,
    dt: float,
    q_initial: np.ndarray,
) -> None:
    """
    Drive the arm from q_initial to q_desired with a PD controller.
    """
    meshcat = get_meshcat()
    diagram, plant = create_IIWA14_diagram_with_pd_controller(
        controller_gain=controller_gain,
        damping_gain=damping_gain,
        q_desired=q_desired,
        dt=dt,
        meshcat=meshcat,
    )
    # Plant discrete state is [q, v]; PDController's is the previous measured q.
    simulator = simulate(diagram, iiwa_s0(q_initial) + [q_initial], 5.0)
    plot_log(diagram, simulator, "log_plant")
    q_final = get_positions(plant, simulator)
    print(f"   Initial joint positions: {q_initial}")
    print(f"   Target joint positions:  {q_desired}")
    print(f"   Final joint positions:   {q_final}")
    show_meshcat()


def create_IIWA14_diagram_with_waypoints(
    waypoints: list[np.ndarray],
    controller_gain: float = 10000,
    damping_gain: float = 3000,
    epsilon: float = 0.01,
    dt: float = 0.01,
    meshcat: Meshcat | None = None,
) -> tuple[Diagram, MultibodyPlant]:
    """
    Drive the arm through a sequence of joint-space waypoints: an Observer
    keeps the positions out of the 14-D plant state, a SimpleTrajectoryFollower
    over the waypoints outputs the current target, and a PDController2 turns
    target and actual into joint torques.

    The follower only advances once the arm is within epsilon of the current
    waypoint, so this controller must be stiff enough that its gravity sag
    stays well under epsilon.
    """
    builder = DiagramBuilder()
    plant, scene_graph = _iiwa_plant(builder)

    nq = plant.num_positions()
    observer = builder.AddSystem(Observer(14, list(range(nq))))
    follower = builder.AddSystem(SimpleTrajectoryFollower(waypoints, epsilon))
    controller = builder.AddSystem(
        PDController2(nq, controller_gain, damping_gain, dt)
    )

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


def test_square_iiwa(
    waypoints: list[np.ndarray] | None = None,
    controller_gain: float = 10000,
    damping_gain: float = 3000,
    epsilon: float = 0.01,
    T: float = 5.0,
) -> None:
    """
    Draw SQUARE_IIWA: start at the first joint-space corner and chase the rest
    with the PD waypoint follower.
    """
    if waypoints is None:
        waypoints = SQUARE_IIWA
    meshcat = get_meshcat()
    if meshcat is not None:
        # Look at the workspace in front of the base, where the 0.4 m square lives.
        meshcat.SetCameraPose(
            np.array([1.15, -0.75, 0.55]),
            np.array([0.45, 0.0, 0.75]),
        )
    diagram, plant = create_IIWA14_diagram_with_waypoints(
        waypoints=waypoints,
        controller_gain=controller_gain,
        damping_gain=damping_gain,
        epsilon=epsilon,
        meshcat=meshcat,
    )
    q0 = waypoints[0]
    # Plant discrete state is [q, v]; follower index; PDController2's previous q.
    simulator = simulate(diagram, iiwa_s0(q0) + [[0.0], q0], T)
    plot_log(diagram, simulator, "log_plant")
    q_final = get_positions(plant, simulator)
    print(f"   Initial joint positions: {q0}")
    print(f"   Final joint positions:   {q_final}")
    print(f"   Last waypoint:           {waypoints[-1]}")
    print(f"   Error to last waypoint:  {np.linalg.norm(q_final - waypoints[-1]):.4f}")
    if meshcat is not None:
        html_path = __file__.replace("iiwa_1.py", "iiwa_square_meshcat.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(meshcat.StaticHtml())
        print(f"   Saved meshcat recording: {html_path}")
        print(f"   Meshcat URL:             {meshcat.web_url()}")
    show_meshcat()


if __name__ == "__main__":
    test_square_iiwa()
    from utils.viz import keep_meshcat_open

    keep_meshcat_open()


