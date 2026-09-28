"""Render test4's stacking run to an mp4, from a camera placed in the scene.

    HEADLESS=1 uv run python make_movie.py [a b c] [--still]

Drake renders the frames itself, so no Meshcat window or screen recording is
involved; ffmpeg has to be on the path to encode them.
"""

import subprocess
import sys

import numpy as np
from manipulation.station import CameraConfig
from pydrake.all import RigidTransform, RotationMatrix, Simulator
from pydrake.common.schema import Transform

import pick_and_place as pp

WIDTH, HEIGHT, FPS = 1280, 720, 30
EYE = np.array([1.25, -0.85, 0.55])
TARGET = np.array([0.5, -0.08, 0.18])


def look_at(eye, target) -> RigidTransform:
    """A camera pose at `eye` looking at `target`: +z forward, +y down."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return RigidTransform(RotationMatrix(np.column_stack([right, down, forward])), eye)


def scenario_with_camera(make_scenario=pp.make_scenario):
    scenario = make_scenario()
    camera = CameraConfig()
    camera.name = "movie"
    camera.width, camera.height, camera.fps = WIDTH, HEIGHT, FPS
    camera.X_PB = Transform(look_at(EYE, TARGET))
    camera.background.set([0.92, 0.93, 0.95, 1.0])
    scenario.cameras = {"movie": camera}
    return scenario


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    a, b, c = args if len(args) == 3 else ("red", "blue", "yellow")
    still = "--still" in sys.argv

    pp.make_scenario = scenario_with_camera
    system, duration = pp.build_system_tracking_waypoints(pp.plan(a, b, c), 0.05)
    simulator = Simulator(system)
    context = simulator.get_mutable_context()
    pp.start_the_integrator(system, context)
    station = system.GetSubsystemByName("station")
    image_port = station.GetOutputPort("movie.rgb_image")
    station_context = station.GetMyContextFromRoot(context)

    def frame() -> bytes:
        return np.ascontiguousarray(image_port.Eval(station_context).data[:, :, :3]).tobytes()

    if still:
        from PIL import Image

        simulator.AdvanceTo(12.0)
        Image.frombytes("RGB", (WIDTH, HEIGHT), frame()).save("movie_still.png")
        return

    out = f"stack_{a}_{b}_{c}.mp4"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
        stdin=subprocess.PIPE,
    )
    n_frames = int((duration + 2.0) * FPS) + 1
    for i in range(n_frames):
        simulator.AdvanceTo(i / FPS)
        ffmpeg.stdin.write(frame())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"wrote {out}: {n_frames} frames, {n_frames / FPS:.1f} s")


if __name__ == "__main__":
    main()
