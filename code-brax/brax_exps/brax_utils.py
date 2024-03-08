from __future__ import division
import jax.numpy as np
import numpy as onp
import jax
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm
from pytinyrenderer import TinyRenderCamera as Camera
from pytinyrenderer import TinyRenderLight as Light
from pytinyrenderer import TinySceneRenderer as Renderer
from brax.io import image
import wandb
import brax
from typing import List, Tuple, Optional
from brax import math
from PIL import Image, ImageFont, ImageDraw
from brax.physics.base import vec_to_arr
from optax._src.schedule import linear_schedule, cosine_decay_schedule, join_schedules


def create_dtype_tree(pytree):
    def get_dtype(value):
        return value.dtype if isinstance(value, jnp.ndarray) else None

    return jax.tree_map(get_dtype, pytree)


def tree_type_difference(tree_1, tree_2):
    def get_dtype(value):
        return value.dtype if isinstance(value, jnp.ndarray) else None

    def get_type_diff(l1, l2):
        types_match = type(l1) == type(l2)
        dtypes_match = get_dtype(l1) == get_dtype(l2)
        shapes_match = l1.shape == l2.shape
        if hasattr(l1, "weak_type"):
            l1_weak_type = l1.weak_type
        elif hasattr(l1, "_unstack"):
            l1_weak_type = l1._unstack()[0].weak_type
        else:
            l1_weak_type = "none"
        if hasattr(l2, "weak_type"):
            l2_weak_type = l2.weak_type
        elif hasattr(l2, "_unstack"):
            l2_weak_type = l2._unstack()[0].weak_type
        else:
            l2_weak_type = "none"
        weak_types_match = l1_weak_type == l2_weak_type
        all_match = dtypes_match and shapes_match and weak_types_match and types_match
        if all_match:
            return "SAME"
        else:
            return f"DIFF: ({type(l1)} vs {type(l2)}), ({get_dtype(l1)} vs {get_dtype(l2)}), ({l1.shape} vs {l2.shape}), ({l1_weak_type} vs {l2_weak_type})"

    return jax.tree_map(get_type_diff, tree_1, tree_2)


class TextureRGB888:
    def __init__(self, pixels):
        self.pixels = pixels
        self.width = int(onp.sqrt(len(pixels) / 3))
        self.height = int(onp.sqrt(len(pixels) / 3))


class Grid(TextureRGB888):
    def __init__(self, grid_size, color):
        grid = onp.zeros((grid_size, grid_size, 3), dtype=onp.int32)
        grid[:, :] = onp.array(color)
        grid[0] = onp.zeros((grid_size, 3), dtype=onp.int32)
        grid[:, 0] = onp.zeros((grid_size, 3), dtype=onp.int32)
        super().__init__(list(grid.ravel()))


class GridThick(TextureRGB888):
    def __init__(self, grid_size, color, thickness):
        assert thickness < grid_size
        grid = onp.zeros((grid_size, grid_size, 3), dtype=onp.int32)
        grid[:, :] = onp.array(color)
        grid[:thickness] = onp.zeros((thickness, grid_size, 3), dtype=onp.int32)
        grid[:, -thickness:] = onp.zeros((grid_size, thickness, 3), dtype=onp.int32)
        super().__init__(list(grid.ravel()))


_BASIC = TextureRGB888([133, 118, 102])
_TARGET = TextureRGB888([255, 34, 34])
_GOAL = TextureRGB888([214, 39, 40])
_OBJECT1 = TextureRGB888([31, 119, 180])
_OBJECT2 = TextureRGB888([255, 127, 14])
_GROUND = Grid(100, [200, 200, 200])
_THICK_GROUND = GridThick(100, [200, 200, 200], thickness=4)


def _flatten_vectors(vectors):
    """Returns the flattened array of the vectors."""
    return sum(map(lambda v: [v.x, v.y, v.z], vectors), [])


def _scene(sys: brax.System, qp: brax.QP, use_thick_ground=False) -> Tuple[Renderer, List[int]]:
    """Converts a brax System and qp to a pytinyrenderer scene and instances."""
    scene = Renderer()
    instances = []
    mesh_geoms = {g.name: g for g in sys.config.mesh_geometries}
    for i, body in enumerate(sys.config.bodies):
        tex = _TARGET if body.name.lower() == "target" else _BASIC
        tex = _GOAL if body.name.lower() == "goal" else tex
        tex = _OBJECT1 if body.name.lower() == "object" else tex
        tex = _OBJECT2 if body.name.lower() == "object2" else tex
        if body.name.lower() == 'hidden':
            continue
        for col in body.colliders:
            col_type = col.WhichOneof("type")
            if col_type == "capsule":
                half_height = col.capsule.length / 2 - col.capsule.radius
                model = scene.create_capsule(
                    col.capsule.radius, half_height, 2, tex.pixels, tex.width, tex.height
                )
            elif col_type == "box":
                hs = col.box.halfsize
                model = scene.create_cube(
                    [hs.x, hs.y, hs.z], _BASIC.pixels, tex.width, tex.height, 16.0
                )
            elif col_type == "sphere":
                model = scene.create_capsule(
                    col.sphere.radius, 0, 2, tex.pixels, tex.width, tex.height
                )
            elif col_type == "plane":
                if use_thick_ground:
                    tex = _THICK_GROUND
                else:
                    tex = _GROUND
                model = scene.create_cube(
                    [1000.0, 1000.0, 0.0001], tex.pixels, tex.width, tex.height, 8192
                )
            elif col_type == "mesh":
                mesh = col.mesh
                g = mesh_geoms[mesh.name]
                scale = mesh.scale if mesh.scale else 1
                model = scene.create_mesh(
                    onp.array(_flatten_vectors(g.vertices)) * scale,
                    _flatten_vectors(g.vertex_normals),
                    [0] * len(g.vertices) * 2,
                    g.faces,
                    tex.pixels,
                    tex.width,
                    tex.height,
                    1.0,
                )
            else:
                raise RuntimeError(f"unrecognized collider: {col_type}")

            instance = scene.create_object_instance(model)
            off = onp.array([col.position.x, col.position.y, col.position.z])
            pos = onp.array(qp.pos[i]) + math.rotate(off, qp.rot[i])
            rot = math.euler_to_quat(vec_to_arr(col.rotation))
            rot = math.quat_mul(qp.rot[i], rot)
            scene.set_object_position(instance, list(pos))
            scene.set_object_orientation(instance, [rot[1], rot[2], rot[3], rot[0]])
            instances.append(instance)

    return scene, instances


def get_camera_light(up, sys, qp, width, height, ssaa, hfov=58.0, ant=False):
    eye = image._eye(sys, qp)
    vfov = int(hfov * height / width)
    direction = [0.57735, 0.57735, 0.57735]
    if ant:
        eye[0] = 5
        eye[1] = 5
        eye[2] = 10
        target = [2, 2, 0]
    else:
        eye[0] = -eye[0] / 1.5
        eye[1] = -eye[1] / 1.5
        target = [qp.pos[0, 0], qp.pos[0, 1] - .75, 0]
    camera = Camera(
        viewWidth=width * ssaa,
        viewHeight=height * ssaa,
        position=eye,
        target=target,
        up=up,
        hfov=hfov,
        vfov=vfov,
    )
    light = Light(
        direction=direction, ambient=0.8, diffuse=0.8, specular=0.6, shadowmap_center=target
    )
    return camera, light


def render_array(
    sys: brax.System,
    qp: brax.QP,
    width: int,
    height: int,
    light: Optional[Light] = None,
    camera: Optional[Camera] = None,
    ssaa: int = 2,
    use_thick_ground: bool = False,
) -> onp.ndarray:
    """Renders an RGB array of a brax system and QP."""
    if (len(qp.pos.shape), len(qp.rot.shape)) != (2, 2):
        raise RuntimeError("unexpected shape in qp")

    scene, instances = _scene(sys, qp, use_thick_ground)
    target = [qp.pos[0, 0], qp.pos[0, 1], 0]
    if light is None:
        direction = [0.57735, -0.57735, 0.57735]
        light = Light(
            direction=direction, ambient=0.8, diffuse=0.8, specular=0.6, shadowmap_center=target
        )
    if camera is None:
        eye, up = _eye(sys, qp), _up(sys)
        hfov = 58.0
        vfov = hfov * height / width
        camera = Camera(
            viewWidth=width * ssaa,
            viewHeight=height * ssaa,
            position=eye,
            target=target,
            up=up,
            hfov=hfov,
            vfov=vfov,
        )
    img = scene.get_camera_image(instances, light, camera).rgb
    arr = onp.reshape(onp.array(img, dtype=onp.uint8), (camera.view_height, camera.view_width, -1))
    if ssaa > 1:
        arr = onp.asarray(Image.fromarray(arr).resize((width, height)))
    return arr


def add_label(image_np, r, fontsize=20):
    # Convert numpy array to PIL Image
    image = Image.fromarray(image_np)

    # Make the image editable
    txt = ImageDraw.Draw(image)

    # Define the font for the text
    # You may need to provide full path to the font file
    # In this example, a default PIL font is used
    # fnt = ImageFont.load_default()
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", fontsize)

    # Create the label
    text = f"(r={r})"

    # Add the text to the bottom left of your image
    txt.text((10, image.height - 20), text, font=fnt, fill=(0, 0, 0))

    # Convert image back to numpy array
    image_np = np.array(image)

    return image_np


def add_reward_label(frame, state):
    if "reward_dist" in state.metrics:  # pusher
        text = f"r: {state.reward:.2f}, ctrl: {state.metrics['reward_ctrl']:.2f}, dist: {state.metrics['reward_dist']:.2f}, near: {state.metrics['reward_near']:.2f}"
        fontsize = 20
    elif "reward_forward" in state.metrics:
        reward, reward_ctrl, reward_fwd, reward_contact = (
            state.reward,
            state.metrics["reward_ctrl"],
            state.metrics["reward_forward"],
            state.metrics["reward_contact"],
        )
        x, y, dx, dy = (
            state.metrics["x_position"],
            state.metrics["y_position"],
            state.metrics["x_velocity"],
            state.metrics["y_velocity"],
        )
        text = f"r: {reward:.2f}, ctrl: {reward_ctrl:.2f}, dist: {reward_fwd:.2f}, near: {reward_contact:.2f}, x,y,dx/dt,dy/dt {x:.2f}, {y:.2f}, {dx:.2f}, {dy:.2f}"
        fontsize = 10
    else:
        raise NotImplementedError("Not implemented for envs except ant, pusher")
    return add_label(frame, text, fontsize)


def make_video(
    params,
    make_inference_fn,
    env,
    T,
    flip_camera,
    curr_seed,
    use_antialising=False,
    n_seeds=5,
    hfov=29.0,
    width=800,
    height=600,
    save_frames=None,
    frame_name='',
    video_name='clip.mp4'
):
    # Can turn on antialising for smoother video
    # but about 4x slower to render
    if use_antialising:
        ssaa = 2
    else:
        ssaa = 1
    if save_frames is None:
        save_frames = []
    # we use hfov=50 for ANT videos
    if not flip_camera:
        hfov = 50.0
    inference_fn = make_inference_fn(params)
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    # Render in high-resolution for figures
    frames_to_render = []

    all_frames = []
    all_states = []
    up = image._up(env.sys)
    for i in tqdm(range(n_seeds)):
        print(f"seed {i}")
        rollout = []
        rng = jax.random.PRNGKey(seed=curr_seed + i)
        state = jit_env_reset(rng=rng)
        states = []
        for _ in range(T):
            rollout.append(state)
            act_rng, rng = jax.random.split(rng)
            act, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_env_step(state, act)
            states.append(state)

        frames_to_render = [(t, rollout[t]) for t in save_frames]
        frames = []
        # all_frames = render_frames_parallel(env, rollout, width=800, height=600)

        for s in  tqdm(rollout):
            # Use multiprocessing for this
            camera, light = get_camera_light(
                up, env.sys, s.qp, width, height, ssaa, hfov, ant=not flip_camera
            )
            rendered_array = render_array(
                env.sys,
                s.qp,
                width,
                height,
                camera=camera,
                light=light,
                ssaa=ssaa,
                use_thick_ground=not flip_camera,
            )
            frames.append(rendered_array)
        all_frames.extend(frames)
        all_states.extend(states)
    all_frames = [add_reward_label(frame, state) for frame, state in zip(all_frames, all_states)]
    clip = ImageSequenceClip(all_frames, fps=15)  # type:ignore
    clip.write_videofile(video_name, fps=15)

    # These are high-definition frames rendered separately for figures
    for t, s in frames_to_render:
        camera, light = get_camera_light(
            up, env.sys, s.qp, width=1600, height=1200, ssaa=2, hfov=hfov, ant=not flip_camera
        )
        rendered_array = render_array(
            env.sys,
            s.qp,
            width=1600,
            height=1200,
            camera=camera,
            light=light,
            ssaa=ssaa,
            use_thick_ground=not flip_camera,
        )

        img = Image.fromarray(rendered_array, 'RGB')
        img.save(f'{frame_name}_frame_{t}.png')

    return wandb.Video(video_name)


def warmup_flat_cosine_decay_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    flat_steps: int,
    total_steps: int,
    end_value: float = 0.0,
):

    """Linear warmup followed by cosine decay.

    Args:
      init_value: Initial value for the scalar to be annealed.
      peak_value: Peak value for scalar to be annealed at end of warmup.
      warmup_steps: Positive integer, the length of the linear warmup.
      decay_steps: Positive integer, the total length of the schedule. Note that
        this includes the warmup time, so the number of steps during which cosine
        annealing is applied is `decay_steps - warmup_steps`.
      end_value: End value of the scalar to be annealed.
    Returns:
      schedule: A function that maps step counts to values.
    """
    schedules = [
        linear_schedule(init_value=init_value, end_value=peak_value, transition_steps=warmup_steps),
        linear_schedule(
            init_value=peak_value,
            end_value=peak_value,
            transition_steps=flat_steps,
            transition_begin=warmup_steps,
        ),
        cosine_decay_schedule(
            init_value=peak_value,
            decay_steps=total_steps - warmup_steps - flat_steps,
            alpha=end_value / peak_value,
        ),
    ]
    return join_schedules(schedules, [warmup_steps, warmup_steps + flat_steps])
