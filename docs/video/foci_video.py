"""
FOCI - narrated explainer video (Manim Community Edition)
=========================================================

Animated explanation of FOCI: Trajectory Optimization on Gaussian Splats for
Gaussian Robots, narrated with Microsoft Edge TTS. This single file contains
every scene and a small script that renders them all and joins them (video +
narration) into one video.

Requirements
------------
    - Python >= 3.12
    - pip install manim manim-voiceover edge-tts
      (manim also installs numpy, Pillow and PyAV, used for GIFs, videos and
      to join the scenes)
    - A LaTeX distribution (MiKTeX or TeX Live), needed by MathTex
    - Internet connection the first time, to generate the narration

Media files (not included)
--------------------------
The intro GIFs and the experiment videos are not part of this file. Fill in
their paths in the "Parameters" section below (INTRO_GIF_LEFT,
INTRO_GIF_RIGHT and EXPERIMENTS). Any path left empty is skipped, so the
video still renders without them.

Render everything and join it into one video
--------------------------------------------
    python foci_video.py            # low quality (480p15), fast
    python foci_video.py h          # 1080p60
    python foci_video.py k          # 4K (2160p60)
    python foci_video.py k --join-only   # only join scenes already rendered

The result is saved as FOCI_<quality>.mp4 next to this file.

Render a single scene (with preview)
------------------------------------
    manim -pql foci_video.py Hook
    manim -pql foci_video.py DualRepresentation
    manim -pql foci_video.py RobotModeling
    manim -pql foci_video.py CollisionOverlap1D
    manim -pql foci_video.py TrajectorySpline
    manim -pql foci_video.py CostFunction
    manim -pql foci_video.py Experiments

Scenes are saved in media/videos/foci_video/<quality>/<SceneName>.mp4 and the
narration is cached in media/voiceovers/ (only sentences whose text, voice or
speed change are generated again).
"""

# Star import first, so the explicit imports below are never shadowed by it
from manim import *

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import edge_tts
import numpy as np
from manim_voiceover import VoiceoverScene
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import (
    SpeechService,
    initialize_speech_service,
    path_to_string,
)
from PIL import Image, ImageSequence


# ===========================================================================
# Shared helpers: animation speed, colors, Gaussians, B-splines
# ===========================================================================

# ---------------------------------------------------------------------------
# Animation speed for text
# ---------------------------------------------------------------------------
# WRITE_SPEED: how much faster Write(...) is than Manim's default
#   (Manim uses 1 s for short texts and 2 s for long ones; 2.0 -> 0.5 s / 1 s).
# FADE_OUT_TIME: duration in seconds of FadeOut(...) (Manim's default is 1 s).
# Passing run_time=... explicitly in a call still overrides both.
WRITE_SPEED = 2.0
FADE_OUT_TIME = 0.5

_ManimWrite = Write
_ManimFadeOut = FadeOut


def Write(mobject, **kwargs):
    anim = _ManimWrite(mobject, **kwargs)
    if "run_time" not in kwargs:
        anim.run_time /= WRITE_SPEED
    return anim


def FadeOut(*mobjects, **kwargs):
    kwargs.setdefault("run_time", FADE_OUT_TIME)
    return _ManimFadeOut(*mobjects, **kwargs)


# ---------------------------------------------------------------------------
# Shared colors and helpers
# ---------------------------------------------------------------------------

ENV_COLOR = ManimColor("#FF6A4D")  # environment / obstacle Gaussians (warm orange-red)
ROBOT_COLOR = ManimColor("#4FC3F7")  # robot Gaussians (cyan-blue)
ARM_COLOR = GREY_B  # color of the robot's rigid "skeleton"
ENV_LABEL_COLOR = ManimColor("#C8645A")
ROBOT_LABEL_COLOR = ManimColor("#6FA8C0")

# Colors for the 4 control points of a cubic segment (P0, P1, P2, P3)
POINT_COLORS = [
    ManimColor("#E8335A"),
    ManimColor("#3BA7E8"),
    ManimColor("#2FD4A0"),
    ManimColor("#F2C14E"),
]


def lerp(p, q, alpha):
    """Linear interpolation between two points."""
    return np.array(p) + alpha * (np.array(q) - np.array(p))


def unit(v):
    """Normalize a vector, safely handling the zero-vector case."""
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def fit_to_box(mobj, max_width, max_height):
    """Scale a mobject down (never up) so it fits inside a given box."""
    width_scale = max_width / mobj.width if mobj.width > max_width else 1
    height_scale = max_height / mobj.height if mobj.height > max_height else 1
    scale = min(width_scale, height_scale)
    if scale < 1:
        mobj.scale(scale)
    return mobj


def gaussian_splat(
    center, width, height, angle=0.0, color=ENV_COLOR, n_rings=7, strength=1.0
):
    """
    A 2D Gaussian drawn as nested iso-density contours (like a heat map):
    faint and wide on the outside, brighter and whiter towards the mean.
    `width`/`height` are the size of the outermost contour, `angle` rotates
    the principal axes (anisotropic covariance).
    """
    color = ManimColor(color)
    rings = VGroup()
    for i in range(n_rings):
        k = i / max(n_rings - 1, 1)  # 0 = outermost contour, 1 = innermost
        s = 1.0 - 0.85 * k  # relative size of this contour
        ring_color = interpolate_color(color, WHITE, 0.55 * k**2)
        line_color = interpolate_color(color, WHITE, 0.35 + 0.3 * k)
        ring = Ellipse(width=width * s, height=height * s)
        ring.set_fill(ring_color, opacity=strength * (0.10 + 0.20 * k))
        ring.set_stroke(
            line_color,
            width=1.0 + 0.5 * k,
            opacity=strength * (0.55 if i == 0 else 0.25 + 0.25 * k),
        )
        rings.add(ring)
    rings.rotate(angle)
    rings.move_to(center)
    return rings


def gaussian_blob(center, radius=0.5, color=ROBOT_COLOR, opacity=0.5):
    """An isotropic (spherical) Gaussian."""
    return gaussian_splat(
        center, 2 * radius, 2 * radius, color=color, strength=min(1.0, opacity / 0.5)
    )


def random_cloud(n, color, center=ORIGIN, spread=3.0, seed=0):
    """A cloud of anisotropic Gaussians simulating a 3DGS scene."""
    rng = np.random.default_rng(seed)
    group = VGroup()
    for _ in range(n):
        offset = np.array(
            [
                rng.uniform(-spread, spread),
                rng.uniform(-spread * 0.55, spread * 0.55),
                0.0,
            ]
        )
        pos = np.array(center, dtype=float) + offset
        w = rng.uniform(0.35, 1.0)
        h = w * rng.uniform(0.35, 1.0)
        group.add(
            gaussian_splat(
                pos,
                w,
                h,
                angle=rng.uniform(0, PI),
                color=color,
                n_rings=5,
                strength=rng.uniform(0.5, 1.0),
            )
        )
    return group


def corridor_group(center, color=ENV_COLOR, spacing=1.1, gap=1.9):
    """A structured environment: two walls of anisotropic Gaussians (a corridor)."""
    center = np.array(center, dtype=float)
    top_angles = [0, 0, 80, 75, 82]  # degrees
    bottom_angles = [25, 28, 30, 25, 28]
    xs = (np.arange(len(top_angles)) - (len(top_angles) - 1) / 2) * spacing
    group = VGroup()
    for x, a in zip(xs, top_angles):
        group.add(
            gaussian_splat(
                center + np.array([x, gap / 2, 0]), 1.2, 0.62, a * DEGREES, color
            )
        )
    for x, a in zip(xs, bottom_angles):
        group.add(
            gaussian_splat(
                center + np.array([x, -gap / 2, 0]), 1.2, 0.62, a * DEGREES, color
            )
        )
    return group


def simple_robot_group(center, color=ROBOT_COLOR):
    """A 2-link arm: white skeleton with one elongated Gaussian along each link."""
    center = np.array(center, dtype=float)
    j0 = center + LEFT * 1.4 + DOWN * 0.5
    j1 = center + LEFT * 0.1 + UP * 0.55
    j2 = center + RIGHT * 1.4 + DOWN * 0.4

    splats = VGroup()
    for a, b in ((j0, j1), (j1, j2)):
        d = b - a
        splats.add(
            gaussian_splat(
                (a + b) / 2,
                np.linalg.norm(d) * 1.3,
                0.85,
                np.arctan2(d[1], d[0]),
                color,
            )
        )
    skeleton = VGroup(
        Line(j0, j1, color=WHITE, stroke_width=6),
        Line(j1, j2, color=WHITE, stroke_width=6),
    )
    joints = VGroup(*[Dot(p, color=WHITE, radius=0.07) for p in (j0, j1, j2)])
    return VGroup(splats, skeleton, joints)


# ---------------------------------------------------------------------------
# Helpers for the collision scene: 1D normal density
# ---------------------------------------------------------------------------


def npdf(x, mu, sigma):
    """1D normal density N(x; mu, sigma^2)."""
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


# ---------------------------------------------------------------------------
# Helpers for the trajectory scene: cubic B-splines and the MINVO basis
# ---------------------------------------------------------------------------

# Rows = basis functions of P0..P3, columns = coefficients of [t^3, t^2, t, 1], t in [0,1]
BSPLINE_M = (
    np.array(
        [
            [-1, 3, -3, 1],
            [3, -6, 0, 4],
            [-3, 3, 3, 1],
            [1, 0, 0, 0],
        ],
        dtype=float,
    )
    / 6.0
)
# MINVO basis for n = 3 on t in [0,1] (Tordesillas & How, 2022)
MINVO_M = np.array(
    [
        [
            -3.4416308968564117698,
            6.9895481477801393311,
            -4.4622887507045296829,
            0.91437149978080234369,
        ],
        [6.6792587327074839365, -11.845989901556746915, 5.2523596690684613009, 0.0],
        [
            -6.6792587327074839365,
            8.1917862965657040064,
            -1.5981560640774179483,
            0.085628500219197656307,
        ],
        [3.4416308968564117698, -3.3353445427890959785, 0.80808514571348655231, 0.0],
    ]
)


def _tpow(t):
    return np.array([t**3, t**2, t, 1.0])


def bspline_segment_point(P4, t):
    """Point at t in [0,1] of the uniform cubic B-spline segment with control points P4."""
    return np.asarray(P4, dtype=float).T @ BSPLINE_M @ _tpow(t)


def bspline_segment_samples(P4, n=40):
    return [bspline_segment_point(P4, t) for t in np.linspace(0, 1, n)]


def bspline_curve_points(ctrl, samples_per_seg=40):
    pts = []
    for i in range(len(ctrl) - 3):
        seg = bspline_segment_samples(ctrl[i : i + 4], samples_per_seg)
        pts += seg if i == 0 else seg[1:]
    return pts


def bspline_to_minvo(P4):
    """MINVO control points V of the same cubic segment: V^T A_mv = P^T A_bs."""
    P4 = np.asarray(P4, dtype=float)
    return (P4.T @ BSPLINE_M @ np.linalg.inv(MINVO_M)).T


def convex_hull_2d(points):
    """Monotone-chain convex hull of 2D points (z ignored). Returns 3D vertices."""
    pts = sorted({(round(p[0], 9), round(p[1], 9)) for p in points})
    if len(pts) <= 2:
        return [np.array([x, y, 0.0]) for x, y in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return [np.array([x, y, 0.0]) for x, y in hull]


def polygon_area(vertices):
    x = np.array([v[0] for v in vertices])
    y = np.array([v[1] for v in vertices])
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def control_handle(point, color=WHITE, radius=0.1):
    """Hollow circle used to draw a control point."""
    return Circle(
        radius=radius,
        stroke_color=color,
        stroke_width=3,
        fill_color=config.background_color,
        fill_opacity=1,
    ).move_to(point)


def basis_panel(matrix, heading_text, position):
    """Small framed plot of the 4 basis functions of a cubic segment."""
    axes = Axes(
        x_range=[0, 1, 1], y_range=[0, 1, 1], x_length=2.6, y_length=1.15, tips=False
    )
    axes.set_opacity(0)
    curves = VGroup(
        *[
            axes.plot(
                lambda t, row=row: float(max(row @ _tpow(t), 0.0)),
                x_range=[0, 1, 0.01],
                color=col,
                stroke_width=3,
            )
            for row, col in zip(matrix, POINT_COLORS)
        ]
    )
    frame = SurroundingRectangle(
        axes, buff=0.12, corner_radius=0.1, stroke_color=GREY_C, stroke_width=1.5
    )
    heading = Text(heading_text, font_size=18, weight=BOLD).next_to(frame, UP, buff=0.1)
    return VGroup(frame, axes, curves, heading).move_to(position)


# ===========================================================================
# GIFs and videos inside a scene
# ===========================================================================

# Do NOT use FadeIn/FadeOut (or any animation) directly on these objects:
# during an animation Manim pauses their updaters and animates copies, so
# the GIF/video jumps back when it ends. Use media_fade_in / media_fade_out.

class _MediaMobject(ImageMobject):
    """
    Base for animated images. Keeps its own opacity instead of using
    ImageMobject.set_opacity, which writes into the pixel array in place
    (it would make the stored frames themselves transparent).
    """

    def _init_media(self, first_frame, **kwargs):
        self._raw = first_frame
        self._opacity = 1.0
        ImageMobject.__init__(self, first_frame, **kwargs)
        self._fade_driver = None

    def _show(self, frame):
        self._raw = frame
        if self._opacity >= 1.0:
            self.pixel_array = frame
        else:
            out = frame.copy()
            out[:, :, 3] = (frame[:, :, 3] * self._opacity).astype(np.uint8)
            self.pixel_array = out

    def set_opacity(self, alpha):
        self._opacity = float(alpha)
        self._show(self._raw)
        return self


class AnimatedImage(_MediaMobject):
    """Sequence of images played in a loop (e.g. a GIF)."""

    def __init__(self, frames, durations, loop=True, **kwargs):
        self._init_media(frames[0], **kwargs)
        self._frames = frames
        self._ends = np.cumsum(durations)  # end time of each frame (s)
        self._total = float(self._ends[-1])
        self._loop = loop
        self._t = 0.0
        self.add_updater(self._advance)

    @classmethod
    def from_gif(cls, path, **kwargs):
        frames, durations = [], []
        with Image.open(path) as im:
            for frame in ImageSequence.Iterator(im):
                frames.append(np.array(frame.convert("RGBA")))
                durations.append(frame.info.get("duration", 100) / 1000.0)
        return cls(frames, durations, **kwargs)

    @property
    def duration(self):
        return self._total

    def _advance(self, mob, dt):
        self._t += dt
        t = self._t % self._total if self._loop else min(self._t, self._total - 1e-6)
        i = min(int(np.searchsorted(self._ends, t, side="right")), len(self._frames) - 1)
        if self._frames[i] is not self._raw:
            self._show(self._frames[i])


class _SharedDecoder:
    """
    Holds the PyAV container. Manim deep-copies mobjects in some operations,
    and PyAV objects cannot be copied, so every copy shares this same object.
    """

    def __init__(self, path, max_width):
        import av

        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.max_width = max_width
        if self.container.duration is not None:
            self.duration = self.container.duration / 1e6
        else:
            self.duration = float(self.stream.duration * self.stream.time_base)
        self._frames = self.container.decode(self.stream)
        self.current = self._next()
        self.upcoming = self._next()

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    def _to_array(self, frame):
        if frame.width > self.max_width:
            h = round(frame.height * self.max_width / frame.width)
            frame = frame.reformat(width=self.max_width, height=h - h % 2)
        return frame.to_ndarray(format="rgba")

    def _next(self):
        try:
            frame = next(self._frames)
        except StopIteration:
            return None
        return float(frame.time or 0.0), self._to_array(frame)

    def advance_to(self, t):
        """Move forward to the frame shown at time t. Returns True if it changed."""
        changed = False
        while self.upcoming is not None and self.upcoming[0] <= t:
            self.current, self.upcoming = self.upcoming, self._next()
            changed = True
        return changed


class VideoClip(_MediaMobject):
    """
    Video (mp4, mov, ...) decoded on the fly with PyAV: it is never fully
    loaded in memory. It plays once and stays on its last frame. The video's
    own audio is ignored (the narration comes from manim-voiceover).

    max_width: frames are downscaled to this width in pixels to save memory
        (3840 = 4K; lower it if rendering is slow).
    speed: 2.0 plays the video at double speed, etc.
    target_duration: if given, speeds the video up (never slows it down) so
        it fits in that many seconds.
    max_duration: if given, only the first `max_duration` seconds are used.
    """

    def __init__(
        self,
        path,
        max_width=3840,
        speed=1.0,
        target_duration=None,
        max_duration=None,
        **kwargs,
    ):
        self._decoder = _SharedDecoder(path, max_width)
        self._length = self._decoder.duration  # seconds of source video used
        if max_duration:
            self._length = min(self._length, max_duration)
        if target_duration:
            speed = max(speed, self._length / target_duration)
        self._speed = speed
        self._init_media(self._decoder.current[1], **kwargs)
        self._t = 0.0
        self.add_updater(self._advance)

    @property
    def duration(self):
        """Duration on screen (taking `speed` into account)."""
        return self._length / self._speed

    @property
    def remaining(self):
        """Seconds (on screen) left until the end of the video."""
        return max(0.0, (self._length - self._t) / self._speed)

    def _advance(self, mob, dt):
        self._t = min(self._t + dt * self._speed, self._length)
        if self._decoder.advance_to(self._t):
            self._show(self._decoder.current[1])


# ---------------------------------------------------------------------------
# Fades that don't pause the media
# ---------------------------------------------------------------------------


def _media_fade(scene, mobs, target, run_time):
    # Only one fade driver per media object at a time
    for m in mobs:
        if m._fade_driver is not None:
            scene.remove(m._fade_driver)
    tracker = ValueTracker(mobs[0]._opacity)
    driver = Mobject()
    driver.add_updater(lambda _: [m.set_opacity(tracker.get_value()) for m in mobs])
    for m in mobs:
        m._fade_driver = driver
    scene.add(driver)
    return tracker.animate(run_time=run_time).set_value(target)


def media_fade_in(scene, *mobs, run_time=0.5):
    """Animation (for scene.play) that fades GIFs/videos in without pausing them."""
    for m in mobs:
        m.set_opacity(0.0)
    scene.add(*mobs)
    return _media_fade(scene, mobs, 1.0, run_time)


def media_fade_out(scene, *mobs, run_time=0.5):
    """Animation (for scene.play) that fades GIFs/videos out."""
    return _media_fade(scene, mobs, 0.0, run_time)


def media_remove(scene, *mobs):
    """Remove the GIFs/videos (and their fade driver) from the scene."""
    for m in mobs:
        if m._fade_driver is not None:
            scene.remove(m._fade_driver)
            m._fade_driver = None
    scene.remove(*mobs)


# ===========================================================================
# Narration: Microsoft Edge TTS service for manim-voiceover
# ===========================================================================

# Free and without an API key, but it needs internet the first time each
# sentence is generated; afterwards the audio is reused from
# media/voiceovers/. List voices with:  edge-tts --list-voices

class EdgeTTSService(SpeechService):
    def __init__(
        self,
        voice: str = "en-US-AndrewNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        max_retries: int = 10,
        request_timeout: float = 30.0,
        **kwargs,
    ):
        """
        Args:
            voice: voice name (e.g. "en-US-AndrewNeural", "es-ES-AlvaroNeural").
            rate: relative speed, e.g. "+10%" or "-5%".
            pitch: relative pitch, e.g. "+0Hz", "-20Hz".
            volume: relative volume, e.g. "+0%".
        """
        initialize_speech_service(self, kwargs)
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    def generate_from_text(self, text, cache_dir=None, path=None, **kwargs):
        if cache_dir is None:
            cache_dir = self.cache_dir

        input_text = remove_bookmarks(text)
        # Everything that changes the audio goes into the cache key
        input_data = {
            "input_text": input_text,
            "service": "edge_tts",
            "voice": self.voice,
            "rate": self.rate,
            "pitch": self.pitch,
            "volume": self.volume,
        }

        cached_result = self.get_cached_result(input_data, cache_dir)
        if cached_result is not None:
            return cached_result

        if path is None:
            audio_path = self.get_audio_basename(input_data) + ".mp3"
        else:
            audio_path = path_to_string(path)

        # The Edge TTS server fails intermittently ("No audio was received",
        # dropped connections, or requests that hang), so every attempt has a
        # hard time limit and any error triggers a retry.
        out_file = Path(cache_dir) / audio_path
        for attempt in range(1, self.max_retries + 1):
            try:
                communicate = edge_tts.Communicate(
                    input_text,
                    self.voice,
                    rate=self.rate,
                    pitch=self.pitch,
                    volume=self.volume,
                    connect_timeout=10,
                    receive_timeout=15,
                )
                # Own thread + event loop, so it works even if a loop is already running
                with ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(
                        asyncio.run,
                        asyncio.wait_for(
                            communicate.save(str(out_file)),
                            timeout=self.request_timeout,
                        ),
                    ).result()
                if out_file.stat().st_size == 0:
                    raise edge_tts.exceptions.NoAudioReceived("empty file")
                break
            except Exception as e:
                out_file.unlink(missing_ok=True)
                if attempt == self.max_retries:
                    raise
                logger.warning(
                    f"Edge TTS failed ({type(e).__name__}, attempt "
                    f"{attempt}/{self.max_retries}), retrying..."
                )
                time.sleep(min(2.0 * attempt, 10))

        return {
            "input_text": text,
            "input_data": input_data,
            "original_audio": audio_path,
        }


# ===========================================================================
# Parameters
# ===========================================================================

# ---- Voice settings (shared by every scene) --------------------------------
VOICE = "en-US-AndrewNeural"
RATE = "+10%"


def make_speech_service():
    return EdgeTTSService(voice=VOICE, rate=RATE)


# ---- Intro GIFs (first scene) ----------------------------------------------
# Paths to the two GIFs shown under the title (left / right). Empty = skipped.
INTRO_GIF_LEFT = ""
INTRO_GIF_RIGHT = ""
INTRO_GIF_WIDTH = 6.5  # width of each GIF in Manim units (the frame is ~14.2 wide)
INTRO_GIF_HOLD = 2.0  # extra seconds on the title screen after the narration

# ---- Experiment videos (last scene) ----------------------------------------
# (video path, robot type shown in the top-left corner), in the order they
# appear. Entries with an empty path are skipped. The videos play at their
# original speed, without their own audio. Videos longer than
# EXPERIMENT_MAX_SECONDS are cut: only their first seconds are shown.
# The robot label only changes (with a short transition) when the type changes.
EXPERIMENTS = [
    ("", "Quadruped"),
    ("", "Quadruped"),
    ("", "Quadruped"),
    ("", "Quadruped"),
    ("", "Quadruped"),
    ("", "Quadruped"),
    ("", "Robotic arm"),
    ("", "Robotic arm"),
    ("", "Standard UAV"),
]
EXPERIMENT_MAX_SECONDS = 7.2  # the short videos last 7.2 s, so they stay whole


# ===========================================================================
# Scenes
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Hook / motivation
# ---------------------------------------------------------------------------


class Hook(VoiceoverScene):
    def construct(self):
        self.set_speech_service(make_speech_service())

        title = Text("FOCI", font_size=72)
        subtitle = Text(
            "Trajectory Optimization on Gaussian Splats for Gaussian Robots",
            font_size=32,
        )
        subtitle.next_to(title, DOWN)

        # Looping GIFs below the title, side by side at the same height
        gifs = [
            AnimatedImage.from_gif(path)
            for path in (INTRO_GIF_LEFT, INTRO_GIF_RIGHT)
            if path
        ]
        if gifs:
            # Title block at the top, leaving the rest of the frame for the GIFs
            VGroup(title, subtitle).to_edge(UP, buff=0.4)
            for gif in gifs:
                gif.set(width=INTRO_GIF_WIDTH)
            gif_row = Group(*gifs).arrange(RIGHT, buff=0.5)
            # Centered vertically in the space left below the subtitle
            free_top = subtitle.get_bottom()[1]
            free_bottom = -config.frame_height / 2
            gif_row.move_to([0, (free_top + free_bottom) / 2, 0])
        else:
            VGroup(title, subtitle).move_to(ORIGIN)

        gif_in = [media_fade_in(self, *gifs)] if gifs else []
        gif_out = [media_fade_out(self, *gifs)] if gifs else []

        with self.voiceover(
            text="This is FOCI: trajectory optimization on Gaussian splats, "
            "for Gaussian robots."
        ):
            # media_fade_in instead of FadeIn: FadeIn would pause the GIFs and make
            # them jump back to their first frame when the fade ends
            self.play(Write(title), FadeIn(subtitle, shift=DOWN), *gif_in)
        self.wait(INTRO_GIF_HOLD)  # extra time to watch the GIFs
        self.play(FadeOut(title), FadeOut(subtitle), *gif_out)
        if gifs:
            media_remove(self, *gifs)

        scene_cloud = random_cloud(60, ENV_COLOR, spread=4.0, seed=1)
        question = Text(
            "How do we plan trajectories in an environment modelled as a Gaussian Splat?",
            font_size=34,
        )
        fit_to_box(question, config.frame_width - 1.0, 1.0).to_edge(UP)
        with self.voiceover(
            text="Gaussian splats model a scene as thousands of small Gaussians. "
            "So how can a robot plan a trajectory directly in one?"
        ):
            self.play(LaggedStartMap(FadeIn, scene_cloud, lag_ratio=0.02), run_time=2)
            self.play(Write(question))

        self.play(FadeOut(scene_cloud), FadeOut(question))

def highlight_sequence(scene, items, highlight=YELLOW, caption_size=24):
    """
    Highlight items one after another, each with its own narration.
    items: list of (mobject, caption_text or None, narration).
    The previous item goes back to white in the same animation in which the
    next one lights up, so there are no dead "un-highlight" steps.
    Returns the last caption on screen (or None).
    """
    prev, caption = None, None
    for mob, caption_text, narration in items:
        anims = [mob.animate.set_color(highlight)]
        if prev is not None:
            anims.append(prev.animate.set_color(WHITE))
        if caption_text is not None:
            new_caption = Text(caption_text, font_size=caption_size).to_edge(DOWN)
            anims.append(
                FadeIn(new_caption)
                if caption is None
                else ReplacementTransform(caption, new_caption)
            )
            caption = new_caption
        with scene.voiceover(text=narration):
            scene.play(*anims, run_time=0.6)
        prev = mob
    anims = [prev.animate.set_color(WHITE)]
    if caption is not None:
        anims.append(FadeOut(caption))
    scene.play(*anims)


# ---------------------------------------------------------------------------
# 2. Core idea: robot + environment as Gaussians
# ---------------------------------------------------------------------------


class DualRepresentation(VoiceoverScene):
    def construct(self):
        self.set_speech_service(make_speech_service())

        title = Text(
            "FOCI represents both the robot and the environment with Gaussians",
            font_size=26,
        ).to_edge(UP)

        env_group = corridor_group(LEFT * 3.4)
        env_label = Text("Environment", font_size=30, color=ENV_LABEL_COLOR).next_to(
            env_group, DOWN, buff=0.35
        )

        robot_group = simple_robot_group(RIGHT * 3.6, color=ROBOT_COLOR)
        robot_label = Text("Robot", font_size=30, color=ROBOT_LABEL_COLOR).next_to(
            robot_group, DOWN, buff=0.35
        )

        with self.voiceover(
            text="The key idea is to describe the environment with Gaussians,"
        ):
            self.play(Write(title))
            self.play(
                LaggedStartMap(FadeIn, env_group, lag_ratio=0.08),
                Write(env_label),
                run_time=1.5,
            )

        with self.voiceover(text="and the robot too."):
            self.play(FadeIn(robot_group), Write(robot_label))

        with self.voiceover(
            text="Then, checking for collisions is just measuring how much they overlap."
        ):
            pass  # nothing moves: the scene just stays while the sentence is spoken

        self.play(
            FadeOut(VGroup(env_group, env_label, robot_group, robot_label, title))
        )


# ---------------------------------------------------------------------------
# 3. Robot modeling flexibility: one arm, three different Gaussian treatments
# ---------------------------------------------------------------------------


class RobotModeling(VoiceoverScene):
    def construct(self):
        self.set_speech_service(make_speech_service())

        title = Text("Gaussian modeling of a robot", font_size=28).to_edge(UP)

        base = LEFT * 5.0 + DOWN * 0.3
        j1 = LEFT * 2.6 + UP * 1.0
        j2 = LEFT * 0.2 + UP * 0.2
        j3 = RIGHT * 1.8 + UP * 0.9

        mount = Square(
            side_length=0.25, color=WHITE, fill_color=WHITE, fill_opacity=1
        ).move_to(base)

        sticks = VGroup(
            Line(base, j1, color=ARM_COLOR, stroke_width=10),
            Line(j1, j2, color=ARM_COLOR, stroke_width=10),
            Line(j2, j3, color=ARM_COLOR, stroke_width=10),
        )
        joints = VGroup(*[Dot(p, color=WHITE) for p in (j1, j2, j3)])

        # Gripper: a simple "U" shape attached at the end of the third stick
        dir3 = unit(j3 - j2)
        angle3 = np.arctan2(dir3[1], dir3[0])
        palm_half, prong_len = 0.35, 0.8
        local_pts = [
            np.array([prong_len, palm_half, 0]),
            np.array([0, palm_half, 0]),
            np.array([0, -palm_half, 0]),
            np.array([prong_len, -palm_half, 0]),
        ]
        rot = np.array(
            [
                [np.cos(angle3), -np.sin(angle3), 0],
                [np.sin(angle3), np.cos(angle3), 0],
                [0, 0, 1],
            ]
        )
        gripper = VMobject(color=ARM_COLOR, stroke_width=10)
        gripper.set_points_as_corners([j3 + rot @ p for p in local_pts])

        for m in (mount, sticks, gripper, joints):
            m.set_z_index(2)

        with self.voiceover(text="Any robot can be modelled with Gaussians."):
            self.play(Write(title))
            self.play(FadeIn(mount), Create(sticks), Create(gripper), FadeIn(joints))

        # Stick 1: 3 equally spaced spherical Gaussians
        pts1 = [lerp(base, j1, t) for t in (1 / 6, 3 / 6, 5 / 6)]
        group1 = VGroup(
            *[gaussian_blob(p, radius=0.5, color=ROBOT_COLOR) for p in pts1]
        )
        caption1 = Text("3 equally spaced spherical Gaussians", font_size=26).to_edge(
            DOWN
        )
        with self.voiceover(text="Evenly spaced spheres on one link,"):
            self.play(FadeIn(group1), Write(caption1))

        # Stick 2: 2 unevenly spaced spherical Gaussians
        pts2 = [lerp(j1, j2, 0.3), lerp(j1, j2, 0.75)]
        group2 = VGroup(
            gaussian_blob(pts2[0], radius=0.6, color=ROBOT_COLOR),
            gaussian_blob(pts2[1], radius=0.75, color=ROBOT_COLOR),
        )
        caption2 = Text("2 unevenly spaced spherical Gaussians", font_size=26).to_edge(
            DOWN
        )
        with self.voiceover(text="spheres of different sizes on another,"):
            self.play(FadeIn(group2), ReplacementTransform(caption1, caption2))

        # Stick 3 + gripper: a single non-spherical Gaussian covering both
        tip_point = j3 + dir3 * prong_len
        span = np.linalg.norm(tip_point - j2)
        ellipse_center = (j2 + tip_point) / 2
        group3 = gaussian_splat(
            ellipse_center, span + 1.0, 1.5, angle=angle3, color=ROBOT_COLOR
        )
        caption3 = Text(
            "1 non-spherical Gaussian covering the gripper", font_size=26
        ).to_edge(DOWN)
        with self.voiceover(text="or a single elongated Gaussian."):
            self.play(FadeIn(group3), ReplacementTransform(caption2, caption3))

        final_caption = Text(
            "Shape and number of Gaussians are free to choose",
            font_size=24,
        ).to_edge(DOWN)
        with self.voiceover(text="The shape and number of Gaussians are up to you."):
            self.play(ReplacementTransform(caption3, final_caption))

        self.play(
            FadeOut(
                VGroup(
                    mount,
                    sticks,
                    gripper,
                    joints,
                    group1,
                    group2,
                    group3,
                    final_caption,
                    title,
                )
            )
        )


# ---------------------------------------------------------------------------
# 4. How collision is measured: overlap integral between the two fields
# ---------------------------------------------------------------------------


class CollisionOverlap1D(VoiceoverScene):
    def construct(self):
        self.set_speech_service(make_speech_service())

        title = Text(
            """
            Collision is measured using the Field Overlap Collision Integral:
            A measure of how much the gaussians of the environment overlap with the ones of the robot
            """,
            font_size=22,
        ).to_edge(UP)

        # ---- The two fields (paper notation) -------------------------------
        p_def = MathTex(
            r"p(\boldsymbol{x},t)",
            r"=\sum_{n=1}^{N}\mathcal{N}\big(\boldsymbol{x};\boldsymbol{\mu}_n(t),"
            r"\boldsymbol{\Sigma}_n(t)\big)",
            font_size=30,
        )
        r_def = MathTex(
            r"r(\boldsymbol{x},\boldsymbol{q}(t))",
            r"=\sum_{m=1}^{M}\mathcal{N}\big(\boldsymbol{x};"
            r"\bar{\boldsymbol{\mu}}_m(\boldsymbol{q}(t)),"
            r"\bar{\boldsymbol{\Sigma}}_m(\boldsymbol{q}(t))\big)",
            font_size=30,
        )
        p_def[0].set_color(ENV_COLOR)
        r_def[0].set_color(ROBOT_COLOR)
        defs = VGroup(p_def, r_def).arrange(RIGHT, buff=0.8)
        fit_to_box(defs, config.frame_width - 1.0, 1.0)
        defs.next_to(title, DOWN, buff=0.35)

        with self.voiceover(
            text="The overlap is the Field Overlap Collision Integral. "
            "The environment and the robot each define a field: a sum of Gaussians."
        ):
            self.play(Write(title))
            self.play(Write(p_def), Write(r_def))

        # ---- 1D illustration ------------------------------------------------
        axes = Axes(
            x_range=[-5, 6, 1],
            y_range=[0, 1.2, 0.5],
            x_length=11,
            y_length=3.6,
            axis_config={"include_tip": False, "include_ticks": False},
        ).shift(DOWN * 0.95)
        axes.get_y_axis().set_opacity(0)  # only the x axis is meaningful here
        note = (
            Text("1D illustration", font_size=18, color=GREY_B)
            .next_to(axes, UR, buff=0)
            .shift(DOWN * 0.3 + LEFT * 0.2)
        )

        env_mu = [-3.3, -1.9, 0.1]
        env_sigma = [0.55, 0.45, 0.6]
        rob_offset = [-0.45, 0.45]
        rob_sigma = [0.45, 0.38]
        c = ValueTracker(4.2)  # position of the robot (depends on q(t))

        def p_field(x):
            return sum(npdf(x, m, s) for m, s in zip(env_mu, env_sigma))

        def r_field(x):
            cv = c.get_value()
            return sum(npdf(x, cv + o, s) for o, s in zip(rob_offset, rob_sigma))

        def collision_value():
            # Closed form: sum_n sum_m N(mu_bar_m; mu_n, Sigma_n + Sigma_bar_m)
            cv = c.get_value()
            return sum(
                npdf(cv + o, m, np.sqrt(s**2 + sb**2))
                for m, s in zip(env_mu, env_sigma)
                for o, sb in zip(rob_offset, rob_sigma)
            )

        xr = [-5, 6, 0.02]

        env_parts = VGroup(
            *[
                axes.plot(
                    lambda x, m=m, s=s: npdf(x, m, s),
                    x_range=xr,
                    color=ENV_COLOR,
                    stroke_width=1.5,
                    stroke_opacity=0.45,
                )
                for m, s in zip(env_mu, env_sigma)
            ]
        )
        env_sum = axes.plot(p_field, x_range=xr, color=ENV_COLOR, stroke_width=4)
        env_label = MathTex(r"p(x,t)", font_size=30, color=ENV_COLOR).next_to(
            axes.c2p(-2.6, p_field(-1.9)), UP, buff=0.25
        )

        rob_parts = always_redraw(
            lambda: VGroup(
                *[
                    axes.plot(
                        lambda x, o=o, s=s: npdf(x, c.get_value() + o, s),
                        x_range=xr,
                        color=ROBOT_COLOR,
                        stroke_width=1.5,
                        stroke_opacity=0.45,
                    )
                    for o, s in zip(rob_offset, rob_sigma)
                ]
            )
        )
        rob_sum = always_redraw(
            lambda: axes.plot(r_field, x_range=xr, color=ROBOT_COLOR, stroke_width=4)
        )
        rob_label = always_redraw(
            lambda: MathTex(r"r(x,q(t))", font_size=30, color=ROBOT_COLOR).next_to(
                axes.c2p(c.get_value(), r_field(c.get_value() + 0.45)), UP, buff=0.25
            )
        )

        with self.voiceover(
            text="In this 1D example, the environment is in red and the robot in blue."
        ):
            self.play(Create(axes), FadeIn(note), run_time=0.8)
            self.play(Create(env_parts), Create(env_sum), Write(env_label), run_time=1)
            self.play(Create(rob_parts), Create(rob_sum), Write(rob_label), run_time=1)

        # The integrand p * r: the "collision volume" where both fields overlap
        integrand = always_redraw(
            lambda: axes.get_area(
                axes.plot(
                    lambda x: p_field(x) * r_field(x), x_range=xr, stroke_opacity=0
                ),
                x_range=[-5, 6],
                color=YELLOW,
                opacity=0.65,
            )
        )
        value_tex = MathTex(
            r"\int p(x,t)\,r(x,q(t))\,dx \;=",
            font_size=32,
        )
        value_num = DecimalNumber(collision_value(), num_decimal_places=3, font_size=32)
        value_group = VGroup(value_tex, value_num).arrange(RIGHT, buff=0.2)
        value_group.to_edge(DOWN, buff=0.3)
        value_num.add_updater(lambda d: d.set_value(collision_value()))

        self.add(integrand)
        with self.voiceover(
            text="Their product, in yellow, is the overlap. "
            "As the robot moves into the obstacles, its integral grows."
        ) as tracker:
            self.play(FadeIn(value_group), run_time=0.6)
            self.play(
                c.animate.set_value(-1.2),
                run_time=max(tracker.duration - 0.6, 2.5),
                rate_func=smooth,
            )

        value_num.clear_updaters()
        self.play(
            FadeOut(
                VGroup(
                    axes,
                    note,
                    env_parts,
                    env_sum,
                    env_label,
                    rob_parts,
                    rob_sum,
                    rob_label,
                    integrand,
                    value_group,
                )
            )
        )
        self.remove(rob_parts, rob_sum, rob_label, integrand)

        # ---- Closed form (Eq. 2 of the paper) -------------------------------
        eq0 = MathTex(
            r"\int_{\mathbb{R}^3} p(\boldsymbol{x},t)\,"
            r"r(\boldsymbol{x},\boldsymbol{q}(t))\,d\boldsymbol{x}",
            font_size=34,
        )
        eq1 = MathTex(
            r"=\sum_{n=1}^{N}\sum_{m=1}^{M}\int_{\mathbb{R}^3}"
            r"\mathcal{N}\big(\boldsymbol{x};\boldsymbol{\mu}_n(t),\boldsymbol{\Sigma}_n(t)\big)\,"
            r"\mathcal{N}\big(\boldsymbol{x};\bar{\boldsymbol{\mu}}_m(\boldsymbol{q}(t)),"
            r"\bar{\boldsymbol{\Sigma}}_m(\boldsymbol{q}(t))\big)\,d\boldsymbol{x}",
            font_size=34,
        )
        eq2 = MathTex(
            r"=\sum_{n=1}^{N}\sum_{m=1}^{M}"
            r"\mathcal{N}\big(\bar{\boldsymbol{\mu}}_m(\boldsymbol{q}(t));"
            r"\boldsymbol{\mu}_n(t),\boldsymbol{\Sigma}_n(t)+"
            r"\bar{\boldsymbol{\Sigma}}_m(\boldsymbol{q}(t))\big)",
            font_size=34,
        )
        eqs = VGroup(eq0, eq1, eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        fit_to_box(eqs, config.frame_width - 1.2, 4.0)
        eqs.next_to(defs, DOWN, buff=0.6)

        caption1 = Text(
            "Both fields are sums of Gaussians, so it splits into pairs (n, m)",
            font_size=24,
        ).to_edge(DOWN)
        with self.voiceover(
            text="Since both fields are sums of Gaussians, the integral splits into pairs,"
        ):
            self.play(Write(eq0), Write(eq1), FadeIn(caption1))

        caption2 = Text(
            "Closed form: each pair is a single Gaussian evaluation, no numerical integration",
            font_size=24,
        ).to_edge(DOWN)
        box = SurroundingRectangle(eq2, color=YELLOW, buff=0.12)
        with self.voiceover(
            text="and each pair has a closed form. So the cost is exact, smooth and cheap."
        ):
            self.play(Write(eq2), ReplacementTransform(caption1, caption2))
            self.play(Create(box), run_time=0.6)

        self.play(FadeOut(VGroup(title, defs, eqs, box, caption2)))


# ---------------------------------------------------------------------------
# 5. The trajectory: cubic B-spline (left) and its MINVO basis (right)
# ---------------------------------------------------------------------------


class TrajectorySpline(VoiceoverScene):
    def construct(self):
        self.set_speech_service(make_speech_service())

        title = Text(
            "The trajectory is a cubic B-spline defined by control points",
            font_size=26,
        ).to_edge(UP)
        divider = DashedLine(UP * 3.0, DOWN * 3.7, color=GREY_D, stroke_width=1.5)
        left_heading = Text("B-spline", font_size=24, weight=BOLD).move_to(
            LEFT * 3.6 + UP * 2.85
        )

        # ---- Left: B-spline with its control polygon ------------------------
        ctrl0 = [
            [-6.3, -0.7, 0],
            [-6.5, 1.1, 0],
            [-5.4, 2.3, 0],
            [-3.9, 2.2, 0],
            [-4.6, 0.2, 0],
            [-2.8, -1.0, 0],
            [-1.0, -0.5, 0],
            [-0.7, 1.6, 0],
            [-2.0, 2.2, 0],
        ]
        handles = VGroup(*[control_handle(p) for p in ctrl0])
        handles.set_z_index(3)

        def ctrl():
            return [h.get_center() for h in handles]

        polygon = always_redraw(
            lambda: VMobject(
                stroke_color=GREY_B, stroke_width=1.5
            ).set_points_as_corners(ctrl())
        )
        curve = always_redraw(
            lambda: VMobject(stroke_color=WHITE, stroke_width=5).set_points_as_corners(
                bspline_curve_points(ctrl())
            )
        )

        def knot_points():
            c = ctrl()
            pts = [bspline_segment_point(c[i : i + 4], 0) for i in range(len(c) - 3)]
            pts.append(bspline_segment_point(c[-4:], 1))
            return pts

        knots = always_redraw(
            lambda: VGroup(
                *[control_handle(p, radius=0.055) for p in knot_points()]
            ).set_z_index(2)
        )
        left_basis = basis_panel(BSPLINE_M, "Basis Functions", LEFT * 3.6 + DOWN * 2.75)

        with self.voiceover(
            text="The trajectory is a cubic B-spline: a chain of polynomial segments "
            "shaped by control points."
        ):
            self.play(Write(title), Write(left_heading), Create(divider), run_time=0.8)
            self.play(
                LaggedStartMap(FadeIn, handles, lag_ratio=0.1),
                Create(polygon),
                run_time=1,
            )
            self.play(Create(curve), FadeIn(left_basis), run_time=1.5)
            self.play(FadeIn(knots), run_time=0.5)

        # Local support: moving one control point only changes nearby segments
        with self.voiceover(
            text="Moving a control point only reshapes nearby segments."
        ):
            self.play(handles[4].animate.shift(RIGHT * 0.7 + DOWN * 0.3), run_time=1)
            self.play(handles[7].animate.shift(LEFT * 0.4 + UP * 0.4), run_time=0.8)

        # ---- Pick one segment: it depends only on 4 control points ----------
        k = 1
        P4 = [np.array(p) for p in ctrl()[k : k + 4]]
        seg_highlight = VMobject(stroke_color=YELLOW, stroke_width=8)
        seg_highlight.set_points_as_corners(bspline_segment_samples(P4, 60))
        seg_highlight.set_z_index(1)

        # ---- Right: the same segment in the MINVO basis ---------------------
        right_heading = Text("MINVO Basis", font_size=24, weight=BOLD).move_to(
            RIGHT * 3.6 + UP * 2.85
        )

        V4 = bspline_to_minvo(P4)
        seg = bspline_segment_samples(P4, 80)

        # Magnify the segment into the right panel
        everything = np.array(list(P4) + list(V4) + seg)
        lo, hi = everything.min(axis=0), everything.max(axis=0)
        box_w, box_h = 5.0, 3.0
        scale = min(box_w / max(hi[0] - lo[0], 1e-6), box_h / max(hi[1] - lo[1], 1e-6))
        target = RIGHT * 3.6 + UP * 0.45

        def to_panel(p):
            return (np.asarray(p) - (lo + hi) / 2) * scale + target

        big_seg = VMobject(stroke_color=WHITE, stroke_width=6)
        big_seg.set_points_as_corners([to_panel(p) for p in seg])
        big_seg.set_z_index(1)

        bs_pts = [to_panel(p) for p in P4]
        bs_dots = VGroup(
            *[
                control_handle(p, color=col, radius=0.11)
                for p, col in zip(bs_pts, POINT_COLORS)
            ]
        ).set_z_index(3)
        bs_hull_v = convex_hull_2d(bs_pts)
        bs_hull = Polygon(
            *bs_hull_v,
            stroke_color=GREY_B,
            stroke_width=2,
            fill_color=GREY,
            fill_opacity=0.15,
        )
        bs_label = Text("B-spline hull", font_size=18, color=GREY_B)
        bs_label.next_to(bs_hull, RIGHT, buff=0.15)

        with self.voiceover(
            text="Each segment depends on four control points, "
            "and lies inside their convex hull, which is quite loose."
        ):
            self.play(
                Create(seg_highlight),
                *[handles[k + i].animate.set_stroke(POINT_COLORS[i]) for i in range(4)],
                Write(right_heading),
                run_time=0.8,
            )
            self.play(TransformFromCopy(seg_highlight, big_seg), run_time=1)
            self.play(
                TransformFromCopy(VGroup(*[handles[k + i] for i in range(4)]), bs_dots),
                FadeIn(bs_hull),
                FadeIn(bs_label),
                run_time=1,
            )

        mv_pts = [to_panel(p) for p in V4]
        mv_dots = VGroup(
            *[Dot(p, radius=0.09, color=col) for p, col in zip(mv_pts, POINT_COLORS)]
        ).set_z_index(3)
        mv_hull_v = convex_hull_2d(mv_pts)
        mv_hull = Polygon(
            *mv_hull_v,
            stroke_color=TEAL_B,
            stroke_width=3,
            fill_color=TEAL,
            fill_opacity=0.3,
        )
        mv_hull.set_z_index(0)
        ratio_text = Text(
            "MINVO hull area",
            font_size=18,
            color=TEAL_B,
        ).move_to(RIGHT * 3.6 + DOWN * 1.55)
        right_basis = basis_panel(
            MINVO_M, "MINVO Basis Functions", RIGHT * 3.6 + DOWN * 2.75
        )
        with self.voiceover(
            text="The MINVO basis describes the same curve with a smaller n-simplex."
        ):
            self.play(TransformFromCopy(bs_dots, mv_dots), run_time=1)
            self.play(
                Create(mv_hull), FadeIn(ratio_text), FadeIn(right_basis), run_time=1
            )

        note = Text(
            "Same curve, tighter enclosure: bounding the MINVO control points bounds the whole segment",
            font_size=20,
        ).to_edge(DOWN, buff=0.15)
        with self.voiceover(
            text="So bounding its control points tightly bounds the whole segment."
        ):
            self.play(FadeOut(left_basis), FadeOut(right_basis), FadeIn(note))

        self.play(*[FadeOut(m) for m in self.mobjects])


# ---------------------------------------------------------------------------
# 6. The algorithm: the full optimization problem (objective + constraints)
# ---------------------------------------------------------------------------


class CostFunction(VoiceoverScene):
    def construct(self):
        self.set_speech_service(make_speech_service())

        title = Text(
            "FOCI's full trajectory optimization problem", font_size=28
        ).to_edge(UP)

        # ---- Objective -----------------------------------------------------
        objective_heading = Text("Objective", font_size=26, color=YELLOW).next_to(
            title, DOWN, buff=0.4
        )

        min_label = MathTex(r"\min_{\boldsymbol{q}}")
        term_collision = MathTex(
            r"\omega_1 \cdot \frac{1}{KNM} \sum_{k=1}^{K}\sum_{n=1}^{N}\sum_{m=1}^{M}"
            r"\mathcal{N}\big(\bar{\boldsymbol{\mu}}_m(\boldsymbol{q}(t_k));\boldsymbol{\mu}_n(t_k),"
            r"\boldsymbol{\Sigma}_{mn}(t_k)\big)"
        )
        term_jerk_real = MathTex(
            r"+\ \omega_2 \cdot \frac{T}{(K-1)\,|\mathcal{J}\setminus\bar{\mathcal{J}}|}"
            r"\sum_{k=1}^{K}\sum_{j\notin\bar{\mathcal{J}}} \left\lVert \dddot{\boldsymbol{q}}_j(t_k) \right\rVert^2"
        )
        term_jerk_virtual = MathTex(
            r"+\ \omega_3 \cdot \frac{T}{(K-1)\,|\bar{\mathcal{J}}|}"
            r"\sum_{k=1}^{K}\sum_{j\in\bar{\mathcal{J}}} \left\lVert \dddot{\boldsymbol{q}}_j(t_k) \right\rVert^2"
        )
        term_goal = MathTex(
            r"+\ \omega_4 \left\lVert \boldsymbol{f}_{\text{task}}(\boldsymbol{q}(T)) - \boldsymbol{p}_{\text{goal}} \right\rVert^2"
        )

        objective_terms = [term_collision, term_jerk_real, term_jerk_virtual, term_goal]
        objective_group = VGroup(min_label, *objective_terms).arrange(
            DOWN, aligned_edge=LEFT, buff=0.3
        )
        fit_to_box(objective_group, max_width=config.frame_width - 1.5, max_height=4.3)
        objective_group.next_to(objective_heading, DOWN, buff=0.4)

        with self.voiceover(text="Putting it all together, FOCI minimizes four terms:"):
            self.play(Write(title), Write(objective_heading), run_time=0.8)
            self.play(Write(objective_group), run_time=1.5)

        highlight_sequence(
            self,
            [
                (
                    term_collision,
                    "Penalizes Gaussian overlap between the robot and the environment (collision)",
                    "the collision overlap, averaged over time,",
                ),
                (
                    term_jerk_real,
                    "Penalizes abrupt motion (jerk) of the real joints",
                    "the jerk of the real joints,",
                ),
                (
                    term_jerk_virtual,
                    "Penalizes abrupt motion (jerk) of the virtual joints (e.g. the robot's base)",
                    "the jerk of the virtual joints, like a mobile base,",
                ),
                (
                    term_goal,
                    "Drives the task-space point (e.g. the end-effector) toward the goal",
                    "and the distance to the goal.",
                ),
            ],
        )

        self.play(FadeOut(objective_group), FadeOut(objective_heading))

        # ---- Constraints -----------------------------------------------------
        constraints_heading = Text("Constraints", font_size=26, color=YELLOW).next_to(
            title, DOWN, buff=0.4
        )

        c_start = MathTex(r"\boldsymbol{q}(0) = \boldsymbol{q}_{\text{start}}")
        c_vel_virtual = MathTex(
            r"\left\lVert [\dot{\boldsymbol{q}}_j(t)]_{j\in\bar{\mathcal{J}}} \right\rVert^2 \le v_{\text{max}}^2,"
            r"\ \forall t \in [0,T]"
        )
        c_acc_virtual = MathTex(
            r"\left\lVert [\ddot{\boldsymbol{q}}_j(t)]_{j\in\bar{\mathcal{J}}} \right\rVert^2 \le a_{\text{max}}^2,"
            r"\ \forall t \in [0,T]"
        )
        c_vel_real = MathTex(
            r"\left| \dot{\boldsymbol{q}}_j(t) \right| \le \omega_{j,\text{max}},"
            r"\ \forall j \notin \bar{\mathcal{J}},\ \forall t \in [0,T]"
        )
        c_acc_real = MathTex(
            r"\left| \ddot{\boldsymbol{q}}_j(t) \right| \le \alpha_{j,\text{max}},"
            r"\ \forall j \notin \bar{\mathcal{J}},\ \forall t \in [0,T]"
        )

        constraints = [c_start, c_vel_virtual, c_acc_virtual, c_vel_real, c_acc_real]
        constraints_group = VGroup(*constraints).arrange(
            DOWN, aligned_edge=LEFT, buff=0.35
        )
        fit_to_box(
            constraints_group, max_width=config.frame_width - 1.5, max_height=4.3
        )
        constraints_group.next_to(constraints_heading, DOWN, buff=0.4)

        with self.voiceover(text="Subject to a few constraints:"):
            self.play(Write(constraints_heading), run_time=0.5)
            self.play(Write(constraints_group), run_time=1.5)

        # Group the constraints by meaning: start / velocities / accelerations
        highlight_sequence(
            self,
            [
                (
                    c_start,
                    None,
                    "a fixed start,",
                ),
                (
                    VGroup(c_vel_virtual, c_vel_real),
                    None,
                    "bounded velocities,",
                ),
                (
                    VGroup(c_acc_virtual, c_acc_real),
                    None,
                    "and bounded accelerations, for both virtual and real joints.",
                ),
            ],
        )

        self.play(
            FadeOut(constraints_group), FadeOut(constraints_heading), FadeOut(title)
        )


# ---------------------------------------------------------------------------
# 7. Experiments: example videos after the explanation
# ---------------------------------------------------------------------------


class Experiments(VoiceoverScene):
    def construct(self):
        self.set_speech_service(make_speech_service())

        experiments = [(path, robot) for path, robot in EXPERIMENTS if path]
        if not experiments:
            logger.warning("EXPERIMENTS has no video paths: skipping this scene.")
            self.wait(0.1)
            return

        heading = Text("Experiments", font_size=40)
        with self.voiceover(text="Let's see FOCI in action."):
            self.play(Write(heading))

        def robot_label(text):
            # Same place and size the "Experiments" heading used to have
            return Text(text, font_size=24).to_corner(UL, buff=0.3)

        label, current_robot = None, None  # robot type shown in the corner
        for i, (path, robot) in enumerate(experiments):
            clip = VideoClip(path, max_duration=EXPERIMENT_MAX_SECONDS)
            # As big as possible below the label, keeping aspect ratio
            clip.set(width=config.frame_width - 1.0)
            if clip.height > config.frame_height - 1.4:
                clip.set(height=config.frame_height - 1.4)
            clip.move_to(DOWN * 0.3)

            anims = [media_fade_in(self, clip, run_time=0.4)]
            if i == 0:
                # The big "Experiments" heading is replaced by the first robot type
                label = robot_label(robot)
                anims += [FadeOut(heading), FadeIn(label)]
            elif robot != current_robot:
                new_label = robot_label(robot)
                anims.append(FadeTransform(label, new_label))
                label = new_label
            current_robot = robot
            self.play(*anims, run_time=0.4)

            self.wait(clip.remaining)  # rest of the video
            self.play(media_fade_out(self, clip, run_time=0.4))
            media_remove(self, clip)

        self.play(FadeOut(label))


# ===========================================================================
# Render all the scenes and join them into a single video
# ===========================================================================

# Scenes in the order they appear in the final video
SCENES = [
    "Hook",
    "DualRepresentation",
    "RobotModeling",
    "CollisionOverlap1D",
    "TrajectorySpline",
    "CostFunction",
    "Experiments",
]

# Manim quality flag -> name of the folder where Manim saves the scenes
QUALITIES = {"l": "480p15", "m": "720p30", "h": "1080p60", "k": "2160p60"}


def render_scenes(quality):
    """Render every scene with Manim (same as `manim -q<quality> file Scene...`)."""
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "manim", f"-q{quality}", __file__, *SCENES]
    subprocess.run(cmd, check=True)


def join_scenes(quality):
    """
    Concatenate the rendered scenes (video + narration) into one video without
    re-encoding: packets are copied and their timestamps shifted. The shift is
    kept in seconds because video and audio use different time bases.
    """
    import av

    base = Path(__file__).resolve().parent
    folder = base / "media" / "videos" / Path(__file__).stem / QUALITIES[quality]
    output = base / f"FOCI_{QUALITIES[quality]}.mp4"

    videos = []
    for name in SCENES:
        path = folder / f"{name}.mp4"
        if path.exists():
            videos.append(path)
        else:
            print(f"WARNING: {path.name} is missing, skipping it")
    if not videos:
        raise SystemExit(f"No rendered scenes in {folder}")

    def add_stream(out, kind):
        """Create an output stream copying the first one of that kind found."""
        for v in videos:
            with av.open(str(v)) as c:
                streams = getattr(c.streams, kind)
                if streams:
                    return out.add_stream_from_template(streams[0])
        return None

    with av.open(str(output), "w") as out:
        out_streams = {}
        for kind in ("video", "audio"):
            stream = add_stream(out, kind)
            if stream is not None:
                out_streams[kind] = stream

        offset = 0.0  # seconds
        for path in videos:
            with av.open(str(path)) as inp:
                end = offset
                inputs = list(inp.streams.video[:1]) + list(inp.streams.audio[:1])
                for packet in inp.demux(*inputs):
                    if packet.dts is None:
                        continue
                    tb = packet.stream.time_base
                    kind = packet.stream.type
                    shift = int(round(offset / tb))
                    pts, dur = packet.pts, packet.duration or 0
                    packet.pts = pts + shift
                    packet.dts = packet.dts + shift
                    packet.stream = out_streams[kind]
                    out.mux(packet)
                    if kind == "video":
                        end = max(end, float((pts + dur) * tb) + offset)
                # The next scene starts when this one's image ends
                offset = end

    print(f"\nDone: {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render all FOCI scenes and join them into one video."
    )
    parser.add_argument(
        "quality",
        nargs="?",
        default="l",
        choices=QUALITIES,
        help="l=480p15, m=720p30, h=1080p60, k=2160p60 (default: l)",
    )
    parser.add_argument(
        "--join-only",
        action="store_true",
        help="do not render, only join the scenes already rendered",
    )
    args = parser.parse_args()

    if not args.join_only:
        render_scenes(args.quality)
    join_scenes(args.quality)
