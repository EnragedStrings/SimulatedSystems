import pygame
import math
import sys
from functools import partial
import os
from datetime import datetime, timezone

import threading
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat

import math
from typing import List, Tuple
import json

from direct.task import Task
import numpy as np

def load_json_to_array(filepath):
    """
    Load a JSON file that contains a top-level array.
    Returns the array as a Python list, or None if file is empty, missing, or invalid.
    """
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            return None
        return data
    except (json.JSONDecodeError, IOError):
        return None

def save_array_to_json(array, filepath):
    """
    Save a Python list to a JSON file with pretty formatting.
    Does nothing if the array is None.
    """
    if array is None:
        print(f"[Warning] Not saving to {filepath}: array is None.")
        return
    if not isinstance(array, list):
        raise ValueError("Input data must be a list.")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(array, f, indent=4, ensure_ascii=False)

tmp = load_json_to_array('import.json')[0]
if tmp is not None:
    importArray = tmp

devmode = importArray['MISC']['DEVMODE']

config_path = os.path.join(os.path.dirname(__file__), 'import.json')

models = [
    "luke2.bam",
    "elmendorf.bam",
    "elmendorf2.bam",
    "luke3.bam"
]
currentModel = models[3]

from panda3d.core import getModelPath
from panda3d.core import (
    loadPrcFileData,
    FrameBufferProperties,
    GraphicsOutput,
    GraphicsPipe,
    WindowProperties,
    Texture,
    AmbientLight,
    DirectionalLight,
    Vec4,
    Vec3,
    Fog
)
#getModelPath().appendDirectory(f"models/{currentModel}")
from direct.showbase.ShowBase import ShowBase
import random

def make_thermal_lut():
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        norm = i / 255.0
        # piecewise as before, but on a single value
        if norm < 0.25:
            b = int(255 * (norm / 0.25))
            lut[i] = (0, 0, b)
        elif norm < 0.5:
            m = (norm - 0.25) / 0.25
            r = int(255 * m)
            b = int(255 * (1 - m))
            lut[i] = (r, 0, b)
        elif norm < 0.75:
            m = (norm - 0.5) / 0.25
            g = int(255 * m)
            lut[i] = (255, g, 0)
        else:
            m = (norm - 0.75) / 0.25
            v = int(255 * m)
            lut[i] = (255, 255, v)
    return lut

thermal_lut = make_thermal_lut()

def make_vignette_mask(w, h, strength=0.6):
    yy, xx = np.indices((h, w))
    cx, cy = w / 2, h / 2
    d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    maxd = np.sqrt(cx*cx + cy*cy)
    mask = 1.0 - strength * (d / maxd)
    return np.clip(mask, 0, 1).astype(np.float32)

# Configure Panda3D for offscreen rendering
loadPrcFileData('', 'window-type offscreen\n'
                     'framebuffer-srgb true\n'
                     'sync-video false\n'
                     'show-frame-rate-meter false\n')

class MultiCamOffscreenRenderer(ShowBase):
    def __init__(self, width, height, import_array):
        super().__init__(windowType='offscreen')
        self.width = width
        self.height = height
        self.cfg = import_array

        # Load aircraft
        self.aircraft_np = self.loader.loadModel(self.cfg["Aircraft"]["modelPath"])
        self.aircraft_np.reparentTo(self.render)
        self.aircraft_np.setScale(self.cfg["Aircraft"]["scale"])
        self.aircraft_np.setPos(*self.cfg["Aircraft"]["Position"])
        self.aircraft_np.setHpr(*self.cfg["Aircraft"].get("Rotation", (0,0,0)))
        
        if devmode == False:
            # 1) Create/load your global node
            terrain = self.loader.loadModel(f"models/{currentModel}")
            terrain.reparentTo(self.render)
            terrain.setScale(100)
            terrain.setPos(0, 0, -1)

        # Keep the camera definitions around
        self.cam_defs = self.cfg["Cameras"]

        # Shared lights & fog
        self._setup_global_lighting_and_fog()

        # Create all cameras & skyboxes
        self.cameras = {
            cam_def["id"]: self._make_offscreen_camera(cam_def)
            for cam_def in self.cam_defs
        }
        
    def _setup_global_lighting_and_fog(self):
        # Ambient + directional
        amb = AmbientLight('ambient')
        amb.setColor(Vec4(*self.cfg["Rendering"]["ambientColor"], 1))
        self.render.setLight(self.render.attachNewNode(amb))

        sun = DirectionalLight('sun')
        sun_dir = Vec3(*self.cfg["Rendering"]["sunDirection"])
        sun.setDirection(sun_dir)
        sun.setColor(Vec4(*self.cfg["Rendering"]["sunColor"], 1))
        self.render.setLight(self.render.attachNewNode(sun))

        # Global fog
        f = Fog("globalFog")
        f.setColor(Vec4(*self.cfg["Rendering"]["fogColor"], 1))
        near = self.cfg["Rendering"]["renderDistance"] - self.cfg["Rendering"]["fogDistance"]
        far  = self.cfg["Rendering"]["renderDistance"]
        f.setLinearRange(near, far)
        self.render.setFog(f)

    def _make_offscreen_camera(self, cam_def):
        """Create buffer, camera, skybox for one cam definition."""
        # 1) Framebuffer & texture
        fb = FrameBufferProperties()
        fb.setRgbColor(True)
        fb.setAlphaBits(1)
        fb.setDepthBits(1)

        wp = WindowProperties.size(self.width, self.height)
        buffer = self.graphicsEngine.makeOutput(
            self.pipe, f"offbuf_{cam_def['id']}", -2,
            fb, wp,
            GraphicsPipe.BF_refuse_window,
            self.win.getGsg(), self.win
        )
        buffer.setClearColorActive(True)
        buffer.setClearColor(Vec4(*self.cfg["Rendering"]["clearColor"]))
        buffer.setClearDepthActive(True)

        tex = Texture()
        buffer.addRenderTexture(tex, GraphicsOutput.RTM_copy_ram, GraphicsOutput.RTP_color)

        # 2) Camera node parented to the aircraft, at offset
        cam_np = self.makeCamera(buffer)
        cam_np.reparentTo(self.aircraft_np)
        cam_np.setPos(*cam_def["offset"])
        cam_np.setHpr(*cam_def.get("hpr", (0,0,0)))

        lens = cam_np.node().getLens()

        # Per-camera FOV, fallback to a default if missing
        cam_fov = cam_def.get("fov",
                    self.cfg["Rendering"].get("defaultFov", 50))
        lens.setFov(cam_fov)

        lens.setNearFar(
            0.1,
            self.cfg["Rendering"]["renderDistance"]
        )
        lens.setNearFar(0.1, self.cfg["Rendering"]["renderDistance"])

        # 3) Skybox for this camera
        sky = self.loader.loadModel(cam_def["skyboxModel"])
        sky.reparentTo(cam_np)        # parent to camera so it follows orientation
        sky.setScale(self.cfg["Rendering"]["skydomeScale"])
        sky.setBin('background', 0)
        sky.setDepthWrite(False)
        sky.setLightOff()
        sky.setFogOff()
        sky.setCompass()
        sky.setPos(0,0,0)

        return {
            "buffer": buffer,
            "tex": tex,
            "cam_np": cam_np,
            "sky": sky
        }

    def render_all(self, updated_import_array=None):
        """
        Optionally replace self.cfg (and self.cam_defs), then:
        - Update aircraft pos/hpr
        - Update each camera's pos/hpr/FOV
        - Render all cameras
        Returns dict of Pygame surfaces by cam id.
        """
        if updated_import_array:
            self.cfg = updated_import_array
            self.cam_defs = self.cfg["Cameras"]

        # 1) Update the aircraft's dynamic transform, if provided
        self.aircraft_np.setPos(*self.cfg["Aircraft"]["Position"])
        self.aircraft_np.setHpr(*self.cfg["Aircraft"].get("Rotation", (0,0,0)))

        surfaces = {}
        for cam_def in self.cam_defs:
            cid = cam_def["id"]
            cam_data = self.cameras[cid]
            cam_np   = cam_data["cam_np"]

            # 2) Update each camera's relative transform
            cam_np.setPos(*cam_def.get("offset", cam_np.getPos()))
            cam_np.setHpr(*cam_def.get("hpr",    cam_np.getHpr()))

            # 3) Update per-camera FOV
            lens = cam_np.node().getLens()
            lens.setFov(cam_def.get("fov", lens.getFov()))

            # 4) Render and read back
            self.graphicsEngine.renderFrame()
            tex = cam_data["tex"]
            img = tex.getRamImageAs('RGB')
            w, h = tex.getXSize(), tex.getYSize()
            surf = pygame.image.frombuffer(img.getData(), (w, h), 'RGB')
            surfaces[cid] = pygame.transform.flip(surf, False, True)

        return surfaces

# Constants for logical resolution (fixed 20:8 aspect ratio)
LOGICAL_WIDTH = 1920
ASPECT_RATIO = 20 / 8
LOGICAL_HEIGHT = int(LOGICAL_WIDTH / ASPECT_RATIO)  # 768

# Start with 1920x768 window
WINDOW_WIDTH, WINDOW_HEIGHT = LOGICAL_WIDTH, LOGICAL_HEIGHT

GRID_X = WINDOW_WIDTH/16
GRID_Y = WINDOW_HEIGHT/9

# Initialize pygame
pygame.init()
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
window.fill((0, 0, 0))  # Black background behind scaled screen

#window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.NOFRAME | pygame.RESIZABLE)

dev = " |  DEVMODE" and devmode or ""
pygame.display.set_caption(f"F-35 PCD | TS//SCI  |  HCS//NOFORN//ORCON {dev}")

def dprint(str):
    if devmode:
        print(str)
        
dprint("STARTING IN DEV MODE")

# Logical drawing surface (fixed resolution)
draw_surface = pygame.Surface((LOGICAL_WIDTH, LOGICAL_HEIGHT))

# Font setup
font = pygame.font.SysFont("Arial", 24)
PCDFont = pygame.font.Font("PCD.ttf", 24)
PCDThin = pygame.font.Font("PCD.ttf", 24)
PCDThin2 = pygame.font.Font("PCD.ttf", 33)
PCDLarge = pygame.font.Font("PCD.ttf", 35)
PCDMassive = pygame.font.Font("PCD.ttf", 55)
PCDFont.set_bold(True)
PCDLarge.set_bold(True)
PCDMassive.set_bold(True)
is_fullscreen = False

def toggle_fullscreen():
    global window, is_fullscreen
    global WINDOW_WIDTH, WINDOW_HEIGHT, GRID_X, GRID_Y

    if not is_fullscreen:
        # go fullscreen at native desktop resolution
        info = pygame.display.Info()
        WINDOW_WIDTH, WINDOW_HEIGHT = info.current_w, info.current_h
        flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
        window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), flags)
        is_fullscreen = True
    else:
        # back to your original logical size
        WINDOW_WIDTH, WINDOW_HEIGHT = LOGICAL_WIDTH, LOGICAL_HEIGHT
        # you can add pygame.RESIZABLE or other flags here if you like
        window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        is_fullscreen = False
    window.fill((0, 0, 0))
    # recalc any derived values
    GRID_X = WINDOW_WIDTH / 16
    GRID_Y = WINDOW_HEIGHT / 9

def _obs_cam_loop(draw_surf: pygame.Surface, fps: float):
    """
    Internal loop: grabs each frame from draw_surf and
    sends it to the first available virtual camera (OBS Virtual Camera).
    """
    w, h = draw_surf.get_size()
    # RGB is fine for OBS on Windows
    with pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=PixelFormat.RGB, device=None) as cam:
        print(f"🔴 Streaming to OBS Virtual Camera on device {cam.device}")
        clock = pygame.time.Clock()
        while True:
            # You can break this loop by calling thread.join(timeout)
            frame = pygame.surfarray.array3d(draw_surf)
            frame = np.swapaxes(frame, 0, 1)  # -> (H, W, 3)
            cam.send(frame)
            cam.sleep_until_next_frame()
            clock.tick(fps)
    # (Won’t actually reach here unless cam.close() is called)
    pygame.quit()
    sys.exit()

def start_obs_cam_thread(draw_surf: pygame.Surface, fps: float = 30.0) -> threading.Thread:
    """
    Launch the OBS virtual‑camera stream in its own daemon thread.

    Args:
      draw_surf: your off‑screen pygame.Surface that you render into each frame.
      fps:       the target framerate for the virtual camera.

    Returns:
      The Thread object (daemon=True), already started.
    """
    thread = threading.Thread(
        target=_obs_cam_loop,
        args=(draw_surf, fps),
        daemon=True,
        name="OBSVirtualCamThread"
    )
    thread.start()
    return thread

def get_zulu_time():
    """
    Returns the current Zulu (UTC) time formatted as 'HH:MM:SS'.
    """
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def render_multicolor_text(surface, font, text_lines, start_pos, line_spacing=5, centered=False, align = "C", bg = None):
    """
    Renders multi-line, multi-color text onto a surface.
    
    :param surface: The surface to draw on
    :param font: A pygame.font.Font object
    :param text_lines: A list of lines, where each line is a list of (text, color) tuples
    :param start_pos: Tuple (x, y) where to begin rendering
    :param line_spacing: Pixels of vertical spacing between lines
    """
    x_start, y = start_pos
    index = 0
    max_width = 0
    totalHeight = 0
    x = x_start
    for line in text_lines:
        for text, color in line:
            tw, th = font.size(text)
            totalHeight += th
            if tw > max_width:
                max_width = tw
                
    max_width += 2
    totalHeight += 2
    bgx = 0
    bgy = 0
    if align == "C":
        max_width += 4
        bgx = x - (max_width/2)
        bgy = y - (totalHeight/2)
    elif align == "L":
        max_width += 10
        bgx = x
        bgy = y - (totalHeight/2)
    elif align == "R":
        max_width += 10
        bgx = x - max_width
        bgy = y - (totalHeight/2)
    elif align == "T":
        max_width += 4
        totalHeight += 2
        bgx = x - (max_width/2)
        bgy = y
    elif align == "B":
        max_width += 4
        totalHeight += 2
        bgx = x - (max_width/2)
        bgy = y - totalHeight
    elif align == "TL":
        max_width += 10
        totalHeight += 2
        bgx = x
        bgy = y
    elif align == "TR":
        max_width += 10
        totalHeight += 2
        bgx = x - max_width
        bgy = y
    elif align == "BL":
        max_width += 10
        totalHeight += 2
        bgx = x
        bgy = y - totalHeight
    elif align == "BR":
        max_width += 10
        totalHeight += 2
        bgx = x - max_width
        bgy = y - totalHeight
    if bg is not None:
        pygame.draw.rect(surface, bg, pygame.Rect(bgx, bgy, max_width, totalHeight), 0)
        
    for line in text_lines:
        x = x_start
        max_height = 0
        for text, color in line:
            text_surf = font.render(text, True, color)
            if centered:
                cx = 0
                cy = 0
                bx = 8
                by = 5
                if align == "C":
                    cx = x - (text_surf.get_width()/2)
                    cy = y - ((len(text_lines)*text_surf.get_height())/2)
                elif align == "L":
                    cx = x + bx
                    cy = y - ((len(text_lines)*text_surf.get_height())/2)
                elif align == "R":
                    cx = x - text_surf.get_width() - bx
                    cy = y - ((len(text_lines)*text_surf.get_height())/2)
                elif align == "T":
                    cx = x - (text_surf.get_width()/2)
                    cy = y + by
                elif align == "B":
                    cx = x - (text_surf.get_width()/2)
                    cy = y - (len(text_lines)*text_surf.get_height())
                elif align == "TL":
                    cx = x + bx
                    cy = y + by
                elif align == "TR":
                    cx = x - text_surf.get_width() - bx
                    cy = y + by
                elif align == "BL":
                    cx = x + bx
                    cy = y - (len(text_lines)*text_surf.get_height())
                elif align == "BR":
                    cx = x - text_surf.get_width() - bx
                    cy = y - (len(text_lines)*text_surf.get_height())
                surface.blit(text_surf, (cx, cy))
            else:
                surface.blit(text_surf, (x, y))
            x += text_surf.get_width()
            max_height = max(max_height, text_surf.get_height())
                
        y += max_height + line_spacing
        index += 1
    return (bgx, bgy, max_width, totalHeight)

def load_bold_font(path, size):
    f = pygame.font.Font(path, size)
    f.set_bold(True)
    return f

def apply_tflir_fast(src_surf, thermal_lut, vignette_mask, noise_strength=0.03):
    # 1) Grab three-channel uint8 array [H,W,3]
    arr = pygame.surfarray.array3d(src_surf)
    # 2) Compute luminance 0–255
    #    weight per channel, then convert to uint8 index
    lum = ((0.3 * arr[...,0] + 0.59 * arr[...,1] + 0.11 * arr[...,2])
           .clip(0,255).astype(np.uint8))
    # 3) Map through LUT → [H,W,3] uint8
    therm = thermal_lut[lum]
    # 4) Add tiny noise (float32), then clamp
    noise = (np.random.randn(*therm.shape) * noise_strength * 255).astype(np.int16)
    therm = np.clip(therm.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # 5) Apply vignette (on float32), then to uint8
    therm = (therm.astype(np.float32) * vignette_mask[...,None]).clip(0,255).astype(np.uint8)
    # 6) Blit back to Pygame surface
    out = pygame.Surface(src_surf.get_size())
    pygame.surfarray.blit_array(out, therm)
    return out

def apply_bw_effect(src_surf: pygame.Surface, whiteHot=True) -> pygame.Surface:
    """
    Convert a Pygame surface to grayscale.
    If whiteHot is False, invert the grayscale to make bright areas dark and vice versa.
    """
    # 1) Extract a (H, W, 3) uint8 array
    arr = pygame.surfarray.array3d(src_surf)

    # 2) Compute luminance using standard weights
    lum = (0.3 * arr[..., 0] +
           0.59 * arr[..., 1] +
           0.11 * arr[..., 2]).clip(0, 255).astype(np.uint8)

    # 3) Invert if whiteHot is False
    if not whiteHot:
        lum = 255 - lum

    # 4) Stack back into 3 channels
    gray3 = np.stack([lum, lum, lum], axis=-1)

    # 5) Build output Surface
    out = pygame.Surface(src_surf.get_size())
    pygame.surfarray.blit_array(out, gray3)
    return out

# Colors
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
GRAY  = (120, 120, 120)
DARK_GRAY = (50, 50, 50)
CYAN = (0, 255, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
PURPLE = (224, 127, 247)
BLUE = (0, 0, 255)

MASTER_MODE = "NAV"

# Mouse input scaled to logical coordinates
def get_logical_mouse_pos(window_size, mouse_pos):
    w, h = window_size
    x, y = mouse_pos
    current_ratio = w / h

    if current_ratio > ASPECT_RATIO:
        # Too wide (pillarbox)
        scale_height = h
        scale_width = int(scale_height * ASPECT_RATIO)
    else:
        # Too tall (letterbox)
        scale_width = w
        scale_height = int(scale_width / ASPECT_RATIO)

    x_offset = (w - scale_width) // 2
    y_offset = (h - scale_height) // 2

    if not (x_offset <= x <= x_offset + scale_width) or not (y_offset <= y <= y_offset + scale_height):
        return None  # Mouse is outside render area

    logical_x = int((x - x_offset) * LOGICAL_WIDTH / scale_width)
    logical_y = int((y - y_offset) * LOGICAL_HEIGHT / scale_height)

    return logical_x, logical_y

# Draw helpers
def draw_pixel(surface, x, y, color):
    if 0 <= x < LOGICAL_WIDTH and 0 <= y < LOGICAL_HEIGHT:
        surface.set_at((x, y), color)

def grab_PMD_data(location, replacement = None):
    if str(location) == "" or str(location) == None or location == 0 and replacement != None:
        return replacement
    else:
        return location

# Simple button class
class Button:
    def __init__(self, x, y, w, h, text, color, selectedColor, on_click):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.on_click = on_click
        self.hovered = False
        self.pressed = False
        self.color = color
        self.selectedColor = selectedColor

    def draw(self, surface):
        color = self.selectedColor if self.pressed else self.color
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, WHITE, self.rect, 2)
        text_surf = font.render(self.text, True, WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def update(self, mouse_pos, mouse_down, mouse_up):
        if self.rect.collidepoint(mouse_pos):
            self.hovered = True
            if mouse_down:
                self.pressed = True
            elif mouse_up and self.pressed:
                self.pressed = False
                self.on_click()
            elif not mouse_down:
                self.pressed = False
        else:
            self.hovered = False
            self.pressed = False

# Instantiate a test button
def button_action():
    print("Test button clicked!")
    
buttons = [
    {}, #Panel 1
    {}, #Panel 2
    {}, #Panel 3
    {}, #Panel 4
    {}, #ICAWS
    {}, #Overlay
    {}, #Popup
]

def draw_lines(surface, color, closed, mirror, points, width, position, scale):
    x, y = position
    new_pointset = []
    inverted_pointset = []
    for px, py in points:
        point = (x + (px*scale), y + (py*scale))
        new_pointset.append(point)
        if mirror:
            ipoint = (x - (px*scale), y + (py*scale))
            inverted_pointset.append(ipoint)
    pygame.draw.lines(surface, color, closed, new_pointset, width)
    if mirror:
        pygame.draw.lines(surface, color, closed, inverted_pointset, width)

def draw_filled_polygon(surface, color, points, width=0, fill_percent=1.0, fill_color=(0, 255, 0)):
    if not 0 <= fill_percent <= 1:
        raise ValueError("fill_percent must be between 0.0 and 1.0")

    if fill_percent == 0:
        return  # No fill needed

    # Find bounding box
    min_x = math.floor(min(p[0] for p in points))
    max_x = math.floor(max(p[0] for p in points))
    min_y = math.floor(min(p[1] for p in points))
    max_y = math.floor(max(p[1] for p in points))

    # Height of the polygon
    height = max_y - min_y
    if height == 0:
        return  # Avoid division by zero

    # Calculate y position to fill up to
    fill_height = int(height * fill_percent)
    fill_top_y = max_y - fill_height

    # Create a surface for the polygon shape
    temp_surf = pygame.Surface((max_x - min_x + 1, max_y - min_y + 1), pygame.SRCALPHA)
    shifted_points = [(x - min_x, y - min_y) for (x, y) in points]
    pygame.draw.polygon(temp_surf, (255, 255, 255), shifted_points)

    # Use a mask to identify the filled region
    mask = pygame.mask.from_surface(temp_surf)

    # Render the fill region
    for y in range(fill_top_y, max_y + 1):
        row_y = y - min_y
        if 0 <= row_y < mask.get_size()[1]:
            for x in range(min_x, max_x + 1):
                col_x = x - min_x
                if 0 <= col_x < mask.get_size()[0] and mask.get_at((col_x, row_y)):
                    surface.set_at((x, y), fill_color)
                    
    # Draw the outline or solid polygon first
    pygame.draw.polygon(surface, color, points, width)

def draw_gauge_circle(
    surface,
    x, y, radius,
    arc_color, arc_width,            # color & thickness of the main 3/4 arc
    fill_color=None,                 # optional circle fill behind it
    text=None, font=None, text_color=(0,0,0),
    tick_length=10, tick_color=None, # radial tick at the end‐of‐arc
    red_arc_length=None,             # if set, pixels of extra arc‐length beyond 3/4
    red_arc_color=(255,0,0),
    rotation_deg=0                   # rotate the gauge itself
):
    """
    Draws a rotatable 3/4‐circle gauge, mirrored horizontally:
      • starts at 6 o'clock (pointing down)
      • sweeps *clockwise* 270° to 9 o'clock (pointing left)
      • draws an outward tick at that end point
      • optionally extends a red arc beyond it (along the circle)
      • text is still upright at the top

    Key change: sweep is negative.
    """
    # 1) optional filled background
    if fill_color is not None:
        pygame.draw.circle(surface, fill_color, (x, y), radius)

    # 2) compute start/end angles
    base_start = math.pi / 2                # 6 o'clock
    rot_rad    = math.radians(rotation_deg)
    start_ang  = base_start + rot_rad
    # sweep *clockwise* 270° → subtract 3π/2
    end_ang    = start_ang - 3 * math.pi/2

    # bounding box
    rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)

    # draw the main 3/4 arc (pygame arcs always go CCW from start→end;
    # by making end<start, it effectively draws CW)
    pygame.draw.arc(surface, arc_color, rect, end_ang, start_ang, arc_width)

    # 3) radial tick at end_ang
    ex = x + math.cos(end_ang) * radius
    ey = y + math.sin(end_ang) * radius
    dx, dy = math.cos(end_ang), math.sin(end_ang)
    tick_c = tick_color or arc_color
    tx, ty = ex + dx * tick_length, ey + dy * tick_length
    pygame.draw.line(surface, tick_c, (ex, ey), (tx, ty), arc_width)

    # 4) extra red arc (along circumference), if requested
    if red_arc_length is not None:
        extra_ang = red_arc_length / float(radius)
        # continue further *clockwise*
        red_end = end_ang - extra_ang
        pygame.draw.arc(surface, red_arc_color, rect, red_end, end_ang, arc_width)

    # 5) text always upright at top of circle
    if text and font:
        render_multicolor_text(surface, font, [[(text, text_color)]], (x, y-radius+2), 0, True, "R")
        #txt_surf = font.render(text, True, text_color)
        #tw, th   = txt_surf.get_size()
        #txt_x    = x - tw // 2
        #txt_y    = (y - radius) - th // 2
        #surface.blit(txt_surf, (txt_x, txt_y))
        
def draw_arrow(
    surface, start, angle_deg, length,
    line_width, head_width, head_height,
    line_style, head_style):
    """
    Draw a rotated arrow composed of:
      - a rectangular shaft
      - a triangular head

    Parameters:
    -----------
    surface      : pygame.Surface
    start        : (x, y) start point of the arrow
    angle_deg    : float, arrow angle in degrees (0° is to the right, increases clockwise)
    length       : float, total arrow length (shaft + head)

    line_width   : float, thickness of the shaft
    head_width   : float, width of the base of the triangular head
    head_height  : float, length of the triangular head along the arrow

    line_style   : (color, fill)
                   color: RGB tuple for the shaft
                   fill: 0 => filled; >1 => border thickness

    head_style   : (color, fill)
                   same scheme for the triangular head
    """

    x0, y0 = start
    # Unit direction vector
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    # Perpendicular for width
    px, py = -dy, dx

    # Shaft runs from start to base of head
    shaft_len = length - head_height
    bx = x0 + dx * shaft_len
    by = y0 + dy * shaft_len

    # Build shaft polygon (a skinny rectangle)
    half_lw = line_width / 2
    shaft_pts = [
        (x0 + px * half_lw, y0 + py * half_lw),
        (x0 - px * half_lw, y0 - py * half_lw),
        (bx - px * half_lw, by - py * half_lw),
        (bx + px * half_lw, by + py * half_lw),
    ]

    # Build head triangle
    tip_x = x0 + dx * length
    tip_y = y0 + dy * length
    half_hw = head_width / 2
    head_pts = [
        (bx + px * half_hw, by + py * half_hw),
        (bx - px * half_hw, by - py * half_hw),
        (tip_x, tip_y)
    ]

    # Draw shaft
    line_color, line_fill = line_style
    if line_fill == 0:
        pygame.draw.polygon(surface, line_color, shaft_pts)
    else:
        pygame.draw.polygon(surface, line_color, shaft_pts, int(line_fill))

    # Draw head
    head_color, head_fill = head_style
    if head_fill == 0:
        pygame.draw.polygon(surface, head_color, head_pts)
    else:
        pygame.draw.polygon(surface, head_color, head_pts, int(head_fill))

def draw_striped_square_border(surface, x, y, w, h, border_width, stripe1, stripe2, direction='right'):
    """
    Draws a diagonal-striped square border around the outside of a rectangle.

    Parameters:
        surface:        Pygame surface to draw on.
        x, y, w, h:     Rectangle's position and size.
        border_width:   Thickness of the border (outside the rectangle).
        stripe1:        Tuple of (color, width) for the first stripe.
        stripe2:        Tuple of (color, width) for the second stripe.
        direction:      'right' (/) or 'left' (\) for stripe direction.
    """
    color1, width1 = stripe1
    color2, width2 = stripe2
    total_stripe_width = width1 + width2

    total_border = border_width * 2
    outer_w = w + total_border
    outer_h = h + total_border

    # Temporary surface
    border_surf = pygame.Surface((outer_w, outer_h), pygame.SRCALPHA)
    border_surf.fill((0, 0, 0, 0))  # Fully transparent

    # Drawing diagonal stripes
    offset = -outer_h if direction == 'right' else -outer_w
    toggle = True
    current_color = color1

    while offset < (outer_w if direction == 'right' else outer_h) * 2:
        stripe_width = width1 if toggle else width2

        if direction == 'right':  # /
            points = [
                (offset, 0),
                (offset + stripe_width, 0),
                (offset + outer_h + stripe_width, outer_h),
                (offset + outer_h, outer_h)
            ]
        else:  # \
            points = [
                (outer_w - offset, 0),
                (outer_w - (offset + stripe_width), 0),
                (outer_w - (offset + stripe_width) - outer_h, outer_h),
                (outer_w - offset - outer_h, outer_h)
            ]

        pygame.draw.polygon(border_surf, current_color, points)

        offset += stripe_width
        toggle = not toggle
        current_color = color1 if toggle else color2

    # Cut inner rectangle
    inner_rect = pygame.Rect(border_width, border_width, w, h)
    pygame.draw.rect(border_surf, (0, 0, 0, 0), inner_rect)

    # Blit onto destination surface
    surface.blit(border_surf, (x - border_width, y - border_width))

def getSubportal(inta):
    if inta < 5:
        return (inta, False, 0)
    elif inta > 4:
        index = inta - 5
        portal = math.floor(index/2)
        sub = index - (portal*2)
        return (portal, True, sub)

def setButton(panel, index, position, callback):
    buttons[panel][index] = {"position" : position, "callback" : callback}
    return buttons[panel][index]

def getButtonCount(panel):
    count = 0
    for btn_data in get_buttons(panel).values():
        # skip anything that isn’t a proper { "position":…, "callback":… } dict
        if not isinstance(btn_data, dict):
            continue
        position = btn_data.get("position")
        callback = btn_data.get("callback")
        # if either field is missing, skip
        if position is None or callback is None:
            continue
        count += 1
    return count
    
def renderNavButton(surface, x, y, text, color = None):
    render_multicolor_text(surface, PCDFont, [[(text, GREEN)], [(f"  {MASTER_MODE}  ", GREEN)]], (x, y), 1, True, "T", color)
    
def renderMenuButton(surface, x, y, text):
    render_multicolor_text(surface, PCDFont, [[(text, GREEN)],[("MENU",  MAGENTA)],[("POPUP", MAGENTA)]], (x, y), 1, True)

def render_menu(name, surface, x, y, w, h, portal):
    pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
    texts = [
        "ASR",
        "CKLST",
        "CNI",
        "DAS",
        "DIM",
        "EFI",
        "ENG",
        "FCS",
        "FTA",
        "FTI",
        "FUEL",
        "HUD",
        "ICAWS",
        "PHM",
        "SMS",
        "SRCH",
        "TFLIR",
        "TSD-1",
        "TSD-2",
        "TSD-3",
        "TWD",
        "WPN-A",
        "WPN-S"
    ]
    
    pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
    px, py, pw, ph = pos
    cx, cy = centered
    renderMenuButton(surface, cx, cy, name)
    renderFunc = portal['CurrentPage']['render']
    #print(renderFunc)
    setButton(portal['loc'], "0", pos, lambda: set_portal_render(portal['loc'], renderFunc))
    
    grid, size = get_portal_grid(portal['loc'])
    boxWidth, boxHeight, boxColumns, boxRows = grid
    ind = 0
    for by in range(1, boxRows):
        for bx in range(0, boxColumns):
            position, cent = get_portal_grid_box(portal['loc'], (bx, by))
            gx, gy, gw, gh = position
            
            if len(texts) > ind and ind >= 0 and texts[ind] != None and texts[ind] != "":

                string = texts[ind]
                cx, cy = cent
                text = PCDFont.render(string+">", True, CYAN)
                surface.blit(text, (cx - text.get_width()//2, cy - text.get_height()//2))

                setButton(portal['loc'], ind+1, position, partial(set_portal_page, portal['loc'], string))
            ind += 1

def render_PHM(surface, x, y, w, h, portal, sub = 0):
    inta, bool, sub = getSubportal(sub)
    if bool and portal['hidden'] == False and portal['EXP'] == 0:
        name = portal['subs'][sub]['name']
        px, py, pw, ph = get_subportal_position(portal['loc'], sub)
        pygame.draw.rect(surface, BLACK, pygame.Rect(px, py, pw, ph))
        
        #render_multicolor_text(surface, PCDLarge, [[(f"DISP {name.upper()}", GREEN)]], (px + (pw/2), py + (ph/2)), 0, True)
    elif bool == False and portal['hidden'] == False:
        BLOffset = (GRID_Y) * min(portal['EXP'], 1)
        pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
        name = portal['CurrentPage']['name']
        pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
        ncx, ncy = centered
        
        render_multicolor_text(surface, PCDFont, [[("STATUS>", CYAN)]], (x+(w/2), y), 0, True, "T")
        render_multicolor_text(surface, PCDFont, [[("SUBSYS>", CYAN)]], (x+w, y), 0, True, "TR")
        
        vehicleSystems = importArray["PHM"]["VEHICLE"]
        missionSystems = importArray["PHM"]["MISSION"]
        gap = 60
        pygame.draw.line(surface, GRAY, (x+(w/2), y+gap), (x+(w/2), y+500), 2)
        
        vsLines = []
        vsStatus = []
        for s2, s1 in enumerate(vehicleSystems):
            s2 = vehicleSystems[s1]

            # Calculate how many dots are needed
            
            # Clamp to at least 0 (in case stringLength is too short)
            
            result_string = f"{s1}"
            color = GREEN
            status = "GO"
            if s2 == 0:
                color = YELLOW
                status = "DEGRD"
            elif s2 == -1:
                color = GRAY
                status = "INOP"
            vsLines.append([(result_string, GREEN)])
            vsStatus.append([(status, color)])
        msLines = []
        msStatus = []
        for s2, s1 in enumerate(missionSystems):
            s2 = missionSystems[s1]

            # Calculate how many dots are needed
            
            # Clamp to at least 0 (in case stringLength is too short)
            
            result_string = f"{s1}"
            color = GREEN
            status = "GO"
            if s2 == 0:
                color = YELLOW
                status = "DEGRD"
            elif s2 == -1:
                color = GRAY
                status = "INOP"
            msLines.append([(result_string, GREEN)])
            msStatus.append([(status, color)])
            
        lineSpace = 14
        
        utext = pygame.font.Font("PCD.ttf", 24)
        utext.set_bold(True)
        utext.set_underline(True)
        
        render_multicolor_text(surface, utext, [[("VEHICLE SYSTEMS", GREEN)]], (x+(w/2)-15, y+gap), 0, True, "TR")
        render_multicolor_text(surface, utext, [[("MISSION SYSTEMS", GREEN)]], (x+(w/2)+15, y+gap), 0, True, "TL")
         
        render_multicolor_text(surface, PCDFont, vsLines, (x+(w/2)-200, y+gap+30), lineSpace, True, "TL")
        render_multicolor_text(surface, PCDFont, vsStatus, (x+(w/2)-75, y+gap+30), lineSpace, True, "TL")
        
        render_multicolor_text(surface, PCDFont, msLines, (x+(w/2)+30, y+gap+30), lineSpace, True, "TL")
        render_multicolor_text(surface, PCDFont, msStatus, (x+(w/2)+130, y+gap+30), lineSpace, True, "TL")
        
        setButton(portal['loc'], 1, pos, lambda: set_portal_render(portal['loc'], partial(render_menu, portal["CurrentPage"]['name'])))
        renderNavButton(surface, ncx, y, portal["CurrentPage"]['name'], BLACK)
        
def render_TFLIR(surface, x, y, w, h, portal, sub = 0):
    def scale_to_fit(src_surf: pygame.Surface, dst_rect: pygame.Rect):
        """
        Scale src_surf to fit inside dst_rect while preserving aspect ratio.
        Returns the scaled surface and the topleft position (x, y) to blit it.
        """
        src_w, src_h = src_surf.get_size()
        dst_w, dst_h = dst_rect.size

        # Compute aspect ratios
        src_ratio = src_w / src_h
        dst_ratio = dst_w / dst_h

        # Decide whether to fit by width or height
        if dst_ratio < src_ratio:
            # destination is wider than source → scale by height
            scale_factor = dst_h / src_h
        else:
            # destination is narrower → scale by width
            scale_factor = dst_w / src_w

        # Convert to integer pixel sizes
        new_w = int(src_w * scale_factor)
        new_h = int(src_h * scale_factor)

        # Center in dst_rect
        pos_x = dst_rect.x + (dst_w - new_w) // 2
        pos_y = dst_rect.y + (dst_h - new_h) // 2

        scaled_surf = pygame.transform.smoothscale(src_surf, (new_w, new_h))
        return scaled_surf, (pos_x, pos_y)
    def crop_surface(surface: pygame.Surface, rect: pygame.Rect) -> pygame.Surface:
        """
        Returns a new Surface containing only the part of `surface` inside `rect`.
        If `rect` extends outside `surface`, the out-of-bounds areas will be transparent
        (or black, if the source has no alpha channel).

        Args:
            surface (pygame.Surface): The source surface to crop.
            rect (pygame.Rect): The region to keep (in surface coordinates).

        Returns:
            pygame.Surface: A new surface of size (rect.width, rect.height) with the
                            cropped image.
        """
        # Create a new surface the size of the crop
        # Preserve alpha if the source has it
        flags = pygame.SRCALPHA if surface.get_flags() & pygame.SRCALPHA else 0
        cropped = pygame.Surface((rect.width, rect.height), flags, surface.get_bitsize())
        
        # Fill with transparent or black
        if flags & pygame.SRCALPHA:
            cropped.fill((0, 0, 0, 0))
        else:
            cropped.fill((0, 0, 0))
        
        # Blit the source surface onto our cropped surface, using rect as the source area
        cropped.blit(surface, (0, 0), rect)
        return cropped
    def grabEOTSImage(bx, by, bw, bh):
        global frames
        panda_surf = frames['EOTS']
        #panda_surf = panda.render_frame(importArray)
        src_w, src_h = panda_surf.get_size()
        dst_w, dst_h = bw, bh
        src_ratio = src_w / src_h
        dst_ratio = dst_w / dst_h

        if dst_ratio > src_ratio:
            # target is wider => match widths, overflow vertically
            scale_factor = dst_w / src_w
        else:
            # target is taller or equal => match heights, overflow horizontally
            scale_factor = dst_h / src_h

        new_w = int(src_w * scale_factor)
        new_h = int(src_h * scale_factor)
        scaled = pygame.transform.smoothscale(panda_surf, (new_w, new_h))

        # 2) Center-crop to exactly (w, h)
        crop_x = (new_w - dst_w) // 2
        crop_y = (new_h - dst_h) // 2
        crop_rect = pygame.Rect(crop_x, crop_y, dst_w, dst_h)
        cropped = crop_surface(scaled, crop_rect)
        height, width = cropped.get_size()
        vignette_mask = make_vignette_mask(width, height)
        
        #tflir = apply_tflir_fast(cropped, thermal_lut, vignette_mask)
        tflir = apply_bw_effect(cropped, importArray["Rendering"]["whiteHot"])
        #tflir = cropped
        return tflir
    
    inta, bool, sub = getSubportal(sub)
    if bool and portal['hidden'] == False and portal['EXP'] == 0:
        name = portal['subs'][sub]['name']
        px, py, pw, ph = get_subportal_position(portal['loc'], sub)
        pygame.draw.rect(surface, BLACK, pygame.Rect(px, py, pw, ph))
        
        if devmode:
            render_multicolor_text(surface, PCDLarge, [[(name, GREEN)], [("OFFLINE", GREEN)]], (px + (pw/2), py + (ph/2)), 1, True, "C")
        else:
            surface.blit(grabEOTSImage(px, py, pw, ph), (px, py))
        
        #render_multicolor_text(surface, PCDLarge, [[(f"DISP {name.upper()}", GREEN)]], (px + (pw/2), py + (ph/2)), 0, True)
    elif bool == False and portal['hidden'] == False:
        if devmode:
            BLOffset = (GRID_Y) * min(portal['EXP'], 1)
            pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
            name = portal['CurrentPage']['name']
            pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
            ncx, ncy = centered
            render_multicolor_text(surface, PCDMassive, [[(name, GREEN)], [("OFFLINE", GREEN)]], (x + (w/2), y + (h/2)), 1, True, "C")
        else:
            BLOffset = (GRID_Y) * min(portal['EXP'], 1)
            pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
            name = portal['CurrentPage']['name']
            pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
            ncx, ncy = centered
            
            surface.blit(grabEOTSImage(x, y, w, h), (x, y))
            
            count = 6
            gap = w/count
            FLIRFont = load_bold_font("PCD.ttf", 20)
            render_multicolor_text(surface, FLIRFont, [[("NAV", GREEN)]], (x+(gap*1.75), y), 0, True, "T", BLACK)
            render_multicolor_text(surface, FLIRFont, [[("LASER", CYAN)]], (x+(gap*3), y), 0, True, "T", BLACK)
            render_multicolor_text(surface, FLIRFont, [[("SYSTEM", CYAN)]], (x+(gap*4.25), y), 0, True, "T", BLACK)
            render_multicolor_text(surface, FLIRFont, [[("CNTL>", CYAN)]], (x+(gap*5.5), y), 0, True, "T", BLACK)
            render_multicolor_text(surface, FLIRFont, [[("IRST", CYAN)]], (x, y+150), 0, True, "L", BLACK)
            render_multicolor_text(surface, FLIRFont, [[(" E 4 ", GREEN)]], (x+w, y+50), 0, True, "R", BLACK)
            render_multicolor_text(surface, FLIRFont, [[("CAPTR", GRAY)]], (x+w, y+(w/2)+50), 0, True, "R", BLACK)
            
            cursorSize = min(w,h)-200
            ag = 30
            cornerLength = 40
            cursorWidth = 2
            
            cx, cy, cs = (x+(w/2), y+(h/2), cursorSize/2)
            cursorColor = importArray["Rendering"]["whiteHot"] and WHITE or importArray["Rendering"]["whiteHot"] == False and BLACK
            
            #TOP LEFT
            pygame.draw.line(surface, cursorColor, (cx-cs, cy-cs), (cx-cs, cy-cs+cornerLength), cursorWidth)
            pygame.draw.line(surface, cursorColor, (cx-cs, cy-cs), (cx-cs+cornerLength, cy-cs), cursorWidth)
            #TOP RIGHT
            pygame.draw.line(surface, cursorColor, (cx+cs, cy-cs), (cx+cs, cy-cs+cornerLength), cursorWidth)
            pygame.draw.line(surface, cursorColor, (cx+cs, cy-cs), (cx+cs-cornerLength, cy-cs), cursorWidth)
            #BOTTOM LEFT
            pygame.draw.line(surface, cursorColor, (cx-cs, cy+cs), (cx-cs, cy+cs-cornerLength), cursorWidth)
            pygame.draw.line(surface, cursorColor, (cx-cs, cy+cs), (cx-cs+cornerLength, cy+cs), cursorWidth)
            #BOTTOM RIGHT
            pygame.draw.line(surface, cursorColor, (cx+cs, cy+cs), (cx+cs, cy+cs-cornerLength), cursorWidth)
            pygame.draw.line(surface, cursorColor, (cx+cs, cy+cs), (cx+cs-cornerLength, cy+cs), cursorWidth)
            
            #CENTER
            pygame.draw.line(surface, cursorColor, (cx-cs, cy), (cx-ag, cy), cursorWidth)
            pygame.draw.line(surface, cursorColor, (cx+cs, cy), (cx+ag, cy), cursorWidth)
            pygame.draw.line(surface, cursorColor, (cx, cy-cs), (cx, cy-ag), cursorWidth)
            pygame.draw.line(surface, cursorColor, (cx, cy+cs), (cx, cy+ag), cursorWidth)
        
        setButton(portal['loc'], 1, pos, lambda: set_portal_render(portal['loc'], partial(render_menu, portal["CurrentPage"]['name'])))
        renderNavButton(surface, ncx, y, portal["CurrentPage"]['name'], BLACK)
        
def render_DAS(surface, x, y, w, h, portal, sub = 0):
    def scale_to_fit(src_surf: pygame.Surface, dst_rect: pygame.Rect):
        """
        Scale src_surf to fit inside dst_rect while preserving aspect ratio.
        Returns the scaled surface and the topleft position (x, y) to blit it.
        """
        src_w, src_h = src_surf.get_size()
        dst_w, dst_h = dst_rect.size

        # Compute aspect ratios
        src_ratio = src_w / src_h
        dst_ratio = dst_w / dst_h

        # Decide whether to fit by width or height
        if dst_ratio < src_ratio:
            # destination is wider than source → scale by height
            scale_factor = dst_h / src_h
        else:
            # destination is narrower → scale by width
            scale_factor = dst_w / src_w

        # Convert to integer pixel sizes
        new_w = int(src_w * scale_factor)
        new_h = int(src_h * scale_factor)

        # Center in dst_rect
        pos_x = dst_rect.x + (dst_w - new_w) // 2
        pos_y = dst_rect.y + (dst_h - new_h) // 2

        scaled_surf = pygame.transform.smoothscale(src_surf, (new_w, new_h))
        return scaled_surf, (pos_x, pos_y)
    def crop_surface(surface: pygame.Surface, rect: pygame.Rect) -> pygame.Surface:
        """
        Returns a new Surface containing only the part of `surface` inside `rect`.
        If `rect` extends outside `surface`, the out-of-bounds areas will be transparent
        (or black, if the source has no alpha channel).

        Args:
            surface (pygame.Surface): The source surface to crop.
            rect (pygame.Rect): The region to keep (in surface coordinates).

        Returns:
            pygame.Surface: A new surface of size (rect.width, rect.height) with the
                            cropped image.
        """
        # Create a new surface the size of the crop
        # Preserve alpha if the source has it
        flags = pygame.SRCALPHA if surface.get_flags() & pygame.SRCALPHA else 0
        cropped = pygame.Surface((rect.width, rect.height), flags, surface.get_bitsize())
        
        # Fill with transparent or black
        if flags & pygame.SRCALPHA:
            cropped.fill((0, 0, 0, 0))
        else:
            cropped.fill((0, 0, 0))
        
        # Blit the source surface onto our cropped surface, using rect as the source area
        cropped.blit(surface, (0, 0), rect)
        return cropped
    def grabEOTSImage(bx, by, bw, bh):
        global frames
        panda_surf = frames[importArray["Position"]["DAS"]["CAMERA"]]
        src_w, src_h = panda_surf.get_size()
        dst_w, dst_h = bw, bh
        src_ratio = src_w / src_h
        dst_ratio = dst_w / dst_h

        if dst_ratio > src_ratio:
            # target is wider => match widths, overflow vertically
            scale_factor = dst_w / src_w
        else:
            # target is taller or equal => match heights, overflow horizontally
            scale_factor = dst_h / src_h

        new_w = int(src_w * scale_factor)
        new_h = int(src_h * scale_factor)
        scaled = pygame.transform.smoothscale(panda_surf, (new_w, new_h))

        # 2) Center-crop to exactly (w, h)
        crop_x = (new_w - dst_w) // 2
        crop_y = (new_h - dst_h) // 2
        crop_rect = pygame.Rect(crop_x, crop_y, dst_w, dst_h)
        cropped = crop_surface(scaled, crop_rect)
        height, width = cropped.get_size()
        vignette_mask = make_vignette_mask(width, height)
        
        #tflir = apply_tflir_fast(cropped, thermal_lut, vignette_mask)
        tflir = cropped
        #tflir = cropped
        return tflir
    
    inta, bool, sub = getSubportal(sub)
    if bool and portal['hidden'] == False and portal['EXP'] == 0:
        name = portal['subs'][sub]['name']
        px, py, pw, ph = get_subportal_position(portal['loc'], sub)
        pygame.draw.rect(surface, BLACK, pygame.Rect(px, py, pw, ph))
        
        if devmode:
            render_multicolor_text(surface, PCDLarge, [[(name, GREEN)], [("OFFLINE", GREEN)]], (px + (pw/2), py + (ph/2)), 1, True, "C")
        else:
            surface.blit(grabEOTSImage(px, py, pw, ph), (px, py))
        
        #render_multicolor_text(surface, PCDLarge, [[(f"DISP {name.upper()}", GREEN)]], (px + (pw/2), py + (ph/2)), 0, True)
    elif bool == False and portal['hidden'] == False:
        if devmode:
            BLOffset = (GRID_Y) * min(portal['EXP'], 1)
            pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
            name = portal['CurrentPage']['name']
            pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
            ncx, ncy = centered
            render_multicolor_text(surface, PCDMassive, [[(name, GREEN)], [("OFFLINE", GREEN)]], (x + (w/2), y + (h/2)), 1, True, "C")
        else:
            BLOffset = (GRID_Y) * min(portal['EXP'], 1)
            pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
            name = portal['CurrentPage']['name']
            pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
            ncx, ncy = centered
            
            surface.blit(grabEOTSImage(x, y, w, h), (x, y))
            
            count = 6
            gap = w/count
            FLIRFont = load_bold_font("PCD.ttf", 20)
            render_multicolor_text(surface, FLIRFont, [[("OPER", CYAN)]], (x+(gap*1.75), y), 0, True, "T", BLACK)
            render_multicolor_text(surface, FLIRFont, [[("CNTL>", CYAN)]], (x+(gap*5.5), y), 0, True, "T", BLACK)
            render_multicolor_text(surface, FLIRFont, [[(" E 4 ", GREEN)]], (x+w, y+50), 0, True, "R", BLACK)
            render_multicolor_text(surface, FLIRFont, [[("CAPTR", GRAY)]], (x+w, y+(w/2)+50), 0, True, "R", BLACK)
            
            cursorSize = min(w,h)-200
            ag = 30
            cursorWidth = 2
            
            cx, cy, cs = (x+(w/2), y+(h/2), cursorSize/2)
            cursorColor = importArray["Rendering"]["whiteHot"] and WHITE or importArray["Rendering"]["whiteHot"] == False and BLACK
            
            #CENTER
            pygame.draw.line(surface, cursorColor, (cx-cs, cy), (cx-ag, cy), cursorWidth)
            pygame.draw.line(surface, cursorColor, (cx+cs, cy), (cx+ag, cy), cursorWidth)
            pygame.draw.line(surface, cursorColor, (cx, cy-cs), (cx, cy-ag), cursorWidth)
            pygame.draw.line(surface, cursorColor, (cx, cy+cs), (cx, cy+ag), cursorWidth)
        
        setButton(portal['loc'], 1, pos, lambda: set_portal_render(portal['loc'], partial(render_menu, portal["CurrentPage"]['name'])))
        renderNavButton(surface, ncx, y, portal["CurrentPage"]['name'], BLACK)
  
def render_FCS(surface, x, y, w, h, portal, sub = 0):
    inta, bool, sub = getSubportal(sub)
    vertical = [
        (0, -5.2),
        (-1, -6),
        (-2, -5),
        (-3, -6),
        (-4, -5),
        (-4, 0),
        (-4.5, 1),
        (-4.5, -1),
        (-8, 3),
        (-8, 4),
        (-9, 10),
        (-9, 12),
        
        #Wings
        (-9.5, 14),
        (-25, 21),
        (-25.5, 30),
        
        #Vert Stabs
        (-8, 33),
        (-6, 29),
        (-5.5, 31),
        (-5.5, 36),
        (-6, 40),
        (-10, 42),
        (-10, 37),
        (-8, 33),
        (-10, 37),
        
        #Horizontal Stab
        (-10, 40),
        (-16, 42),
        (-16, 47),
        (-4, 49),
        (-3.5, 43),
        (-3.5, 42),
        
        #Engine
        (-2, 40.5),
        (0, 40.5),
        (-2, 40.5),
        (-1.5, 42),
        (0, 42)
    ]
    lef = [
        (-11, 15),
        (-11, 17.5),
        (-25, 23.5)
    ]
    flaps = [
        (-20, 31),
        (-20, 27),
        (-7.5, 29),
        (-7.5, 32)
    ]
    rudders = [
        (-6, 40),
        (-6.5, 42.5),
        (-10, 44),
        (-10, 42)
    ]
    
    horizontal = [
        (-2, 1),
        (5, -2),
        (7, -2),
        (15, -6),
        (20, -6),
        (25, -4),
        (30, -4),
        (45, -3),
        (50, -2.5),
        (60, -2.5),
        
        #Tail
        (58.5, 0),
        (67, -10),
        (73, -10.2),
        (70, 0.5),
        (58.5, 0),
        (70, 0.5),
        
        #Stab
        (70, 1),
        (78, 2.5),
        (61, 2),
        (70, 1),
        (78, 2.5),
        (68, 2.25),
        
        #Belly
        (68, 6),
        (72, 4.5),
        (72, 2.2),
        (72, 4.5),
        (68, 6),
        (60, 7),
        (44, 7),
        (20, 6),
        (10, 4.5),
        (-2, 1),
        (5, -2),
        (7, -2),
        (8.5, -2.5),
        (20, -2.5),
        (25, -4),
        (20, -2.5),
        (15, -2.5),
        (14, -5.3)
    ]
    
    if bool and portal['hidden'] == False and portal['EXP'] == 0:
        name = portal['subs'][sub]['name']
        px, py, pw, ph = get_subportal_position(portal['loc'], sub)
        pygame.draw.rect(surface, BLACK, pygame.Rect(px, py, pw, ph))
        render_multicolor_text(surface, PCDLarge, [[(f"DISP {name.upper()}", GREEN)]], (px + (pw/2), py + (ph/2)), 0, True)
    elif bool == False and portal['hidden'] == False:
        BLOffset = (GRID_Y) * min(portal['EXP'], 1)
        pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
        name = portal['CurrentPage']['name']
        pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
        cx, cy = centered
        
        x1, y1, x2, y2, scale, scale2 = (x+w - (w/2.2), y+h-275, x+w-400, y+100, 5, 5)
        if portal["EXP"]>0:
            x1 = x+w - (w/2)
            y1 = y+h-375
            y2 = y+150
            scale = 6
            if portal["EXP"]==2:
                x2 = x+(w/2)-300
                y2 = y+130
                scale2 = 8
                scale = 8
                y1 = y+h-435

        draw_lines(surface, GRAY, False, True, vertical, 2, (x1, y1), scale)
        draw_lines(surface, GRAY, False, True, lef, 2, (x1, y1), scale)
        draw_lines(surface, GRAY, False, True, flaps, 2, (x1, y1), scale)
        draw_lines(surface, GRAY, False, True, rudders, 2, (x1, y1), scale)

        draw_lines(surface, GRAY, False, False, horizontal, 2, (x2, y2), scale2)
        
        threeGreenSize = 0.8
        threeGreenWidth = 4.5*scale*threeGreenSize
        threeGreenHight = 7*scale*threeGreenSize
        
        noseHeight = -2*scale
        mainHeight = 21*scale
        mainSpace = 6*scale
        
        NLGColor = importArray['gear']['NLG_UPLOCK'] and GRAY or importArray['gear']['NLG_DOWNLOCK'] and GREEN or YELLOW
        LMLGColor = importArray['gear']['LMLG_UPLOCK'] and GRAY or importArray['gear']['LMLG_DOWNLOCK'] and GREEN or YELLOW
        RMLGColor = importArray['gear']['RMLG_UPLOCK'] and GRAY or importArray['gear']['RMLG_DOWNLOCK'] and GREEN or YELLOW
        
        def getArrow(direction, var):
            if direction == "H":
                if math.floor(var) > 0:
                    return "→"
                elif math.floor(var) < 0:
                    return "←"
                elif math.floor(var) == 0:
                    return ""
            elif direction == "V":
                if math.floor(var) > 0:
                    return "↑"
                elif math.floor(var) < 0:
                    return "↓"
                elif math.floor(var) == 0:
                    return ""
            else:
                return ""
        pygame.draw.rect(surface, NLGColor, pygame.rect.Rect(x1-(threeGreenWidth/2), y1+noseHeight, threeGreenWidth, threeGreenHight), 0)
        
        pygame.draw.rect(surface, LMLGColor, pygame.rect.Rect(x1-mainSpace-threeGreenWidth, y1+mainHeight, threeGreenWidth, threeGreenHight), 0)
        pygame.draw.rect(surface, RMLGColor, pygame.rect.Rect(x1+mainSpace, y1+mainHeight, threeGreenWidth, threeGreenHight), 0)
        
        arrowFont = pygame.font.SysFont("Arial", 20)
        arrowFont.set_bold(True)
        
        #Center Text
        render_multicolor_text(surface, load_bold_font("PCD.ttf", 20), [[("CG 16", GREEN)], [("10 C", GREEN)]], (x1, y1+(10*scale)), 1.5*scale, True, "T")
        
        #Left LEF
        render_multicolor_text(surface, load_bold_font("PCD.ttf", 20), [[(str(abs(math.floor(importArray['FCS']['L_LEF']))), GREEN)]], (x1-(14*scale), y1+(16*scale)), 1.5*scale, True, "T")
        render_multicolor_text(surface, arrowFont, [[(getArrow("V", importArray['FCS']['L_LEF']), GREEN)]], (x1-(20*scale), y1+(16*scale)), 1.5*scale, True, "T")
        
        #Right LEF
        render_multicolor_text(surface, load_bold_font("PCD.ttf", 20), [[(str(abs(math.floor(importArray['FCS']['R_LEF']))), GREEN)]], (x1+(14*scale), y1+(16*scale)), 1.5*scale, True, "T")
        render_multicolor_text(surface, arrowFont, [[(getArrow("V", importArray['FCS']['R_LEF']), GREEN)]], (x1+(20*scale), y1+(16*scale)), 1.5*scale, True, "T")
        
        #Left Aileron
        render_multicolor_text(surface, load_bold_font("PCD.ttf", 20), [[(str(abs(math.floor(importArray['FCS']['L_AILERON']))), GREEN)]], (x1-(11*scale), y1+(28*scale)), 1.5*scale, True, "T")
        render_multicolor_text(surface, arrowFont, [[(getArrow("V", importArray['FCS']['L_AILERON']), GREEN)]], (x1-(17*scale), y1+(26*scale)), 1.5*scale, True, "T")
        
        #Right Aileron
        render_multicolor_text(surface, load_bold_font("PCD.ttf", 20), [[(str(abs(math.floor(importArray['FCS']['R_AILERON']))), GREEN)]], (x1+(11*scale), y1+(28*scale)), 1.5*scale, True, "T")
        render_multicolor_text(surface, arrowFont, [[(getArrow("V", importArray['FCS']['R_AILERON']), GREEN)]], (x1+(17*scale), y1+(26*scale)), 1.5*scale, True, "T")
        
        #Left Rudder
        render_multicolor_text(surface, load_bold_font("PCD.ttf", 20), [[(str(abs(math.floor(importArray['FCS']['L_RUDDER']))), GREEN)]], (x1-(7*scale), y1+(34*scale)), 1.5*scale, True, "T")
        render_multicolor_text(surface, arrowFont, [[(getArrow("H", importArray['FCS']['L_RUDDER']), GREEN)]], (x1-(9.5*scale), y1+(36*scale)), 1.5*scale, True, "T")
        
        #Right Rudder
        render_multicolor_text(surface, load_bold_font("PCD.ttf", 20), [[(str(abs(math.floor(importArray['FCS']['R_RUDDER']))), GREEN)]], (x1+(7*scale), y1+(34*scale)), 1.5*scale, True, "T")
        render_multicolor_text(surface, arrowFont, [[(getArrow("H", importArray['FCS']['R_RUDDER']), GREEN)]], (x1+(9.5*scale), y1+(36*scale)), 1.5*scale, True, "T")
        
        #Left Stab
        render_multicolor_text(surface, load_bold_font("PCD.ttf", 20), [[(str(abs(math.floor(importArray['FCS']['L_STAB']))), GREEN)]], (x1-(10*scale), y1+(44*scale)), 1.5*scale, True, "T")
        render_multicolor_text(surface, arrowFont, [[(getArrow("V", importArray['FCS']['L_STAB']), GREEN)]], (x1-(6*scale), y1+(42*scale)), 1.5*scale, True, "T")
        
        #Right Stab
        render_multicolor_text(surface, load_bold_font("PCD.ttf", 20), [[(str(abs(math.floor(importArray['FCS']['R_STAB']))), GREEN)]], (x1+(10*scale), y1+(44*scale)), 1.5*scale, True, "T")
        render_multicolor_text(surface, arrowFont, [[(getArrow("V", importArray['FCS']['R_STAB']), GREEN)]], (x1+(6*scale), y1+(42*scale)), 1.5*scale, True, "T")
        
        render_multicolor_text(surface, PCDFont, [[("NOSE", CYAN)], [("DOOR", CYAN)]], (x+150, y), 1, True, "T")
        
        render_multicolor_text(surface, PCDFont, [[("EXER", GRAY)], [("MODE", GRAY)]], (x+300, y+20), 1, True, "T")
        
        gx, gy = (x+100, y+225)
        textGap = 75
        if portal["EXP"] > 0:
            textGap = 100
            gx, gy = (x1-150, y1+30)
            if portal["EXP"]==2:
                gx, gy = (x1-200, y1+30)
        render_multicolor_text(surface, load_bold_font("PCD.ttf", 17), [[("GWT   37.1", GREEN)], [("G-LIM    9", GREEN)]], (gx, gy), 10, True, "TL")
        

        render_multicolor_text(surface, PCDFont, [[("A/P", CYAN)]], (x, y+300+(textGap*0)), 1, True, "L")
        
        render_multicolor_text(surface, PCDFont, [[("ALT", CYAN)], [("PA", CYAN)]], (x, y+300+(textGap*1)), 1, True, "L")
        
        render_multicolor_text(surface, PCDFont, [[("INTEG", CYAN)], [("FCS", CYAN)], [("FADEC", CYAN)]], (x, y+300+(textGap*2.25)), 1, True, "L")
        
        render_multicolor_text(surface, PCDFont, [[("TRIM", CYAN)], [("RESET", CYAN)]], (x+w, y+300+(textGap*2.25)), 1, True, "R")
        
        HOTAS = importArray["FCS"]["HOTAS"]
        pitch = max(min(HOTAS["PITCH"],1), -1)
        yaw = max(min(HOTAS["YAW"],1), -1)
        roll = max(min(HOTAS["ROLL"],1), -1)
        
        TRIM = importArray["FCS"]["TRIM"]
        tpitch = max(min(TRIM["PITCH"],1), -1)
        tyaw = max(min(TRIM["YAW"],1), -1)
        troll = max(min(TRIM["ROLL"],1), -1)
        
        lineThickness = 1
        
        boxSize = 100
        boxPosition = (x+w-boxSize-20, y+((h/2)-(boxSize/2)), boxSize, boxSize)
        bx, by, bw, bh = boxPosition
        bxc, byc = (bx+(bw/2), by+(bh/2))
        pygame.draw.rect(surface, GRAY, pygame.Rect(bx+(bw/4), by+(bh/4), bw/2, bh/2), lineThickness)
        
        render_multicolor_text(surface, load_bold_font("PCD.ttf", 20), [[("MAN -6.5", GREEN)]], (bxc, by), 0, True, "B")
        
        gapWidth=10
        pygame.draw.rect(surface, BLACK, pygame.Rect(bx+(bw/2)-(gapWidth/2), by, gapWidth, bh), 0)
        pygame.draw.rect(surface, BLACK, pygame.Rect(bx, by+(bh/2)-(gapWidth/2), bw, gapWidth), 0)
        
        pygame.draw.line(surface, GRAY, (bx+(bw/2), by), (bx+(bw/2), by+bh-1), lineThickness)
        pygame.draw.line(surface, GRAY, (bx, by+(bh/2)), (bx+bw-1, by+(bh/2)), lineThickness)
        
        tbx, tby = ((bxc) + (troll*((boxSize-(gapWidth*2))/2)), (byc) + (tpitch*((boxSize-(gapWidth*2))/2)))
        
        pygame.draw.line(surface, GREEN, (tbx-gapWidth, tby), (tbx+gapWidth, tby), lineThickness)
        pygame.draw.line(surface, GREEN, (tbx, tby-gapWidth), (tbx, tby+gapWidth), lineThickness)
        
        indicatorBoxSize = 16
        ibx, iby = ((bxc - (indicatorBoxSize/2)) + (roll*((boxSize-indicatorBoxSize)/2)), (byc - (indicatorBoxSize/2)) + (pitch*((boxSize-indicatorBoxSize)/2)))
        
        pygame.draw.rect(surface, CYAN, pygame.Rect(ibx, iby, indicatorBoxSize, indicatorBoxSize), lineThickness)
        
        pygame.draw.rect(surface, GRAY, pygame.Rect(boxPosition), lineThickness)
        
        separator = 14
        pygame.draw.line(surface, GRAY, (bx, by+bh+separator), (bx+bw-1, by+bh+separator), lineThickness)
        
        tbr = (bxc) + (tyaw*((boxSize-(gapWidth*2))/2))
        ibr = (bxc) + (yaw*((boxSize-indicatorBoxSize)/2)) - (indicatorBoxSize/2)
        sepY = by+bh+separator
        pygame.draw.line(surface, GREEN, (tbr-gapWidth, sepY), (tbr+gapWidth, sepY), lineThickness)
        pygame.draw.line(surface, GREEN, (tbr, sepY-gapWidth), (tbr, sepY+gapWidth), lineThickness)
        
        pygame.draw.rect(surface, CYAN, pygame.Rect(ibr, sepY-(indicatorBoxSize/2), indicatorBoxSize, indicatorBoxSize), 1)
        
        pygame.draw.line(surface, GRAY, (bx, by+bh+separator-5), (bx, by+bh+separator+5), lineThickness)
        pygame.draw.line(surface, GRAY, (bx+bw-1, by+bh+separator-5), (bx+bw-1, by+bh+separator+5), lineThickness)
        
        
        
        setButton(portal['loc'], 1, pos, lambda: set_portal_render(portal['loc'], partial(render_menu, portal["CurrentPage"]['name'])))
        renderNavButton(surface, cx, y, portal["CurrentPage"]['name'])
    
def render_FUEL(surface, x, y, w, h, portal, sub = 0):
    inta, bool, sub = getSubportal(sub)
    if bool and portal['hidden'] == False and portal['EXP'] == 0:
        name = portal['subs'][sub]['name']
        px, py, pw, ph = get_subportal_position(portal['loc'], sub)
        pygame.draw.rect(surface, DARK_GRAY, pygame.Rect(px, py, pw, ph))
        render_multicolor_text(surface, PCDLarge, [[(f"DISP {name.upper()}", GREEN)]], (px + (pw/2), py + (ph/2)), 0, True)
    elif bool == False and portal['hidden'] == False:
        BLOffset = (GRID_Y) * min(portal['EXP'], 1)
        pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
        name = portal['CurrentPage']['name']
        pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
        cx, cy = centered
    
        setButton(portal['loc'], 1, pos, lambda: set_portal_render(portal['loc'], partial(render_menu, portal["CurrentPage"]['name'])))
        
        scalex = 1
        scaley = 0.9
        if portal["EXP"]==2:
            scaley=1.2
            scalex=1.3
        
        b1w = 120*scalex
        b1x = (w/2)-(b1w/2) + x
        b1y = (30*scaley) + y + 30
        b1h = (60*scaley)
        
        b2y = b1y+b1h
        b2h = (70*scaley)
        
        b3y = b2y+b2h+(10*scaley)
        b3w = b1w/2
        b3b = (3*scalex)
        
        b4y = b3y+b2h+(25*scaley)
        b5y = b4y+b2h+(25*scaley)
        b6y = b5y+b2h+(10*scaley)
        
        b6b = (20*scalex)
        
        b7b = 15*scalex
        b7y = b4y+(5*scaley)
        b7y2 = b6y+(20*scaley)
        b7tri = 40*scalex
        
        fuelColor = PURPLE
        fuelFont = pygame.font.Font("PCD.ttf", 20)
        labelFont = pygame.font.Font("PCD.ttf", 15)
        labelOffset = 4
        
        F1Max = 3200
        F1IMax = 1200
        F2Max = 900
        F3Max = 900
        F4Max = 900
        F5Max = 500
        WingMax = 1000
        
        F1P = importArray['fuels']['F1']/F1Max
        F1IP = importArray['fuels']['F1I']/F1IMax
        F2LP = importArray['fuels']['F2L']/F2Max
        F2RP = importArray['fuels']['F2R']/F2Max
        F3LP = importArray['fuels']['F3L']/F3Max
        F3RP = importArray['fuels']['F3R']/F3Max
        F4LP = importArray['fuels']['F4L']/F4Max
        F4RP = importArray['fuels']['F4R']/F4Max
        F5LP = importArray['fuels']['F5L']/F5Max
        F5RP = importArray['fuels']['F5R']/F5Max
        LWP = importArray['fuels']['LW']/WingMax
        RWP = importArray['fuels']['RW']/WingMax
        
        totalInt = math.floor((importArray['fuels']['F1'] + importArray['fuels']['F1I'] + importArray['fuels']['F2L'] + importArray['fuels']['F2R'] + importArray['fuels']['F3L'] + importArray['fuels']['F3R'] + importArray['fuels']['F4L'] + importArray['fuels']['F4R'] + importArray['fuels']['F5L'] + importArray['fuels']['F5R'] + importArray['fuels']['LW'] + importArray['fuels']['RW']) / 100)/10
        totalExt = 0
        total = totalInt+totalExt
        # F1
        draw_filled_polygon(surface, WHITE, [(b1x, b1y), (b1x+b1w, b1y), (b1x+b1w, b1y+b1h), (b1x, b1y+b1h)], 1, F1P, fuelColor)
        render_multicolor_text(surface, fuelFont, [[(str(importArray['fuels']['F1']), WHITE)]], (b1x+(b1w/2), b2y), 0, True, "B", None)
        render_multicolor_text(surface, labelFont, [[("F1", WHITE)]], (b1x-labelOffset, b1y), 0, True, "TL", None)
        
        # F1I
        draw_filled_polygon(surface, WHITE, [(b1x, b2y), (b1x+b1w, b2y), (b1x+b1w, b2y+b2h), (b1x, b2y+b2h)], 1, F1IP, fuelColor)
        render_multicolor_text(surface, fuelFont, [[(str(importArray['fuels']['F1I']), WHITE)]], (b1x+(b1w/2), b2y+(b2h/2)), 0, True, "C", None)

        # F2 L/R
        draw_filled_polygon(surface, WHITE, [(b1x, b3y), (b1x+b3w-b3b, b3y), (b1x+b3w-b3b, b3y+b2h), (b1x, b3y+b2h)], 1, F2LP, fuelColor)
        draw_filled_polygon(surface, WHITE, [(b1x+b3w+b3b, b3y), (b1x+b1w, b3y), (b1x+b1w, b3y+b2h), (b1x+b3w+b3b, b3y+b2h)], 1, F2RP, fuelColor)
        render_multicolor_text(surface, fuelFont, [[(str(importArray['fuels']['F2L']), WHITE)]], (b1x+b3w-b3b, b3y+(b2h/2)), 0, True, "R", None)
        render_multicolor_text(surface, fuelFont, [[(str(importArray['fuels']['F2R']), WHITE)]], (b1x+b1w, b3y+(b2h/2)), 0, True, "R", None)

        render_multicolor_text(surface, labelFont, [[("F2L", WHITE)]], (b1x-labelOffset, b3y), 0, True, "TL", None)
        render_multicolor_text(surface, labelFont, [[("F2R", WHITE)]], (b1x+b3w+b3b-labelOffset, b3y), 0, True, "TL", None)

        # F3 L/R
        draw_filled_polygon(surface, WHITE, [(b1x, b4y), (b1x+b3w-b3b, b4y), (b1x+b3w-b3b, b4y+b2h), (b1x, b4y+b2h)], 1, F3LP, fuelColor)
        draw_filled_polygon(surface, WHITE, [(b1x+b3w+b3b, b4y), (b1x+b1w, b4y), (b1x+b1w, b4y+b2h), (b1x+b3w+b3b, b4y+b2h)], 1, F3RP, fuelColor)
        render_multicolor_text(surface, fuelFont, [[(str(importArray['fuels']['F3L']), WHITE)]], (b1x+b3w-b3b, b4y+(b2h/2)), 0, True, "R", None)
        render_multicolor_text(surface, fuelFont, [[(str(importArray['fuels']['F3R']), WHITE)]], (b1x+b1w, b4y+(b2h/2)), 0, True, "R", None)

        render_multicolor_text(surface, labelFont, [[("F3L", WHITE)]], (b1x-labelOffset, b4y), 0, True, "TL", None)
        render_multicolor_text(surface, labelFont, [[("F3R", WHITE)]], (b1x+b3w+b3b-labelOffset, b4y), 0, True, "TL", None)

        # F4 L/R
        draw_filled_polygon(surface, WHITE, [(b1x, b5y), (b1x+b3w-b3b, b5y), (b1x+b3w-b3b, b5y+b2h), (b1x, b5y+b2h)], 1, F4LP, fuelColor)
        draw_filled_polygon(surface, WHITE, [(b1x+b3w+b3b, b5y), (b1x+b1w, b5y), (b1x+b1w, b5y+b2h), (b1x+b3w+b3b, b5y+b2h)], 1, F4RP, fuelColor)
        render_multicolor_text(surface, fuelFont, [[(str(importArray['fuels']['F4L']), WHITE)]], (b1x+b3w-b3b, b5y+(b2h/2)), 0, True, "R", None)
        render_multicolor_text(surface, fuelFont, [[(str(importArray['fuels']['F4R']), WHITE)]], (b1x+b1w, b5y+(b2h/2)), 0, True, "R", None)

        render_multicolor_text(surface, labelFont, [[("F4L", WHITE)]], (b1x-labelOffset, b5y), 0, True, "TL", None)
        render_multicolor_text(surface, labelFont, [[("F4R", WHITE)]], (b1x+b3w+b3b-labelOffset, b5y), 0, True, "TL", None)

        # F5 L/R
        draw_filled_polygon(surface, WHITE, [(b1x, b6y), (b1x+b3w-b6b, b6y), (b1x+b3w-b6b, b6y+b2h), (b1x, b6y+b2h)], 1, F5LP, fuelColor)
        draw_filled_polygon(surface, WHITE, [(b1x+b3w+b6b, b6y), (b1x+b1w, b6y), (b1x+b1w, b6y+b2h), (b1x+b3w+b6b, b6y+b2h)], 1, F5RP, fuelColor)
        render_multicolor_text(surface, labelFont, [[(str(importArray['fuels']['F5L']), WHITE)]], (b1x+b3w-b6b+labelOffset, b6y+(b2h/2)), 0, True, "R", None)
        render_multicolor_text(surface, labelFont, [[(str(importArray['fuels']['F5R']), WHITE)]], (b1x+b3w+b6b-labelOffset, b6y+(b2h/2)), 0, True, "L", None)

        render_multicolor_text(surface, labelFont, [[("F5L", WHITE)]], (b1x-labelOffset, b6y), 0, True, "TL", None)
        render_multicolor_text(surface, labelFont, [[("F5R", WHITE)]], (b1x+b3w+b6b-labelOffset, b6y), 0, True, "TL", None)

        # LW/RW
        draw_filled_polygon(surface, WHITE, [(b1x-b7b, b7y), (b1x-b7b, b7y2), (b1x-b7b-b7tri, b7y2-b7tri), (b1x-b7b-b7tri, b7y+b7tri)], 1, LWP, fuelColor)
        draw_filled_polygon(surface, WHITE, [(b1x+b1w+b7b, b7y), (b1x+b1w+b7b, b7y2), (b1x+b1w+b7b+b7tri, b7y2-b7tri), (b1x+b1w+b7b+b7tri, b7y+b7tri)], 1, RWP, fuelColor)
        render_multicolor_text(surface, fuelFont, [[(str(importArray['fuels']['LW']), WHITE)]], (b1x-b7b+labelOffset, b5y), 0, True, "R", None)
        render_multicolor_text(surface, fuelFont, [[(str(importArray['fuels']['RW']), WHITE)]], (b1x+b1w+b7b-labelOffset, b5y), 0, True, "L", None)
        
        render_multicolor_text(surface, labelFont, [[("LW", WHITE)]], (b1x-b7b, b7y+b7tri), 0, True, "R", None)
        render_multicolor_text(surface, labelFont, [[("RW", WHITE)]], (b1x+b1w+b7b, b7y+b7tri), 0, True, "L", None)
        
        border = 20
        grid, size = get_portal_grid(portal['loc'])
        boxWidth, boxHeight, boxColumns, boxRows = grid
        dumpPosition, cent = get_portal_grid_box(portal['loc'], (0, 2.5))
        MFSOVPosition, cent = get_portal_grid_box(portal['loc'], (0, 6.3))
        REFUELPosition, cent = get_portal_grid_box(portal['loc'], (boxColumns-1, 6.3))
        dx, dy, dw, dh = dumpPosition
        mx, my, mw, mh = MFSOVPosition
        rx, ry, rw, rh = REFUELPosition
        dx = dx + border
        mx = mx + border
        dw = dw*0.8
        dh = dh*0.7
        textoffset = border/2
        
        draw_striped_square_border(surface, dx, dy, dw, dh, 10, (YELLOW, 10), (BLACK, 10), "right")
        
        if portal["EXP"]>0:
            draw_striped_square_border(surface, mx, my, dw, dh, 10, (YELLOW, 10), (BLACK, 10), "right")
            draw_striped_square_border(surface, rx, ry, dw, dh, 10, (YELLOW, 10), (BLACK, 10), "right")
            render_multicolor_text(surface, PCDThin, [[("MFSOV", CYAN)]], (mx, my + (mh/2)-textoffset), 0, True, "L", None)
            render_multicolor_text(surface, PCDThin, [[("EMER", CYAN)], [("REFUEL", CYAN)]], (rx + (rw/2)-textoffset, ry + (rh/2)-textoffset), 0, True, "C", None)
        
        render_multicolor_text(surface, PCDThin, [[("DUMP", CYAN)]], (dx, dy + (dh/2)), 0, True, "L", None)

        pygame.draw.rect(surface, GREEN, pygame.rect.Rect(b1x+b1w+border, b1y+border, dw*1.2, b1h*1.75), 1)
        render_multicolor_text(surface, PCDThin, [[(f"TOT:{total}", GREEN)], [(f"INT:{totalInt}", GREEN)], [(f"EXT:{totalExt}", GREEN)]], (b1x+b1w+border-5, b1y+border+((b1h*1.75)/2)), 5, True, "L", None)
        
        render_multicolor_text(surface, PCDThin, [[("REFUEL", CYAN)]], (b1x, y), 0, True, "TR", None)
        render_multicolor_text(surface, PCDThin, [[("PRE", CYAN)], [("CONTACT", CYAN)]], (b1x+b1w, y), 0, True, "TL", None)
        
        render_multicolor_text(surface, PCDThin, [[("LRP>", CYAN)]], (x, 225), 0, True, "L", None)
        render_multicolor_text(surface, PCDThin, [[("GW:  38.8", CYAN)], [("INLET: 43", GREEN)], [("FEED:  45", GREEN)]], (x+50, 225), 1, True, "L", None)
        
        render_multicolor_text(surface, PCDThin, [[("DUMPCO", CYAN)], [(" 1.0", CYAN)]], (x, y+400), 1, True, "L")
        render_multicolor_text(surface, PCDThin, [[("JOKER", CYAN)], [("5.0 ", CYAN)]], (x+w, dy+(dh/2)), 1, True, "R")
        render_multicolor_text(surface, PCDThin, [[("BINGO", CYAN)], [("2.0 ", CYAN)]], (x+w, dy+(dh*2)), 1, True, "R")
        
        render_multicolor_text(surface, PCDThin, [[("FUEL", GRAY)], [("XFER", GRAY)], [("LEFT", GRAY)]], (x, y+475), 1, True, "L")
        render_multicolor_text(surface, PCDThin, [[("FUEL", GRAY)], [("XFER", GRAY)], [("RIGHT", GRAY)]], (x+w, y+475), 1, True, "R")
        
        renderNavButton(surface, cx, y, portal["CurrentPage"]['name'])

def render_DIM(surface, x, y, w, h, portal, sub = 0):
    inta, bool, sub = getSubportal(sub)
    if bool and portal['hidden'] == False and portal['EXP'] == 0:
        name = portal['subs'][sub]['name']
        px, py, pw, ph = get_subportal_position(portal['loc'], sub)
        pygame.draw.rect(surface, DARK_GRAY, pygame.Rect(px, py, pw, ph))
        render_multicolor_text(surface, PCDLarge, [[(f"DISP {name.upper()}", GREEN)]], (px + (pw/2), py + (ph/2)), 0, True)
    elif bool == False and portal['hidden'] == False:
        BLOffset = (GRID_Y) * min(portal['EXP'], 1)
        pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
        name = portal['CurrentPage']['name']
        pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
        cx, cy = centered
        bfont = pygame.font.Font("PCD.ttf", 20)
        ufont = pygame.font.Font("PCD.ttf", 20)
        tfont = pygame.font.Font("PCD.ttf", 18)
        ufont.set_underline(True)
        ufont.set_bold(True)
        bfont.set_bold(True)
        tfont.set_bold(True)

        render_multicolor_text(surface, bfont, [[("INBOX", WHITE)], [("OUTBOX", CYAN)]], (x+115, y), 1, True, "TL")
        pygame.draw.rect(surface, WHITE, pygame.rect.Rect(x+120, y+3, 57, 20), 2)
        render_multicolor_text(surface, bfont, [[("PRESERVE", GRAY)]], (x+205, y), 1, True, "TL")
        render_multicolor_text(surface, bfont, [[("ASGN", GRAY)], [("SELF", GRAY)]], (x+w, y+150), 1, True, "R")
        render_multicolor_text(surface, bfont, [[("FAC-A>", CYAN)]], (x+w, y+300), 0, True, "R")
        render_multicolor_text(surface, bfont, [[("REPORT>", CYAN)]], (x+w, y+375), 0, True, "R")
        render_multicolor_text(surface, bfont, [[("IFDLZ>", CYAN)]], (x+w, y+450), 0, True, "R")
        render_multicolor_text(surface, bfont, [[("DELETE", GRAY)]], (x, y+450), 0, True, "L")
        render_multicolor_text(surface, bfont, [[("VIEW>", CYAN)]], (x, y+375), 0, True, "L")
        
        t1x = x+10
        t1y = y+150
        t2x = x+10
        t2y = t1y+40
        tscale = 20
        pygame.draw.polygon(surface, CYAN, [(t1x, t1y), (t1x+tscale, t1y), (t1x+(tscale/2), t1y-tscale)], 0)
        pygame.draw.polygon(surface, CYAN, [(t2x, t2y), (t2x+tscale, t2y), (t2x+(tscale/2), t2y+tscale)], 0)
        
        render_multicolor_text(surface, ufont, [
            [("MSN ASSIGN - OWN - CTOTAL", GREEN)], 
            [("MSN ASSIGN - DTM - CTOTAL", GREEN)],
            [("MSN ASSIGN - FLT - CTOTAL", GREEN)],
            [("IMAGES", GREEN)],
            [("GPS DATA", GREEN)]
            ], (x+115, t1y-tscale-10), 25, True, "TL")
        
        render_multicolor_text(surface, tfont, [
            [("LAT 34.9141821640", WHITE)],
            [("LON -117.8667496240", WHITE)],
            [("WAYPOINT 1", WHITE)],
            [("BRNG 254 DIST 13", WHITE)],
            [("ETA  12 : 22 : 30", WHITE)]
            ], (x+115, y+330), 1, True, "TL")

        setButton(portal['loc'], 1, pos, lambda: set_portal_render(portal['loc'], partial(render_menu, portal["CurrentPage"]['name'])))
        renderNavButton(surface, cx, y, portal["CurrentPage"]['name'])

def render_CKLST(surface, x, y, w, h, portal, sub = 0):
    inta, bool, sub = getSubportal(sub)
    if bool and portal['hidden'] == False and portal['EXP'] == 0:
        name = portal['subs'][sub]['name']
        px, py, pw, ph = get_subportal_position(portal['loc'], sub)
        pygame.draw.rect(surface, DARK_GRAY, pygame.Rect(px, py, pw, ph))
        render_multicolor_text(surface, PCDLarge, [[(f"DISP {name.upper()}", GREEN)]], (px + (pw/2), py + (ph/2)), 0, True)
    elif bool == False and portal['hidden'] == False:
        BLOffset = (GRID_Y) * min(portal['EXP'], 1)
        pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
        name = portal['CurrentPage']['name']
        pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
        cx, cy = centered
        
        lineSpace = 75
        if portal["EXP"]>0:
            lineSpace = 95
        render_multicolor_text(surface, PCDFont, [[("COCKPIT", CYAN)], [("CHECK", CYAN)]], (x, y+100+(lineSpace*0)), 1, True, "L")
        render_multicolor_text(surface, PCDFont, [[("ENGINE", CYAN)], [("START", CYAN)]], (x, y+100+(lineSpace*1)), 1, True, "L")
        render_multicolor_text(surface, PCDFont, [[("TAXI", CYAN)], [("CHECK", CYAN)]], (x, y+100+(lineSpace*2)), 1, True, "L")
        render_multicolor_text(surface, PCDFont, [[("TAKE", CYAN)], [("OFF", CYAN)]], (x, y+100+(lineSpace*3)), 1, True, "L")
        render_multicolor_text(surface, PCDFont, [[("LANDING", CYAN)]], (x, y+100+(lineSpace*4)), 1, True, "L")
        render_multicolor_text(surface, PCDFont, [[("POST", CYAN)], [("LANDING", CYAN)]], (x, y+100+(lineSpace*5)), 1, True, "L")
        
        cockpit = [
            ("SAFETY PINS", "REMOVED"),
            ("ICC 1 2 and 3 switches", "ON"),
            ("CABIN PRESSURE knob", "NORM"),
            ("BATT switch", "OFF"),
            ("IPP switch", "AUTO"),
            ("HMD", "OFF"),
            ("Harness and restraint lines", "CONNECT"),
            ("ENGINE switch", "OFF"),
            ("DEFOG", "AS REQUIRED"),
            ("Throttle", "IDLE"),
            ("BOS", "NORM"),
            ("LAND/TAXI LIGHT", "OFF"),
            ("PARKING BRAKE", "ON"),
            ("LDG GEAR handle", "DN"),
            ("JETTISON", "SEL"),
            ("PCD", "AS REQUIRED"),
            ("MASTER ARM", "OFF"),
            ("AUTO RECOVERY", "NORM"),
            ("AIRCRAFT ZEROIZE", "NORM")
        ]
        stringLength = 36
        checklistLines = []
        for index, (s1, s2) in enumerate(cockpit):
            # Calculate how many dots are needed
            dots_needed = stringLength - len(s1) - len(s2)
            
            # Clamp to at least 0 (in case stringLength is too short)
            dots_needed = max(0, dots_needed)
            
            result_string = f"{s1}{'.' * dots_needed}{s2}"
            checklistLines.append([(result_string, WHITE)])
            
        render_multicolor_text(surface, pygame.font.Font("PCD.ttf", 20), checklistLines, (x+100, y+85), 1, True, "TL")
        render_multicolor_text(surface, pygame.font.Font("PCD.ttf", 20), [[("COCKPIT CHECK", GREEN)]], (x+100, y+50), 0, True, "TL")
        
        setButton(portal['loc'], 1, pos, lambda: set_portal_render(portal['loc'], partial(render_menu, portal["CurrentPage"]['name'])))
        renderNavButton(surface, cx, y, portal["CurrentPage"]['name'])

def render_ENG(surface, x, y, w, h, portal, sub = 0):
    inta, bool, sub = getSubportal(sub)
    if bool and portal['hidden'] == False and portal['EXP'] == 0:
        name = portal['subs'][sub]['name']
        px, py, pw, ph = get_subportal_position(portal['loc'], sub)
        pygame.draw.rect(surface, DARK_GRAY, pygame.Rect(px, py, pw, ph))
        render_multicolor_text(surface, PCDLarge, [[(f"DISP {name.upper()}", GREEN)]], (px + (pw/2), py + (ph/2)), 0, True)
    elif bool == False and portal['hidden'] == False:
        BLOffset = (GRID_Y) * min(portal['EXP'], 1)
        pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
        name = portal['CurrentPage']['name']
        pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
        cx, cy = centered
        
        def lerp(start, end, t):
            """
            Linearly interpolates from start to end by t.
            t should be between 0 and 1.
            """
            return start + (end - start) * t

        
        maxThrust = 100
        currentThrust = importArray['engine']['THRUST']
        maxEGT = 1100
        currentEGT = importArray['engine']['EGT']
        maxNozzle = 105
        currentNozzle = importArray['engine']['NOZZLE']
        maxN1RPM = 100
        currentN1RPM = importArray['engine']['N1RPM']
        maxN2RPM = 100
        currentN2RPM = importArray['engine']['N2RPM']
        maxOil = 150
        currentOil = importArray['engine']['OIL']
        
        maxAngle = 180
        minAngle = -90
        
        thrustAngle = lerp(minAngle, maxAngle, currentThrust/maxThrust)
        egtAngle = lerp(minAngle, maxAngle, currentEGT/maxEGT)
        nozzleAngle = lerp(minAngle, maxAngle, currentNozzle/maxNozzle)
        n1Angle = lerp(minAngle, maxAngle, currentN1RPM/maxN1RPM)
        n2Angle = lerp(minAngle, maxAngle, currentN2RPM/maxN2RPM)
        oilAngle = lerp(minAngle, maxAngle, currentOil/maxOil)
        
        gaugeSize = 35
        gaugeBorder = 40
        gaugeY = 120
        if portal["EXP"]==2:
            gaugeSize=50
        
        #Thrust Gauge
        draw_gauge_circle(surface, x+((w/4)*1)-gaugeBorder, y+gaugeY, gaugeSize, GREEN, 2, None, str(currentThrust), PCDFont, GREEN, 10, GREEN, 20, RED, 0)
        render_multicolor_text(surface, PCDFont, [[("THRUST", GREEN)]], (x+((w/4)*1)-gaugeBorder, y+gaugeY+gaugeSize), 0, True, "T")
        draw_arrow(surface, (x+((w/4)*1)-gaugeBorder, y+gaugeY), thrustAngle, gaugeSize*0.9, 1, 7, 7, (GREEN, 0), (GREEN, 0))
        
        #EGT Gauge
        draw_gauge_circle(surface, x+((w/2)), y+gaugeY, gaugeSize, GREEN, 2, None, str(currentEGT), PCDFont, GREEN, 10, GREEN, 20, RED, 0)
        render_multicolor_text(surface, PCDFont, [[("EGT", GREEN)]], (x+((w/2)), y+gaugeY+gaugeSize), 0, True, "T")
        draw_arrow(surface, (x+((w/2)), y+gaugeY), egtAngle, gaugeSize*0.9, 1, 7, 7, (GREEN, 0), (GREEN, 0))

        #Nozzle Gauge
        draw_gauge_circle(surface, x+((w/4)*3)+gaugeBorder, y+gaugeY, gaugeSize, GREEN, 2, None, str(currentNozzle), PCDFont, GREEN, 10, GREEN, 0)
        render_multicolor_text(surface, PCDFont, [[("NOZZLE", GREEN)]], (x+((w/4)*3)+gaugeBorder, y+gaugeY+gaugeSize), 0, True, "T")
        draw_arrow(surface, (x+((w/4)*3)+gaugeBorder, y+gaugeY), nozzleAngle, gaugeSize*0.9, 1, 7, 7, (GREEN, 0), (GREEN, 0))

        #N1 Gauge
        draw_gauge_circle(surface, x+((w/4)*1)-gaugeBorder, y+h-gaugeY, gaugeSize, GREEN, 2, None, str(currentN1RPM), PCDFont, GREEN, 10, GREEN, 20, RED, 0)
        render_multicolor_text(surface, PCDFont, [[("N1 RPM", GREEN)]], (x+((w/4)*1)-gaugeBorder, y+h-gaugeY+gaugeSize), 0, True, "T")
        draw_arrow(surface, (x+((w/4)*1)-gaugeBorder, y+h-gaugeY), n1Angle, gaugeSize*0.9, 1, 7, 7, (GREEN, 0), (GREEN, 0))
        
        #N2 Gauge
        draw_gauge_circle(surface, x+((w/2)), y+h-gaugeY, gaugeSize, GREEN, 2, None, str(currentN2RPM), PCDFont, GREEN, 10, GREEN, 20, RED, 0)
        render_multicolor_text(surface, PCDFont, [[("N2 RPM", GREEN)]], (x+((w/2)), y+h-gaugeY+gaugeSize), 0, True, "T")
        draw_arrow(surface, (x+((w/2)), y+h-gaugeY), n2Angle, gaugeSize*0.9, 1, 7, 7, (GREEN, 0), (GREEN, 0))

        #Oil Gauge
        draw_gauge_circle(surface, x+((w/4)*3)+gaugeBorder, y+h-gaugeY, gaugeSize, GREEN, 2, None, str(currentOil), PCDFont, GREEN, 10, GREEN, 20, RED, 0)
        render_multicolor_text(surface, PCDFont, [[("OIL", GREEN)]], (x+((w/4)*3)+gaugeBorder, y+h-gaugeY+gaugeSize), 0, True, "T")
        draw_arrow(surface, (x+((w/4)*3)+gaugeBorder, y+h-gaugeY), oilAngle, gaugeSize*0.9, 1, 7, 7, (GREEN, 0), (GREEN, 0))

        #Middle Numbers
        textOffset = 5
        render_multicolor_text(surface, PCDFont, [[(f"FF {importArray['engine']['FF']}", GREEN)]], (x+((w/4)*1)-gaugeBorder, y+(h/2)+textOffset), 0, True, "C")
        render_multicolor_text(surface, PCDFont, [[(f"HYDA {importArray['engine']['HYDA']}", GREEN)]], (x+((w/4)*2), y+(h/2)+textOffset), 0, True, "C")
        render_multicolor_text(surface, PCDFont, [[(f"HYDB {importArray['engine']['HYDB']}", GREEN)]], (x+((w/4)*3)+gaugeBorder, y+(h/2)+textOffset), 0, True, "C")


        setButton(portal['loc'], 1, pos, lambda: set_portal_render(portal['loc'], partial(render_menu, portal["CurrentPage"]['name'])))
        renderNavButton(surface, cx, y, portal["CurrentPage"]['name'])
        
def render_blank(surface, x, y, w, h, portal, sub=0):
    inta, bool, sub = getSubportal(sub)
    if bool and portal['hidden'] == False and portal['EXP'] == 0:
        name = portal['subs'][sub]['name']
        px, py, pw, ph = get_subportal_position(portal['loc'], sub)
        #render_multicolor_text(surface, PCDLarge, [[(f"DISP {name.upper()}", GREEN)]], (px + (pw/2), py + (ph/2)), 0, True)
    elif bool == False and portal['hidden'] == False:
        name = portal['CurrentPage']['name']
        pos, centered = get_portal_grid_box(portal['loc'], (0, 0))
        px, py, pw, ph = pos
        cx, cy = centered
        renderNavButton(surface, cx, y, "BNK")
        setButton(portal['loc'], 1, pos, lambda: set_portal_render(portal['loc'], partial(render_menu, "NAV")))
        #render_multicolor_text(surface, PCDMassive, [[(f"DISP {name.upper()}", GREEN)]], (x + (w/2), y + (h/2)), 0, True)
    
def draw_dashed_line(surface, color, start_pos, end_pos, dash_length=10, gap_length=5, width=1):
    """
    Draws a dashed or dotted line between two points.

    Parameters:
        surface     : The pygame surface to draw on.
        color       : The color of the line (RGB or RGBA tuple).
        start_pos   : Starting position (x, y).
        end_pos     : Ending position (x, y).
        dash_length : Length of each dash (or dot).
        gap_length  : Length of the gap between dashes.
        width       : Thickness of the line.
    """

    # Vector from start to end
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    length = math.hypot(dx, dy)

    # Normalize direction vector
    if length == 0:
        return  # No line to draw
    dx /= length
    dy /= length

    # Position pointer
    x, y = start_pos
    drawn = 0

    while drawn < length:
        dash_end_x = x + dx * min(dash_length, length - drawn)
        dash_end_y = y + dy * min(dash_length, length - drawn)
        pygame.draw.line(surface, color, (x, y), (dash_end_x, dash_end_y), width)

        # Move to next dash start
        x += dx * (dash_length + gap_length)
        y += dy * (dash_length + gap_length)
        drawn += dash_length + gap_length
    
def draw_triangle(
    surface: pygame.Surface,
    position: Tuple[float, float],
    color: Tuple[int, int, int],
    scale: float = 50.0,
    rotation: float = 0.0,
    width: int = 0
) -> None:
    """
    Draws an equilateral triangle centered at `position` on the given surface.

    Args:
        surface:   The pygame Surface to draw on.
        position:  (x, y) coordinates of the triangle's center.
        color:     RGB color tuple.
        scale:     Side length of the triangle, in pixels.
        rotation:  Rotation angle in degrees, counter‑clockwise.
        width:     Line thickness. 0 (default) means filled.
    """
    cx, cy = position
    L = scale
    # height of equilateral triangle
    h = math.sqrt(3) / 2 * L

    # define vertices relative to triangle center (centroid at origin)
    verts: List[Tuple[float, float]] = [
        ( 0,      -2/6 * h),  # top vertex
        (-L/2,   +1/3 * h),   # bottom‑left
        (+L/2,   +1/3 * h),   # bottom‑right
    ]

    # precompute rotation
    theta = math.radians(rotation)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # rotate and translate vertices
    transformed: List[Tuple[float, float]] = []
    for x, y in verts:
        xr = x * cos_t - y * sin_t
        yr = x * sin_t + y * cos_t
        transformed.append((cx + xr, cy + yr))

    # draw the triangle
    return pygame.draw.polygon(surface, color, transformed, width)
    
def rectToDim(rect):
    return (rect.x, rect.y, rect.w, rect.h)
    
def draw_borders(surface):
    for i in range(0, 4):
        #pygame.draw.polygon(surface, GRAY, [(100, 100), (200, 100), (150, 200)], 2)
        trueBorders = False
        x, y, w, h, portal = get_portal_positions(i, portals[i]['rendering'] is not portals[i]['CurrentPage']['render'])
        if portal["hidden"] == False:
            tabWidth = (GRID_X*4)/5
            tabTop = tabWidth-20
            tabHeight = 40
            LR = 1 - (portal['loc'] % 2)
            
            if portal['rendering'] == portal['CurrentPage']['render']:
                if portal["EXP"] == 0:
                    # ADD LOGIC HERE LATER FOR BUTTONS TO SWITCH THE MAIN PAGE TO THE SUBPORTAL AND VISE VERSA
                    setButton(5, 100 + (i*2), (x, y+h, GRID_X*2, GRID_Y*2), partial(swap_portal_subportal, portal, 0))
                    setButton(5, 101 + (i*2), (x+(w/2), y+h, GRID_X*2, GRID_Y*2), partial(swap_portal_subportal, portal, 1))
                    for k in range(2):
                        name = portal['subs'][k]['name']
                        cx = x + (w/4) + (GRID_Y*3*k)
                        cy = y + h + (GRID_Y*2)
                        render_multicolor_text(surface, PCDFont, [[(name.upper(), CYAN)]], (cx, cy), 1, True, "B", BLACK)
                    pygame.draw.rect(surface, WHITE, pygame.Rect(x, y+h, GRID_X*2, GRID_Y*2), width=1)
                    pygame.draw.rect(surface, WHITE, pygame.Rect(x+(w/2), y+h, GRID_X*2, GRID_Y*2), width=1)
                    
                if portal['EXP'] == 1:
                    tabStartX = (w/2) - ((tabWidth*3))/2
                    for j in range(1, 3):
                        tx = x + tabStartX + (tabWidth*j)
                        rect = pygame.draw.lines(surface, CYAN, False, [(tx, y+h), (tx + ((tabWidth-tabTop)/2), y+h-tabHeight), (tx + ((tabWidth-tabTop)/2)+tabTop, y+h-tabHeight), (tx+tabWidth, y+h)], 2)
                        rx, ry, rw, rh = rect.x, rect.y, rect.w, rect.h
                        setButton(5, 100 + (j-1) + (i*2), (rx, ry, rw, rh), partial(swap_portal_subportal, portal, j-1))
                        ctx = tx+(tabWidth/2)
                        name = portal['subs'][j-1]['name'].upper()
                        render_multicolor_text(surface, PCDFont, [[(name, CYAN)]], (ctx, y+h-(tabHeight/2)), 0, True)
                        
                if portal['EXP'] == 2:
                    
                    tabStartX = ((w/4)*max(LR*3, 1)) - ((tabWidth*3))/2
                    for j in range(0, 3):
                        tx = x + tabStartX + (tabWidth*j)
                        ctx = tx+(tabWidth/2)
                        rect = pygame.draw.lines(surface, CYAN, False, [(tx, y+h), (tx + ((tabWidth-tabTop)/2), y+h-tabHeight), (tx + ((tabWidth-tabTop)/2)+tabTop, y+h-tabHeight), (tx+tabWidth, y+h)], 2)
                        rx, ry, rw, rh = rect.x, rect.y, rect.w, rect.h
                        name = ""
                        if j == 0:
                            if portal['loc'] == 0 or portal['loc'] == 2:
                                name = portals[portal['loc']+1]['CurrentPage']['name'].upper()
                                setButton(5, 200 + (i*2), (rx, ry, rw, rh), partial(swap_portals, portal, portals[portal['loc']+1]))
                            else:
                                name = portals[portal['loc']-1]['CurrentPage']['name'].upper()
                                setButton(5, 200 + (i*2), (rx, ry, rw, rh), partial(swap_portals, portal, portals[portal['loc']-1]))
                        else:
                            if portal['loc'] == 0 or portal['loc'] == 2:
                                name = portals[portal['loc']+1]['subs'][j-1]['name'].upper()
                                setButton(5, 100 + (j-1) + ((i+1)*2), (rx, ry, rw, rh), partial(swap_portal_altsubportal, portal, portals[portal['loc']+1], j-1))
                            else:
                                name = portals[portal['loc']-1]['subs'][j-1]['name'].upper()
                                setButton(5, 100 + (j-1) + ((i-1)*2), (rx, ry, rw, rh), partial(swap_portal_altsubportal, portal, portals[portal['loc']-1], j-1))
                        render_multicolor_text(surface, PCDFont, [[(name, CYAN)]], (ctx, y+h-(tabHeight/2)), 0, True)

                        
                    tabStartX = ((w/4)*max((1 - LR)*3, 1)) - ((tabWidth*3))/2
                    for j in range(1, 3):
                        tx = x + tabStartX + (tabWidth*j)
                        ctx = tx+(tabWidth/2)
                        rect = pygame.draw.lines(surface, CYAN, False, [(tx, y+h), (tx + ((tabWidth-tabTop)/2), y+h-tabHeight), (tx + ((tabWidth-tabTop)/2)+tabTop, y+h-tabHeight), (tx+tabWidth, y+h)], 2)
                        rx, ry, rw, rh = rect.x, rect.y, rect.w, rect.h
                        setButton(5, 100 + (j-1) + (i*2), (rx, ry, rw, rh), partial(swap_portal_subportal, portal, j-1))
                        name = portal['subs'][j-1]['name'].upper()
                        render_multicolor_text(surface, PCDFont, [[(name, CYAN)]], (ctx, y+h-(tabHeight/2)), 0, True)
            else:
                setButton(5, 100 + (i*2), (0, 0, 0, 0), partial(swap_portal_subportal, portal, 0))
                setButton(5, 101 + (i*2), (0, 0, 0, 0), partial(swap_portal_subportal, portal, 1))
            grid, pos = get_portal_grid(i, True)
            boxWidth, boxHeight, boxColumns, boxRows = grid
            
            blPos, blCent = get_portal_grid_box(i, (0, boxRows-1), True)
            brPos, brCent = get_portal_grid_box(i, (boxColumns-1, boxRows-1), True)
            U = 22
            D = 22
            L = -32
            R = 32
            
            VL = -20
            VR = 20
            
            HD = 10
            
            blX, blY = blCent
            brX, brY = brCent
            
            blU = (blX+VL, blY+U)
            blD = (blX+VL, blY+D)
            blL = (blX+L, blY+HD)
            blR = (blX+R, blY+HD)
            
            brU = (brX+VR, brY+U)
            brD = (brX+VR, brY+D)
            brL = (brX+L, brY+HD)
            brR = (brX+R, brY+HD)
            triSize = 50
            triWidth = 2
            
            if portal['hidden']:
                setButton(6, (portal['loc']*2)+0, (0, 0, 0, 0), lambda: print())
                setButton(6, (portal['loc']*2)+1, (0, 0, 0, 0), lambda: print())
            elif portal['rendering'] != portal['CurrentPage']['render']:
                setButton(6, (portal['loc']*2)+0, (0, 0, 0, 0), lambda: print())
                setButton(6, (portal['loc']*2)+1, (0, 0, 0, 0), lambda: print())
            else:
                TRICOLOR = GREEN
                if portal['loc'] == 0 or portal['loc'] == 2:
                    if portal['EXP'] == 0:
                        setButton(6, (portal['loc']*2)+0, rectToDim(draw_triangle(surface, blD, TRICOLOR, triSize, 180, triWidth)), partial(set_portal_exp, portal['loc'], 1))
                        setButton(6, (portal['loc']*2)+1, rectToDim(draw_triangle(surface, brR, TRICOLOR, triSize, 90, triWidth)), partial(set_portal_exp, portal['loc'], 2))
                    elif portal['EXP'] == 1:
                        setButton(6, (portal['loc']*2)+0, rectToDim(draw_triangle(surface, blU, TRICOLOR, triSize, 0, triWidth)), partial(set_portal_exp, portal['loc'], 0))
                        setButton(6, (portal['loc']*2)+1, rectToDim(draw_triangle(surface, brR, TRICOLOR, triSize, 90, triWidth)), partial(set_portal_exp, portal['loc'], 2))         
                    elif portal['EXP'] == 2:
                        setButton(6, (portal['loc']*2)+0, rectToDim(draw_triangle(surface, blU, TRICOLOR, triSize, 0, triWidth)), partial(set_portal_exp, portal['loc'], 0))
                        setButton(6, (portal['loc']*2)+1, rectToDim(draw_triangle(surface, brU, TRICOLOR, triSize, 0, triWidth)), partial(set_portal_exp, portal['loc'], 0))
                else:
                    if portal['EXP'] == 0:
                        setButton(6, (portal['loc']*2)+0, rectToDim(draw_triangle(surface, blL, TRICOLOR, triSize, -90, triWidth)), partial(set_portal_exp, portal['loc'], 2))
                        setButton(6, (portal['loc']*2)+1, rectToDim(draw_triangle(surface, brD, TRICOLOR, triSize, 180, triWidth)), partial(set_portal_exp, portal['loc'], 1))
                    elif portal['EXP'] == 1:
                        setButton(6, (portal['loc']*2)+0, rectToDim(draw_triangle(surface, blL, TRICOLOR, triSize, -90, triWidth)), partial(set_portal_exp, portal['loc'], 2))
                        setButton(6, (portal['loc']*2)+1, rectToDim(draw_triangle(surface, brU, TRICOLOR, triSize, 0, triWidth)), partial(set_portal_exp, portal['loc'], 0))
                    elif portal['EXP'] == 2:
                        setButton(6, (portal['loc']*2)+0, rectToDim(draw_triangle(surface, blU, TRICOLOR, triSize, 0, triWidth)), partial(set_portal_exp, portal['loc'], 0))
                        setButton(6, (portal['loc']*2)+1, rectToDim(draw_triangle(surface, brU, TRICOLOR, triSize, 0, triWidth)), partial(set_portal_exp, portal['loc'], 0))
                
            pygame.draw.rect(surface, WHITE, pygame.Rect(x, y, w, h), width=1)

def render_ICAWS(surface, position):
    x, y, w, h = position
    pygame.draw.rect(surface, BLACK, pygame.Rect(x, y, w,h))
    ICAWS_font = pygame.font.Font("PCD.ttf", 15)
    ICAWS_thin = pygame.font.Font("PCD.ttf", 20)
    ICAWS2 = pygame.font.Font("PCD.ttf", 20)
    ICAWS2.set_bold(True)
    ICAWS_font.set_bold(True)
    #render_multicolor_text(surface, ICAWS_font, [[("R", RED)], [("G", GREEN)], [("B", BLUE)]], (x + 20, y + (h/2)), 4, True)
    #render_multicolor_text(surface, ICAWS_font, [[("R", RED)], [("G", GREEN)], [("B", BLUE)]], (x + w - 20, y + (h/2)), 4, True)
    pygame.draw.line(surface, WHITE, (x, y), (x+w, y), 1)
    
    # 0 Draws white squares on either side
    wall_width = GRID_X*0.5
    pygame.draw.rect(surface, GRAY, pygame.Rect(x, y, wall_width, h), 0)
    pygame.draw.rect(surface, GRAY, pygame.Rect(x+w-wall_width, y, wall_width, h), 0)
    SWAPFONT = PCDFont
    
    if devmode:
        render_multicolor_text(surface, ICAWS_thin, [[(str(math.floor(clock.get_fps())), BLACK)]], (x, y), False)
    
    # 1 Draws VTOL
    borderSize = 5
    draw_striped_square_border(surface, x+wall_width+borderSize, y+borderSize, h-(borderSize*2), h-(borderSize*2), borderSize, (BLACK, 3), (YELLOW, 5), "left")
    draw_arrow(surface, (x+wall_width+(borderSize/2)+(h/2), y + (borderSize/2) + (h/2)), 200, h/3, 1, 10, 10, (GREEN, 0), (GREEN, 0))
    
    draw_gauge_circle(surface, x+wall_width+(borderSize/2)+(h/2), y + (borderSize/2) + (h/2), (h/2)-10, GREEN, 2, None, "130", ICAWS_font, GREEN, 4, GREEN, 20, GREEN, 0)
    
    # 2 Draws FUEL area
    render_multicolor_text(surface, ICAWS_thin, [[("B", GREEN)], [("U", GREEN)], [("R", GREEN)], [("N", GREEN)]], (x + GRID_X*1.5, y + h/2), 0, True, "C")
    
    # 5 Draws ICAWS text
    render_multicolor_text(surface, PCDThin, [[("ICAWS", CYAN)]], (x + GRID_X*4.6, y + (h/2)), 0, True, "C")
    
    # 8 Draws SW AP button
    trigap = 34
    buttonSize = 50
    draw_triangle(surface, (x + w/2 - trigap, y + h/2 - 1), CYAN, 50, -90, 2)
    draw_triangle(surface, (x + w/2 + trigap, y + h/2 - 1), CYAN, 50, 90, 2)
    render_multicolor_text(surface, SWAPFONT, [[("SW AP", CYAN)]], (x + w/2, y + h/2), 0, True, "C", BLACK)
    setButton(4, 1, (x + w/2 - buttonSize, y, buttonSize*2, h), swapPortals)
    
    # 9 COMMS
    CommsA = grab_PMD_data(importArray['comms']['A'], "#######")
    CommsB = grab_PMD_data(importArray['comms']['B'], "#######")
    CommsC = grab_PMD_data(importArray['comms']['C'], "#######")
    render_multicolor_text(surface, ICAWS2, [[(f"A  V {CommsA}  KEWD", CYAN)], [(f"B  V {CommsB}  COM", CYAN)], [(f"C  V {CommsC}  COM", CYAN)]], (x + GRID_X*8.5, y + h/2 - 5), 7, True, "L")
    
    # 10 RECORDER
    render_multicolor_text(surface, ICAWS2, [[("RECORD", CYAN)], [("0:00:00", GREEN)], [("", BLACK)]], (x + GRID_X*10.7, y + h/2 - 5), 7, True, "C")
    
    # 12 Draws MENU button
    menuX = GRID_X*12 + 20
    menuSize = GRID_Y*0.8
    menuLineGap = menuSize/5
    menuYGap = (h - menuSize)/2
    for m in range(6):
        pygame.draw.line(surface, GRAY, (menuX + (m*menuLineGap), y + menuYGap), (menuX + (m*menuLineGap), y + h - menuYGap), 1)
        pygame.draw.line(surface, GRAY, (menuX, y + menuYGap + (m*menuLineGap)), (menuX + menuSize, y + menuYGap + (m*menuLineGap)), 1)
    render_multicolor_text(surface, PCDThin2, [[('MENU', CYAN)]], (menuX + (menuSize/2), y + (h/2)), 0, True, "C")
    
    # 15 Draws ZULU/WIND
    render_multicolor_text(surface, ICAWS2, [[("", BLACK)], [(f"{get_zulu_time()}Z", CYAN)], [("WIND:", CYAN)], [("258/  1", CYAN)]], (x + GRID_X*14.6, y + h/2 - 5), 1, True, "L")

pages = {
    "FCS":{
        "render" : render_FCS,
        "name" : "FCS",
        "subable" : True,
        "allowMult" : False,
        "requires":("FCS", 0)
    },
    "ASR":{
        "render" : render_FCS,
        "name" : "ASR",
        "subable" : True,
        "allowMult" : False,
    },
    "CNI":{
        "render" : render_FCS,
        "name" : "CNI",
        "subable" : True,
        "allowMult" : False,
        "requires":("CNI", 0)
    },
    "DAS":{
        "render" : render_DAS,
        "name" : "DAS",
        "subable" : True,
        "allowMult" : False,
        "requires":("DAS", 1)
    },
    "CKLST":{
        "render" : render_CKLST,
        "name" : "CKLST",
        "subable" : True,
        "allowMult" : False,
    },
    "DIM":{
        "render" : render_DIM,
        "name" : "DIM",
        "subable" : True,
        "allowMult" : False,
    },
    "EFI":{
        "render" : render_FCS,
        "name" : "EFI",
        "subable" : True,
        "allowMult" : False,
    },
    "FUEL":{
        "render" : render_FUEL,
        "name" : "FUEL",
        "subable" : True,
        "allowMult" : False,
        "requires":("FUEL", -1)
    },
    "ENG":{
        "render" : render_ENG,
        "name" : "ENG",
        "subable" : True,
        "allowMult" : False,
        "requires":("ENGINE", -1)
    },
    "TFLIR":{
        "render" : render_TFLIR,
        "name" : "TFLIR",
        "subable" : True,
        "allowMult" : False,
        "requires":("EOTS", 0)
    },
    "PHM":{
        "render" : render_PHM,
        "name" : "PHM",
        "subable" : True,
        "allowMult" : False,
    },
    "blank":{
        "render" : render_blank,
        "name" : "blank",
        "subable" : True,
        "allowMult" : True,
    }
}

portals = [
    {
        "CurrentPage":None,
        "rendering":None,
        "EXP":1,
        "hidden":False,
        "loc":0,
        "subs" : [None, None],
    },
    {
        "CurrentPage":None,
        "rendering":None,
        "EXP":1,
        "hidden":False,
        "loc":1,
        "subs" : [None, None],
    },
    {
        "CurrentPage":None,
        "rendering":None,
        "EXP":1,
        "hidden":False,
        "loc":2,
        "subs" : [None, None],
    },
    {
        "CurrentPage":None,
        "rendering":None,
        "EXP":1,
        "hidden":False,
        "loc":3,
        "subs" : [None, None],
    },
]

def clear_duplicate_pages(page):
    if pages[page]['allowMult'] == False:
        for i in range(4):
            if portals[i]['CurrentPage'] == pages[page]:
                set_portal_page(i, 'blank')
            if portals[i]['subs'][0] == pages[page]:
                set_portal_subportal(i, 0, 'blank')
            if portals[i]['subs'][1] == pages[page]:
                set_portal_subportal(i, 1, 'blank')

def set_portal_page(portal, page):
    try:
        clear_duplicate_pages(page)
        portals[portal]['CurrentPage'] = pages[page]
        portals[portal]['rendering'] = pages[page]['render']
        buttons[portal] = {}
    except KeyError:
        #print(f"{page} Does Not Exist")
        pass
    
def set_portal_render(portal, func):
    portals[portal]['rendering'] = func
    buttons[portal] = {}
    
def set_portal_subportal(portal, sub, page):
    try:
        if pages[page]["subable"]:
            clear_duplicate_pages(page)
            portals[portal]['subs'][sub] = pages[page]
    except:
        #print(f"{page} Does Not Exist")
        pass

def swap_portals(portal1, portal2):
    p1name = portal1['CurrentPage']['name']
    p2name = portal2['CurrentPage']['name']
    set_portal_page(portal1['loc'], p2name)
    set_portal_page(portal2['loc'], p1name)

def swap_portal_subportal(portal, sub):
    portalName = portal['CurrentPage']['name']
    if portal['CurrentPage']['subable']:
        subpage = portal['subs'][sub]
        subpageName = subpage['name']
        set_portal_page(portal['loc'], subpageName)
        set_portal_subportal(portal['loc'], sub, portalName)
    elif portal['subs'][sub]['name'] != "blank":
        subpage = portal['subs'][sub]
        subpageName = subpage['name']
        set_portal_page(portal['loc'], subpageName)
        set_portal_subportal(portal['loc'], sub, "blank")

def swap_portal_altsubportal(portal, portal2, sub):
    portalName = portal['CurrentPage']['name']
    if portal['CurrentPage']['subable']:
        subpage = portal2['subs'][sub]
        subpageName = subpage['name']
        set_portal_page(portal['loc'], subpageName)
        set_portal_subportal(portal2['loc'], sub, portalName)
    elif portal['subs'][sub]['name'] != "blank":
        subpage = portal2['subs'][sub]
        subpageName = subpage['name']
        set_portal_page(portal['loc'], subpageName)
        set_portal_subportal(portal2['loc'], sub, "blank")
        
def set_portal_exp(portal, exp):
    if portals[portal]['hidden'] == False:
        if portal == 0 or portal == 2:
            portals[portal]['EXP'] = exp
            if exp == 2:
                portals[portal+1]['hidden'] = True
            else:
                portals[portal+1]['hidden'] = False
        else:
            portals[portal]['EXP'] = exp
            if exp == 2:
                portals[portal-1]['hidden'] = True
            else:
                portals[portal-1]['hidden'] = False

set_portal_page(0, "blank")
set_portal_subportal(0, 0, "blank")
set_portal_subportal(0, 1, "blank")
set_portal_page(1, "blank")
set_portal_subportal(1, 0, "blank")
set_portal_subportal(1, 1, "blank")
set_portal_page(2, "blank")
set_portal_subportal(2, 0, "blank")
set_portal_subportal(2, 1, "blank")
set_portal_page(3, "blank")
set_portal_subportal(3, 0, "blank")
set_portal_subportal(3, 1, "blank")

set_portal_page(0, "PHM")
set_portal_subportal(0, 0, "FCS")
set_portal_subportal(0, 1, "DIM")
set_portal_page(1, "FUEL")
set_portal_subportal(1, 0, "DAS")
set_portal_subportal(1, 1, "TFLIR")
set_portal_page(2, "CKLST")


set_portal_page(3, "ENG")



def get_portal_positions(inta, fullBool=False):
    index, bool, sub = getSubportal(inta)
    portal = portals[index]
    x = 0
    y = 0
    w = 0
    h = 0
    if portal["hidden"] == False:
        exp = portal["EXP"]
        if bool:
            if exp > 0:
                x = 0
                y = 0,
                w = 0,
                h = 0,
            else:
                x = inta * (GRID_X*4) + (sub * (GRID_X*2))
                y = GRID_Y * 7
                w = GRID_X*2
                h = GRID_Y*2
        else:
            if fullBool == True:
                if exp < 2:
                    x = (inta) * (GRID_X*4)
                    y = GRID_Y
                    w = GRID_X*4
                    h = GRID_Y*8
                elif exp == 2:
                    if inta == 0 or inta == 2:
                        x = (inta) * (GRID_X*4)
                    else:
                        x = (inta-1) * (GRID_X*4)
                    y = GRID_Y
                    w = GRID_X*8
                    h = GRID_Y*8
            elif fullBool == False:
                if exp == 0:
                    x = (inta) * (GRID_X*4)
                    y = GRID_Y
                    w = GRID_X*4
                    h = GRID_Y*6
                elif exp == 1:
                    x = (inta) * (GRID_X*4)
                    y = GRID_Y
                    w = GRID_X*4
                    h = GRID_Y*8
                elif exp == 2:
                    if inta == 0 or inta == 2:
                        x = (inta) * (GRID_X*4)
                    else:
                        x = (inta-1) * (GRID_X*4)
                    y = GRID_Y
                    w = GRID_X*8
                    h = GRID_Y*8
    else:
        x=0
        y=0
        w=0
        h=0
    return (x, y, w, h, portal)

def get_subportal_position(pint, sub):
    x, y, w, h, portal = get_portal_positions(pint)
    px = 0
    py = 0
    pw = 0
    ph = 0
    if portal["hidden"] == False and portal['EXP'] == 0:
        px = x + (sub*GRID_X*2)
        py = y + h
        pw = GRID_X*2
        ph = GRID_Y*2
    return (px, py, pw, ph)

def get_ICAWS_position():
    return (0, 0, GRID_X*16, GRID_Y)

def inBounds(mouse, bounds):
    x, y = mouse
    bx, by, bw, bh = bounds
    if (x >= bx and x <= bx+bw) and (y >= by and y <= by+bh):
        return True
    else:
        return False
    
def get_portal_grid(inta, fullBool=False):
    x,y,w,h,portal = get_portal_positions(inta, fullBool)
    boxWidth = (GRID_X*4)/5
    boxHeight = GRID_Y
    boxColumns = math.floor(w/boxWidth)
    boxRows = math.floor(h/boxHeight)
    return (boxWidth, boxHeight, boxColumns, boxRows), (x, y, w, h)

def get_portal_grid_box(inta, gridPos, fullBool=False):
    grid, dim = get_portal_grid(inta, fullBool)
    boxWidth, boxHeight, boxColumns, boxRows = grid
    x, y, w, h = dim
    gx, gy = gridPos
    return (x + (gx*boxWidth), y + (gy*boxHeight), boxWidth, boxHeight), (x + (boxWidth*(gx+0.5)), y + (boxHeight*(gy+0.5)))

def swapPortals():
    p0 = portals[0]
    p1 = portals[1]
    p2 = portals[2]
    p3 = portals[3]
    portals[0] = p3
    portals[1] = p2
    portals[2] = p1
    portals[3] = p0
    for i in range(4):
        portals[i]['loc'] = i

def get_buttons(index):
    panel_buttons = buttons[index] if 0 <= index < len(buttons) else {}
    return panel_buttons

def execute_buttons(index):
    found = False
    for btn_data in get_buttons(index).values():
        # skip anything that isn’t a proper { "position":…, "callback":… } dict
        if not isinstance(btn_data, dict):
            continue
        position = btn_data.get("position")
        callback = btn_data.get("callback")
        # if either field is missing, skip
        if position is None or callback is None:
            continue
        # now check bounds and fire
        if inBounds(logical_mouse, position):
            found = True
            callback()
            #break
    return found

def set_master_mode(string):
    MASTER_MODE = string
import pygame

class Cursor:
    def __init__(self, color=(0, 255, 0), size=35, gap=10, width=2, dash_length = 2, gap_length = 4, wait_time=50):
        self.position = (-100, -100)
        self.temp_timer = 0
        self.color = color
        self.size = size
        self.gap = gap
        self.dash_length = dash_length
        self.gap_length = gap_length
        self.width = width
        self.wait_time = wait_time
        
        #self.dash_length = self.width
        #self.gap_length = self.width

    def set(self, pos=(-100, -100), temporary=False):
        """Sets the cursor position. If temporary is True, it will disappear after a delay."""
        self.position = pos
        if temporary:
            self.temp_timer = 1  # Start counting

    def draw(self, surface):
        x, y = self.position
        if x == -100 and y == -100:
            return  # Cursor hidden

        # Draw the four bracket lines around the cursor
        #pygame.draw.line(surface, self.color, (x - self.size, y), (x - self.gap, y), self.width)
        #pygame.draw.line(surface, self.color, (x + self.gap, y), (x + self.size, y), self.width)
        #pygame.draw.line(surface, self.color, (x, y - self.size), (x, y - self.gap), self.width)
        #pygame.draw.line(surface, self.color, (x, y + self.gap), (x, y + self.size), self.width)

        draw_dashed_line(surface, RED, (x - self.size, y), (x - self.gap, y), self.dash_length, self.gap_length, self.width)
        draw_dashed_line(surface, RED, (x + self.gap, y), (x + self.size, y), self.dash_length, self.gap_length, self.width)
        draw_dashed_line(surface, RED, (x, y - self.size), (x, y - self.gap), self.dash_length, self.gap_length, self.width)
        draw_dashed_line(surface, RED, (x, y + self.gap), (x, y + self.size), self.dash_length, self.gap_length, self.width)

        # Manage temporary timeout
        if 0 < self.temp_timer < self.wait_time:
            self.temp_timer += 1
            #print(self.temp_timer)
        elif self.temp_timer >= self.wait_time:
            self.set()  # Reset to hidden
            self.temp_timer = 0

    def is_visible(self):
        return self.position != (-100, -100)

    def reset(self):
        """Manually hide and reset the cursor."""
        self.set()
        self.temp_timer = 0

def draw_gui(surface, frame_count, mouse, mouse_down):
    surface.fill(BLACK)
    
    for i in range(0, 4):
        x, y, w, h, portal = get_portal_positions(i)
        sub1 = (i*2)+5
        sub2 = (i*2)+6
        buttons[i] = {}
        if portal["hidden"] == False:
            portal['rendering'](surface, x, y, w, h, portal)
            if portal['rendering'] == portal['CurrentPage']['render']:
                portal['subs'][0]['render'](surface, x, y, w, h, portal, sub1)
                portal['subs'][1]['render'](surface, x, y, w, h, portal, sub2)

    render_ICAWS(surface, get_ICAWS_position())
    draw_borders(surface)
        
    # Other graphics
    #pygame.draw.rect(surface, RED, pygame.Rect(50, 50, 300, 150), width=5)
    #pygame.draw.line(surface, RED, (200, 300), (700, 500), width=8)

    # Text
    #text_surface = font.render("Aspect-Ratio Locked 20:8", True, WHITE)
    #surface.blit(text_surface, (LOGICAL_WIDTH//2 - text_surface.get_width()//2, 30))

    # Button
    #test_button.draw(surface)

# Main loop
clock = pygame.time.Clock()
running = True
frame_count = 0
mouse_down = False
mouse_up = False
waitBool = False
cursor = Cursor()
if devmode == False:
    panda = MultiCamOffscreenRenderer(1024, 1024, importArray)
frames = {}

#cam_thread = start_obs_cam_thread(draw_surface, fps=30.0)

while running:
    tmp = load_json_to_array('import.json')[0]
    if tmp is not None:
        importArray = tmp
    
    if devmode == False:
        frames = panda.render_all(importArray)

    # 4) Save each camera’s view to disk
    #output_dir = os.path.join(os.path.dirname(__file__), "renders")
    #os.makedirs(output_dir, exist_ok=True)
    #for cam_id, surf in frames.items():
    #    filename = os.path.join(output_dir, f"{cam_id}.png")
    #    pygame.image.save(surf, filename)
    #    print(f"Saved {cam_id} → {filename}")
        
    mouse_up = False
    if waitBool:
        mouse_down = False
    for event in pygame.event.get():
        #print(event)
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and waitBool == False:
            mouse_down = True
            waitBool = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            mouse_down = False
            mouse_up = True
            waitBool = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_F11:
                toggle_fullscreen()

    #print(mouse_down, waitBool)

    # Get logical mouse position
    mouse_pos = pygame.mouse.get_pos()
    logical_mouse = get_logical_mouse_pos(window.get_size(), mouse_pos)

    if logical_mouse and mouse_up:
        cursor.set(logical_mouse, True)
        mouse_up = False
        index = 6
        if execute_buttons(index) == False:
            index = 5
            if execute_buttons(index) == False:
                if inBounds(logical_mouse, (0, 0, WINDOW_WIDTH, GRID_Y)):
                    index = 4
                    execute_buttons(index)
                else:
                    for i in range(0, 4):
                        x, y, w, h, portal = get_portal_positions(i, True)
                        if inBounds(logical_mouse, (x,y,w,h)):
                            index = i
                            execute_buttons(index)
    elif logical_mouse and waitBool:
        cursor.set(logical_mouse, True)
    # Draw GUI to logical surface
    window.fill((0, 0, 0))
    draw_gui(draw_surface, frame_count, logical_mouse, mouse_down)
    cursor.draw(draw_surface)

    # Get current window size
    WINDOW_WIDTH, WINDOW_HEIGHT = window.get_size()
    current_ratio = WINDOW_WIDTH / WINDOW_HEIGHT

    if current_ratio > ASPECT_RATIO:
        scale_height = WINDOW_HEIGHT
        scale_width = int(scale_height * ASPECT_RATIO)
    else:
        scale_width = WINDOW_WIDTH
        scale_height = int(scale_width / ASPECT_RATIO)

    x_offset = (WINDOW_WIDTH - scale_width) // 2
    y_offset = (WINDOW_HEIGHT - scale_height) // 2

    scaled_surface = pygame.transform.smoothscale(draw_surface, (scale_width, scale_height))
    window.blit(scaled_surface, (x_offset, y_offset))

    pygame.display.flip()
    clock.tick(60)
    frame_count += 1

pygame.quit()
panda.userExit()
sys.exit()