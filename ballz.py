#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Battle Balls — Offline Renderer (Video + Audio)

Generates a vertical video of an auto-played fight with title, names,
side-by-side health bars, arena taking most of the screen, and a winner
obliteration finish. Audio is synthesized to match events.

Deps: pillow, numpy, moviepy, tqdm (pip install pillow numpy moviepy tqdm)
Requires: ffmpeg in PATH.

Usage example:
python battle_balls_render.py --imgA a.png --imgB b.jpg --nameA "Draco" --nameB "Nimbus" \
  --title "Battle Balls — Ep 1" --out fight.mp4 --fps 30 --width 1080 --height 1920
"""
import os, sys, math, random, tempfile, shutil, argparse, time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip, AudioArrayClip

# ----------------------- Config defaults -----------------------
MAX_HP = 5
R_MAX, R_MIN = 44, 22
BASE_SPEED = 6.3
SPEED_EXP = 1.85
SPEED_CAP_BASE = 13.5
KNOCKBACK = 1.35
KNOCKBACK_SAW = 1.85
COLLISION_COOLDOWN_MS = 140
ITEM_SIZE = 28  # px on arena canvas
ARENA_W, ARENA_H = 720, 1280   # internal sim canvas (scaled to export)

# Item spawn cadence (off-screen sim)
HEART_SPAWN_P = 0.01  # per frame probability

# Fonts: we only rely on default system fallback
def load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        # Try some common fonts, fallback to PIL default
        for name in ["DejaVuSans-Bold.ttf", "Arial.ttf", "Helvetica.ttf"]:
            try:
                return ImageFont.truetype(name, size=size)
            except Exception:
                pass
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

# ----------------------- Helpers -----------------------
def radius_for_hp(hp: int) -> float:
    hp = max(1, min(MAX_HP, hp))
    return R_MIN + (R_MAX - R_MIN) * (hp - 1) / (MAX_HP - 1)

def target_speed_for(hp: int) -> float:
    r = radius_for_hp(hp)
    return BASE_SPEED * (R_MAX / r) ** SPEED_EXP

def speed_cap_for(hp: int) -> float:
    r = radius_for_hp(hp)
    return SPEED_CAP_BASE * (R_MAX / r) ** SPEED_EXP

def clamp(v, lo, hi): return max(lo, min(hi, v))

def lerp(a, b, t): return a + (b - a) * t

def make_radial_bg(w, h) -> Image.Image:
    # subtle gradient + grid drawn later
    img = Image.new("RGB", (w, h), "#0a0c17")
    draw = ImageDraw.Draw(img)
    # vertical gradient
    for y in range(h):
        t = y / (h - 1)
        c = int(lerp(0x0f, 0x1a, t))
        draw.line([(0, y), (w, y)], fill=(c, c+2, c+16))
    return img

def draw_grid_overlay(draw: ImageDraw.ImageDraw, w, h):
    step = max(32, min(48, int(min(w, h) / 12)))
    grid_col = (255, 255, 255, 34)
    for x in range(step, w, step):
        draw.line([(x, 0), (x, h)], fill=grid_col, width=1)
    for y in range(step, h, step):
        draw.line([(0, y), (w, y)], fill=grid_col, width=1)
    # border
    draw.rectangle([1, 1, w-2, h-2], outline=(255, 255, 255, 48), width=2)

def circle_mask(diam: int) -> Image.Image:
    m = Image.new("L", (diam, diam), 0)
    ImageDraw.Draw(m).ellipse([0, 0, diam-1, diam-1], fill=255)
    return m

def fit_img_in_circle(src: Optional[Image.Image], diam: int, tint: Tuple[int,int,int]) -> Image.Image:
    """Return square image with circular mask applied. If src missing, use tinted gradient."""
    if diam < 2: diam = 2
    if src is None:
        # simple fallback gradient tile
        tile = Image.new("RGB", (diam, diam), (0, 0, 0))
        g = ImageDraw.Draw(tile)
        for i in range(diam):
            a = int(48 + 120 * (i / max(1, diam-1)))
            g.line([(0,i),(diam,i)], fill=(min(255, tint[0]+a//6), min(255, tint[1]+a//6), min(255, tint[2]+a//6)))
        tile.putalpha(circle_mask(diam))
        return tile.convert("RGBA")
    # scale preserving aspect to cover circle
    iw, ih = src.size
    s = max(diam/iw, diam/ih)
    img = src.resize((int(iw*s), int(ih*s)), Image.LANCZOS)
    # center crop
    x0 = (img.width - diam)//2
    y0 = (img.height - diam)//2
    img = img.crop((x0, y0, x0+diam, y0+diam))
    img.putalpha(circle_mask(diam))
    return img.convert("RGBA")

# ----------------------- Entities -----------------------
@dataclass
class Ball:
    name: str
    x: float
    y: float
    vx: float
    vy: float
    hp: int
    sharp: bool
    img: Optional[Image.Image]
    color: Tuple[int,int,int]
    removed: bool = False

@dataclass
class Item:
    kind: str   # 'saw' or 'heart'
    x: float
    y: float

# ----------------------- Physics -----------------------
def rescale_speed_to_target(b: Ball):
    t = target_speed_for(b.hp)
    s = math.hypot(b.vx, b.vy)
    if s < 1e-6:
        a = random.random() * math.tau
        b.vx, b.vy = math.cos(a)*t, math.sin(a)*t
    else:
        k = t / s
        b.vx, b.vy = b.vx*k, b.vy*k

def clamp_vel(b: Ball):
    cap = speed_cap_for(b.hp)
    s = math.hypot(b.vx, b.vy)
    if s > cap:
        k = cap / (s or 1e-6)
        b.vx, b.vy = b.vx*k, b.vy*k

def circle_collision(a: Ball, b: Ball) -> bool:
    if a.removed or b.removed: return False
    return math.hypot(b.x-a.x, b.y-a.y) < radius_for_hp(a.hp) + radius_for_hp(b.hp)

def resolve_elastic(a: Ball, b: Ball, factor=1.0):
    r1, r2 = radius_for_hp(a.hp), radius_for_hp(b.hp)
    dx, dy = b.x-a.x, b.y-a.y
    dist = math.hypot(dx, dy) or 1e-6
    nx, ny = dx/dist, dy/dist
    tx, ty = -ny, nx
    v1n = a.vx*nx + a.vy*ny
    v1t = a.vx*tx + a.vy*ty
    v2n = b.vx*nx + b.vy*ny
    v2t = b.vx*tx + b.vy*ty
    v1n_after = v2n * factor
    v2n_after = v1n * factor
    a.vx, a.vy = v1t*tx + v1n_after*nx, v1t*ty + v1n_after*ny
    b.vx, b.vy = v2t*tx + v2n_after*nx, v2t*ty + v2n_after*ny
    overlap = r1 + r2 - dist
    sep = overlap/2 + 0.2
    a.x -= nx*sep; a.y -= ny*sep
    b.x += nx*sep; b.y += ny*sep
    clamp_vel(a); clamp_vel(b)

# ----------------------- Audio synth -----------------------
def synth_audio(events: List[Tuple[float,str]], duration: float, sr=48000) -> np.ndarray:
    """
    events: list of (t, type) where type in {'saw','heart','hit','boom'}
    Returns stereo float32 in [-1,1].
    """
    n = int(duration*sr)+sr//10
    audio = np.zeros((n,), dtype=np.float32)

    def env_exp(length, start=1.0):
        t = np.linspace(0, 1, int(length*sr), endpoint=False)
        return (start * np.exp(-6*t)).astype(np.float32)

    def add_tone(t0, dur, freq_start, freq_end=None, amp=0.08, wave="sine"):
        N = int(dur*sr)
        if N <= 0: return
        i0 = int(t0*sr)
        if i0+N > n: return
        if freq_end is None: freq_end = freq_start
        f = np.linspace(freq_start, freq_end, N)
        phase = 2*np.pi*np.cumsum(f)/sr
        if wave == "sine":
            sig = np.sin(phase)
        elif wave == "square":
            sig = np.sign(np.sin(phase))
        elif wave == "saw":
            sig = 2.0*(phase/(2*np.pi) - np.floor(0.5 + phase/(2*np.pi)))
        else:
            sig = np.sin(phase)
        sig *= env_exp(dur) * amp
        audio[i0:i0+N] += sig.astype(np.float32)

    def add_noise(t0, dur, amp=0.16):
        N = int(dur*sr)
        if N <= 0: return
        i0 = int(t0*sr)
        if i0+N > n: return
        sig = (np.random.rand(N).astype(np.float32)*2 - 1)
        env = np.linspace(1, 0, N).astype(np.float32)
        audio[i0:i0+N] += sig * env * amp

    for t, typ in events:
        if typ == "saw":
            add_tone(t, 0.08, 220, 800, 0.06, "square")
            add_tone(t+0.06, 0.11, 360, 1000, 0.05, "saw")
        elif typ == "heart":
            add_tone(t, 0.07, 523, None, 0.06, "sine")
            add_tone(t+0.09, 0.08, 659, None, 0.06, "sine")
            add_tone(t+0.18, 0.10, 784, None, 0.07, "sine")
        elif typ == "hit":
            add_tone(t, 0.06, 120, None, 0.10, "sine")
        elif typ == "boom":
            add_tone(t, 0.65, 320, 1250, 0.06, "sine")
            add_noise(t+0.14, 0.32, 0.18)

    # soft limiter
    m = np.max(np.abs(audio)) or 1.0
    audio = np.clip(audio / (1.2*m), -1, 1)
    # stereo
    stereo = np.stack([audio, audio], axis=1)
    return stereo.astype(np.float32)

# ----------------------- Rendering -----------------------
def draw_item(draw: ImageDraw.ImageDraw, kind: str, x: float, y: float):
    if kind == "saw":
        # simple saw: disc + spokes
        r1 = int(ITEM_SIZE*0.6)
        draw.ellipse([x-r1, y-r1, x+r1, y+r1], fill=(220,220,220), outline=(0,0,0))
        for i in range(12):
            a = i * (math.tau/12)
            x1, y1 = x + math.cos(a)*r1, y + math.sin(a)*r1
            x2, y2 = x + math.cos(a)*(r1+8), y + math.sin(a)*(r1+8)
            draw.line([x1,y1,x2,y2], fill=(255,214,102), width=2)
    else:
        # heart
        s = ITEM_SIZE*0.26
        pts = []
        # Using a simple heart poly approximation
        for t in np.linspace(0, math.tau, 60):
            ix = 16*math.sin(t)**3
            iy = 13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t)
            pts.append((x+ix*s*0.08, y-iy*s*0.08))
        draw.polygon(pts, fill=(255,111,174), outline=(0,0,0))

def draw_ball(arena_img: Image.Image, b: Ball, cache_imgs):
    if b.removed: return
    draw = ImageDraw.Draw(arena_img)
    r = int(radius_for_hp(b.hp))
    # shadow
    draw.ellipse([b.x-r-2, b.y-r-2, b.x+r+2, b.y+r+2], fill=(0,0,0))
    # image circle (cache per HP)
    key = (id(b.img), b.hp)
    if key not in cache_imgs:
        tinted = fit_img_in_circle(b.img, 2*r, b.color)
        cache_imgs[key] = tinted
    circ = cache_imgs[key]
    arena_img.alpha_composite(circ, (int(b.x-r), int(b.y-r)))
    # rim
    color = (255,209,102) if b.sharp else (255,255,255)
    draw.ellipse([b.x-r, b.y-r, b.x+r, b.y+r], outline=color, width=3 if b.sharp else 2)

def draw_funky_title(canvas: Image.Image, title: str, y_center: int):
    ex = ImageDraw.Draw(canvas)
    base = 64
    title = (title or "Battle Balls")[:120]
    font = load_font(base)
    # Compute widths per letter to center
    widths = [font.getlength(ch) for ch in title]
    total_w = sum(widths)*0.9
    x = canvas.width//2 - int(total_w/2)
    t = time.time()*1000.0  # just for slight variation
    for i, ch in enumerate(title):
        wob = math.sin((t*0.004)+(i*0.6))*6
        rot = math.sin((t*0.002)+(i*0.8))*0.08
        # render letter on its own layer for rotation
        w = int(widths[i]*1.2)+4
        h = base+20
        im = Image.new("RGBA", (w, h), (0,0,0,0))
        d = ImageDraw.Draw(im)
        # gradient-ish fill by overlay strokes
        d.text((2,h//2), ch, font=font, anchor="lm", fill=(255,236,120,255))
        d.text((0,h//2), ch, font=font, anchor="lm", fill=(255,122,233,120))
        d.text((1,h//2), ch, font=font, anchor="lm", fill=(138,247,255,120))
        im = im.rotate(math.degrees(rot), resample=Image.BICUBIC, expand=1, center=(0,h//2))
        canvas.alpha_composite(im, (int(x), int(y_center + wob - h//2)))
        x += int(widths[i]*0.56)

def round_rect(draw: ImageDraw.ImageDraw, xy, r, fill=None, outline=None, width=1):
    (x1,y1,x2,y2) = xy
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)

def draw_bars_row(canvas: Image.Image, y: int, nameA: str, nameB: str, fracA: float, fracB: float, sawA: bool, sawB: bool):
    ex = ImageDraw.Draw(canvas)
    pad, gap = 18, 22
    cellW = (canvas.width - pad*2 - gap)//2
    barH, radius = 30, 12
    font_label = load_font(22)
    def cell(x, label, frac, c1, c2, saw):
        ex.text((x, y-6), label, fill=(232,246,255), font=font_label, anchor="ls")
        tx, ty, tw, th = x, y+4, cellW, barH
        round_rect(ex, (tx,ty,tx+tw,ty+th), radius, fill=(0,0,0,160))
        # fill gradient approx: two rectangles
        w = int(clamp(frac,0,1)*tw)
        if w>0:
            ex.rectangle([tx,ty,tx+w,ty+th], fill=c1)
        if saw:
            bx, by, bw, bh = tx+tw-64, ty-18, 60, th+18
            round_rect(ex, (bx,by,bx+bw,by+bh), 10, fill=(255,209,102), outline=(0,0,0,160), width=2)
            ex.text((bx+bw//2, ty+th//2), "SAW", fill=(17,17,17), font=load_font(18), anchor="mm")
    cell(pad, nameA, fracA, (69,255,154), (138,255,255), sawA)
    cell(pad+cellW+gap, nameB, fracB, (255,159,211), (255,122,217), sawB)

# ----------------------- Simulation + Render -----------------------
def render_video(args):
    random.seed(args.seed)

    # Load input images
    imgA = Image.open(args.imgA).convert("RGB") if args.imgA else None
    imgB = Image.open(args.imgB).convert("RGB") if args.imgB else None

    # Prepare arena background (static)
    arena_bg = make_radial_bg(ARENA_W, ARENA_H).convert("RGBA")
    arena_grid = Image.new("RGBA", (ARENA_W, ARENA_H), (0,0,0,0))
    draw_grid_overlay(ImageDraw.Draw(arena_grid), ARENA_W, ARENA_H)

    # Export canvas
    W, H = args.width, args.height
    TITLE_H = 120
    MARGIN = 18

    # Sim state
    balls = [
        Ball(args.nameA or "Ball A", ARENA_W*0.30, ARENA_H*0.35, 0, 0, MAX_HP, False, imgA, (158,240,255)),
        Ball(args.nameB or "Ball B", ARENA_W*0.70, ARENA_H*0.65, 0, 0, MAX_HP, False, imgB, (255,122,217))
    ]
    # set initial speeds
    def set_speed(b: Ball, dx, dy):
        t = target_speed_for(b.hp)
        L = math.hypot(dx, dy) or 1
        b.vx, b.vy = (dx/L)*t, (dy/L)*t
    set_speed(balls[0], +1, -0.6)
    set_speed(balls[1], -0.95, +0.85)

    items: List[Item] = [Item('saw', random.uniform(60,ARENA_W-60), random.uniform(60,ARENA_H-60))]
    last_hit_ms = -9999
    winner: Optional[str] = None

    # FX state
    particles = []  # dicts with x,y,vx,vy,age,life,size,color
    shockwaves = []
    screen_flash = 0.0

    # Audio events
    audio_events: List[Tuple[float,str]] = []

    # Output frames to temp dir to avoid RAM blowup
    tmpdir = tempfile.mkdtemp(prefix="battle_balls_frames_")
    frames_paths = []
    fps = args.fps
    dt = 1.0/fps
    t_sec = 0.0
    outro_left = 0.0

    # Pre-cache circle images per HP
    circle_cache = {}

    # Pre-render static title-less background for export (we draw title each frame for wiggle)
    total_frames_est = fps*20  # just for progress bar smoothing
    pbar = tqdm(total=total_frames_est, desc="Sim & draw", unit="f")

    try:
        frame_idx = 0
        while True:
            # --- physics step ---
            for b in balls:
                if b.removed: continue
                r = radius_for_hp(b.hp)
                b.x += b.vx*dt*60
                b.y += b.vy*dt*60
                if b.x < r: b.x = r; b.vx = abs(b.vx)
                if b.x > ARENA_W - r: b.x = ARENA_W - r; b.vx = -abs(b.vx)
                if b.y < r: b.y = r; b.vy = abs(b.vy)
                if b.y > ARENA_H - r: b.y = ARENA_H - r; b.vy = -abs(b.vy)
                clamp_vel(b)

            # items pickup / spawn
            for b in balls:
                if b.removed: continue
                r = radius_for_hp(b.hp)
                for it in list(items):
                    if math.hypot(b.x-it.x, b.y-it.y) < r + (ITEM_SIZE*0.6 if it.kind=='saw' else ITEM_SIZE*0.4):
                        if it.kind == 'saw':
                            # only one can be sharp; remove all saws
                            for bb in balls: bb.sharp = False
                            b.sharp = True
                            items = [i for i in items if i.kind!='saw']
                            audio_events.append((t_sec,'saw'))
                        elif it.kind == 'heart':
                            if b.hp < MAX_HP:
                                b.hp += 1
                                rescale_speed_to_target(b)
                                audio_events.append((t_sec,'heart'))
                            items.remove(it)
                # spawn hearts occasionally
            if random.random() < HEART_SPAWN_P and not any(i.kind=='heart' for i in items):
                items.append(Item('heart', random.uniform(60,ARENA_W-60), random.uniform(60,ARENA_H-60)))

            # collisions
            kb = KNOCKBACK
            if circle_collision(balls[0], balls[1]):
                now_ms = t_sec*1000.0
                if now_ms - last_hit_ms > COLLISION_COOLDOWN_MS:
                    a, b = balls[0], balls[1]
                    if not a.removed and not b.removed:
                        if a.sharp and not b.sharp:
                            b.hp = max(0, b.hp-1); a.sharp=False
                            rescale_speed_to_target(b); rescale_speed_to_target(a)
                            audio_events.append((t_sec,'hit')); kb = KNOCKBACK_SAW
                        elif b.sharp and not a.sharp:
                            a.hp = max(0, a.hp-1); b.sharp=False
                            rescale_speed_to_target(a); rescale_speed_to_target(b)
                            audio_events.append((t_sec,'hit')); kb = KNOCKBACK_SAW
                    last_hit_ms = now_ms
                resolve_elastic(balls[0], balls[1], kb)

            # winner
            if winner is None and ((balls[0].hp<=0 and not balls[0].removed) or (balls[1].hp<=0 and not balls[1].removed)):
                loser = balls[0] if (balls[0].hp<=0 and not balls[0].removed) else balls[1]
                win = balls[1] if loser is balls[0] else balls[0]
                winner = win.name
                # obliteration FX
                screen_flash = 0.35
                shockwaves.append({'x': loser.x, 'y': loser.y, 'r':6, 'dr':22, 'life':0.7, 'age':0})
                for _ in range(220):
                    ang = random.random()*math.tau
                    spd = random.uniform(90,420)
                    particles.append({'x':loser.x, 'y':loser.y, 'vx':math.cos(ang)*spd, 'vy':math.sin(ang)*spd,
                                      'age':0.0, 'life':random.uniform(0.5,1.1), 'size':random.uniform(1.5,3.5),
                                      'color': (255,255,255) if random.random()<0.5 else win.color})
                audio_events.append((t_sec,'boom'))
                loser.removed = True
                outro_left = random.uniform(0.8, 1.6)

            # FX update
            for p in list(particles):
                p['age'] += dt
                p['x'] += p['vx']*dt
                p['y'] += p['vy']*dt
                p['vy'] += 40*dt
                if p['age'] > p['life']: particles.remove(p)
            for s in list(shockwaves):
                s['age'] += dt
                s['r'] += s['dr']*dt*60
                if s['age'] > s['life']: shockwaves.remove(s)
            if screen_flash > 0.0:
                screen_flash = max(0.0, screen_flash - dt)

            # --- draw frame ---
            # Arena layer
            arena = arena_bg.copy()
            arena.alpha_composite(arena_grid)
            ad = ImageDraw.Draw(arena, "RGBA")
            # items
            for it in items:
                draw_item(ad, it.kind, it.x, it.y)
            # balls
            for b in balls: draw_ball(arena, b, circle_cache)
            # debris (particles) as small circles
            for p in particles:
                a = int(255 * max(0, 1 - p['age']/p['life']))
                r = p['size']
                col = (*p['color'], a)
                ad.ellipse([p['x']-r, p['y']-r, p['x']+r, p['y']+r], fill=col)

            # Compose export frame
            frame = Image.new("RGBA", (W, H), (15,16,32))
            # Title
            draw_funky_title(frame, args.title or "Battle Balls", y_center=TITLE_H//2)
            ImageDraw.Draw(frame).line([(MARGIN, TITLE_H),(W-MARGIN, TITLE_H)], fill=(255,255,255,34), width=2)

            # Bars row
            fracA = (balls[0].hp if not balls[0].removed else 0)/MAX_HP
            fracB = (balls[1].hp if not balls[1].removed else 0)/MAX_HP
            sawA = balls[0].sharp and not balls[0].removed
            sawB = balls[1].sharp and not balls[1].removed
            draw_bars_row(frame, TITLE_H+52, balls[0].name, balls[1].name, fracA, fracB, sawA, sawB)

            # Arena placement (dominant)
            avail_top = TITLE_H + 52 + 30 + 18
            availH = H - avail_top - MARGIN
            availW = W - MARGIN*2
            srcW, srcH = ARENA_W, ARENA_H
            src_aspect = srcW/srcH
            dst_aspect = availW/availH
            dw, dh = availW, availH
            if src_aspect > dst_aspect:
                dh = int(dw/src_aspect)
            else:
                dw = int(dh*src_aspect)
            dx = (W-dw)//2
            dy = int(avail_top + (availH - dh)//2)

            arena_scaled = arena.resize((dw, dh), Image.LANCZOS)
            frame.alpha_composite(arena_scaled, (dx, dy))

            # Shockwaves on export coords (scale)
            exd = ImageDraw.Draw(frame, "RGBA")
            for s in shockwaves:
                t_ = max(0.0, 1.0 - s['age']/s['life'])
                a = int(90 * t_)
                lw = max(1, int(6 * t_))
                exd.ellipse([dx + (s['x']/srcW)*dw - s['r']*(dw/srcW),
                             dy + (s['y']/srcH)*dh - s['r']*(dw/srcW),
                             dx + (s['x']/srcW)*dw + s['r']*(dw/srcW),
                             dy + (s['y']/srcH)*dh + s['r']*(dw/srcW)],
                            outline=(255,255,255,a), width=lw)

            # Winner caption
            if winner:
                d = ImageDraw.Draw(frame)
                f = load_font(60)
                d.text((W//2, dy + int(dh*0.12)), f"{winner} WINS!", anchor="mm",
                       font=f, fill=(255,255,255), stroke_width=3, stroke_fill=(0,0,0))

            # screen flash
            if screen_flash > 0.0:
                a = int(min(0.75, screen_flash*1.8)*255)
                overlay = Image.new("RGBA", (W, H), (255,255,255,a))
                frame = Image.alpha_composite(frame, overlay)

            # Save frame
            out_path = os.path.join(tmpdir, f"f_{frame_idx:06d}.png")
            frame.convert("RGB").save(out_path, "PNG", compress_level=4)
            frames_paths.append(out_path)

            # early stop after outro completes
            if winner:
                outro_left -= dt
                if outro_left <= 0.0:
                    break

            t_sec += dt
            frame_idx += 1
            # progress bar trick: update capped estimation
            if frame_idx <= total_frames_est:
                pbar.update(1)

        pbar.close()

        duration = len(frames_paths)/fps

        # ------------------ Audio build ------------------
        print("Synthesizing audio…")
        audio = synth_audio(audio_events, duration, sr=48000)  # stereo float32
        audio_clip = AudioArrayClip(audio, fps=48000)

        # ------------------ Compose video ----------------
        print("Encoding video…")
        clip = ImageSequenceClip(frames_paths, fps=fps)
        clip = clip.set_audio(audio_clip).set_duration(duration)

        out = args.out
        if args.webm:
            clip.write_videofile(
                out if out.lower().endswith(".webm") else out + ".webm",
                codec="libvpx-vp9", audio_codec="libopus",
                bitrate="6M", fps=fps, threads=os.cpu_count() or 4,
                preset="good"
            )
        else:
            clip.write_videofile(
                out if out.lower().endswith(".mp4") else out + ".mp4",
                codec="libx264", audio_codec="aac",
                bitrate="8M", fps=fps, threads=os.cpu_count() or 4,
                preset="medium"
            )
        print("Done:", out)
    finally:
        # Cleanup temp frames
        shutil.rmtree(tmpdir, ignore_errors=True)

# ----------------------- CLI -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Battle Balls — Offline Renderer (Video + Audio)")
    ap.add_argument("--imgA", type=str, required=False, help="Image for Ball A")
    ap.add_argument("--imgB", type=str, required=False, help="Image for Ball B")
    ap.add_argument("--nameA", type=str, default="Ball A", help="Name for Ball A")
    ap.add_argument("--nameB", type=str, default="Ball B", help="Name for Ball B")
    ap.add_argument("--title", type=str, default="Battle Balls", help="Video title")
    ap.add_argument("--out", type=str, default="battle_balls", help="Output file path (extension added if missing)")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second")
    ap.add_argument("--width", type=int, default=1080, help="Export width (px)")
    ap.add_argument("--height", type=int, default=1920, help="Export height (px)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--webm", action="store_true", help="Export WebM VP9/Opus instead of MP4")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    render_video(args)
