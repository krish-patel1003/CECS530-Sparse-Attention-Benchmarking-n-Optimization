"""Build the walkthrough video:  narration (macOS `say`) + matplotlib animations + ffmpeg.

    python video/make_video.py --preview      # contact sheets of every beat (fast check)
    python video/make_video.py                # full render -> video/Sparse_Attention_Walkthrough.mp4

Each beat's visual lasts exactly as long as its narration (+ a short pause), so picture and
voice stay in sync.  Output: H.264/AAC MP4 with chapter markers and a soft subtitle track, plus
a separate .srt file.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from script import CHAPTERS  # noqa: E402

VOICE, RATE = "Samantha", 175
FPS = 24
PAUSE = 0.35          # silence after each beat (s)
CHAPTER_PAUSE = 0.6   # extra silence at the end of a chapter
OUT = HERE / "Sparse_Attention_Walkthrough.mp4"


def sh(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def duration(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(path)]).decode().strip())


def display_text(t):
    for a, b in [("C P U", "CPU"), ("G P U", "GPU"), ("G P T", "GPT"), ("C S R", "CSR"), ("C S V", "CSV")]:
        t = t.replace(a, b)
    return t


# ------------------------------------------------------------------ timeline
def build_beats(work):
    beats = []
    for ci, (ctitle, items) in enumerate(CHAPTERS):
        for bi, (key, text) in enumerate(items):
            beats.append(dict(chapter_no=ci + 1, chapter=ctitle, key=key, text=text,
                              last_in_chapter=(bi == len(items) - 1)))
    tts = work / "tts"
    tts.mkdir(parents=True, exist_ok=True)
    for b in beats:
        h = hashlib.sha1(f"{VOICE}|{RATE}|{b['text']}".encode()).hexdigest()[:12]
        aiff = tts / f"{h}.aiff"
        if not aiff.exists():
            subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), b["text"]], check=True)
        b["aiff"] = aiff
        b["speech"] = duration(aiff)
        pad = PAUSE + (CHAPTER_PAUSE if b["last_in_chapter"] else 0)
        b["frames"] = math.ceil((b["speech"] + pad) * FPS)
        b["dur"] = b["frames"] / FPS
    t = 0.0
    total = sum(b["dur"] for b in beats)
    for i, b in enumerate(beats):
        b["idx"], b["t0"], b["t1"] = i, t, t + b["dur"]
        t += b["dur"]
    starts, seen = [], set()
    for b in beats:
        if b["chapter_no"] not in seen:
            seen.add(b["chapter_no"])
            starts.append(b["t0"] / total)
    for b in beats:
        b["ticks"] = starts[1:]
        b["total"] = total
    return beats, total


# ------------------------------------------------------------------ frame rendering (worker)
_fig = None


def _init():
    global _fig
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _fig = plt.figure(figsize=(19.2, 10.8), dpi=100)


def _render_chunk(args):
    import visuals
    key, frames, nframes, outdir, t0, dur, total, info_base = args
    for f in frames:
        p = f / max(nframes - 1, 1)
        info = dict(info_base, progress=(t0 + p * dur) / total)
        visuals.render(_fig, key, p, info)
        _fig.savefig(outdir / f"f{f:05d}.png", dpi=100, facecolor=_fig.get_facecolor())
    return len(frames)


def render_frames(beats, work, workers):
    tasks = []
    for b in beats:
        d = work / "frames" / f"b{b['idx']:02d}"
        if d.exists() and len(list(d.glob("f*.png"))) == b["frames"]:
            continue
        d.mkdir(parents=True, exist_ok=True)
        info = dict(chapter_no=b["chapter_no"], chapter=b["chapter"], ticks=b["ticks"])
        fr = list(range(b["frames"]))
        step = 48
        for s in range(0, len(fr), step):
            tasks.append((b["key"], fr[s:s + step], b["frames"], d, b["t0"], b["dur"], b["total"], info))
    if not tasks:
        return
    done, totalf = 0, sum(len(t[1]) for t in tasks)
    with Pool(workers, initializer=_init) as pool:
        for n in pool.imap_unordered(_render_chunk, tasks):
            done += n
            if done % 960 < n:
                print(f"  rendered {done}/{totalf} frames", flush=True)


# ------------------------------------------------------------------ encoding
def encode(beats, work, total):
    seg = work / "segments"
    seg.mkdir(exist_ok=True)
    vlist, alist = [], []
    for b in beats:
        v = seg / f"v{b['idx']:02d}.mp4"
        a = seg / f"a{b['idx']:02d}.wav"
        fdir = work / "frames" / f"b{b['idx']:02d}"
        sh(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(fdir / "f%05d.png"), "-c:v", "libx264", "-preset", "medium",
            "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), "-frames:v", str(b["frames"]), str(v)])
        sh(["ffmpeg", "-y", "-i", str(b["aiff"]), "-af", f"apad=whole_dur={b['dur']:.6f}", "-t", f"{b['dur']:.6f}",
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(a)])
        vlist.append(v)
        alist.append(a)
    (work / "v.txt").write_text("".join(f"file '{p}'\n" for p in vlist))
    (work / "a.txt").write_text("".join(f"file '{p}'\n" for p in alist))
    sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "v.txt"), "-c", "copy", str(work / "video.mp4")])
    sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(work / "a.txt"), "-c", "copy", str(work / "audio.wav")])

    # subtitles: split each beat into sentences / clauses, timed by character share of its speech
    srt, n = [], 1

    def ts(x):
        h, r = divmod(x, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s - int(s)) * 1000)):03d}".replace(",1000", ",999")

    for b in beats:
        text = display_text(b["text"])
        parts = re.split(r"(?<=[.!?:])\s+", text)
        chunks = []
        for s in parts:
            while len(s) > 95:
                cut = s.rfind(",", 0, 95)
                cut = cut if cut > 30 else s.rfind(" ", 0, 90)
                chunks.append(s[:cut + 1].strip())
                s = s[cut + 1:].strip()
            if s:
                chunks.append(s)
        tot = sum(len(c) for c in chunks)
        t = b["t0"]
        for c in chunks:
            d = b["speech"] * len(c) / tot
            srt.append(f"{n}\n{ts(t)} --> {ts(t + d)}\n{c}\n")
            n += 1
            t += d
    srt_path = OUT.with_suffix(".srt")
    srt_path.write_text("\n".join(srt))

    meta = [";FFMETADATA1", "title=Sparse Attention Benchmarking and Optimization — Project Walkthrough",
            "artist=Krish Patel, Aarsh Sanjay Sharma"]
    seen = {}
    for b in beats:
        seen.setdefault(b["chapter_no"], [b["chapter"], b["t0"], b["t1"]])
        seen[b["chapter_no"]][2] = b["t1"]
    for no, (title, a, z) in seen.items():
        meta += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(a*1000)}", f"END={int(z*1000)}", f"title={no:02d} {title}"]
    (work / "meta.txt").write_text("\n".join(meta) + "\n")
    sh(["ffmpeg", "-y", "-i", str(work / "video.mp4"), "-i", str(work / "audio.wav"), "-i", str(srt_path),
        "-i", str(work / "meta.txt"), "-map", "0:v", "-map", "1:a", "-map", "2", "-map_metadata", "3", "-map_chapters", "3",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
        "-movflags", "+faststart", str(OUT)])


# ------------------------------------------------------------------ preview
def preview(beats, work):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import visuals
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    pdir = work / "preview"
    pdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for b in beats:
        info = dict(chapter_no=b["chapter_no"], chapter=b["chapter"], ticks=b["ticks"], progress=b["t1"] / b["total"])
        for tag, p in (("mid", 0.45), ("end", 1.0)):
            visuals.render(fig, b["key"], p, info)
            path = pdir / f"b{b['idx']:02d}_{tag}.png"
            fig.savefig(path, dpi=50, facecolor=fig.get_facecolor())
            paths.append(path)
    ims = [Image.open(p) for p in paths]
    w, h = ims[0].size
    per = 12
    for s in range(0, len(ims), per):
        sheet = Image.new("RGB", (w * 3, h * 4), "white")
        for k, im in enumerate(ims[s:s + per]):
            sheet.paste(im, ((k % 3) * w, (k // 3) * h))
        sheet.save(pdir / f"sheet_{s // per:02d}.jpg", quality=85)
    print("preview sheets in", pdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(HERE / "_work"))
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 1))
    ap.add_argument("--clean-frames", action="store_true")
    a = ap.parse_args()
    work = Path(a.work)
    if a.clean_frames and (work / "frames").exists():
        shutil.rmtree(work / "frames")
    beats, total = build_beats(work)
    print(f"{len(beats)} beats, {total/60:.2f} min, {sum(b['frames'] for b in beats)} frames")
    if a.preview:
        preview(beats, work)
        return
    render_frames(beats, work, a.workers)
    encode(beats, work, total)
    print("wrote", OUT, f"({duration(OUT)/60:.2f} min)")


if __name__ == "__main__":
    main()
