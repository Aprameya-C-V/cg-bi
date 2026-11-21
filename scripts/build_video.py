import asyncio, json, subprocess
from pathlib import Path
from cgbi.data import CLASSES, load_manifest
from cgbi.prompts import VIDEO
from cgbi.providers import GeminiVision

D=Path("data/video"); D.mkdir(parents=True, exist_ok=True)
man=load_manifest()
picks=[s for s in man if "crack" in s.classes][:2]+[s for s in man if "thunderstrike" in s.classes][:2]+[s for s in man if "surface_injure" in s.classes][:2]+[s for s in man if "corrosion" in s.classes][:2]
listfile=D/"frames.txt"
lines=[]
for s in picks:
    lines.append(f"file '{s.image.resolve()}'\nduration 1")
lines.append(f"file '{picks[-1].image.resolve()}'")
listfile.write_text("\n".join(lines)+"\n")
out=D/"sweep.mp4"
subprocess.run(["ffmpeg","-y","-loglevel","error","-f","concat","-safe","0","-i",str(listfile),"-vf","fps=1,scale=1024:1024","-c:v","libx264","-pix_fmt","yuv420p",str(out)],check=True)
labels=[{"second":i,"classes":s.classes} for i,s in enumerate(picks)]
(D/"labels.json").write_text(json.dumps(labels,indent=2))
print("video",out, out.stat().st_size//1024,"KB", [s.classes for s in picks])

async def main():
    m=GeminiVision("gemini-3.5-flash-lite")
    try:
        r=await m.inspect_video(out, VIDEO)
    finally:
        await m.aclose()
    print(json.dumps(r, indent=2)[:1200])
    (D/"result.json").write_text(json.dumps(r,indent=2))
asyncio.run(main())
