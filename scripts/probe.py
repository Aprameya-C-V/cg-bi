import base64, json, os, time
import httpx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data/InsulatorDataSet-master"

def gemini_key():
    return Path.home().joinpath(".config/gemini-api-key").read_text().strip()

def go_key():
    return json.loads(Path.home().joinpath(".local/share/opencode/auth.json").read_text())["opencode-go"]["key"]

PROMPT = """You inspect aerial images of power line insulators.
Return JSON only: {"defect_present": bool, "defect_type": "none|broken|flashover|contamination|other",
"severity": "none|low|medium|high", "boxes": [{"label": str, "ymin": int, "xmin": int, "ymax": int, "xmax": int}],
"notes": str}. Coordinates are 0-1000 normalized."""

def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()

def gemini(img):
    body = {"contents":[{"parts":[{"text":PROMPT},{"inline_data":{"mime_type":"image/jpeg","data":b64(img)}}]}],
            "generationConfig":{"temperature":0,"responseMimeType":"application/json","maxOutputTokens":400}}
    t=time.perf_counter()
    r=httpx.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent",
                 headers={"x-goog-api-key":gemini_key()}, json=body, timeout=90)
    ms=(time.perf_counter()-t)*1000
    print("gemini", r.status_code, round(ms), r.text[:400])

def deepseek(img):
    body={"model":"deepseek-v4.1-flash","temperature":0,"response_format":{"type":"json_object"},
          "messages":[{"role":"user","content":[{"type":"text","text":PROMPT},
          {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64(img)}}]}]}
    t=time.perf_counter()
    r=httpx.post("https://opencode.ai/inference/go/openai/v1/chat/completions",
                 headers={"Authorization":"Bearer "+go_key(),"content-type":"application/json",
                          "User-Agent":"cgbi/0.1","x-opencode-session":"cgbi-p1"},
                 json=body, timeout=90)
    ms=(time.perf_counter()-t)*1000
    print("deepseek", r.status_code, round(ms), r.text[:400])

def main():
    defective = sorted((IMG_DIR/"Defective_Insulators/images").glob("*"))[0]
    normal = sorted((IMG_DIR/"Normal_Insulators/images").glob("*"))[0]
    print("defective", defective.name, "normal", normal.name)
    gemini(defective); deepseek(defective); gemini(normal); deepseek(normal)

if __name__ == "__main__":
    main()
