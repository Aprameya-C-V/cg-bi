import base64, json, os, time
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/InsulatorDataSet-master"
KEY_G=Path.home().joinpath(".config/gemini-api-key").read_text().strip()
KEY_GO=[l.split("=",1)[1].strip().strip('"') for l in open("/Volumes/MacMiniSSD/Portfolio Projects/hbc-rag/keys.env") if l.startswith("HBC_GO_API_KEY=")][0]

PROMPT="""You inspect aerial images of high-voltage insulators on transmission towers.
Damage appears as a broken or missing ceramic disc, a burn/flashover mark, or heavy dark contamination on the discs.
Look closely along the whole insulator string, including near image edges.
Return JSON only: {"defect_present": bool, "defect_type": "none|broken|flashover|contamination|other",
"severity": "none|low|medium|high", "boxes": [{"label": str, "ymin": int, "xmin": int, "ymax": int, "xmax": int}],
"notes": str}. Coordinates 0-1000 normalized."""

def b64(p): return base64.b64encode(p.read_bytes()).decode()

def gemini(p, model="gemini-3.5-flash-lite"):
    body={"contents":[{"parts":[{"text":PROMPT},{"inline_data":{"mime_type":"image/jpeg","data":b64(p)}}]}],
          "generationConfig":{"temperature":0,"responseMimeType":"application/json","maxOutputTokens":400}}
    r=httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                 headers={"x-goog-api-key":KEY_G},json=body,timeout=90)
    if r.status_code!=200: return {"error":r.status_code,"body":r.text[:120]}
    return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])

def deepseek(p):
    body={"model":"deepseek-v4.1-flash","temperature":0,"response_format":{"type":"json_object"},
          "messages":[{"role":"user","content":[{"type":"text","text":PROMPT},
          {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64(p)}}]}]}
    r=httpx.post("https://opencode.ai/inference/go/openai/v1/chat/completions",
                 headers={"Authorization":"Bearer "+KEY_GO,"content-type":"application/json","User-Agent":"gridinspect/0.1","x-opencode-session":"gridinspect-p1"},
                 json=body,timeout=90)
    if r.status_code!=200: return {"error":r.status_code,"body":r.text[:120]}
    return json.loads(r.json()["choices"][0]["message"]["content"])

def main():
    defs=sorted((DATA/"Defective_Insulators/images").glob("*"))[:4]
    nors=sorted((DATA/"Normal_Insulators/images").glob("*"))[:2]
    for p in defs+nors:
        g=gemini(p)
        print(p.parent.parent.name[:4], p.name, "gemini", g.get("defect_present"), g.get("defect_type"), str(g.get("notes"))[:40], g.get("error",""))
    p=defs[0]
    d=deepseek(p)
    print("deepseek", p.name, json.dumps(d)[:200])

if __name__=="__main__": main()
