import json, base64, glob, xml.etree.ElementTree as ET
from pathlib import Path
import httpx
from cgbi.config import gemini_key

D=Path("data/wtbd/WT blade defect dataset")
P="""You inspect aerial photos of wind turbine blades.
Defect classes: craze (fine network of surface cracks), hide_craze (subtle concealed crazing), corrosion,
surface_injure (surface damage or erosion), thunderstrike (burn marks or lightning damage), crack (visible crack).
Return JSON only: {"defects": [class names present], "severity": "low|medium|high", "notes": "short"}
Use only those class names. If no defect is visible return an empty list."""
def gold(f):
    r=ET.parse(D/"Annotations"/f"{f}.xml").getroot()
    return sorted({o.findtext("name") for o in r.findall("object")})
for name in ["162","604","88","610","176"]:
    img=D/"JPEGImages"/f"{name}.jpg"
    body={"contents":[{"parts":[{"text":P},{"inline_data":{"mime_type":"image/jpeg","data":base64.b64encode(img.read_bytes()).decode()}}]}],
          "generationConfig":{"temperature":0,"responseMimeType":"application/json","maxOutputTokens":300}}
    r=httpx.post("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent",
                 headers={"x-goog-api-key":gemini_key()},json=body,timeout=90)
    try:
        txt=r.json()["candidates"][0]["content"]["parts"][0]["text"]
        pred=json.loads(txt)
    except Exception as e:
        pred={"error":str(e),"body":r.text[:150]}
    print(name,"gold",gold(name),"pred",pred)
