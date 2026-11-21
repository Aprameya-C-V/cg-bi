import asyncio, json, random, xml.etree.ElementTree as ET
from pathlib import Path
from gridinspect.providers import GeminiVision

D=Path("data/wtbd/WT blade defect dataset")
CLASSES=["craze","hide_craze","corrosion","surface_injure","thunderstrike","crack"]
PROMPT="""You inspect aerial photos of wind turbine blades.
Defect classes: craze (fine network of surface cracks), hide_craze (subtle concealed crazing), corrosion,
surface_injure (surface damage or erosion), thunderstrike (burn marks or lightning damage), crack (visible crack).
Return JSON only: {"defects": [class names present], "severity": "low|medium|high", "notes": "short"}
Use only those class names."""
def gold(n):
    r=ET.parse(D/"Annotations"/f"{n}.xml").getroot()
    return sorted({o.findtext("name") for o in r.findall("object")})
async def run(model,names):
    sem=asyncio.Semaphore(4)
    async def one(n):
        async with sem:
            r=await model.inspect_image(D/"JPEGImages"/f"{n}.jpg",PROMPT)
            try: pred=json.loads(r.raw["candidates"][0]["content"]["parts"][0]["text"])["defects"]
            except Exception: pred=[]
            return n,gold(n),[p for p in pred if p in CLASSES],r
    return await asyncio.gather(*(one(n) for n in names))
def stats(rows):
    exact=sum(1 for _,g,p,_ in rows if set(g)==set(p))/len(rows)
    tp=sum(len(set(g)&set(p)) for _,g,p,_ in rows); fp=sum(len(set(p)-set(g)) for _,g,p,_ in rows); fn=sum(len(set(g)-set(p)) for _,g,p,_ in rows)
    prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
    crit=[(g,p) for _,g,p,_ in rows if ("crack" in g or "thunderstrike" in g)]
    return {"exact":round(exact,3),"f1":round(2*prec*rec/(prec+rec),3) if prec+rec else 0,"critical_recall":round(sum(1 for g,p in crit if set(g)&set(p))/len(crit),3) if crit else 0,"cost":round(sum(r.cost_usd for _,_,_,r in rows),5)}
async def main():
    random.seed(11)
    imgs=random.sample(sorted(p.stem for p in (D/"JPEGImages").glob("*.jpg")),15)
    for model in [GeminiVision("gemini-3.5-flash-lite"), GeminiVision("gemini-3.8-flash")]:
        try: rows=await run(model,imgs)
        finally: await model.aclose()
        print(model.model, stats(rows))
        for n,g,p,_ in rows[:4]: print("  ",n,g,p)
asyncio.run(main())
