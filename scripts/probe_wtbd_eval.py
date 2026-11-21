import asyncio, json, random, xml.etree.ElementTree as ET
from pathlib import Path
from gridinspect.providers import GeminiVision, DeepSeekVision

D=Path("data/wtbd/WT blade defect dataset")
CLASSES=["craze","hide_craze","corrosion","surface_injure","thunderstrike","crack"]
PROMPT="""You inspect aerial photos of wind turbine blades.
Defect classes: craze (fine network of surface cracks), hide_craze (subtle concealed crazing), corrosion,
surface_injure (surface damage or erosion), thunderstrike (burn marks or lightning damage), crack (visible crack).
Return JSON only: {"defects": [class names present], "severity": "low|medium|high", "notes": "short"}
Use only those class names. If no defect is visible return an empty list."""

def gold(name):
    r=ET.parse(D/"Annotations"/f"{name}.xml").getroot()
    return sorted({o.findtext("name") for o in r.findall("object")})

async def run(model, names):
    sem=asyncio.Semaphore(6)
    async def one(n):
        async with sem:
            r=await model.inspect_image(D/"JPEGImages"/f"{n}.jpg", PROMPT)
            try: pred=json.loads(r.raw["candidates"][0]["content"]["parts"][0]["text"])["defects"]
            except Exception:
                try: pred=json.loads(r.raw["choices"][0]["message"]["content"])["defects"]
                except Exception: pred=[]
            return n, gold(n), [p for p in pred if p in CLASSES], r
    return await asyncio.gather(*(one(n) for n in names))

async def main():
    random.seed(7)
    all_imgs=sorted(p.stem for p in (D/"JPEGImages").glob("*.jpg"))
    sample=random.sample(all_imgs, 40)
    for cls in (GeminiVision, DeepSeekVision):
        m=cls()
        try:
            rows=await run(m, sample)
        finally:
            await m.aclose()
        exact=sum(1 for _,g,p,_ in rows if set(g)==set(p))/len(rows)
        tp=sum(len(set(g)&set(p)) for _,g,p,_ in rows); fp=sum(len(set(p)-set(g)) for _,g,p,_ in rows); fn=sum(len(set(g)-set(p)) for _,g,p,_ in rows)
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        f1=2*prec*rec/(prec+rec) if prec+rec else 0
        crit=[(g,p) for _,g,p,_ in rows if ("crack" in g or "thunderstrike" in g)]
        crit_rec=sum(1 for g,p in crit if set(g)&set(p))/len(crit) if crit else 0
        cost=sum(r.cost_usd for _,_,_,r in rows)
        print(m.model, json.dumps({"exact":round(exact,3),"precision":round(prec,3),"recall":round(rec,3),"f1":round(f1,3),"critical_recall":round(crit_rec,3),"n_critical":len(crit),"cost_40":round(cost,5)}))
        for n,g,p,_ in rows[:8]:
            print("  ",n,"gold",g,"pred",p)

asyncio.run(main())
