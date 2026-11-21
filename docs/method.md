# Method

## Question

Can a multimodal model draft a usable blade inspection finding from a drone image, and does the cheap model hold up against the frontier-quality one?

## Data

Wind Turbine Blade Defect dataset (WTBD), DOI 10.6084/m9.figshare.30210175, CC BY 4.0. 1,065 drone images at 1024x1024 with VOC annotations across six classes: craze, hide_craze, corrosion, surface_injure, thunderstrike, crack. Every image is defective, so the study measures classification and recall, not specificity.

Sampling is stratified by class, 13 images per class, deduplicated across multi-class images, for 75 evaluated images.

## Models

- `gemini-3.5-flash-lite`: image and video input, JSON response mode, temperature 0.
- `deepseek-v4.1-flash`: image input through an OpenAI-compatible gateway, JSON object mode, temperature 0.

Both receive the same prompt: a fixed taxonomy with one-line definitions, and a JSON schema of defects, severity, location and notes.

## Metrics

- Exact set match: predicted class set equals the annotated set.
- Class-set micro precision, recall and F1 over class instances.
- Per-class recall.
- Critical recall: share of images annotated crack or thunderstrike where the model names either class.
- Cost per 1,000 images from measured tokens at list prices.
- Latency p50 and p95, including retries.

## Video

Eight held-out frames, one per second, encoded as an 8-second MP4. Gemini returns a per-second condition log. Frame-level exact match and critical recall are reported descriptively; eight frames is a demonstration, not a benchmark.

## Business case

Model cost is measured. Labor inputs (images per month, manual minutes per finding, review seconds, loaded hourly cost) are inputs in the report's calculator, so any reader can substitute their own. No labor number is presented as measured.

## Threats to validity

- No healthy images, so no specificity and no auto-clear claim.
- 75 images and a single seed.
- One annotation source; craze and hide_craze are ambiguous.
- p95 includes 429 retries, so it measures the served path.
- Video results come from eight frames.
