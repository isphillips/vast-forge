import base64, json, os
B = "c:/Users/Manny/dev/vast-forge/bakeoff/bake"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forge-bakeoff.html")
stats = json.load(open(f"{B}/bake_stats.json"))
# shape-stage numbers from the first run's log (the relaunch reused the cached shape meshes)
SHAPE = {"mustache": (88, 776682, 7.6), "car": (84, 344664, 7.6), "turtle": (87, 577042, 7.6)}
# NOTE: the handoff's two image URLs were swapped, so the files tagged "car" are the Leonardo mask and the files
# tagged "turtle" are the car. Tags are kept as the file names; labels are what the image actually is.
TAGS = [("mustache", "Mustache and glasses", "Hair, thin rims, eyebrows"),
        ("turtle", "Autumn Jones car", "Dense object; hardest for TRELLIS"),
        ("car", "Leonardo mask", "Frontal view; ambiguous depth")]

def b64(p): return base64.b64encode(open(p, "rb").read()).decode()
models = {f"{t}_{e}": b64(f"{B}/q_{t}_{e}.glb.gz") for t, _, _ in TAGS for e in ("trellis", "hy")}
models["car_hy2"] = b64(f"{B}/q_mask_seed1_hy.glb.gz")          # mask, Hunyuan re-roll with seed 1
SEED1 = json.load(open(f"{B}/mask_seed1_stats.json")); SEED1_SHAPE_S = 89
thumbs = {t: "data:image/jpeg;base64," + b64(f"{B}/thumb_{t}.jpg") for t, _, _ in TAGS}

def n(v): return f"{v:,}"
def yesno(b): return '<span class="ok">closed</span>' if b else '<span class="bad">open</span>'
rows = []
for t, label, _ in TAGS:
    s = stats[t]; tr, hy = s["trellis"], s["hy"]; sh_s, raw, sh_gb = SHAPE[t]
    tr_time = f"{s['trellis_s']} s"
    if t == "mustache": tr_time += "<sup>a</sup>"
    if t == "car": tr_time += "<sup>b</sup>"  # tag "car" = mask image
    span = 3 if t == "car" else 2
    rows.append(f'''
          <tr class="grp"><td class="k" rowspan="{span}"><img src="{thumbs[t]}" alt="" class="rowthumb"><span>{label}</span></td>
            <td><span class="eng tr">TRELLIS.2</span></td><td class="r">{n(tr['faces'])}</td><td class="r">{n(tr['open_edges'])}<sup>c</sup></td><td class="r">{n(tr['parts'])}</td><td>{yesno(tr['watertight'])}</td><td class="r">{tr['mb']} MB</td><td class="r">{tr['low_alpha_pct']}%</td><td class="r">{tr_time}</td><td class="r">n/a<sup>d</sup></td></tr>
          <tr><td><span class="eng hy">Hunyuan3D 2.1</span></td><td class="r">{n(hy['faces'])}</td><td class="r">{n(hy['open_edges'])}</td><td class="r">{n(hy['parts'])}</td><td>{yesno(hy['watertight'])}</td><td class="r">{hy['mb']} MB</td><td class="r">n/a<sup>e</sup></td><td class="r">{sh_s + s['hy_paint_s']} s <span class="sub">{sh_s} shape + {s['hy_paint_s']} paint</span></td><td class="r">{s['hy_paint_peak_gb']} GB <span class="sub">{sh_gb} shape</span></td></tr>''')
    if t == "car":
        q = SEED1
        rows.append(f'''
          <tr><td><span class="eng hy">Hunyuan3D 2.1 · seed 1</span><sup>f</sup></td><td class="r">{n(q['faces'])}</td><td class="r">{n(q['open_edges'])}</td><td class="r">{n(q['parts'])}</td><td>{yesno(q['watertight'])}</td><td class="r">{q['mb']} MB</td><td class="r">n/a<sup>e</sup></td><td class="r">{SEED1_SHAPE_S + q['hy_paint_s']} s <span class="sub">{SEED1_SHAPE_S} shape + {q['hy_paint_s']} paint</span></td><td class="r">{q['hy_paint_peak_gb']} GB <span class="sub">7.6 shape</span></td></tr>''')
rows_html = "".join(rows)
raw_html = " · ".join(f"{ {'mustache': 'mustache', 'turtle': 'car', 'car': 'mask'}[t] } {n(SHAPE[t][1])}" for t, l, _ in TAGS)

tabs_html = "".join(f'''<button class="tab" data-tag="{t}" aria-pressed="false"><img src="{thumbs[t]}" alt=""><span><b>{label}</b><small>{hint}</small></span></button>''' for t, label, hint in TAGS)

html = r'''<title>Forge Bake-off</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {
    --bg: #F6F3FC; --panel: #FFFFFF; --panel-2: #F0EBFA; --line: #D9D0EC; --line-soft: #E7E1F3;
    --ink: #1B1233; --dim: #5E5478; --mute: #8A80A6;
    --violet: #7C3AED; --teal: #0FA99F; --gold: #B8860B;
    --violet-ink: #6D28D9; --teal-ink: #0B7F78; --gold-ink: #8A6508;
    --teal-soft: rgba(15,169,159,0.14); --gold-soft: rgba(184,134,11,0.14);
    --good: #1B9E5A; --warn: #B7791F; --bad: #C8304A;
    --stage: #EDE7F7;
    --sans: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
    --display: "Sora", "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #0F0A1F; --panel: #1A1230; --panel-2: #221840; --line: #33284F; --line-soft: #271D3E;
      --ink: #F1ECFA; --dim: #A79BC4; --mute: #7A6F98;
      --violet: #A05CFF; --teal: #2DD4BF; --gold: #FFD24A;
      --violet-ink: #C9A6FF; --teal-ink: #7FE9DC; --gold-ink: #FFE08A;
      --teal-soft: rgba(45,212,191,0.16); --gold-soft: rgba(255,210,74,0.16);
      --good: #2FE08A; --warn: #FFB020; --bad: #FF5470;
      --stage: #140E28;
    }
  }
  :root[data-theme="dark"] {
    --bg: #0F0A1F; --panel: #1A1230; --panel-2: #221840; --line: #33284F; --line-soft: #271D3E;
    --ink: #F1ECFA; --dim: #A79BC4; --mute: #7A6F98;
    --violet: #A05CFF; --teal: #2DD4BF; --gold: #FFD24A;
    --violet-ink: #C9A6FF; --teal-ink: #7FE9DC; --gold-ink: #FFE08A;
    --teal-soft: rgba(45,212,191,0.16); --gold-soft: rgba(255,210,74,0.16);
    --good: #2FE08A; --warn: #FFB020; --bad: #FF5470;
    --stage: #140E28;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 var(--sans); }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 36px 22px 80px; }
  h1, h2, h3 { font-family: var(--display); text-wrap: balance; margin: 0; }
  h1 { font-size: 34px; font-weight: 700; letter-spacing: -0.01em; }
  h2 { font-size: 21px; font-weight: 600; margin: 0 0 6px; }
  h3 { font-size: 15px; font-weight: 600; }
  p { max-width: 72ch; margin: 0; }
  section { margin-top: 44px; }
  .lede { color: var(--dim); font-size: 16px; margin-top: 10px; }
  .eyebrow { font: 500 11px/1 var(--mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--mute); margin-bottom: 8px; }
  code { font-family: var(--mono); font-size: 0.92em; background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 4px; padding: 1px 5px; }
  sup { font: 500 10px/0 var(--mono); color: var(--mute); margin-left: 1px; }
  .eng { font: 500 12px/1 var(--mono); letter-spacing: 0.04em; padding: 4px 8px; border-radius: 4px; white-space: nowrap; }
  .eng.tr { background: var(--gold-soft); color: var(--gold-ink); }
  .eng.hy { background: var(--teal-soft); color: var(--teal-ink); }
  .ok { color: var(--good); font-weight: 600; } .bad { color: var(--bad); font-weight: 600; }

  .verdict { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 22px; }
  .verdict > div { background: var(--panel); border: 1px solid var(--line-soft); border-radius: 10px; padding: 14px 16px; }
  .verdict .lab { font: 500 11px/1 var(--mono); letter-spacing: 0.1em; text-transform: uppercase; color: var(--mute); }
  .verdict .val { font-family: var(--display); font-size: 22px; font-weight: 600; margin-top: 8px; line-height: 1.2; }
  .verdict .val b { color: var(--teal-ink); font-weight: 600; }
  .verdict .val i { color: var(--gold-ink); font-style: normal; }
  .verdict .sub { color: var(--dim); font-size: 13px; margin-top: 4px; }

  .lab-box { background: var(--panel); border: 1px solid var(--line-soft); border-radius: 12px; overflow: hidden; }
  .tabs { display: flex; gap: 8px; padding: 12px; border-bottom: 1px solid var(--line-soft); flex-wrap: wrap; }
  .tab { display: flex; align-items: center; gap: 10px; text-align: left; font: 500 13px/1.25 var(--sans); color: var(--ink); background: var(--panel-2); border: 1px solid var(--line-soft); border-radius: 8px; padding: 6px 12px 6px 6px; cursor: pointer; flex: 1 1 200px; }
  .tab img { width: 40px; height: 40px; border-radius: 6px; object-fit: cover; background: #fff; }
  .tab small { display: block; color: var(--mute); font: 400 11.5px/1.3 var(--mono); margin-top: 3px; }
  .tab[aria-pressed="true"] { border-color: var(--violet); box-shadow: inset 0 0 0 1px var(--violet); }
  .tab:focus-visible, .hud button:focus-visible { outline: 2px solid var(--teal); outline-offset: 2px; }
  .stage { position: relative; display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line-soft); }
  .pane { position: relative; height: 460px; background: var(--stage); }
  .pane canvas { display: block; width: 100%; height: 100%; }
  .pane .tag { position: absolute; left: 12px; top: 12px; font: 500 12px/1 var(--mono); letter-spacing: 0.06em; padding: 7px 10px; border-radius: 6px; pointer-events: none; }
  .pane.l .tag { background: var(--gold-soft); color: var(--gold-ink); }
  .pane.r .tag { background: var(--teal-soft); color: var(--teal-ink); }
  .variant { position: absolute; right: 12px; top: 12px; display: flex; gap: 4px; }
  .variant button { font: 500 12px/1 var(--mono); color: var(--dim); background: color-mix(in srgb, var(--panel) 85%, transparent); border: 1px solid var(--line); border-radius: 6px; padding: 7px 9px; cursor: pointer; }
  .variant button[aria-pressed="true"] { border-color: var(--teal); color: var(--teal-ink); }
  .status { position: absolute; right: 12px; bottom: 12px; font: 500 12px/1 var(--mono); color: var(--dim); background: color-mix(in srgb, var(--panel) 85%, transparent); border: 1px solid var(--line-soft); border-radius: 8px; padding: 8px 10px; max-width: calc(100% - 24px); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; pointer-events: none; }
  .hud { display: flex; gap: 8px; flex-wrap: wrap; padding: 12px; border-top: 1px solid var(--line-soft); align-items: center; }
  .hud button { font: 500 12px/1 var(--sans); color: var(--ink); background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; cursor: pointer; }
  .hud button[aria-pressed="true"] { border-color: var(--teal); color: var(--teal-ink); }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; margin-left: auto; color: var(--dim); font-size: 13px; }
  .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .sw { width: 18px; height: 3px; border-radius: 2px; display: inline-block; }
  @media (max-width: 720px) { .stage { grid-template-columns: 1fr; } .pane { height: 320px; } .legend { margin-left: 0; } }

  .tbl { overflow-x: auto; border: 1px solid var(--line-soft); border-radius: 10px; background: var(--panel); margin-top: 16px; }
  table { border-collapse: collapse; width: 100%; min-width: 900px; font-size: 13.5px; }
  th { text-align: left; font: 500 11px/1.2 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; color: var(--mute); padding: 12px 12px 10px; border-bottom: 1px solid var(--line); }
  th.r { text-align: right; }
  td { padding: 10px 12px; border-bottom: 1px solid var(--line-soft); vertical-align: middle; }
  tr.grp td { border-top: 1px solid var(--line); }
  tr:last-child td { border-bottom: 0; }
  td.k { white-space: nowrap; font-weight: 500; }
  td.k img { vertical-align: middle; margin-right: 10px; }
  .rowthumb { width: 36px; height: 36px; border-radius: 6px; object-fit: cover; background: #fff; }
  td.r { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  td .sub { display: block; color: var(--mute); font: 400 11px/1.3 var(--mono); margin-top: 2px; }
  .notes { color: var(--mute); font-size: 12.5px; margin-top: 12px; max-width: none; line-height: 1.6; }
  .notes sup { margin-right: 3px; }

  .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; margin-top: 16px; }
  .card { background: var(--panel); border: 1px solid var(--line-soft); border-radius: 10px; padding: 16px 18px; }
  .card h3 { margin-bottom: 8px; }
  .card p, .card li { color: var(--dim); font-size: 13.5px; }
  .card p b, .rec p b { color: var(--ink); font-weight: 600; }
  .rec { border-left: 3px solid var(--teal); padding-left: 18px; margin-top: 16px; }
  .rec p { margin-top: 8px; color: var(--dim); }
  .rec p:first-child { margin-top: 0; }
  .steps { margin: 10px 0 0; padding-left: 20px; color: var(--dim); font-size: 14px; max-width: 76ch; }
  .steps li { margin: 6px 0; }
  .h3gap { margin-top: 24px; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Vidrip · Facet Forge · engine bake-off · 12 Sep 2026 · one RTX 3090 forge node</div>
    <h1>Forge Bake-off</h1>
    <p class="lede">The same three source images forged twice on one box: TRELLIS.2, which the forge ships today, and Hunyuan3D 2.1, the candidate. Both were held to the 120,000-face budget and are rendered here the way the app shades them (plain diffuse, no gloss). Left is TRELLIS, right is Hunyuan, and they share one camera, so drag either side. Turn on <em>open edges</em> and look at the mustache tips, the eyebrows, the car's grille and the mask's shell. On the mask, the right pane has two Hunyuan results to switch between: the first run, which misread the depth, and a re-roll with a fixed seed.</p>
    <div class="verdict">
      <div><div class="lab">Closed surfaces</div><div class="val"><b>3 of 3</b> Hunyuan · <i>0 of 3</i> TRELLIS</div><div class="sub">zero open edges, one piece, every image</div></div>
      <div><div class="lab">Loose pieces</div><div class="val"><b>1</b> vs <i>345 · 1,884 · 4,312</i></div><div class="sub">mustache · car · mask</div></div>
      <div><div class="lab">Time per facet</div><div class="val"><b>151–169 s</b> vs <i>245–410 s</i></div><div class="sub">Hunyuan shape + paint; TRELLIS end to end</div></div>
      <div><div class="lab">Peak GPU memory</div><div class="val"><b>13.8 GB</b> of 24</div><div class="sub">Hunyuan paint; shape stage 7.6 GB</div></div>
    </div>
  </header>

  <section>
    <div class="lab-box">
      <div class="tabs" id="tabs" role="group" aria-label="Source image">__TABS__</div>
      <div class="stage" id="stage">
        <div class="pane l"><canvas id="cL"></canvas><span class="tag">TRELLIS.2 · ships today</span><div class="status" id="sL">loading…</div></div>
        <div class="pane r"><canvas id="cR"></canvas><span class="tag">HUNYUAN3D 2.1 · candidate</span><div class="variant" id="variant" hidden><button data-v="hy" aria-pressed="true">first run</button><button data-v="hy2" aria-pressed="false">seed 1 re-roll</button></div><div class="status" id="sR">loading…</div></div>
      </div>
      <div class="hud">
        <button id="btnEdges" aria-pressed="false">open edges</button>
        <button id="btnWire" aria-pressed="false">wireframe</button>
        <button id="btnTex" aria-pressed="true">texture</button>
        <button id="btnReset">reset view</button>
        <div class="legend">
          <span><i class="sw" style="background: var(--bad)"></i> open edge · a crack or hole boundary</span>
          <span>drag to orbit · wheel to zoom · shift-drag to pan</span>
        </div>
      </div>
    </div>
  </section>

  <section>
    <div class="eyebrow">Measured on the full exports</div>
    <h2>The numbers</h2>
    <p class="lede">Open edges are mesh edges used by only one triangle after vertices are merged by position: zero means a closed surface. Pieces is how many disconnected parts the file has. Grey texels is the share of texture pixels inside the UV islands that came back with alpha under 50%, which the app draws as grey haze. Both engines were measured with the same code on the full GLBs, before the viewer copies were shrunk.</p>
    <div class="tbl">
      <table>
        <thead><tr><th>Image</th><th>Engine</th><th class="r">Faces</th><th class="r">Open edges</th><th class="r">Pieces</th><th>Surface</th><th class="r">File</th><th class="r">Grey texels</th><th class="r">Time</th><th class="r">Peak VRAM</th></tr></thead>
        <tbody>__ROWS__
        </tbody>
      </table>
    </div>
    <p class="notes"><sup>a</sup>Includes this node's first TRELLIS load (weights pulled from cold). <sup>b</sup>The mask was TRELLIS's slowest job here; the server does not report which rung of its out-of-memory retry ladder a job lands on. <sup>c</sup>TRELLIS's open-edge counts include hairline cracks left where the exporter deletes zero-area sliver triangles; the pieces column is the honest measure of fragmentation. <sup>d</sup>The forge server does not report its peak. In production the car image has overflowed the 24 GB shape decoder and fallen back to the 512 pipeline. <sup>e</sup>Hunyuan bakes an opaque JPEG albedo and inpaints uncovered texels, so the alpha-haze mechanism does not exist there. Hunyuan raw shape output before reduction to 120k: __RAW__ faces. <sup>f</sup>Re-roll of the mask's shape stage with seed 1; the first run had no fixed seed. Both Hunyuan stages were timed with their models already resident; TRELLIS times are the server's wall clock per request.</p>
  </section>

  <section>
    <div class="eyebrow">What to look at</div>
    <h2>Three images, three failure modes</h2>
    <div class="grid2">
      <div class="card"><h3>Mustache and glasses · hair and thin rims</h3><p>TRELLIS generates a sparse voxel <b>surface</b>, so hair becomes torn sheets: 345 pieces and thousands of open edges around the eyebrows and mustache tips. Hunyuan reconstructs a closed signed-distance surface, so the same hair arrives as one solid mass, softer at the tips. Judge whether the softening is acceptable against the tearing.</p></div>
      <div class="card"><h3>Autumn Jones car · a dense object</h3><p>In production this image overflows TRELLIS's shape decoder and falls back to the 512 pipeline; here it came out in 245 s with <b>1,884 loose pieces</b>, 30,550 open edges and 4.6% grey texels. Hunyuan held 13.8 GB, kept one closed body whose proportions match the car (2 : 0.8 : 0.6), and finished in 153 s. Compare the grille, the headlights and the wheel arches, then compare texture sharpness on the paint.</p></div>
      <div class="card"><h3>Leonardo mask · a frontal view</h3><p>TRELLIS read it as a head-shaped shell and shredded it: <b>4,312 pieces</b>, 44,726 open edges, 410 s. Hunyuan's first run is closed and single-piece but <b>wrong</b>: it read the frontal image as a whole turtle and extruded a body four times deeper than wide, with the shell texture repeating down its length. A re-roll with four fixed seeds gave heads 1.4 to 1.7 times deeper than wide every time, so the extrusion is a one-in-five outlier; the typical Hunyuan reading is a full head with the shell behind, not a shallow mask. Both results are in the viewer.</p></div>
    </div>
  </section>

  <section>
    <div class="eyebrow">Recommendation</div>
    <h2>Replace TRELLIS.2 with Hunyuan3D 2.1, with a shape sanity check</h2>
    <div class="rec">
      <p><b>On every measured axis Hunyuan wins, and on the ones that hurt today it wins outright.</b> Every export is a single closed surface, which retires the hole-filling, solidify and sliver-crack work the TRELLIS path accumulated. It never approaches the 24 GB ceiling (7.6 GB for shape, 13.8 GB for paint), so the retry ladder and the 512 fallback that produced the blobby car go away. It is faster end to end even before the TRELLIS times are discounted for the cold load and the retries. Files are smaller at the same face budget.</p>
      <p><b>Its failure mode is different, and cheaper to catch.</b> TRELLIS fails on the surface: fragments, cracks, haze. Hunyuan fails on the reading: on a frontal image with ambiguous depth it can hallucinate what is behind, and the mask's first run did exactly that. Because the mesh is closed, the failure shows up in one number: the first run's depth-to-width ratio was 3.9 against 1.4 to 1.7 for four re-rolls. A check after the shape stage that re-rolls with a new seed when the ratio passes about 2.5 costs one extra 85 s pass, was needed once in five here, and would have caught it. TRELLIS's failures have no such single number.</p>
      <p><b>The one thing this page cannot decide is texture fidelity</b>, which is why the models are here to turn. Hunyuan's paint is a multi-view bake (6 views at 512, upscaled) rather than TRELLIS's per-voxel color, so it can be softer on fine print and can blend across seams; the trade is that it is complete, with no grey haze. If the paint reads as good enough on the car and the mask re-roll, switch.</p>
      <p><b>What the switch costs.</b> A separate environment: Hunyuan wants torch 2.5.1 (the box runs 2.6), an 8 GB venv, and about 10 GB of weights next to TRELLIS's. Same box, same 120k-face and 1024-texture budgets, same /generate contract.</p>
    </div>
    <h3 class="h3gap">Notes for the Dockerfile, learned on this box</h3>
    <ol class="steps">
      <li>The base image needs <code>python3.10-venv</code>; the CUDA devel image lacks ensurepip.</li>
      <li>Install <code>basicsr</code>, <code>custom_rasterizer</code> and <code>deepspeed</code> with <code>--no-build-isolation</code>; their setup imports torch, and an isolated build pulls a second, current torch (multi-GB) and then fails.</li>
      <li>Drop <code>bpy</code> from requirements (no Python 3.10 wheel). It is only used to convert the painted OBJ to GLB; trimesh does that in one line, and this run used exactly that.</li>
      <li>Pin <code>setuptools&lt;81</code>: the paint stage imports <code>pkg_resources</code>, which newer setuptools removed.</li>
      <li>Apply the repo's <code>torchvision_fix</code> before importing the paint pipeline (basicsr imports a module torchvision 0.20 removed).</li>
      <li>Call the paint pipeline with <code>use_remesh=False</code>: its remesh is a quadric decimation to a hard-coded 40,000 faces, which would silently discard two thirds of the budget. Reduce to 120k after the shape stage instead, as here.</li>
      <li>After the shape stage, measure the reduced mesh's extents and re-run the shape stage with a new seed when depth/width exceeds about 2.5 (one in five mask runs here); only then paint.</li>
      <li>Serve shape and paint from one process and keep both models resident: shape 7.6 GB and paint 13.8 GB peaks fit together in 24 GB, and the per-request cost is then the 151–169 s measured here.</li>
    </ol>
  </section>
</div>

<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pako/2.1.0/pako.min.js"></script>
<script>window.FORGE_MODELS = __MODELS__;</script>
<script>
(function () {
  var M = window.FORGE_MODELS || {};
  var ENG = { trellis: { canvas: document.getElementById('cL'), status: document.getElementById('sL') },
              hy:      { canvas: document.getElementById('cR'), status: document.getElementById('sR') } };
  var stage = document.getElementById('stage');
  var camera, controls, current = null, loaded = {}, showEdges = false, showWire = false, showTex = true, hyVariant = 'hy';
  var variantBox = document.getElementById('variant');

  function b64ToBuf(b64) {
    var bin = atob(b64), len = bin.length, bytes = new Uint8Array(len);
    for (var i = 0; i < len; i++) bytes[i] = bin.charCodeAt(i);
    if (bytes[0] === 0x1f && bytes[1] === 0x8b && window.pako) { return window.pako.ungzip(bytes).buffer; }
    return bytes.buffer;
  }
  // The sandbox allows data: for images but not for fetch(); hiding createImageBitmap makes GLTFLoader use <img>.
  try { window.createImageBitmap = undefined; } catch (e) { /* ignore */ }

  // Viewer copies ship with KHR_mesh_quantization (int16 positions, uint16 UVs). three r128 only dequantizes in the
  // shader, so turn them back into floats here for bounding boxes, normals and the open-edge pass.
  function dequantize(g) {
    ['position', 'uv'].forEach(function (n) {
      var a = g.attributes[n]; if (!a || !a.normalized) return;
      var src = a.array, out = new Float32Array(src.length);
      var d = (src instanceof Int16Array) ? 32767 : (src instanceof Uint16Array) ? 65535 : (src instanceof Int8Array) ? 127 : 255;
      for (var i = 0; i < src.length; i++) out[i] = Math.max(src[i] / d, -1);
      g.setAttribute(n, new THREE.Float32BufferAttribute(out, a.itemSize));
    });
    g.deleteAttribute('normal'); g.computeVertexNormals(); g.boundingBox = null; g.boundingSphere = null;
  }

  function stripTextures(buffer) {
    var dv = new DataView(buffer);
    if (dv.getUint32(0, true) !== 0x46546C67) return buffer;
    var total = dv.getUint32(8, true), off = 12, chunks = [];
    while (off < total) { var len = dv.getUint32(off, true), type = dv.getUint32(off + 4, true); chunks.push({ type: type, data: buffer.slice(off + 8, off + 8 + len) }); off += 8 + len; }
    var json = JSON.parse(new TextDecoder().decode(chunks[0].data));
    delete json.images; delete json.textures; delete json.samplers;
    (json.materials || []).forEach(function (m) { delete m.normalTexture; delete m.occlusionTexture; delete m.emissiveTexture; if (m.pbrMetallicRoughness) { delete m.pbrMetallicRoughness.baseColorTexture; delete m.pbrMetallicRoughness.metallicRoughnessTexture; } delete m.extensions; });
    var jb = new TextEncoder().encode(JSON.stringify(json));
    var jpad = (4 - jb.length % 4) % 4, jlen = jb.length + jpad;
    var bin = chunks[1] ? new Uint8Array(chunks[1].data) : new Uint8Array(0);
    var bpad = (4 - bin.length % 4) % 4, blen = bin.length + bpad;
    var out = new ArrayBuffer(12 + 8 + jlen + (bin.length ? 8 + blen : 0)), o = new DataView(out), u8 = new Uint8Array(out);
    o.setUint32(0, 0x46546C67, true); o.setUint32(4, 2, true); o.setUint32(8, out.byteLength, true);
    o.setUint32(12, jlen, true); o.setUint32(16, 0x4E4F534A, true); u8.set(jb, 20); for (var i = 0; i < jpad; i++) u8[20 + jb.length + i] = 0x20;
    if (bin.length) { var b0 = 20 + jlen; o.setUint32(b0, blen, true); o.setUint32(b0 + 4, 0x004E4942, true); u8.set(bin, b0 + 8); }
    return out;
  }

  function openEdgeLines(mesh) {
    var g = mesh.geometry, pos = g.attributes.position, idx = g.index;
    if (!idx) return null;
    var key = {}, uniq = [], map = new Int32Array(pos.count), q = 1e-5;
    for (var i = 0; i < pos.count; i++) {
      var k = Math.round(pos.getX(i) / q) + ',' + Math.round(pos.getY(i) / q) + ',' + Math.round(pos.getZ(i) / q);
      if (key[k] === undefined) { key[k] = uniq.length; uniq.push(i); }
      map[i] = key[k];
    }
    var edges = new Map(), arr = idx.array;
    for (var f = 0; f < arr.length; f += 3) {
      var a = map[arr[f]], b = map[arr[f + 1]], c = map[arr[f + 2]], e = [[a, b], [b, c], [c, a]];
      for (var j = 0; j < 3; j++) { var lo = Math.min(e[j][0], e[j][1]), hi = Math.max(e[j][0], e[j][1]); if (lo === hi) continue; var ek = lo * 4294967296 + hi; edges.set(ek, (edges.get(ek) || 0) + 1); }
    }
    var verts = [];
    edges.forEach(function (n, ek) { if (n !== 1) return; var lo = Math.floor(ek / 4294967296), hi = ek - lo * 4294967296, ia = uniq[lo], ib = uniq[hi]; verts.push(pos.getX(ia), pos.getY(ia), pos.getZ(ia), pos.getX(ib), pos.getY(ib), pos.getZ(ib)); });
    var lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    var lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0xff5470, depthTest: false })); lines.renderOrder = 2; lines.userData.count = verts.length / 6;
    return lines;
  }

  function setupPane(e) {
    e.renderer = new THREE.WebGLRenderer({ canvas: e.canvas, antialias: true, alpha: true });
    e.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    e.renderer.outputEncoding = THREE.sRGBEncoding;
    e.scene = new THREE.Scene();
    e.scene.add(new THREE.HemisphereLight(0xfff4ff, 0x2a1a44, 1.1));
    var d1 = new THREE.DirectionalLight(0xffffff, 0.9); d1.position.set(2, 3, 2); e.scene.add(d1);
    var d2 = new THREE.DirectionalLight(0xa05cff, 0.35); d2.position.set(-2, 1, -2); e.scene.add(d2);
  }
  function setup() {
    camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100); camera.position.set(1.15, 0.65, 1.85);
    controls = new THREE.OrbitControls(camera, stage); controls.enableDamping = true;
    setupPane(ENG.trellis); setupPane(ENG.hy);
    resize(); window.addEventListener('resize', resize);
    (function loop() { requestAnimationFrame(loop); controls.update(); ENG.trellis.renderer.render(ENG.trellis.scene, camera); ENG.hy.renderer.render(ENG.hy.scene, camera); })();
  }
  function resize() {
    var w = ENG.trellis.canvas.parentNode.clientWidth, h = ENG.trellis.canvas.parentNode.clientHeight;
    ENG.trellis.renderer.setSize(w, h, false); ENG.hy.renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }

  function applyLook(entry) {
    entry.root.traverse(function (o) {
      if (!o.isMesh) return;
      if (!o.userData.origMat) o.userData.origMat = o.material;
      var m = o.userData.origMat;
      if (showTex) { o.material = m; } else { o.material = o.userData.flat || (o.userData.flat = new THREE.MeshStandardMaterial({ color: 0xd9cff0, roughness: 0.8, metalness: 0.0 })); }
      o.material.wireframe = showWire; o.material.side = THREE.DoubleSide;
      if (o.material.isMeshStandardMaterial) { o.material.metalness = 0; o.material.roughness = 0.9; o.material.metalnessMap = null; o.material.roughnessMap = null; o.material.envMap = null; o.material.flatShading = false; }
      o.material.needsUpdate = true;
    });
    if (entry.edges) entry.edges.visible = showEdges;
  }

  function load(tag, eng) {
    var id = tag + '_' + eng, E = ENG[eng === 'hy2' ? 'hy' : eng];
    if (loaded[id]) { loaded[id].group.visible = true; applyLook(loaded[id]); setStatus(id); return; }
    E.status.textContent = 'parsing…';
    var loader = new THREE.GLTFLoader(), buf = b64ToBuf(M[id].b64), triedStripped = false;
    var onError = function (err) {
      var msg = (err && (err.message || (err.target && err.target.src) || String(err))) || 'unknown error';
      if (!triedStripped) { triedStripped = true; E.status.textContent = 'textures blocked — geometry only…'; try { loader.parse(stripTextures(buf), '', onLoad, onError); return; } catch (e2) { msg = String(e2); } }
      E.status.textContent = 'could not load: ' + String(msg).slice(0, 120); console.error(err);
    };
    var onLoad = function (gltf) {
      var root = gltf.scene;
      root.traverse(function (o) { if (o.isMesh) dequantize(o.geometry); });
      root.updateMatrixWorld(true);
      var box = new THREE.Box3().setFromObject(root);
      var size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
      var scale = 1.2 / Math.max(size.x, size.y, size.z);
      var group = new THREE.Group(); root.position.sub(center); group.scale.setScalar(scale); group.add(root);
      var edgesGroup = new THREE.Group(), openCount = 0;
      root.traverse(function (o) { if (o.isMesh) { var l = openEdgeLines(o); if (l) { openCount += l.userData.count; l.applyMatrix4(o.matrixWorld); edgesGroup.add(l); } } });
      edgesGroup.position.sub(center);
      var eg = new THREE.Group(); eg.scale.setScalar(scale); eg.add(edgesGroup);
      E.scene.add(group); E.scene.add(eg);
      var entry = { group: group, root: root, edges: eg, openCount: openCount, untextured: triedStripped };
      loaded[id] = entry;
      if (current !== tag || (eng !== 'trellis' && eng !== hyVariant)) { group.visible = false; eg.visible = false; return; }
      applyLook(entry); setStatus(id);
    };
    loader.parse(buf, '', onLoad, onError);
  }
  function setStatus(id) {
    var entry = loaded[id], E = ENG[id.split('_')[1] === 'trellis' ? 'trellis' : 'hy'], tris = 0;
    entry.root.traverse(function (o) { if (o.isMesh && o.geometry.index) tris += o.geometry.index.count / 3; });
    E.status.textContent = Math.round(tris).toLocaleString() + ' tris · ' + entry.openCount.toLocaleString() + ' open edges' + (entry.untextured ? ' · untextured' : '');
  }
  function show(tag) {
    current = tag;
    var hasVariant = !!M[tag + '_hy2'];
    variantBox.hidden = !hasVariant;
    if (!hasVariant) hyVariant = 'hy';
    var rightId = tag + '_' + hyVariant;
    Object.keys(loaded).forEach(function (id) { if (id !== tag + '_trellis' && id !== rightId) { loaded[id].group.visible = false; loaded[id].edges.visible = false; } });
    document.querySelectorAll('.tab').forEach(function (t) { t.setAttribute('aria-pressed', String(t.dataset.tag === tag)); });
    variantBox.querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.v === hyVariant)); });
    load(tag, 'trellis'); load(tag, hyVariant);
  }
  variantBox.querySelectorAll('button').forEach(function (b) { b.addEventListener('click', function () { hyVariant = b.dataset.v; show(current); }); });

  document.querySelectorAll('.tab').forEach(function (b) { b.addEventListener('click', function () { show(b.dataset.tag); }); });
  function relook() { if (!current) return; ['trellis', hyVariant].forEach(function (e) { var en = loaded[current + '_' + e]; if (en) applyLook(en); }); }
  document.getElementById('btnEdges').addEventListener('click', function () { showEdges = !showEdges; this.setAttribute('aria-pressed', String(showEdges)); relook(); });
  document.getElementById('btnWire').addEventListener('click', function () { showWire = !showWire; this.setAttribute('aria-pressed', String(showWire)); relook(); });
  document.getElementById('btnTex').addEventListener('click', function () { showTex = !showTex; this.setAttribute('aria-pressed', String(showTex)); relook(); });
  document.getElementById('btnReset').addEventListener('click', function () { camera.position.set(1.15, 0.65, 1.85); controls.target.set(0, 0, 0); controls.update(); });

  try { setup(); show('mustache'); } catch (e) { ENG.trellis.status.textContent = ENG.hy.status.textContent = 'WebGL unavailable'; console.error(e); }
})();
</script>
'''
html = html.replace("__TABS__", tabs_html).replace("__ROWS__", rows_html).replace("__RAW__", raw_html)
html = html.replace("__MODELS__", json.dumps({k: {"b64": v} for k, v in models.items()}, separators=(",", ":")))
open(OUT, "w", encoding="utf-8", newline="\n").write(html)
print(OUT, f"{os.path.getsize(OUT)/1e6:.2f} MB")
