# IMPLEMENTATION-Frontend.md

**HORIZON — Marketing Site & Console**
Problem Statement 26153 · NTRO · SIH

Two parts. **Part A** is a design brief to paste into Claude Design to explore how the product looks before writing any code. **Part B** is the Next.js implementation spec for both the public site and the authenticated console.

Do Part A first. Do not start Part B until you have a visual direction you actually like — retrofitting a design onto a built frontend costs more than exploring first.

---

# PART A — Design brief for Claude Design

Paste everything between the rules below into Claude Design. It is written as a brief, not as instructions to a developer.

---

## Brief: HORIZON

### What this is

HORIZON is a predictive network defence platform. Conventional intrusion detection tells you an attack happened. HORIZON tells you what is *about to* happen: it learns how a network's behaviour evolves over time, simulates the next few minutes forward, and reports how likely the current trajectory is to end in compromise — with a horizon, a confidence band, and a lead time in seconds.

The product is sold to security operations teams at enterprises and operators of critical national infrastructure. The buyer is a SOC lead. The daily user is a security analyst who already has four dashboards open and is deeply sceptical of anything that promises to predict the future.

I need two things designed: a **public marketing site** and the **console** the analyst actually works in.

### The one idea the design should be built around

Everything in this product is organised by a single line: **now**.

To the left of it is observed reality — data that happened, drawn with certainty. To the right is projection — states the model invented, drawn with doubt that widens the further out you go. That boundary is the entire product thesis, and I want it to be the organising device of the visual identity, not just a feature of one chart.

Concretely, this should show up as: a vertical rule that appears in the hero, in every chart, and as a layout axis on at least one marketing section; a hard visual distinction between "observed" and "projected" treatments that stays consistent everywhere (weight, saturation, fill, edge quality — pick a system and hold it); and a widening-uncertainty motif that is real data, not decoration.

The closest existing visual language is a **hurricane forecast cone** — the National Hurricane Center's track maps, where a solid past track meets a widening probabilistic cone. That is the reference to have in mind, not a security dashboard. It is the right metaphor because it is honest: it shows you the projection *and* how much to trust it, which is exactly what we do.

### Hero

The hero should be the forecast itself, live and moving — not a screenshot, not a headline over a gradient.

A single host's risk curve: solid line rising through observed windows, hitting the "now" rule, then continuing as a projected band that fans open across six future windows. Somewhere out at t+3 the band crosses a threshold and the moment is marked. A small readout says how far ahead that is in seconds.

If it animates, it should animate **once** on load, as one orchestrated sequence — the past track drawing in, then the cone opening. Not a loop, not scattered fade-ups on every section.

I would rather the hero be one honest instrument than a headline with three feature cards under it.

### Reference points

Take these as calibration, not as things to copy:

- **National Hurricane Center track maps** — the forecast cone. The core metaphor.
- **Linear** (linear.app) — restraint, type discipline, the confidence to leave space empty. The bar for craft.
- **Cloudflare Radar** (radar.cloudflare.com) — data as the primary content of a public page. Charts that are the argument, not illustrations of it.
- **Observable** / **Datadog** — dense information without feeling cramped; how to make a lot of numbers legible.
- **Stripe docs** — how technical credibility reads visually.

Please **avoid** looking like: a generic dark-mode security SaaS with a neon-cyan glow and a hexagon-grid background; anything with a shield icon; a landing page that is a stack of identical rounded cards with the same soft grey shadow; a warm-cream background with a serif display and a terracotta accent.

### Audience and tone

Write for someone technical who has been pitched a lot of AI security products and believes approximately none of them. The copy should be specific and slightly understated. Numbers beat adjectives. "90 seconds of warning before the scan became an intrusion" is better than "AI-powered predictive threat intelligence."

The product's own honesty is a differentiator worth designing for: we say what we cannot do. There should be a place on the site where the limitations are stated plainly and it reads as confidence rather than hedging.

### Public site sections

1. **Hero** — the live forecast, as above. One line of what it is. One action.
2. **The distinction** — detection versus forecasting. This is the argument the whole product rests on. A conventional IDS fires when evidence exists; we fire before it does. I think this section wants a two-track time comparison rather than a feature table, but decide for yourself.
3. **How it works** — traffic becomes windowed host states, states become learned dynamics, dynamics simulate forward, simulated futures get scored. Four stages. This one genuinely is a sequence, so sequential structure is earned here if you want it.
4. **What you see** — the console. Console frames shown in context, with the explanation panel legible: an analyst gets a risk curve, a predicted attack stage, five named signals with direction, and a lead time.
5. **Evidence** — benchmark numbers against a logistic-regression baseline, and the ablation result showing performance collapses when the dynamics model is removed. This section should feel like a results table in a paper, not marketing.
6. **Honest limits** — what the model does not do. It learns temporal dynamics, not causal structure. Forecasts are reliable to about three minutes and degrade after. Counterfactuals are what the *model* would predict, not what the network would do.
7. **Deployment** — passive tap, fully air-gapped, no external calls. Relevant to critical-infrastructure buyers who cannot send telemetry anywhere.
8. **Footer.**

### Console screens

- **Fleet** — every monitored host, sorted by projected risk rather than current risk. That sort order is the product in one interaction: the top of this list is hosts that are *becoming* a problem.
- **Host detail** — the primary workspace. Forecast chart with the now-line and cone; predicted attack stage per horizon; the five driving signals with direction and magnitude; flagged flows; and a counterfactual control that lets the analyst clamp one behaviour and re-run the simulation.
- **Episode timeline** — a completed incident replayed, with the moment we warned and the moment it actually happened, and the gap between them measured.
- **Benchmarks** — model evaluation as a product page: baselines, ablations, calibration, forecast error against horizon.

### Constraints

Dark and light both need to work — analysts run dark, executives screenshot light. Charts are the hard part; design the data colours to survive both.

Accessible: risk level must never be carried by colour alone. Keyboard focus visible. Reduced motion respected.

### Deliverable

Give me a design plan first — palette as named hex values, typefaces and their roles, a layout concept with wireframes, and the principles that make this specific rather than generic. Tell me which parts you deliberately steered away from and why. Then build the hero and one console screen at full fidelity.

---

*(end of Claude Design brief)*

---

## A.1 Evaluating what comes back

Before building anything, check the returned design against these. Each maps to a way this project can look generic or dishonest.

- Is the **now-line** a real organising device, or one chart detail? If it only appears once, the identity has not landed.
- Do observed and projected have a **consistent, learnable visual distinction**? An analyst should be able to tell at a glance without a legend.
- Does the uncertainty band **widen**? If it is constant width it is decoration and it misrepresents the model.
- Is the hero the **instrument**, or a headline with a chart underneath?
- Is there **one bold element** with everything else quiet, or is boldness spread thin across five sections?
- Can you read a risk level with colour removed?
- Does the copy contain a **number** in the first screen?

If the direction is close but not there, iterate on the design brief rather than fixing it in code.

---

# PART B — Next.js implementation

## 1. Stack

| Concern | Choice | Note |
|---|---|---|
| Framework | Next.js 15, App Router | |
| Language | TypeScript, strict | |
| Styling | Tailwind v4 + CSS variables | Tokens as CSS vars so dark/light is one class toggle |
| Charts | **Visx** (or D3 primitives) | Recharts cannot draw the cone properly — see §4 |
| State | TanStack Query + Zustand | Query for REST, Zustand for the live socket buffer |
| WebSocket | Native, custom hook | Socket.io adds a protocol you do not need |
| Types | `openapi-typescript` from FastAPI's schema | Generated, never hand-written |
| Animation | Motion (framer-motion) | One orchestrated hero sequence only |
| Deploy | `output: 'standalone'`, node in Compose | Must run offline — no Vercel dependency |

```
web/
  app/
    (marketing)/page.tsx            layout.tsx
    (app)/
      fleet/page.tsx
      hosts/[hostId]/page.tsx
      episodes/[id]/page.tsx
      benchmarks/page.tsx
      layout.tsx
    login/page.tsx
    api/auth/[...]/route.ts         httpOnly cookie handling
  components/
    forecast/
      ForecastChart.tsx             the centrepiece
      NowLine.tsx
      UncertaintyCone.tsx
      StageStrip.tsx
      LeadTimeBadge.tsx
    explain/
      SignalList.tsx
      CounterfactualPanel.tsx
      FlaggedFlows.tsx
      SaliencyStrip.tsx
    marketing/ …
    ui/ …
  lib/
    api.ts  ws.ts  types.gen.ts  tokens.ts  format.ts
```

## 2. Types — generated, not written

```bash
npx openapi-typescript http://localhost:8000/openapi.json -o lib/types.gen.ts
```

Make it a `predev` script so it regenerates automatically. Hand-written frontend types drift from the backend within a day, and the drift surfaces as a wrong number on a chart rather than a compile error.

## 3. Design tokens

Two themes, one variable set. Semantics, not hex values, in components.

```css
:root {
  --observed: <hex>;         /* solid, saturated */
  --projected: <hex>;        /* the cone fill */
  --projected-edge: <hex>;
  --now-line: <hex>;
  --threshold: <hex>;
  --risk-low / --risk-elevated / --risk-high: <hex>;
  --surface / --surface-raised / --border / --text / --text-muted: <hex>;
}
[data-theme="dark"] { /* same names, dark values */ }
```

Take the actual values from the Claude Design output. The rule that matters: **every component references `var(--observed)`, never a literal**. Otherwise the light/dark parity in §7 is unachievable.

## 4. ForecastChart — the centrepiece

Everything else is standard product UI. This component is the product.

**Use Visx, not Recharts.** Recharts cannot cleanly render a series that is solid to a point and a widening band after it, with a threshold crossing marked inside the band, without fighting it the whole way. Visx gives you scales and shape generators and stays out of the way.

### Structure

```tsx
<ForecastChart forecast={forecast} history={observedWindows} threshold={0.75} />
```

Layered bottom to top:

1. Grid and axes — time on x, probability on y, `[0, 1]` fixed. Never auto-scale y: a rescaling risk axis makes two hosts incomparable and makes a flat curve look dramatic.
2. **Observed line** — `<LinePath>` over history, solid, `var(--observed)`.
3. **Uncertainty cone** — `<AreaClosed>` from `ci_low` to `ci_high` across k=1..6, `var(--projected)`. Anchor the band's first point at the observed value at t so the cone *opens from* the now-line rather than floating detached.
4. **Projected mean** — dashed line through `p_compromise`.
5. **Now line** — vertical rule at `origin_ts`, labelled.
6. **Threshold** — horizontal rule at 0.75. Where the mean crosses it, a marker.
7. **Lead-time annotation** — from the crossing back to the now-line, with the seconds.
8. **Stage strip** — a thin band under the x-axis, one cell per horizon, coloured by top predicted stage.
9. **Reality overlay** — when actual data for `t+k` arrives, draw the observed value over the earlier prediction.

Layer 9 is the demo's proof moment and the single most persuasive thing on screen. Build it deliberately; do not treat it as a nice-to-have.

### Cone geometry

```ts
const conePoints = [
  { ts: originTs, lo: observedNow, hi: observedNow },   // anchor at now
  ...horizons.map(h => ({ ts: h.ts, lo: h.ci_low, hi: h.ci_high })),
];
```

The anchor point is what makes it read as a cone rather than a floating ribbon. It is one line and it is the difference between the chart communicating the idea and not.

### Do not

- Animate the cone on every data tick. It re-renders every 500 ms during replay and animated transitions turn it into noise.
- Auto-scale y.
- Draw the projected mean solid. The dash is carrying "this is not observed" and it has to stay legible when someone screenshots it in greyscale.

## 5. Live data

```ts
export function useForecastStream(hostIds?: string[]) {
  // ws://…/api/v1/stream/forecast?token=…
  // exponential backoff 1s → 30s, resubscribe filter on reconnect
  // ring buffer, last 200 forecasts per host, in Zustand
}
```

Buffer in a ref and flush to state on `requestAnimationFrame`. At 60× replay speed a forecast arrives every ~500 ms per host, and across 50 hosts naive `setState` per message will drop frames on a demo laptop.

Token in the query string, not a header — browsers cannot set headers on WebSocket upgrade. The backend expects this.

## 6. Console screens

**Fleet.** Table of hosts sorted by **projected** risk (`max(p_compromise)` over horizons), not current risk. Columns: host, current risk, projected peak, predicted stage, lead time, sparkline. The sort order is the product argument — put a one-line note above the table saying so.

**Host detail.** ForecastChart at the top, full width. Below in two columns: left, the driving signals with signed magnitudes and the saliency strip showing which past windows drove the forecast; right, predicted stage per horizon and flagged flows.

The counterfactual control sits under the chart. Select a feature, set a clamp value, `POST /api/v1/counterfactual`, overlay the returned curve as a second dashed line in a distinct colour. **Label it "model-internal what-if"** in the UI itself, not just in the docs. That label is doing real work: it is the difference between an honest claim and an overclaim, and a judge who notices it will trust everything else more.

**Episode.** A completed incident. Scrubber across the episode, two markers — when we warned, when it happened — and the gap measured between them. This is the lead-time metric rendered as a product feature instead of a number in a notebook.

**Benchmarks.** Read `/api/v1/benchmarks`. Baseline table with the logistic-regression row marked as the mandated comparison; the ablation table showing collapse under persistence and time-shuffle; AUC and nRMSE against horizon; a reliability diagram. Render this as a results page, plainly. Its credibility comes from being unadorned.

## 7. Quality floor

Build these in from the start; retrofitting accessibility is miserable.

- **Colour is never the only signal.** Risk level carries a label or shape as well. Test with a greyscale filter — if the fleet table is unreadable, fix it before moving on.
- **Both themes verified on charts.** The cone fill is the fragile element; a fill that reads well on white often disappears on dark.
- **Keyboard:** the chart is focusable, arrow keys step horizons, the focused horizon is announced. The counterfactual control is fully operable without a mouse.
- **Reduced motion:** `prefers-reduced-motion` disables the hero sequence and all chart transitions. The static state must be complete on its own.
- **Loading:** skeletons matching final layout. No spinners over the chart area — layout shift on the centrepiece looks broken.
- **Empty state:** "No traffic ingested yet" with the upload action, not a blank panel.
- **Errors:** state what happened and what to do. Never a bare "Something went wrong."

## 8. Build order

| Day | Work |
|---|---|
| **1** | Scaffold, tokens from the design output, generated types, auth pages, mock WebSocket against `forecast.example.json` |
| **2** | ForecastChart complete against mock data — cone, now-line, threshold, lead time, stage strip |
| **3** | Fleet and host detail; live socket wired to the real backend |
| **4** | Explanation panel, counterfactual, flagged flows, episode view, benchmarks |
| **5** | Marketing site, both themes, accessibility pass, demo rehearsal |

Get `forecast.example.json` from the backend team on day 1 and build against it. The frontend should never be blocked waiting for the pipeline, and if you build against mock data you will find schema problems while they are still cheap.

## 9. Demo-critical

Six things must work on demo day. Rehearse them in order, on the machine you will present from.

1. The hero animates once, cleanly, on a cold load.
2. Replay at 60× runs without dropped frames.
3. The cone opens visibly as risk climbs.
4. The threshold crossing and lead-time badge appear at the right moment.
5. The counterfactual visibly collapses the curve.
6. **The reality overlay lands on the prediction.**

Number six is the proof. Everything before it is a claim. Protect it — pin the demo to a prepared capture slice where you already know the model performs, and have a recorded fallback.

## 10. Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Cone detached from the observed line | Missing anchor point at `origin_ts` | Prepend the anchor (§4) |
| Chart stutters during replay | `setState` per socket message | Ring buffer + rAF flush |
| Risk curve looks dramatic when flat | Auto-scaled y axis | Fix y to `[0, 1]` |
| Cone invisible in dark mode | Fill opacity tuned on light only | Separate `--projected` per theme |
| WebSocket dies at ~60 s | Idle proxy timeout | Client heartbeat matching the backend's 30 s |
| Types disagree with API | Hand-written types | `openapi-typescript` in `predev` |
| Hero janky on the demo laptop | Animating SVG paths and filters together | Pre-compute the path; animate `stroke-dashoffset` only |
