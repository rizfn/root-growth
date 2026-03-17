const TAU = 2 * Math.PI;
const GOLDEN = (Math.sqrt(5) - 1) / 2;
const GOLDEN_OMEGA_CRITICAL = 0.606661063469;

const state = {
  K: 1,
  omega: 0.60666,
  theta0: 0.1234,
  iterations: 3000,
  transient: 700,
  scanSpan: 0.24,
  scanPoints: 250,
  paramGrid: 54
};

const controls = {
  K: d3.select("#kRange"),
  omega: d3.select("#omegaRange"),
  theta0: d3.select("#theta0Range"),
  iterations: d3.select("#iterRange"),
  transient: d3.select("#transientRange"),
  scanSpan: d3.select("#scanSpanRange"),
  scanPoints: d3.select("#scanPointsRange"),
  paramGrid: d3.select("#paramGridRange")
};

const valueEls = {
  K: d3.select("#kValue"),
  omega: d3.select("#omegaValue"),
  theta0: d3.select("#theta0Value"),
  iterations: d3.select("#iterValue"),
  transient: d3.select("#transientValue"),
  scanSpan: d3.select("#scanSpanValue"),
  scanPoints: d3.select("#scanPointsValue"),
  paramGrid: d3.select("#paramGridValue")
};

const statsBox = d3.select("#statsBox");
const MAX_K_PARAM_MAP = 1.4;

const parameterMaps = {
  gridSize: null,
  data: null
};

function frac(x) {
  return ((x % 1) + 1) % 1;
}

function stepWrapped(theta, omega, K) {
  return frac(theta + omega - (K / TAU) * Math.sin(TAU * theta));
}

function simulate(theta0, omega, K, n) {
  const wrapped = [frac(theta0)];
  const unwrapped = [theta0];

  for (let i = 0; i < n; i += 1) {
    const t = wrapped[wrapped.length - 1];
    const nextUnwrapped = unwrapped[unwrapped.length - 1] + omega - (K / TAU) * Math.sin(TAU * t);
    const nextWrapped = frac(nextUnwrapped);
    unwrapped.push(nextUnwrapped);
    wrapped.push(nextWrapped);
  }

  return { wrapped, unwrapped };
}

function rotationNumber(unwrapped, transient) {
  const startIdx = Math.max(0, Math.min(unwrapped.length - 2, transient));
  const endIdx = unwrapped.length - 1;
  const steps = Math.max(1, endIdx - startIdx);
  return (unwrapped[endIdx] - unwrapped[startIdx]) / steps;
}

function metricsForParameters(theta0, omega, K, n, transient) {
  const wrapped = [frac(theta0)];
  const unwrapped = [theta0];
  let lyapunovSum = 0;
  let lyapunovCount = 0;

  for (let i = 0; i < n; i += 1) {
    const t = wrapped[wrapped.length - 1];
    const nextUnwrapped = unwrapped[unwrapped.length - 1] + omega - (K / TAU) * Math.sin(TAU * t);
    const nextWrapped = frac(nextUnwrapped);
    unwrapped.push(nextUnwrapped);
    wrapped.push(nextWrapped);

    if (i >= transient) {
      const derivative = Math.abs(1 - K * Math.cos(TAU * t));
      lyapunovSum += Math.log(Math.max(derivative, 1e-12));
      lyapunovCount += 1;
    }
  }

  const rho = rotationNumber(unwrapped, transient);
  const lambda = lyapunovSum / Math.max(1, lyapunovCount);
  return { rho, lambda };
}

function nearestRational(x, maxDen = 34) {
  let best = { p: 0, q: 1, err: Math.abs(x) };
  for (let q = 1; q <= maxDen; q += 1) {
    const p = Math.round(x * q);
    const val = p / q;
    const err = Math.abs(val - x);
    if (err < best.err) {
      best = { p, q, err };
    }
  }
  return best;
}

function setupSvg(selector) {
  const svg = d3.select(selector);
  const margin = { top: 18, right: 14, bottom: 36, left: 42 };

  function dimensions() {
    const bbox = svg.node().getBoundingClientRect();
    const width = Math.max(260, bbox.width);
    const height = Math.max(230, bbox.height);
    return {
      width,
      height,
      innerW: width - margin.left - margin.right,
      innerH: height - margin.top - margin.bottom,
      margin
    };
  }

  return { svg, dimensions, margin };
}

const mapPlot = setupSvg("#plotMap");
const timePlot = setupSvg("#plotTime");
const returnPlot = setupSvg("#plotReturn");
const scanPlot = setupSvg("#plotScan");
const tonguesPlot = setupSvg("#plotTongues");
const chaosPlot = setupSvg("#plotChaos");

function drawAxes(g, x, y, dims, xLabel, yLabel) {
  g.selectAll("*").remove();
  const gx = g.append("g").attr("transform", `translate(0,${dims.innerH})`).attr("class", "axis");
  const gy = g.append("g").attr("class", "axis");
  gx.call(d3.axisBottom(x).ticks(6));
  gy.call(d3.axisLeft(y).ticks(6));

  g.append("g")
    .attr("class", "grid")
    .attr("transform", `translate(0,${dims.innerH})`)
    .call(d3.axisBottom(x).ticks(6).tickSize(-dims.innerH).tickFormat(""));

  g.append("g")
    .attr("class", "grid")
    .call(d3.axisLeft(y).ticks(6).tickSize(-dims.innerW).tickFormat(""));

  g.append("text")
    .attr("x", dims.innerW / 2)
    .attr("y", dims.innerH + 30)
    .attr("text-anchor", "middle")
    .attr("fill", "#344054")
    .style("font-size", "11px")
    .text(xLabel);

  g.append("text")
    .attr("transform", "rotate(-90)")
    .attr("x", -dims.innerH / 2)
    .attr("y", -31)
    .attr("text-anchor", "middle")
    .attr("fill", "#344054")
    .style("font-size", "11px")
    .text(yLabel);
}

function drawMapAndCobweb(sim) {
  const { svg, dimensions } = mapPlot;
  const dims = dimensions();
  svg.attr("viewBox", `0 0 ${dims.width} ${dims.height}`);
  svg.selectAll("*").remove();

  const g = svg.append("g").attr("transform", `translate(${dims.margin.left},${dims.margin.top})`);
  const x = d3.scaleLinear().domain([0, 1]).range([0, dims.innerW]);
  const y = d3.scaleLinear().domain([0, 1]).range([dims.innerH, 0]);
  drawAxes(g, x, y, dims, "theta_n", "theta_{n+1}");

  const mapCurve = d3.range(0, 1.0001, 1 / 500).map((t) => ({
    x: t,
    y: stepWrapped(t, state.omega, state.K)
  }));

  const line = d3.line().x((d) => x(d.x)).y((d) => y(d.y));
  g.append("path")
    .datum(mapCurve)
    .attr("fill", "none")
    .attr("stroke", "#355070")
    .attr("stroke-width", 2)
    .attr("d", line);

  g.append("line")
    .attr("x1", x(0))
    .attr("x2", x(1))
    .attr("y1", y(0))
    .attr("y2", y(1))
    .attr("stroke", "#8b8f99")
    .attr("stroke-width", 1.4)
    .attr("stroke-dasharray", "5,5");

  const cobN = 60;
  const cobweb = [];
  let t = frac(state.theta0);
  cobweb.push([t, 0]);
  for (let i = 0; i < cobN; i += 1) {
    const next = stepWrapped(t, state.omega, state.K);
    cobweb.push([t, next]);
    cobweb.push([next, next]);
    t = next;
  }

  g.append("path")
    .datum(cobweb)
    .attr("fill", "none")
    .attr("stroke", "#b91c1c")
    .attr("stroke-width", 1)
    .attr("opacity", 0.9)
    .attr("d", d3.line().x((d) => x(d[0])).y((d) => y(d[1])));
}

function drawTimeSeries(sim) {
  const { svg, dimensions } = timePlot;
  const dims = dimensions();
  svg.attr("viewBox", `0 0 ${dims.width} ${dims.height}`);
  svg.selectAll("*").remove();

  const g = svg.append("g").attr("transform", `translate(${dims.margin.left},${dims.margin.top})`);
  const start = Math.min(state.transient, sim.wrapped.length - 2);
  const series = sim.wrapped.slice(start);
  const n = series.length;

  const x = d3.scaleLinear().domain([0, Math.max(1, n - 1)]).range([0, dims.innerW]);
  const y = d3.scaleLinear().domain([0, 1]).range([dims.innerH, 0]);
  drawAxes(g, x, y, dims, "n (after transient)", "theta_n");

  g.append("path")
    .datum(series)
    .attr("fill", "none")
    .attr("stroke", "#0f766e")
    .attr("stroke-width", 1.5)
    .attr("d", d3.line().x((d, i) => x(i)).y((d) => y(d)));
}

function drawReturnMap(sim) {
  const { svg, dimensions } = returnPlot;
  const dims = dimensions();
  svg.attr("viewBox", `0 0 ${dims.width} ${dims.height}`);
  svg.selectAll("*").remove();

  const g = svg.append("g").attr("transform", `translate(${dims.margin.left},${dims.margin.top})`);
  const x = d3.scaleLinear().domain([0, 1]).range([0, dims.innerW]);
  const y = d3.scaleLinear().domain([0, 1]).range([dims.innerH, 0]);
  drawAxes(g, x, y, dims, "theta_n", "theta_{n+1}");

  const start = Math.min(state.transient, sim.wrapped.length - 2);
  const points = [];
  for (let i = start; i < sim.wrapped.length - 1; i += 1) {
    points.push({ x: sim.wrapped[i], y: sim.wrapped[i + 1] });
  }

  g.append("g")
    .selectAll("circle")
    .data(points)
    .join("circle")
    .attr("cx", (d) => x(d.x))
    .attr("cy", (d) => y(d.y))
    .attr("r", 1.6)
    .attr("fill", "#b45309")
    .attr("opacity", 0.5);
}

function computeScan() {
  const span = state.scanSpan;
  const points = Math.max(20, state.scanPoints);
  const left = Math.max(0, state.omega - span / 2);
  const right = Math.min(1, state.omega + span / 2);
  const step = (right - left) / (points - 1);
  const nIter = Math.max(1200, Math.floor(state.iterations * 0.6));
  const transient = Math.min(Math.floor(nIter * 0.45), state.transient);

  const result = [];
  for (let i = 0; i < points; i += 1) {
    const om = left + i * step;
    const sim = simulate(state.theta0, om, state.K, nIter);
    result.push({ omega: om, rho: rotationNumber(sim.unwrapped, transient) });
  }
  return { result, left, right };
}

function drawScan(scan) {
  const { svg, dimensions } = scanPlot;
  const dims = dimensions();
  svg.attr("viewBox", `0 0 ${dims.width} ${dims.height}`);
  svg.selectAll("*").remove();

  const g = svg.append("g").attr("transform", `translate(${dims.margin.left},${dims.margin.top})`);
  const x = d3.scaleLinear().domain([scan.left, scan.right]).range([0, dims.innerW]);
  const y = d3.scaleLinear().domain([0, 1]).range([dims.innerH, 0]);
  drawAxes(g, x, y, dims, "Omega", "rho");

  g.append("path")
    .datum(scan.result)
    .attr("fill", "none")
    .attr("stroke", "#1f2937")
    .attr("stroke-width", 1.4)
    .attr("d", d3.line().x((d) => x(d.omega)).y((d) => y(d.rho)));

  g.append("line")
    .attr("x1", 0)
    .attr("x2", dims.innerW)
    .attr("y1", y(GOLDEN))
    .attr("y2", y(GOLDEN))
    .attr("stroke", "#0f766e")
    .attr("stroke-dasharray", "5,4")
    .attr("stroke-width", 1);

  g.append("line")
    .attr("x1", x(state.omega))
    .attr("x2", x(state.omega))
    .attr("y1", 0)
    .attr("y2", dims.innerH)
    .attr("stroke", "#b91c1c")
    .attr("stroke-width", 1)
    .attr("opacity", 0.8);
}

function classifyTongue(rho, K) {
  const best = nearestRational(rho, 12);
  const tolerance = 0.0022 + 0.0015 * (1 / (1 + 3 * K));
  if (best.err < tolerance) {
    return {
      id: `${best.p}/${best.q}`,
      p: best.p,
      q: best.q,
      err: best.err
    };
  }
  return { id: null, p: null, q: null, err: best.err };
}

function computeParameterMaps() {
  const nK = Math.max(20, state.paramGrid);
  const nOmega = Math.max(20, Math.floor(state.paramGrid * 1.45));
  const nIter = Math.max(900, Math.min(2600, Math.floor(state.iterations * 0.55)));
  const transient = Math.min(state.transient, Math.floor(nIter * 0.45));
  const omegaValues = d3.range(nOmega).map((i) => i / (nOmega - 1));
  const kValues = d3.range(nK).map((i) => (i / (nK - 1)) * MAX_K_PARAM_MAP);
  const cells = [];

  for (let ky = 0; ky < nK; ky += 1) {
    const Kval = kValues[ky];
    for (let ox = 0; ox < nOmega; ox += 1) {
      const omegaVal = omegaValues[ox];
      const m = metricsForParameters(state.theta0, omegaVal, Kval, nIter, transient);
      const tongue = classifyTongue(m.rho, Kval);
      cells.push({
        ix: ox,
        iy: ky,
        omega: omegaVal,
        K: Kval,
        rho: m.rho,
        lambda: m.lambda,
        tongueId: tongue.id,
        q: tongue.q,
        p: tongue.p,
        lockErr: tongue.err
      });
    }
  }

  parameterMaps.gridSize = { nOmega, nK };
  parameterMaps.data = cells;
}

function drawTongueBoundaries(g, dims, x, y, data, nOmega, nK) {
  const idx = (ix, iy) => iy * nOmega + ix;
  const boundaries = [];
  for (let iy = 0; iy < nK; iy += 1) {
    for (let ix = 0; ix < nOmega; ix += 1) {
      const cell = data[idx(ix, iy)];
      if (ix < nOmega - 1) {
        const right = data[idx(ix + 1, iy)];
        if (cell.tongueId !== right.tongueId) {
          const xMid = (x(cell.omega) + x(right.omega)) * 0.5;
          boundaries.push({
            x1: xMid,
            y1: y(cell.K) - (dims.innerH / Math.max(1, nK - 1)) * 0.5,
            x2: xMid,
            y2: y(cell.K) + (dims.innerH / Math.max(1, nK - 1)) * 0.5
          });
        }
      }
      if (iy < nK - 1) {
        const up = data[idx(ix, iy + 1)];
        if (cell.tongueId !== up.tongueId) {
          const yMid = (y(cell.K) + y(up.K)) * 0.5;
          boundaries.push({
            x1: x(cell.omega) - (dims.innerW / Math.max(1, nOmega - 1)) * 0.5,
            y1: yMid,
            x2: x(cell.omega) + (dims.innerW / Math.max(1, nOmega - 1)) * 0.5,
            y2: yMid
          });
        }
      }
    }
  }

  g.append("g")
    .selectAll("line")
    .data(boundaries)
    .join("line")
    .attr("x1", (d) => d.x1)
    .attr("y1", (d) => d.y1)
    .attr("x2", (d) => d.x2)
    .attr("y2", (d) => d.y2)
    .attr("stroke", "#1f2937")
    .attr("stroke-width", 0.35)
    .attr("opacity", 0.6);
}

function drawTonguesMap() {
  if (!parameterMaps.data || !parameterMaps.gridSize) {
    return;
  }

  const { svg, dimensions } = tonguesPlot;
  const dims = dimensions();
  svg.attr("viewBox", `0 0 ${dims.width} ${dims.height}`);
  svg.selectAll("*").remove();

  const g = svg.append("g").attr("transform", `translate(${dims.margin.left},${dims.margin.top})`);
  const x = d3.scaleLinear().domain([0, 1]).range([0, dims.innerW]);
  const y = d3.scaleLinear().domain([0, MAX_K_PARAM_MAP]).range([dims.innerH, 0]);
  drawAxes(g, x, y, dims, "Omega", "K");

  const { nOmega, nK } = parameterMaps.gridSize;
  const dx = dims.innerW / Math.max(1, nOmega - 1);
  const dy = dims.innerH / Math.max(1, nK - 1);
  const qColor = d3.scaleSequential((q) => d3.interpolateSinebow((q - 1) / 11)).domain([1, 12]);

  g.append("g")
    .selectAll("rect")
    .data(parameterMaps.data)
    .join("rect")
    .attr("x", (d) => x(d.omega) - dx * 0.5)
    .attr("y", (d) => y(d.K) - dy * 0.5)
    .attr("width", dx + 0.2)
    .attr("height", dy + 0.2)
    .attr("fill", (d) => {
      if (!d.tongueId) {
        return "#f0ede4";
      }
      return qColor(d.q);
    })
    .attr("opacity", (d) => (d.tongueId ? 0.9 : 0.68));

  drawTongueBoundaries(g, dims, x, y, parameterMaps.data, nOmega, nK);

  g.append("circle")
    .attr("cx", x(state.omega))
    .attr("cy", y(state.K))
    .attr("r", 3.8)
    .attr("fill", "none")
    .attr("stroke", "#b91c1c")
    .attr("stroke-width", 1.4);

  g.append("text")
    .attr("x", dims.innerW - 4)
    .attr("y", 12)
    .attr("text-anchor", "end")
    .style("font-size", "10px")
    .attr("fill", "#475467")
    .text("colored: rational locking, dark lines: tongue boundaries");
}

function drawChaosMap() {
  if (!parameterMaps.data || !parameterMaps.gridSize) {
    return;
  }

  const { svg, dimensions } = chaosPlot;
  const dims = dimensions();
  svg.attr("viewBox", `0 0 ${dims.width} ${dims.height}`);
  svg.selectAll("*").remove();

  const g = svg.append("g").attr("transform", `translate(${dims.margin.left},${dims.margin.top})`);
  const x = d3.scaleLinear().domain([0, 1]).range([0, dims.innerW]);
  const y = d3.scaleLinear().domain([0, MAX_K_PARAM_MAP]).range([dims.innerH, 0]);
  drawAxes(g, x, y, dims, "Omega", "K");

  const { nOmega, nK } = parameterMaps.gridSize;
  const dx = dims.innerW / Math.max(1, nOmega - 1);
  const dy = dims.innerH / Math.max(1, nK - 1);
  const lambdaColor = d3.scaleDiverging(
    [-1.2, 0, 0.25],
    (t) => d3.interpolateRgbBasis(["#1e3a8a", "#f8fafc", "#dc2626"])(t)
  );

  g.append("g")
    .selectAll("rect")
    .data(parameterMaps.data)
    .join("rect")
    .attr("x", (d) => x(d.omega) - dx * 0.5)
    .attr("y", (d) => y(d.K) - dy * 0.5)
    .attr("width", dx + 0.2)
    .attr("height", dy + 0.2)
    .attr("fill", (d) => lambdaColor(d.lambda))
    .attr("opacity", 0.95);

  g.append("path")
    .datum(parameterMaps.data)
    .attr("fill", "none")
    .attr("stroke", "#111827")
    .attr("stroke-width", 1)
    .attr("opacity", 0.7)
    .attr("d", d3.line()
      .defined((d) => Math.abs(d.lambda) < 0.03)
      .x((d) => x(d.omega))
      .y((d) => y(d.K))
    );

  g.append("circle")
    .attr("cx", x(state.omega))
    .attr("cy", y(state.K))
    .attr("r", 3.8)
    .attr("fill", "none")
    .attr("stroke", "#111827")
    .attr("stroke-width", 1.2);

  g.append("text")
    .attr("x", dims.innerW - 4)
    .attr("y", 12)
    .attr("text-anchor", "end")
    .style("font-size", "10px")
    .attr("fill", "#475467")
    .text("blue: regular (lambda < 0), red: chaotic tendency (lambda > 0)");
}

function updateControlReadout() {
  valueEls.K.text(state.K.toFixed(3));
  valueEls.omega.text(state.omega.toFixed(6));
  valueEls.theta0.text(state.theta0.toFixed(4));
  valueEls.iterations.text(state.iterations.toString());
  valueEls.transient.text(state.transient.toString());
  valueEls.scanSpan.text(state.scanSpan.toFixed(2));
  valueEls.scanPoints.text(state.scanPoints.toString());
  valueEls.paramGrid.text(state.paramGrid.toString());
}

function updateStats(rho) {
  const delta = rho - GOLDEN;
  const near = nearestRational(rho, 34);
  const sign = delta >= 0 ? "+" : "-";
  statsBox.html(
    `<strong>Estimated rotation number rho</strong>: ${rho.toFixed(9)}<br>` +
    `<strong>rho - (phi - 1)</strong>: ${sign}${Math.abs(delta).toExponential(3)}<br>` +
    `<strong>Nearest low-order rational</strong>: ${near.p}/${near.q} (error ${near.err.toExponential(2)})<br>` +
    `<strong>Critical golden Omega (K=1)</strong>: ${GOLDEN_OMEGA_CRITICAL.toFixed(12)}`
  );
}

let redrawToken = null;
function requestRedraw() {
  if (redrawToken !== null) {
    cancelAnimationFrame(redrawToken);
  }
  redrawToken = requestAnimationFrame(() => {
    redrawToken = null;
    const sim = simulate(state.theta0, state.omega, state.K, state.iterations);
    const rho = rotationNumber(sim.unwrapped, Math.min(state.transient, state.iterations - 10));
    drawMapAndCobweb(sim);
    drawTimeSeries(sim);
    drawReturnMap(sim);
    const scan = computeScan();
    drawScan(scan);
    drawTonguesMap();
    drawChaosMap();
    updateStats(rho);
  });
}

function requestMapRecomputeAndRedraw() {
  d3.select("#recomputeMaps").text("Computing maps...");
  setTimeout(() => {
    computeParameterMaps();
    d3.select("#recomputeMaps").text("Recompute parameter maps");
    requestRedraw();
  }, 20);
}

function bindControl(control, key, parser) {
  control.on("input", function onInput() {
    state[key] = parser(this.value);
    if (key === "iterations" && state.transient >= state.iterations - 5) {
      state.transient = Math.max(0, state.iterations - 50);
      controls.transient.property("value", state.transient);
    }
    if (key === "transient" && state.transient >= state.iterations - 2) {
      state.transient = Math.max(0, state.iterations - 10);
      controls.transient.property("value", state.transient);
    }
    updateControlReadout();
    requestRedraw();
  });
}

bindControl(controls.K, "K", parseFloat);
bindControl(controls.omega, "omega", parseFloat);
bindControl(controls.theta0, "theta0", parseFloat);
bindControl(controls.iterations, "iterations", (v) => parseInt(v, 10));
bindControl(controls.transient, "transient", (v) => parseInt(v, 10));
bindControl(controls.scanSpan, "scanSpan", parseFloat);
bindControl(controls.scanPoints, "scanPoints", (v) => parseInt(v, 10));
bindControl(controls.paramGrid, "paramGrid", (v) => parseInt(v, 10));

controls.paramGrid.on("change", function onGridChange() {
  state.paramGrid = parseInt(this.value, 10);
  updateControlReadout();
  requestMapRecomputeAndRedraw();
});

d3.select("#presetGolden").on("click", () => {
  state.K = 1;
  state.omega = GOLDEN_OMEGA_CRITICAL;
  controls.K.property("value", state.K);
  controls.omega.property("value", state.omega);
  updateControlReadout();
  requestRedraw();
});

d3.select("#randomIC").on("click", () => {
  state.theta0 = Math.random();
  controls.theta0.property("value", state.theta0);
  updateControlReadout();
  requestRedraw();
});

d3.select("#recomputeMaps").on("click", requestMapRecomputeAndRedraw);

window.addEventListener("resize", requestRedraw);

updateControlReadout();
computeParameterMaps();
requestRedraw();