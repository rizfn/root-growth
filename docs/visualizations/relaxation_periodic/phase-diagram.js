// Clear body to avoid duplicate containers if reloaded
d3.select("body").html("");

// --- 1. Layout setup with strict 50/50 flexbox ---
const container = d3.select("body")
    .append("div")
    .style("display", "flex")
    .style("width", "100vw")
    .style("height", "100vh")
    .style("overflow", "hidden");

const heatmapDiv = container.append("div")
    .attr("id", "heatmapDiv")
    .style("flex", "0 0 50%")
    .style("height", "100%")
    .style("position", "relative")
    .style("overflow", "hidden")
    .style("display", "flex")
    .style("flex-direction", "column"); // Make heatmapDiv a column flexbox

const vizDiv = container.append("div")
    .attr("id", "vizDiv")
    .style("flex", "0 0 50%")
    .style("height", "100%")
    .style("position", "relative")
    .style("overflow", "hidden");

// --- Sliders UI above heatmap ---
const sliderContainer = heatmapDiv
    .append("div")
    .attr("id", "sliderContainer")
    .style("width", "100%")
    .style("flex", "0 0 10%") // Take up 10% of vertical space
    .style("display", "flex")
    .style("align-items", "center")
    .style("justify-content", "center");

sliderContainer.html(`
    <label style="margin-right:32px; font-size:2em;">
        <span>Spacing:</span>
        <input type="range" id="spacingSlider" min="30" max="100" value="50" step="1" style="vertical-align:middle; margin-left:8px;">
        <span id="spacingValue">50</span>
    </label>
    <label style="font-size:2em;">
        <span>Radius:</span>
        <input type="range" id="radiusSlider" min="5" max="40" value="15" step="1" style="vertical-align:middle; margin-left:8px;">
        <span id="radiusValue">15</span>
    </label>
`);


// --- 2. Lattice parameters (fixed size for simulation, but visualization is scaled) ---
const latticeWidth = 500;
const latticeHeight = 500;
let spacing = 50; // horizontal distance between circle centers
let radius = 15;

// --- 3. Parameter grid for simulations ---
const elongationZoneLengths = d3.range(2, 16, 1);
const downwardReturns = d3.range(0.1, 0.51, 0.05);

// --- 4. Precompute lattice circles ---
function computeLattice() {
    const rowHeight = Math.sqrt(3) * spacing / 2;
    let circles = [];
    let row = 0;
    for (let y = radius; y < latticeHeight + rowHeight; y += rowHeight, row++) {
        const offsetX = row % 2 === 0 ? 0 : spacing / 2;
        for (let x = 0; x < latticeWidth + spacing; x += spacing) {
            circles.push({ cx: x + offsetX, cy: y, r: radius });
        }
    }
    return circles;
}
let latticeCircles = computeLattice();

// --- 5. Simulate all parameter combinations ---
const rootSpeed = 2;
let allResults = []; // {elongationZoneLength, downwardReturn, rootLine, slope}

function simulateRoot(elongationZoneLength, downwardReturn, spacing, radius, rootSpeed) {
    const rowHeight = Math.sqrt(3) * spacing / 2;
    let circles = [];
    let row = 0;
    for (let y = radius; y < latticeHeight + rowHeight; y += rowHeight, row++) {
        const offsetX = row % 2 === 0 ? 0 : spacing / 2;
        for (let x = 0; x < latticeWidth + spacing; x += spacing) {
            circles.push({ cx: x + offsetX, cy: y, r: radius });
        }
    }

    let rootLine = [{ x: latticeWidth / 2, y: -10 }];
    let steps = 0;

    function getDirection() {
        if (rootLine.length < 2) return { x: 0, y: 1 };
        const idx = Math.max(0, rootLine.length - elongationZoneLength);
        const from = rootLine[idx];
        const to = rootLine[rootLine.length - 1];
        let dx = to.x - from.x;
        let dy = to.y - from.y;
        let len = Math.sqrt(dx * dx + dy * dy);
        if (len === 0) return { x: 0, y: 1 };
        return { x: dx / len, y: dy / len };
    }

    function smoothRoot() {
        if (rootLine.length < 3) return;
        for (let i = rootLine.length - elongationZoneLength; i < rootLine.length - 1; i++) {
            if (i <= 0 || i >= rootLine.length - 1) continue;
            rootLine[i].x = (rootLine[i - 1].x + rootLine[i].x + rootLine[i + 1].x) / 3;
            rootLine[i].y = (rootLine[i - 1].y + rootLine[i].y + rootLine[i + 1].y) / 3;
        }
    }

    while (true) {
        // Compute direction based on memory
        let dir = getDirection();
        // Add gravity bias
        dir.x = dir.x * (1 - downwardReturn) + 0 * downwardReturn;
        dir.y = dir.y * (1 - downwardReturn) + 1 * downwardReturn;
        // Normalize
        let len = Math.sqrt(dir.x * dir.x + dir.y * dir.y);
        dir.x /= len;
        dir.y /= len;

        let last = rootLine[rootLine.length - 1];
        let next = { x: last.x + dir.x * rootSpeed, y: last.y + dir.y * rootSpeed };

        // Check for collision
        let collided = false, collisionCircle = null;
        for (let c of circles) {
            let dx = next.x - c.cx;
            let dy = next.y - c.cy;
            let dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < c.r + 2) {
                collided = true;
                collisionCircle = c;
                break;
            }
        }

        if (collided && collisionCircle) {
            // Move tip tangentially along the circle
            let dx = last.x - collisionCircle.cx;
            let dy = last.y - collisionCircle.cy;
            let rlen = Math.sqrt(dx * dx + dy * dy);
            dx /= rlen;
            dy /= rlen;
            // Tangent direction
            let tangent = { x: -dy, y: dx };
            // Choose tangent closest to intended direction
            if (tangent.x * dir.x + tangent.y * dir.y < 0) {
                tangent.x *= -1;
                tangent.y *= -1;
            }
            next.x = last.x + tangent.x * rootSpeed;
            next.y = last.y + tangent.y * rootSpeed;
            // Project onto boundary
            let ndx = next.x - collisionCircle.cx;
            let ndy = next.y - collisionCircle.cy;
            let nlen = Math.sqrt(ndx * ndx + ndy * ndy);
            next.x = collisionCircle.cx + ndx / nlen * (collisionCircle.r + 2);
            next.y = collisionCircle.cy + ndy / nlen * (collisionCircle.r + 2);
        }

        // Stop if out of bounds
        if (next.y >= latticeHeight || steps > 2000) break;
        rootLine.push(next);
        smoothRoot();
        steps++;
    }
    // Slope: (y2-y1)/(x2-x1)
    const start = rootLine[0], end = rootLine[rootLine.length - 1];
    let slope = (end.y - start.y) / (end.x - start.x || 1e-6);
    return { elongationZoneLength, downwardReturn, rootLine, slope };
}

// --- Helper to run all simulations ---
function runAllSimulations() {
    latticeCircles = computeLattice();
    allResults = [];
    for (let i = 0; i < elongationZoneLengths.length; ++i) {
        for (let j = 0; j < downwardReturns.length; ++j) {
            const elongationZoneLength = elongationZoneLengths[i];
            const downwardReturn = downwardReturns[j];
            const result = simulateRoot(
                elongationZoneLength,
                downwardReturn,
                spacing,
                radius,
                rootSpeed
            );
            allResults.push(result);
        }
    }
}

// --- Heatmap SVG container (takes 90% of vertical space) ---
const heatmapSvgContainer = heatmapDiv
    .append("div")
    .attr("id", "heatmapSvgContainer")
    .style("flex", "1 1 90%")
    .style("width", "100%")
    .style("height", "100%")
    .style("position", "relative")
    .style("overflow", "hidden");

// --- 6. Draw heatmap ---
function getHeatmapSize() {
    const node = heatmapSvgContainer.node();
    return {
        width: node.clientWidth,
        height: node.clientHeight
    };
}



function drawHeatmap() {
    // Remove any previous SVG
    heatmapSvgContainer.selectAll("svg").remove();

    const { width: heatmapWidth, height: heatmapHeight } = getHeatmapSize();
    const heatmapMargin = { top: 24, right: 20, bottom: 60, left: 80 }; // top margin smaller, sliders are outside

    const svgHeatmap = heatmapSvgContainer.append("svg")
        .attr("width", "100%")
        .attr("height", "100%")
        .attr("preserveAspectRatio", "xMidYMid meet")
        .attr("viewBox", `0 0 ${heatmapWidth} ${heatmapHeight}`);

    const xScale = d3.scaleBand()
        .domain(elongationZoneLengths)
        .range([heatmapMargin.left, heatmapWidth - heatmapMargin.right])
        .padding(0.05);

    const yScale = d3.scaleBand()
        .domain(downwardReturns)
        .range([heatmapHeight - heatmapMargin.bottom, heatmapMargin.top])
        .padding(0.05);

    const slopeExtent = d3.extent(allResults, d => d.slope);
    const colorScale = d3.scaleSequential(d3.interpolateRdBu)
        .domain([slopeExtent[1], slopeExtent[0]]); // Blue = left, Red = right

    svgHeatmap.selectAll("rect")
        .data(allResults)
        .join("rect")
        .attr("x", d => xScale(d.elongationZoneLength))
        .attr("y", d => yScale(d.downwardReturn))
        .attr("width", xScale.bandwidth())
        .attr("height", yScale.bandwidth())
        .attr("fill", d => colorScale(d.slope))
        .attr("stroke", "#222")
        .attr("cursor", "pointer")
        .on("click", function (event, d) {
            drawRoot(d);
        });

    // Axes
    svgHeatmap.append("g")
        .attr("transform", `translate(0,${heatmapHeight - heatmapMargin.bottom})`)
        .call(d3.axisBottom(xScale).tickFormat(d => d));

    svgHeatmap.append("g")
        .attr("transform", `translate(${heatmapMargin.left},0)`)
        .call(d3.axisLeft(yScale).tickFormat(d => d.toFixed(2)));

    // X axis label
    svgHeatmap.append("text")
        .attr("class", "x axis-label")
        .attr("text-anchor", "middle")
        .attr("x", heatmapMargin.left + (heatmapWidth - heatmapMargin.left - heatmapMargin.right) / 2)
        .attr("y", heatmapHeight - 15)
        .attr("font-size", 16)
        .text("Elongation Zone Length");

    // Y axis label
    svgHeatmap.append("text")
        .attr("class", "y axis-label")
        .attr("text-anchor", "middle")
        .attr("transform", `rotate(-90)`)
        .attr("x", -heatmapMargin.top - (heatmapHeight - heatmapMargin.top - heatmapMargin.bottom) / 2)
        .attr("y", 25)
        .attr("font-size", 16)
        .text("Downward Return");
}

// --- 7. Draw root for selected cell ---
function getVizSize() {
    const node = vizDiv.node();
    return {
        width: node.clientWidth,
        height: node.clientHeight
    };
}

const svgViz = vizDiv.append("svg")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("preserveAspectRatio", "xMidYMid meet")
    .style("display", "block")
    .style("margin", "0 auto");

const vizGroup = svgViz.append("g");

function updateVizTransform() {
    const { width, height } = getVizSize();
    const scale = Math.min(
        width / latticeWidth,
        height / latticeHeight
    );
    const tx = (width - latticeWidth * scale) / 2;
    const ty = (height - latticeHeight * scale) / 2;
    vizGroup.attr("transform", `translate(${tx},${ty}) scale(${scale})`);
}

function drawRoot(result) {
    // Draw lattice (scaled)
    const circles = vizGroup.selectAll("circle").data(latticeCircles);
    circles.join("circle")
        .attr("cx", d => d.cx)
        .attr("cy", d => d.cy)
        .attr("r", d => d.r)
        .attr("fill", "#666666");
    // Draw root (scaled)
    vizGroup.selectAll("polyline.root").data([result.rootLine])
        .join("polyline")
        .attr("class", "root")
        .attr("fill", "none")
        .attr("stroke", "#901A1E")
        .attr("stroke-width", 4)
        .attr("points", d => d.map(p => `${p.x},${p.y}`).join(" "));
    updateVizTransform();
}

// --- 8. Initial draw: pick middle cell ---
runAllSimulations();
drawHeatmap();
let selectedResult = allResults[Math.floor(allResults.length / 2)];
drawRoot(selectedResult);

// --- 9. Slider event listeners ---
d3.select("#spacingSlider").on("input", function () {
    spacing = +this.value;
    d3.select("#spacingValue").text(spacing);
    runAllSimulations();
    drawHeatmap();
    // Try to keep the same cell selected
    drawRoot(selectedResult);
});
d3.select("#radiusSlider").on("input", function () {
    radius = +this.value;
    d3.select("#radiusValue").text(radius);
    runAllSimulations();
    drawHeatmap();
    drawRoot(selectedResult);
});

// --- 10. Update selected cell on heatmap click ---
function drawHeatmapWithSelection() {
    drawHeatmap();
    // Add click handler to update selectedResult
    heatmapDiv.select("svg").selectAll("rect")
        .on("click", function (event, d) {
            selectedResult = d;
            drawRoot(d);
        });
}
drawHeatmapWithSelection();

// Redraw on window resize
window.addEventListener("resize", () => {
    drawHeatmapWithSelection();
    updateVizTransform();
});