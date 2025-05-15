// Invasion Percolation Visualization

// --- Layout: Flexbox container ---
const container = d3.select("body")
    .append("div")
    .attr("id", "mainFlexContainer")
    .style("display", "flex")
    .style("align-items", "flex-start")
    .style("height", "100vh")
    .style("box-sizing", "border-box");

// --- Controls Panel (left) ---
const controls = container
    .append("div")
    .attr("id", "controls")
    .style("min-width", "220px")
    .style("max-width", "300px")
    .style("margin", "30px 20px 30px 30px")
    .style("background", "rgba(255,255,255,0.95)")
    .style("padding", "18px 24px 18px 18px")
    .style("border-radius", "10px")
    .style("box-shadow", "0 2px 8px rgba(0,0,0,0.08)")
    .style("user-select", "auto")
    .style("display", "flex")
    .style("flex-direction", "column")
    .style("gap", "28px")
    .style("font-size", "24px");

// Lattice size slider
const latticeRow = controls.append("div")
    .style("display", "flex")
    .style("flex-direction", "column")
    .style("align-items", "stretch")
    .style("gap", "6px");
latticeRow.append("label")
    .attr("for", "latticeSizeSlider")
    .text("Lattice size (L): ");
const latticeSliderWrap = latticeRow.append("div")
    .style("display", "flex")
    .style("align-items", "center");
latticeSliderWrap.append("input")
    .attr("type", "range")
    .attr("min", 10)
    .attr("max", 100)
    .attr("value", 40)
    .attr("id", "latticeSizeSlider")
    .style("flex", "1")
    .style("height", "32px");
latticeSliderWrap.append("span")
    .attr("id", "latticeSizeValue")
    .style("margin-left", "16px")
    .text("40");

// Simulation speed slider (log scale)
const speedRow = controls.append("div")
    .style("display", "flex")
    .style("flex-direction", "column")
    .style("align-items", "stretch")
    .style("gap", "6px");
speedRow.append("label")
    .attr("for", "speedSlider")
    .text("Simulation speed:");
const speedSliderWrap = speedRow.append("div")
    .style("display", "flex")
    .style("align-items", "center");
speedSliderWrap.append("input")
    .attr("type", "range")
    .attr("min", 0)
    .attr("max", 4)
    .attr("step", 0.01)
    .attr("value", 2)
    .attr("id", "speedSlider")
    .style("flex", "1")
    .style("height", "32px");
speedSliderWrap.append("span")
    .attr("id", "speedValue")
    .style("margin-left", "16px")
    .text("1");

// Vertical bias slider
const biasRow = controls.append("div")
    .style("display", "flex")
    .style("flex-direction", "column")
    .style("align-items", "stretch")
    .style("gap", "6px");
biasRow.append("label")
    .attr("for", "biasSlider")
    .text("Vertical bias:");
const biasSliderWrap = biasRow.append("div")
    .style("display", "flex")
    .style("align-items", "center");
biasSliderWrap.append("input")
    .attr("type", "range")
    .attr("min", 0)
    .attr("max", 1)
    .attr("step", 0.01)
    .attr("value", 0)
    .attr("id", "biasSlider")
    .style("flex", "1")
    .style("height", "32px");
biasSliderWrap.append("span")
    .attr("id", "biasValue")
    .style("margin-left", "16px")
    .text("0");

controls.append("button")
    .attr("id", "resetBtn")
    .text("Reset")
    .style("margin-top", "18px")
    .style("font-size", "24px")
    .style("padding", "10px 0")
    .on("click", reset);

// --- SVG Visualization (right) ---
const vizContainer = container
    .append("div")
    .attr("id", "vizContainer")
    .style("flex", "1")
    .style("display", "flex")
    .style("align-items", "center")
    .style("justify-content", "center")
    .style("height", "100vh")
    .style("overflow", "hidden");

const svg = vizContainer
    .append("svg")
    .attr("id", "mainSVG")
    .style("display", "block")
    .style("background", "#fff")

// --- Logic ---
d3.select("#latticeSizeSlider").on("input", function () {
    d3.select("#latticeSizeValue").text(this.value);
    reset();
});

d3.select("#speedSlider").on("input", function () {
    const speed = Math.pow(10, +this.value - 2); // log scale: 0→0.01, 1→0.1, 2→1, 3→10, 4→100
    d3.select("#speedValue").text(speed.toFixed(2));
});

d3.select("#biasSlider").on("input", function () {
    d3.select("#biasValue").text(this.value);
    reset();
});

d3.select(window).on("resize", resizeAndDraw);

let L = 40;
let lattice = [];
let filled = [];
let trapped = [];
let front = new Set();
let animationId = null;

function reset() {
    L = +d3.select("#latticeSizeSlider").property("value");
    lattice = [];
    filled = [];
    trapped = [];
    front = new Set();
    speedAccumulator = 0; // <-- Reset accumulator

    // Get vertical bias value
    const bias = +d3.select("#biasSlider").property("value");

    // Remove all old cells from SVG
    svg.selectAll("*").remove();

    // Initialize lattice: each site has a random number and state
    for (let y = 0; y < L; y++) {
        lattice[y] = [];
        for (let x = 0; x < L; x++) {
            // Apply vertical bias: value = Math.random() - y * bias
            lattice[y][x] = {
                x, y,
                value: Math.random() - y * bias,
                state: "unfilled" // "unfilled", "filled", "trapped"
            };
        }
    }

    // Start with the middle of the top row filled
    let startX = Math.floor(L / 2);
    lattice[0][startX].state = "filled";
    filled = [{ x: startX, y: 0 }];
    updateFront();
    trapped = [];
    resizeAndDraw();
    if (animationId) cancelAnimationFrame(animationId);
    animationId = requestAnimationFrame(step);
}


function resizeAndDraw() {
    // Compute available height (minus some margin for flex container)
    const containerRect = vizContainer.node().getBoundingClientRect();
    const controlsRect = controls.node().getBoundingClientRect();
    // Use the smaller of container height or width for the SVG
    const size = Math.min(containerRect.height, containerRect.width);
    svg.attr("width", size).attr("height", size);
    draw(size);
}

function draw(size) {
    const cellSize = size / L;

    // Flatten lattice for D3
    const flat = [];
    for (let y = 0; y < L; y++) {
        for (let x = 0; x < L; x++) {
            flat.push(lattice[y][x]);
        }
    }

    // Find min/max value for color scaling (for unfilled)
    let minVal = Infinity, maxVal = -Infinity;
    for (const d of flat) {
        if (d.state === "unfilled") {
            if (d.value < minVal) minVal = d.value;
            if (d.value > maxVal) maxVal = d.value;
        }
    }
    // If all unfilled are gone, fallback to [0,1]
    if (!isFinite(minVal) || !isFinite(maxVal) || minVal === maxVal) {
        minVal = 0; maxVal = 1;
    }

    // Draw cells
    svg.selectAll("rect")
        .data(flat, d => d.x + "," + d.y)
        .join(
            enter => enter.append("rect")
                .attr("x", d => d.x * cellSize)
                .attr("y", d => d.y * cellSize)
                .attr("width", cellSize)
                .attr("height", cellSize)
                .attr("stroke", "#fff")
                .attr("stroke-width", 0.5)
                .attr("fill", d => {
                    if (d.state === "filled") return "#901A1E";
                    if (d.state === "trapped") return "#0a5963";
                    // Unfilled: #666666 with alpha based on normalized value
                    // Normalize value to [0,1] for alpha
                    let alpha = 0.3 + 0.7 * ((d.value - minVal) / (maxVal - minVal));
                    return `rgba(102,102,102,${alpha})`;
                }),
            update => update
                .attr("fill", d => {
                    if (d.state === "filled") return "#901A1E";
                    if (d.state === "trapped") return "#0a5963";
                    let alpha = 0.3 + 0.7 * ((d.value - minVal) / (maxVal - minVal));
                    return `rgba(102,102,102,${alpha})`;
                }),
            exit => exit.remove()
        );
}

function neighbours(x, y) {
    return [
        [(x - 1 + L) % L, y], // left (wrap)
        [(x + 1) % L, y],     // right (wrap)
        [x, y - 1],           // up
        [x, y + 1]            // down
    ].filter(([nx, ny]) => ny >= 0 && ny < L);
}


// Update the invasion front (unfilled neighbours of filled)
function updateFront() {
    front.clear();
    for (const { x, y } of filled) {
        for (const [nx, ny] of neighbours(x, y)) {
            const n = lattice[ny][nx];
            if (n.state === "unfilled") {
                front.add(n.y * L + n.x); // unique id
            }
        }
    }
}

// Flood fill from the boundary to mark reachable unfilled sites
function markTrapped() {
    // 0: unvisited, 1: reachable, 2: trapped
    let visited = Array.from({ length: L }, () => Array(L).fill(0));
    let queue = [];

    // Add all unfilled boundary sites to queue (top and bottom rows, all columns)
    for (let x = 0; x < L; x++) {
        if (lattice[0][x].state === "unfilled") { queue.push([x, 0]); visited[0][x] = 1; }
        if (lattice[L - 1][x].state === "unfilled") { queue.push([x, L - 1]); visited[L - 1][x] = 1; }
    }

    // BFS to mark all reachable unfilled sites, using periodic boundary in x
    while (queue.length > 0) {
        const [x, y] = queue.shift();
        for (const [nx, ny] of neighbours(x, y)) {
            if (lattice[ny][nx].state === "unfilled" && visited[ny][nx] === 0) {
                visited[ny][nx] = 1;
                queue.push([nx, ny]);
            }
        }
    }

    // Mark all unfilled but unreachable as trapped
    for (let y = 0; y < L; y++) {
        for (let x = 0; x < L; x++) {
            if (lattice[y][x].state === "unfilled" && visited[y][x] === 0) {
                lattice[y][x].state = "trapped";
                trapped.push({ x, y });
            }
        }
    }
}


let speedAccumulator = 0;

function step() {
    // If no front, stop
    if (front.size === 0) {
        draw(svg.attr("width"));
        return;
    }

    // Get simulation speed (sites per frame, log scale)
    const speed = Math.pow(10, +d3.select("#speedSlider").property("value") - 2);

    // --- Fractional speed handling ---
    speedAccumulator += speed;
    let sitesToFill = Math.floor(speedAccumulator);
    speedAccumulator -= sitesToFill;

    // Always fill at least one site if speed >= 1, otherwise only fill when enough frames have passed
    if (sitesToFill < 1 && speed < 1) {
        animationId = requestAnimationFrame(step);
        return;
    }

    let filledCount = 0;

    while (front.size > 0 && filledCount < sitesToFill) {
        // Find the front site with the lowest value
        let minVal = Infinity;
        let minSite = null;
        for (const idx of front) {
            const x = idx % L, y = Math.floor(idx / L);
            const site = lattice[y][x];
            if (site.value < minVal) {
                minVal = site.value;
                minSite = site;
            }
        }

        if (!minSite) break;

        // Fill the chosen site
        minSite.state = "filled";
        filled.push({ x: minSite.x, y: minSite.y });

        // Update trapped regions
        markTrapped();

        // Update front
        updateFront();

        filledCount++;
    }

    // Draw
    draw(svg.attr("width"));

    // Continue
    animationId = requestAnimationFrame(step);
}

// Initial setup
reset();