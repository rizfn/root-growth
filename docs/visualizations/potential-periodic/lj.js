// 1. Removed unnecessary body clearing line

// 2. Create controls container: only as wide as needed, not full width
const controls = d3.select("body")
    .append("div")
    .attr("id", "controls")
    .style("position", "fixed")
    .style("top", "10px")
    .style("left", "10px")
    .style("z-index", "2")
    .style("background", "rgba(255,255,255,0.95)")
    .style("padding", "10px 20px 10px 10px")
    .style("border-radius", "8px")
    .style("box-shadow", "0 2px 8px rgba(0,0,0,0.08)")
    .style("user-select", "auto")
    .style("display", "inline-block");

// Density slider
controls.append("label")
    .text("Density: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 10)
    .attr("max", 80)
    .attr("value", 40)
    .attr("id", "densitySlider")
    .style("vertical-align", "middle")
    .style("margin-right", "20px");

// Radius slider
controls.append("label")
    .text(" Radius: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 5)
    .attr("max", 40)
    .attr("value", 15)
    .attr("id", "radiusSlider")
    .style("vertical-align", "middle");

// Add Root Speed slider
controls.append("label")
    .text(" Root Speed: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 0.5)
    .attr("max", 8)
    .attr("step", 0.1)
    .attr("value", 2)
    .attr("id", "rootSpeedSlider")
    .style("vertical-align", "middle")
    .style("margin-right", "20px");

// Add Memory Length slider
controls.append("label")
    .text(" Memory Length: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 1)
    .attr("max", 100)
    .attr("step", 1)
    .attr("value", 50)
    .attr("id", "memoryLengthSlider")
    .style("vertical-align", "middle")
    .style("margin-right", "20px");

// Add Gravity slider (gravitational pull)
controls.append("label")
    .text(" Gravity: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 0)
    .attr("max", 1)
    .attr("step", 0.01)
    .attr("value", 0.2)
    .attr("id", "gravitySlider")
    .style("vertical-align", "middle")
    .style("margin-right", "20px");
    
// Add Lennard-Jones epsilon (well depth) slider
controls.append("label")
.text(" LJ ε: ")
.append("input")
.attr("type", "range")
.attr("min", 0.01)
.attr("max", 2)
.attr("step", 0.01)
.attr("value", 0.5)
.attr("id", "ljEpsilonSlider")
.style("vertical-align", "middle")
.style("margin-right", "20px");

// Add Lennard-Jones sigma (distance scale) slider
controls.append("label")
.text(" LJ σ: ")
.append("input")
.attr("type", "range")
.attr("min", 0.01)
.attr("max", 2)
.attr("step", 0.01)
.attr("value", 0.5)
.attr("id", "ljSigmaSlider")
.style("vertical-align", "middle")
.style("margin-right", "20px");

// Add Reset Button
controls.append("button")
    .attr("id", "resetBtn")
    .text("Reset Root")
    .style("margin-left", "20px")
    .on("click", reset);

// Create SVG (instead of canvas)
const svg = d3.select("body")
    .append("svg")
    .attr("id", "mainSVG")
    .style("display", "block")
    .style("position", "absolute")
    .style("top", 0)
    .style("left", 0)
    .style("z-index", "0")
    .attr("width", window.innerWidth)
    .attr("height", window.innerHeight);

// Redraw on resize or slider input
d3.select(window).on("resize", resizeAndDraw);
d3.selectAll("#densitySlider, #radiusSlider, #rootSpeedSlider, #memoryLengthSlider, #gravitySlider, #potentialStrengthSlider").on("input", draw);

function resizeAndDraw() {
    svg.attr("width", window.innerWidth).attr("height", window.innerHeight);
    draw();
}

let circles = [];
let rootLine = [];
let rootVelocity = { x: 0, y: 1 }; // Initial downward
let lastTimestamp = null;

function reset() {
    // Reset the root line to the top center
    rootLine = [{ x: window.innerWidth / 2 + 10, y: 0 }];
    rootVelocity = { x: 0, y: 1 };
    lastTimestamp = null;
    draw();
}

function draw() {
    const controlsRect = document.getElementById("controls").getBoundingClientRect();
    const controlsBottom = controlsRect.bottom;

    const density = +d3.select("#densitySlider").property("value");
    const radius = +d3.select("#radiusSlider").property("value");
    const width = window.innerWidth;
    const height = window.innerHeight;

    const spacing = width / density;
    const rowHeight = Math.sqrt(3) * spacing / 2;

    // 3. Fix circle arrangement: use integer row index for offset calculation
    circles = [];
    let row = 0;
    for (let y = controlsBottom + radius; y < height + rowHeight; y += rowHeight, row++) {
        const offsetX = row % 2 === 0 ? 0 : spacing / 2;
        for (let x = 0; x < width + spacing; x += spacing) {
            circles.push({ cx: x + offsetX, cy: y, r: radius });
        }
    }

    // Data join for circles
    const sel = svg.selectAll("circle").data(circles);

    sel.join(
        enter => enter.append("circle")
            .attr("cx", d => d.cx)
            .attr("cy", d => d.cy)
            .attr("r", d => d.r)
            .attr("fill", "#666666"),
        update => update
            .attr("cx", d => d.cx)
            .attr("cy", d => d.cy)
            .attr("r", d => d.r)
            .attr("fill", "#666666"),
        exit => exit.remove()
    );

    // Draw root line
    svg.selectAll("polyline.root").data([rootLine])
        .join("polyline")
        .attr("class", "root")
        .attr("fill", "none")
        .attr("stroke", "#901A1E")
        .attr("stroke-width", 4)
        .attr("points", d => d.map(p => `${p.x},${p.y}`).join(" "));

    // If rootLine is empty, reset
    if (rootLine.length === 0) reset();
}


function computePotentialForce(tip) {
    // Lennard-Jones parameters from sliders
    const epsilon = +d3.select("#ljEpsilonSlider").property("value");
    const sigma = +d3.select("#ljSigmaSlider").property("value");
    let fx = 0, fy = 0;
    for (let c of circles) {
        let dx = tip.x - c.cx;
        let dy = tip.y - c.cy;
        let dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 1e-3) dist = 1e-3; // Avoid division by zero

        // Lennard-Jones force (derivative of potential)
        // Rest length is the radius of the obstacle: r0 = c.r
        // So, r = dist, and sigma is set relative to c.r
        // We'll use sigma_effective = sigma * c.r
        let sigma_eff = sigma * c.r;
        let r = dist;
        if (r < 1) r = 1; // Avoid singularity at overlap

        let sigOverR = sigma_eff / r;
        let sigOverR6 = Math.pow(sigOverR, 6);
        let sigOverR12 = sigOverR6 * sigOverR6;
        let force = 24 * epsilon * (2 * sigOverR12 - sigOverR6) / r;

        fx += (dx / dist) * force;
        fy += (dy / dist) * force;
    }
    return { x: fx, y: fy };
}


// Gravity force (constant downward acceleration)
function gravityForce() {
    const gravity = +d3.select("#gravitySlider").property("value");
    return { x: 0, y: gravity };
}

// Smooth the elongation zone (moving average)
function smoothRoot() {
    const memoryLength = +d3.select("#memoryLengthSlider").property("value");
    if (rootLine.length < 3) return;
    for (let i = rootLine.length - memoryLength; i < rootLine.length - 1; i++) {
        if (i <= 0 || i >= rootLine.length - 1) continue;
        rootLine[i].x = (rootLine[i - 1].x + rootLine[i].x + rootLine[i + 1].x) / 3;
        rootLine[i].y = (rootLine[i - 1].y + rootLine[i].y + rootLine[i + 1].y) / 3;
    }
}

function growRoot(timestamp) {
    if (rootLine.length === 0) return;

    const rootSpeed = +d3.select("#rootSpeedSlider").property("value");
    const memoryLength = +d3.select("#memoryLengthSlider").property("value");

    // Time step for smoother animation (optional)
    let dt = 1;
    if (lastTimestamp !== null && timestamp !== undefined) {
        dt = Math.min((timestamp - lastTimestamp) / 16, 2); // Clamp to avoid jumps
    }
    lastTimestamp = timestamp;

    // Get tip
    let last = rootLine[rootLine.length - 1];

    // Compute forces
    let potForce = computePotentialForce(last);
    let grav = gravityForce();

    // Add forces to velocity (Euler integration)
    rootVelocity.x += (potForce.x + grav.x) * dt;
    rootVelocity.y += (potForce.y + grav.y) * dt;

    // Normalize velocity to rootSpeed
    let vlen = Math.sqrt(rootVelocity.x * rootVelocity.x + rootVelocity.y * rootVelocity.y);
    if (vlen === 0) {
        rootVelocity = { x: 0, y: 1 };
        vlen = 1;
    }
    rootVelocity.x = (rootVelocity.x / vlen) * rootSpeed;
    rootVelocity.y = (rootVelocity.y / vlen) * rootSpeed;

    // Compute next tip position
    let next = {
        x: last.x + rootVelocity.x * dt,
        y: last.y + rootVelocity.y * dt
    };

    // Only add if within bounds
    if (next.y < window.innerHeight && next.x > 0 && next.x < window.innerWidth) {
        rootLine.push(next);
        smoothRoot();
    }
    draw();
    requestAnimationFrame(growRoot);
}

// Initial setup
resizeAndDraw();
reset();
requestAnimationFrame(growRoot);