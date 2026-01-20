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

// Add Division Zone Length slider
controls.append("label")
    .text(" Division Zone Length: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 1)
    .attr("max", 100)
    .attr("step", 1)
    .attr("value", 30)
    .attr("id", "divisionZoneSlider")
    .style("vertical-align", "middle")
    .style("margin-right", "20px");

// Add Elongation Zone Length slider
controls.append("label")
    .text(" Elongation Zone Length: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 1)
    .attr("max", 100)
    .attr("step", 1)
    .attr("value", 20)
    .attr("id", "elongationZoneSlider")
    .style("vertical-align", "middle")
    .style("margin-right", "20px");

// Add Downward Return Rate slider
controls.append("label")
    .text(" Downward Return: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 0.01)
    .attr("max", 0.5)
    .attr("step", 0.01)
    .attr("value", 0.2)
    .attr("id", "downwardReturnSlider")
    .style("vertical-align", "middle");

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
d3.selectAll("#densitySlider, #radiusSlider, #rootSpeedSlider, #divisionZoneSlider, #elongationZoneSlider, #downwardReturnSlider").on("input", draw);

function resizeAndDraw() {
    svg.attr("width", window.innerWidth).attr("height", window.innerHeight);
    draw();
}


let circles = [];
let rootLine = [];
let rootDirection = { x: 0, y: 1 }; // Downward
const rootSpeed = 2; // pixels per frame

function reset() {
    // Reset the root line to the top center
    rootLine = [{ x: window.innerWidth / 2, y: 0 }];
    rootDirection = { x: 0, y: 1 };
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
            .attr("fill", "#666666"), // Changed to grey
        update => update
            .attr("cx", d => d.cx)
            .attr("cy", d => d.cy)
            .attr("r", d => d.r)
            .attr("fill", "#666666"), // Ensure update is grey
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



function getDirection() {
    const divisionZoneLength = +d3.select("#divisionZoneSlider").property("value");
    if (rootLine.length < 2) return { x: 0, y: 1 };
    const idx = Math.max(0, rootLine.length - divisionZoneLength);
    const from = rootLine[idx];
    const to = rootLine[rootLine.length - 1];
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let len = Math.sqrt(dx * dx + dy * dy);
    if (len === 0) return { x: 0, y: 1 };
    return { x: dx / len, y: dy / len };
}

function smoothRoot() {
    const divisionZoneLength = +d3.select("#divisionZoneSlider").property("value");
    const elongationZoneLength = +d3.select("#elongationZoneSlider").property("value");
    const totalZoneLength = divisionZoneLength + elongationZoneLength;
    if (rootLine.length < 3) return;
    // Simple smoothing: moving average for points within the total zone (division + elongation)
    for (let i = rootLine.length - totalZoneLength; i < rootLine.length - 1; i++) {
        if (i <= 0 || i >= rootLine.length - 1) continue;
        rootLine[i].x = (rootLine[i - 1].x + rootLine[i].x + rootLine[i + 1].x) / 3;
        rootLine[i].y = (rootLine[i - 1].y + rootLine[i].y + rootLine[i + 1].y) / 3;
    }
}

function growRoot() {
    if (rootLine.length === 0) return;

    const rootSpeed = +d3.select("#rootSpeedSlider").property("value");
    const sliderDownwardReturn = +d3.select("#downwardReturnSlider").property("value");

    // Compute direction based on division zone
    let dir = getDirection();
    // Add gravity bias
    dir.x = dir.x * (1 - sliderDownwardReturn) + 0 * sliderDownwardReturn;
    dir.y = dir.y * (1 - sliderDownwardReturn) + 1 * sliderDownwardReturn;
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

    // Only add if within bounds
    if (next.y < window.innerHeight) {
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
