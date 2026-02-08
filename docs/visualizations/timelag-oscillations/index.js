// Create controls container: only as wide as needed, not full width
const controls = d3.select("body")
    .append("div")
    .attr("id", "controls")
    .style("position", "fixed")
    .style("top", "10px")
    .style("left", "10px")
    .style("z-index", "2")
    .style("background", "rgba(255,255,255,0.95)")
    .style("padding", "15px")
    .style("border-radius", "8px")
    .style("box-shadow", "0 2px 8px rgba(0,0,0,0.08)")
    .style("user-select", "auto")
    .style("display", "inline-block");

// Root Speed slider
controls.append("label")
    .text("Root Speed: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 0.5)
    .attr("max", 8)
    .attr("step", 0.1)
    .attr("value", 0.5)
    .attr("id", "rootSpeedSlider")
    .style("vertical-align", "middle")
    .style("margin-right", "20px");

controls.append("span")
    .attr("id", "rootSpeedValue")
    .style("margin-right", "30px")
    .text("0.5");

controls.append("br");

// Division Zone Length slider (physical distance in pixels)
controls.append("label")
    .text("Division Zone Length: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 10)
    .attr("max", 500)
    .attr("step", 10)
    .attr("value", 50)
    .attr("id", "divisionZoneSlider")
    .style("vertical-align", "middle")
    .style("margin-right", "20px")
    .style("margin-top", "10px");

controls.append("span")
    .attr("id", "divisionZoneValue")
    .style("margin-right", "30px")
    .text("50");

// Gravity slider
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

controls.append("span")
    .attr("id", "gravityValue")
    .text("0.2");

// Reset Button
controls.append("button")
    .attr("id", "resetBtn")
    .text("Reset")
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

// Update value displays
d3.select("#rootSpeedSlider").on("input", function() {
    d3.select("#rootSpeedValue").text(this.value);
});
d3.select("#divisionZoneSlider").on("input", function() {
    d3.select("#divisionZoneValue").text(this.value);
});
d3.select("#gravitySlider").on("input", function() {
    d3.select("#gravityValue").text(this.value);
});

// Redraw on resize
d3.select(window).on("resize", resizeAndDraw);

function resizeAndDraw() {
    svg.attr("width", window.innerWidth).attr("height", window.innerHeight);
    draw();
}

let rootLines = [];

// Define 5 initial angles from 90 to 0 degrees (left to right)
const initialAngles = [90, 67.5, 45, 22.5, 0];
const colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"];

function reset() {
    const width = window.innerWidth;
    const controlsRect = document.getElementById("controls").getBoundingClientRect();
    const startY = controlsRect.bottom + 50;
    
    // Create 5 roots with different initial angles, evenly spaced horizontally
    const numRoots = initialAngles.length;
    const spacing = width / (numRoots + 1);
    
    rootLines = initialAngles.map((angleDeg, i) => {
        const angleRad = (angleDeg * Math.PI) / 180;
        const startX = spacing * (i + 1);
        return {
            points: [{ x: startX, y: startY }],
            direction: { x: Math.sin(angleRad), y: Math.cos(angleRad) },
            color: colors[i],
            angle: angleDeg
        };
    });
    
    draw();
}

function draw() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    // Draw root lines
    svg.selectAll("polyline.root").data(rootLines, (d, i) => i)
        .join("polyline")
        .attr("class", "root")
        .attr("fill", "none")
        .attr("stroke", d => d.color)
        .attr("stroke-width", 1)
        .attr("points", d => d.points.map(p => `${p.x},${p.y}`).join(" "));
}

function getDirection(rootLine) {
    const divisionZoneLength = +d3.select("#divisionZoneSlider").property("value");
    if (rootLine.points.length < 2) return rootLine.direction;
    
    // Walk back through points until we've covered divisionZoneLength distance
    let accumulatedDistance = 0;
    let idx = rootLine.points.length - 1;
    
    while (idx > 0 && accumulatedDistance < divisionZoneLength) {
        const p1 = rootLine.points[idx];
        const p2 = rootLine.points[idx - 1];
        const segmentLength = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
        accumulatedDistance += segmentLength;
        idx--;
    }
    
    const from = rootLine.points[Math.max(0, idx)];
    const to = rootLine.points[rootLine.points.length - 1];
    let dx = to.x - from.x;
    let dy = to.y - from.y;
    let len = Math.sqrt(dx * dx + dy * dy);
    if (len === 0) return rootLine.direction;
    return { x: dx / len, y: dy / len };
}

function growRoots() {
    if (rootLines.length === 0) return;

    const rootSpeed = +d3.select("#rootSpeedSlider").property("value");
    const gravitropism = +d3.select("#gravitySlider").property("value");

    rootLines.forEach(rootLine => {
        if (rootLine.points.length === 0) return;

        // Sense direction from division zone (time-lagged response)
        let sensedDir = getDirection(rootLine);
        
        // Compute angle of sensed direction from vertical (downward)
        // atan2(x, y) gives angle from vertical: 0° = down, +π/2 = right, -π/2 = left
        let currentAngle = Math.atan2(sensedDir.x, sensedDir.y);
        
        // Apply gravitropic correction proportional to deviation from vertical
        // The time lag causes oscillations: correction is based on old direction,
        // leading to overshoot when the root is already turning back
        let correctionRate = gravitropism;
        let newAngle = currentAngle * (1 - correctionRate);
        
        // Compute new growth direction
        let dir = {
            x: Math.sin(newAngle),
            y: Math.cos(newAngle)
        };

        let last = rootLine.points[rootLine.points.length - 1];
        let next = { x: last.x + dir.x * rootSpeed, y: last.y + dir.y * rootSpeed };

        // Only add if within bounds
        if (next.y < window.innerHeight && next.x > 0 && next.x < window.innerWidth) {
            rootLine.points.push(next);
        }
    });
    
    draw();
    requestAnimationFrame(growRoots);
}

// Initial setup
resizeAndDraw();
reset();
requestAnimationFrame(growRoots);