// Create controls container
const controls = d3.select("body")
    .append("div")
    .attr("id", "controls")
    .style("position", "fixed")
    .style("top", "0")
    .style("left", "0")
    .style("right", "0")
    .style("z-index", "2")
    .style("background", "rgba(255,255,255,0.95)")
    .style("padding", "10px 15px")
    .style("box-shadow", "0 2px 8px rgba(0,0,0,0.08)")
    .style("user-select", "auto")
    .style("display", "flex")
    .style("align-items", "center")
    .style("gap", "20px")
    .style("flex-wrap", "wrap");

// Equation display
controls.append("div")
    .attr("class", "equation")
    .style("margin", "0")
    .style("padding", "5px 10px")
    .html("dθ/dt = -k·sin(θ(t-τ)) + η·ξ(t)");

// Time lag (tau) slider
const tauContainer = controls.append("div")
    .style("display", "flex")
    .style("align-items", "center")
    .style("gap", "5px");

tauContainer.append("label")
    .text("τ: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 0)
    .attr("max", 50)
    .attr("step", 1)
    .attr("value", 20)
    .attr("id", "tauSlider")
    .style("vertical-align", "middle");

tauContainer.append("span")
    .attr("id", "tauValue")
    .text("20");

// Gravitropic strength (k) slider
const kContainer = controls.append("div")
    .style("display", "flex")
    .style("align-items", "center")
    .style("gap", "5px");

kContainer.append("label")
    .text("k: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 0)
    .attr("max", 0.5)
    .attr("step", 0.01)
    .attr("value", 0.08)
    .attr("id", "kSlider")
    .style("vertical-align", "middle");

kContainer.append("span")
    .attr("id", "kValue")
    .text("0.08");

// Noise strength (eta) slider
const etaContainer = controls.append("div")
    .style("display", "flex")
    .style("align-items", "center")
    .style("gap", "5px");

etaContainer.append("label")
    .text("η: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 0)
    .attr("max", 0.5)
    .attr("step", 0.01)
    .attr("value", 0.05)
    .attr("id", "etaSlider")
    .style("vertical-align", "middle");

etaContainer.append("span")
    .attr("id", "etaValue")
    .text("0.05");

// Time step
const dtContainer = controls.append("div")
    .style("display", "flex")
    .style("align-items", "center")
    .style("gap", "5px");

dtContainer.append("label")
    .text("dt: ")
    .append("input")
    .attr("type", "range")
    .attr("min", 0.1)
    .attr("max", 2)
    .attr("step", 0.1)
    .attr("value", 0.5)
    .attr("id", "dtSlider")
    .style("vertical-align", "middle");

dtContainer.append("span")
    .attr("id", "dtValue")
    .text("0.5");

// Stability parameter display
controls.append("div")
    .attr("id", "stabilityParam")
    .style("font-size", "13px")
    .style("color", "#666")
    .text("τ·k/(π/2) = 1.02");

// Reset Button
controls.append("button")
    .attr("id", "resetBtn")
    .text("Reset")
    .on("click", reset);

// Create SVG
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
function updateStabilityParam() {
    const tau = +d3.select("#tauSlider").property("value");
    const k = +d3.select("#kSlider").property("value");
    const x = (2 * tau * k) / Math.PI;
    d3.select("#stabilityParam").text(`τ·k/(π/2) = ${x.toFixed(2)}`);
}

d3.select("#tauSlider").on("input", function() {
    d3.select("#tauValue").text(this.value);
    updateStabilityParam();
});
d3.select("#kSlider").on("input", function() {
    d3.select("#kValue").text(this.value);
    updateStabilityParam();
});
d3.select("#etaSlider").on("input", function() {
    d3.select("#etaValue").text(this.value);
});
d3.select("#dtSlider").on("input", function() {
    d3.select("#dtValue").text(this.value);
});

// Initialize stability parameter display
updateStabilityParam();

// Redraw on resize
d3.select(window).on("resize", resizeAndDraw);

function resizeAndDraw() {
    svg.attr("width", window.innerWidth).attr("height", window.innerHeight);
    draw();
}

let rootLines = [];

// Define 5 initial angles in degrees
const initialAngles = [90, 67.5, 45, 22.5, 0]; // Left to right: horizontal to vertical
const colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"];

// Helper function to generate Gaussian random number (Box-Muller transform)
function gaussianRandom(mean = 0, stdDev = 1) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return z0 * stdDev + mean;
}

function reset() {
    const width = window.innerWidth;
    const controlsRect = document.getElementById("controls").getBoundingClientRect();
    const startY = controlsRect.bottom + 10;
    
    // Create 5 roots with different initial angles, evenly spaced horizontally
    const numRoots = initialAngles.length;
    const spacing = width / (numRoots + 1);
    
    rootLines = initialAngles.map((angleDeg, i) => {
        const angleRad = (angleDeg * Math.PI) / 180;
        const startX = spacing * (i + 1);
        return {
            points: [{ x: startX, y: startY }],
            // History of theta values for delay
            thetaHistory: [angleRad],
            timeHistory: [0],
            currentTime: 0,
            color: colors[i],
            initialAngle: angleDeg
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

function getDelayedTheta(rootLine, tau) {
    const targetTime = rootLine.currentTime - tau;
    
    // If we don't have history that far back, use initial angle
    if (targetTime <= 0) {
        return rootLine.thetaHistory[0];
    }
    
    // Find the closest time point in history
    let closestIdx = 0;
    let minDiff = Math.abs(rootLine.timeHistory[0] - targetTime);
    
    for (let i = 1; i < rootLine.timeHistory.length; i++) {
        const diff = Math.abs(rootLine.timeHistory[i] - targetTime);
        if (diff < minDiff) {
            minDiff = diff;
            closestIdx = i;
        }
    }
    
    return rootLine.thetaHistory[closestIdx];
}

function growRoots() {
    if (rootLines.length === 0) return;

    const dt = +d3.select("#dtSlider").property("value");
    const tau = +d3.select("#tauSlider").property("value");
    const k = +d3.select("#kSlider").property("value");
    const eta = +d3.select("#etaSlider").property("value");

    rootLines.forEach(rootLine => {
        if (rootLine.points.length === 0) return;

        // Get current theta
        const currentTheta = rootLine.thetaHistory[rootLine.thetaHistory.length - 1];
        
        // Get delayed theta for the DDE
        const delayedTheta = getDelayedTheta(rootLine, tau);
        
        // Solve the DDE: dθ/dt = -k·sin(θ(t-τ)) + η·ξ(t)
        // Negative sign ensures gravitropic correction toward vertical (θ=0)
        // Noise is scaled by sqrt(dt) for proper stochastic integration
        const noise = gaussianRandom(0, 1);
        const dThetaDt = -k * Math.sin(delayedTheta) + (eta / Math.sqrt(dt)) * noise;
        
        // Euler method: θ(t+dt) = θ(t) + dθ/dt * dt
        const newTheta = currentTheta + dThetaDt * dt;
        
        // Update time
        rootLine.currentTime += dt;
        
        // Store theta and time in history
        rootLine.thetaHistory.push(newTheta);
        rootLine.timeHistory.push(rootLine.currentTime);
        
        // Limit history length to prevent memory issues
        const maxHistoryLength = 1000;
        if (rootLine.thetaHistory.length > maxHistoryLength) {
            rootLine.thetaHistory.shift();
            rootLine.timeHistory.shift();
        }
        
        // Compute growth direction based on theta
        // θ = 0 means straight down, positive θ means tilted right
        // Growth speed scales with dt to maintain constant speed in simulation time
        const growthSpeed = 2 * dt; // pixels per simulation time unit
        const dir = {
            x: Math.sin(newTheta),
            y: Math.cos(newTheta)
        };

        let last = rootLine.points[rootLine.points.length - 1];
        let next = { 
            x: last.x + dir.x * growthSpeed, 
            y: last.y + dir.y * growthSpeed 
        };

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
