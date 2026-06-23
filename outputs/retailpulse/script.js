const ranges = {
  "7d": {
    revenue: "$1.84M",
    accuracy: "94.2%",
    churn: "8.7%",
    stockout: "$126K",
    labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    values: [42, 47, 45, 51, 58, 69, 64],
    forecast: [45, 49, 52, 56, 63, 73, 70]
  },
  "30d": {
    revenue: "$7.92M",
    accuracy: "92.8%",
    churn: "9.4%",
    stockout: "$482K",
    labels: ["W1", "W2", "W3", "W4", "W5"],
    values: [168, 184, 176, 213, 231],
    forecast: [174, 188, 196, 224, 246]
  },
  "90d": {
    revenue: "$22.6M",
    accuracy: "91.5%",
    churn: "10.1%",
    stockout: "$1.32M",
    labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    values: [510, 544, 529, 601, 655, 692],
    forecast: [525, 552, 570, 628, 681, 725]
  }
};

const categoryLift = {
  all: 1,
  grocery: 1.18,
  beauty: 0.74,
  home: 0.92
};

const recommendations = [
  {
    title: "Move beverage inventory to Austin 014",
    body: "Forecasted heat-driven demand exceeds current shelf capacity by 22% this weekend.",
    impact: 91,
    value: "+$42K"
  },
  {
    title: "Retain premium skincare customers",
    body: "Send loyalty credit to high-margin buyers with declining purchase frequency.",
    impact: 76,
    value: "1.8K customers"
  },
  {
    title: "Reduce home essentials promo depth",
    body: "Demand is stable without discounting; margin recovery outweighs unit softness.",
    impact: 64,
    value: "+3.1 pts"
  },
  {
    title: "Rebalance frozen foods in Miami",
    body: "Stockout risk reaches the threshold in four days under the current replenishment plan.",
    impact: 58,
    value: "$19K protected"
  }
];

const segments = [
  { name: "Loyal high value", value: 34, color: "#0f8b8d" },
  { name: "Promo responsive", value: 27, color: "#5367b2" },
  { name: "At risk", value: 18, color: "#c04c44" },
  { name: "New shoppers", value: 21, color: "#d79b22" }
];

let activeRange = "7d";
let activeCategory = "all";

const forecastCanvas = document.getElementById("forecastChart");
const segmentCanvas = document.getElementById("segmentChart");
const recommendationRoot = document.getElementById("recommendations");

function formatMetrics(data) {
  document.getElementById("revenueMetric").textContent = data.revenue;
  document.getElementById("accuracyMetric").textContent = data.accuracy;
  document.getElementById("churnMetric").textContent = data.churn;
  document.getElementById("stockoutMetric").textContent = data.stockout;
}

function drawForecast() {
  const data = ranges[activeRange];
  const lift = categoryLift[activeCategory];
  const values = data.values.map((value) => value * lift);
  const forecast = data.forecast.map((value) => value * lift);
  const ctx = forecastCanvas.getContext("2d");
  const width = forecastCanvas.width;
  const height = forecastCanvas.height;
  const pad = { top: 24, right: 28, bottom: 48, left: 56 };
  const chartW = width - pad.left - pad.right;
  const chartH = height - pad.top - pad.bottom;
  const max = Math.max(...values, ...forecast) * 1.18;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#dce3e8";
  ctx.lineWidth = 1;
  ctx.font = "20px system-ui";

  for (let i = 0; i <= 4; i += 1) {
    const y = pad.top + (chartH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }

  function point(value, index, list) {
    const x = pad.left + (chartW / Math.max(1, list.length - 1)) * index;
    const y = pad.top + chartH - (value / max) * chartH;
    return { x, y };
  }

  function drawLine(list, color, dashed = false) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 4;
    ctx.setLineDash(dashed ? [10, 8] : []);
    list.forEach((value, index) => {
      const p = point(value, index, list);
      if (index === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawLine(values, "#5367b2");
  drawLine(forecast, "#0f8b8d", true);

  values.forEach((value, index) => {
    const p = point(value, index, values);
    ctx.fillStyle = "#5367b2";
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.fillStyle = "#66737f";
  ctx.font = "22px system-ui";
  data.labels.forEach((label, index) => {
    const x = pad.left + (chartW / Math.max(1, data.labels.length - 1)) * index;
    ctx.fillText(label, x - 14, height - 15);
  });

  ctx.fillStyle = "#182127";
  ctx.font = "bold 22px system-ui";
  ctx.fillText("Actual", width - 215, 30);
  ctx.fillStyle = "#5367b2";
  ctx.fillRect(width - 248, 17, 22, 8);
  ctx.fillStyle = "#182127";
  ctx.fillText("AI forecast", width - 116, 30);
  ctx.fillStyle = "#0f8b8d";
  ctx.fillRect(width - 152, 17, 22, 8);
}

function drawSegments() {
  const ctx = segmentCanvas.getContext("2d");
  const width = segmentCanvas.width;
  const height = segmentCanvas.height;
  const center = { x: width / 2, y: height / 2 };
  const radius = Math.min(width, height) / 2 - 18;
  let start = -Math.PI / 2;

  ctx.clearRect(0, 0, width, height);
  segments.forEach((segment) => {
    const angle = (segment.value / 100) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(center.x, center.y);
    ctx.fillStyle = segment.color;
    ctx.arc(center.x, center.y, radius, start, start + angle);
    ctx.closePath();
    ctx.fill();
    start += angle;
  });

  ctx.beginPath();
  ctx.fillStyle = "#ffffff";
  ctx.arc(center.x, center.y, radius * 0.58, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#182127";
  ctx.font = "bold 32px system-ui";
  ctx.textAlign = "center";
  ctx.fillText("4", center.x, center.y - 4);
  ctx.fillStyle = "#66737f";
  ctx.font = "18px system-ui";
  ctx.fillText("segments", center.x, center.y + 24);
  ctx.textAlign = "start";
}

function renderRecommendations() {
  recommendationRoot.innerHTML = recommendations
    .map(
      (item) => `
        <div class="recommendation">
          <strong>${item.title}</strong>
          <p>${item.body}</p>
          <div class="impact-row">
            <div class="impact-bar"><span style="width: ${item.impact}%"></span></div>
            <b>${item.value}</b>
          </div>
        </div>
      `
    )
    .join("");
}

function renderSegments() {
  document.getElementById("segmentList").innerHTML = segments
    .map(
      (segment) => `
        <div class="segment-item">
          <span class="swatch" style="background:${segment.color}"></span>
          <span>${segment.name}</span>
          <strong>${segment.value}%</strong>
        </div>
      `
    )
    .join("");
}

function render() {
  formatMetrics(ranges[activeRange]);
  drawForecast();
  drawSegments();
  renderRecommendations();
  renderSegments();
}

document.querySelectorAll("[data-range]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-range]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    activeRange = button.dataset.range;
    render();
  });
});

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
  });
});

document.getElementById("categorySelect").addEventListener("change", (event) => {
  activeCategory = event.target.value;
  drawForecast();
});

document.getElementById("refreshBtn").addEventListener("click", () => {
  const revenue = document.getElementById("revenueMetric");
  revenue.animate(
    [
      { transform: "translateY(0)", opacity: 1 },
      { transform: "translateY(-4px)", opacity: 0.55 },
      { transform: "translateY(0)", opacity: 1 }
    ],
    { duration: 420, easing: "ease-out" }
  );
  drawForecast();
});

render();
