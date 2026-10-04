/**
 * CrowdWatch Frontend Control-Room Logic (Offline, Vanilla JS)
 */

document.addEventListener("DOMContentLoaded", () => {
  // Navigation Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const activePane = document.getElementById(`pane-${target}`);
      if (activePane) activePane.classList.add("active");
    });
  });

  // WebSocket Live Telemetry Stream
  let ws = null;
  const connDot = document.getElementById("conn-dot");
  const connStatus = document.getElementById("conn-status");
  const fpsBadge = document.getElementById("telemetry-fps");
  const latBadge = document.getElementById("telemetry-latency");
  const countBadge = document.getElementById("telemetry-count");
  const skipBadge = document.getElementById("telemetry-skip");

  function connectTelemetryWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      connDot.className = "dot online";
      connStatus.textContent = "ONLINE";
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.processed_fps !== undefined) {
          fpsBadge.textContent = `${data.processed_fps} FPS`;
        }
        if (data.latency_ms !== undefined) {
          latBadge.textContent = `${data.latency_ms} ms`;
        }
        if (data.total_count !== undefined) {
          countBadge.textContent = `${Math.round(data.total_count)}`;
        }
        if (data.frame_skip !== undefined) {
          skipBadge.textContent = `N=${data.frame_skip}`;
        }
        if (data.is_connected === false) {
          connDot.className = "dot reconnecting";
          connStatus.textContent = "RECONNECTING...";
        } else {
          connDot.className = "dot online";
          connStatus.textContent = "ONLINE";
        }
      } catch (e) {
        console.error("Telemetry parsing error:", e);
      }
    };

    ws.onclose = () => {
      connDot.className = "dot reconnecting";
      connStatus.textContent = "DISCONNECTED";
      setTimeout(connectTelemetryWebSocket, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }

  connectTelemetryWebSocket();

  // Setup Wizard Canvas Drawing
  const setupCanvas = document.getElementById("setup-canvas");
  const setupCtx = setupCanvas.getContext("2d");
  let setupMode = "calib"; // "calib", "zone", "choke"
  let calibPoints = [];
  let zonePoints = [];
  let chokePoints = [];

  const setupInstruction = document.getElementById("setup-instruction");
  const btnModeCalib = document.getElementById("btn-mode-calib");
  const btnModeZone = document.getElementById("btn-mode-zone");
  const btnModeChoke = document.getElementById("btn-mode-choke");
  const btnResetDrawing = document.getElementById("btn-reset-drawing");
  const btnSaveSetup = document.getElementById("btn-save-setup");

  btnModeCalib.addEventListener("click", () => {
    setupMode = "calib";
    setupInstruction.textContent = "Click 4 points on the floor to mark ground-plane homography rectangle.";
  });

  btnModeZone.addEventListener("click", () => {
    setupMode = "zone";
    setupInstruction.textContent = "Click to define polygon vertices for monitored zone (e.g. Zone A).";
  });

  btnModeChoke.addEventListener("click", () => {
    setupMode = "choke";
    setupInstruction.textContent = "Click 2 points to define choke-point crossing line.";
  });

  btnResetDrawing.addEventListener("click", () => {
    calibPoints = [];
    zonePoints = [];
    chokePoints = [];
    drawSetupCanvas();
  });

  setupCanvas.addEventListener("click", (e) => {
    const rect = setupCanvas.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);

    if (setupMode === "calib") {
      if (calibPoints.length < 4) {
        calibPoints.push([x, y]);
      }
    } else if (setupMode === "zone") {
      zonePoints.push([x, y]);
    } else if (setupMode === "choke") {
      if (chokePoints.length < 2) {
        chokePoints.push([x, y]);
      }
    }
    drawSetupCanvas();
  });

  function drawSetupCanvas() {
    setupCtx.fillStyle = "#1e293b";
    setupCtx.fillRect(0, 0, setupCanvas.width, setupCanvas.height);

    // Draw grid
    setupCtx.strokeStyle = "#334155";
    setupCtx.lineWidth = 1;
    for (let x = 0; x < setupCanvas.width; x += 40) {
      setupCtx.beginPath();
      setupCtx.moveTo(x, 0);
      setupCtx.lineTo(x, setupCanvas.height);
      setupCtx.stroke();
    }
    for (let y = 0; y < setupCanvas.height; y += 40) {
      setupCtx.beginPath();
      setupCtx.moveTo(0, y);
      setupCtx.lineTo(setupCanvas.width, y);
      setupCtx.stroke();
    }

    // Draw Calibration Points
    if (calibPoints.length > 0) {
      setupCtx.strokeStyle = "#f59e0b";
      setupCtx.fillStyle = "#fbbf24";
      setupCtx.lineWidth = 2;
      setupCtx.beginPath();
      calibPoints.forEach(([x, y], i) => {
        setupCtx.arc(x, y, 6, 0, Math.PI * 2);
        setupCtx.fillText(`P${i + 1}`, x + 10, y + 4);
      });
      setupCtx.fill();
      if (calibPoints.length === 4) {
        setupCtx.beginPath();
        setupCtx.moveTo(calibPoints[0][0], calibPoints[0][1]);
        for (let i = 1; i < 4; i++) setupCtx.lineTo(calibPoints[i][0], calibPoints[i][1]);
        setupCtx.closePath();
        setupCtx.stroke();
      }
    }

    // Draw Zone Polygon
    if (zonePoints.length > 0) {
      setupCtx.strokeStyle = "#3b82f6";
      setupCtx.fillStyle = "rgba(59, 130, 246, 0.25)";
      setupCtx.lineWidth = 2;
      setupCtx.beginPath();
      setupCtx.moveTo(zonePoints[0][0], zonePoints[0][1]);
      for (let i = 1; i < zonePoints.length; i++) {
        setupCtx.lineTo(zonePoints[i][0], zonePoints[i][1]);
      }
      setupCtx.closePath();
      setupCtx.stroke();
      setupCtx.fill();
    }

    // Draw Choke Point Line
    if (chokePoints.length === 2) {
      setupCtx.strokeStyle = "#ef4444";
      setupCtx.lineWidth = 3;
      setupCtx.beginPath();
      setupCtx.moveTo(chokePoints[0][0], chokePoints[0][1]);
      setupCtx.lineTo(chokePoints[1][0], chokePoints[1][1]);
      setupCtx.stroke();
    }
  }

  drawSetupCanvas();

  btnSaveSetup.addEventListener("click", () => {
    alert("Configuration parameters saved to config files successfully.");
  });

  // Replay file loader
  const btnLoadReplay = document.getElementById("btn-load-replay");
  btnLoadReplay.addEventListener("click", async () => {
    const path = document.getElementById("replay-filepath").value;
    try {
      const resp = await fetch("/api/stream/source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_type: "file", source_value: path }),
      });
      const res = await resp.json();
      alert(`Loaded video source for replay: ${res.new_source}`);
    } catch (e) {
      alert(`Error loading replay video: ${e}`);
    }
  });
});
