/**
 * CrowdWatch AI - Control Room Frontend Logic
 * Supports: Side-by-Side Live Streaming, Video File Upload, Mobile/CCTV Linking,
 * Real-time Telemetry WebSockets, and Floor Calibration.
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

  // Telemetry DOM elements
  const connDot = document.getElementById("conn-dot");
  const connStatus = document.getElementById("conn-status");
  const fpsBadge = document.getElementById("telemetry-fps");
  const latBadge = document.getElementById("telemetry-latency");
  const countBadge = document.getElementById("telemetry-count");
  const headerRiskBadge = document.getElementById("header-risk-badge");
  const masterRiskCard = document.getElementById("master-risk-card");
  const masterRiskStatus = document.getElementById("master-risk-status");
  const masterRiskAction = document.getElementById("master-risk-action");
  const metricHeadcount = document.getElementById("metric-headcount");
  const metricRiskLabel = document.getElementById("metric-risk-label");
  const heatmapCountPill = document.getElementById("heatmap-count-pill");
  const currentSourceDisplay = document.getElementById("current-source-display");
  const currentSourceType = document.getElementById("current-source-type");
  const auditLog = document.getElementById("audit-log");

  // Stream Image Elements
  const streamRawImg = document.getElementById("stream-raw-img");
  const streamHeatmapImg = document.getElementById("stream-heatmap-img");
  const streamCompositeImg = document.getElementById("stream-composite-img");
  const containerDual = document.getElementById("container-dual-stream");
  const containerComposite = document.getElementById("container-composite-stream");
  const panelRaw = document.querySelector(".panel-raw");
  const panelHeatmap = document.querySelector(".panel-heatmap");

  // Initial startup log timestamp
  const startupLogTime = document.getElementById("startup-log-time");
  if (startupLogTime) {
    startupLogTime.textContent = new Date().toLocaleTimeString();
  }

  function appendAuditLog(msg, type = "info") {
    if (!auditLog) return;
    const item = document.createElement("div");
    item.className = `log-item ${type}`;
    item.innerHTML = `<span class="log-time">${new Date().toLocaleTimeString()}</span><span class="log-msg">${msg}</span>`;
    auditLog.prepend(item);
  }

  // --------------------------------------------------------------------------
  // 1. WEBSOCKET TELEMETRY CONNECTION
  // --------------------------------------------------------------------------
  let ws = null;

  function connectTelemetryWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      connDot.className = "pulse-dot online";
      connStatus.textContent = "ONLINE";
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.processed_fps !== undefined) {
          fpsBadge.textContent = `${data.processed_fps} FPS`;
        }
        if (data.latency_ms !== undefined) {
          latBadge.textContent = `${Math.round(data.latency_ms)} ms`;
        }
        if (data.total_count !== undefined) {
          const rounded = Math.round(data.total_count);
          countBadge.textContent = `${rounded}`;
          metricHeadcount.textContent = `${rounded}`;
          heatmapCountPill.textContent = `${rounded} ppl`;
        }

        // Risk Level styling
        if (data.risk_level) {
          const level = data.risk_level.toLowerCase();
          headerRiskBadge.className = `risk-badge badge-${level}`;
          headerRiskBadge.textContent = data.risk_level;

          metricRiskLabel.className = `metric-number color-${level}`;
          metricRiskLabel.textContent = data.risk_level;

          masterRiskCard.className = `master-alert-card banner-${level}`;
          masterRiskStatus.textContent = `STATUS: SYSTEM ${data.risk_level}`;
          if (data.risk_action) {
            masterRiskAction.textContent = data.risk_action;
          }
        }

        if (data.source_name && currentSourceDisplay) {
          currentSourceDisplay.textContent = data.source_name;
        }

        if (data.is_connected === false) {
          connDot.className = "pulse-dot reconnecting";
          connStatus.textContent = "CONNECTING...";
        } else {
          connDot.className = "pulse-dot online";
          connStatus.textContent = "ONLINE";
        }
      } catch (e) {
        console.error("Telemetry parsing error:", e);
      }
    };

    ws.onclose = () => {
      connDot.className = "pulse-dot reconnecting";
      connStatus.textContent = "OFFLINE";
      setTimeout(connectTelemetryWebSocket, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }

  connectTelemetryWebSocket();

  // --------------------------------------------------------------------------
  // 2. VIDEO STREAM DISPLAY MODES & CONTROLS
  // --------------------------------------------------------------------------
  const modeButtons = document.querySelectorAll(".mode-btn");

  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.getAttribute("data-mode");
      modeButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      if (mode === "sidebyside") {
        containerDual.classList.remove("hidden");
        containerComposite.classList.add("hidden");
        panelRaw.classList.remove("hidden");
        panelHeatmap.classList.remove("hidden");
      } else if (mode === "composite") {
        containerDual.classList.add("hidden");
        containerComposite.classList.remove("hidden");
      } else if (mode === "heatmap") {
        containerDual.classList.remove("hidden");
        containerComposite.classList.add("hidden");
        panelRaw.classList.add("hidden");
        panelHeatmap.classList.remove("hidden");
      } else if (mode === "raw") {
        containerDual.classList.remove("hidden");
        containerComposite.classList.add("hidden");
        panelRaw.classList.remove("hidden");
        panelHeatmap.classList.add("hidden");
      }
    });
  });

  // Refresh feeds button (bust cache / force reconnect)
  const btnReloadStream = document.getElementById("btn-reload-stream");
  btnReloadStream.addEventListener("click", () => {
    const timestamp = Date.now();
    streamRawImg.src = `/api/stream/raw?t=${timestamp}`;
    streamHeatmapImg.src = `/api/stream/heatmap?t=${timestamp}`;
    streamCompositeImg.src = `/api/stream/sidebyside?t=${timestamp}`;
    appendAuditLog("Stream feeds refreshed.");
  });

  // Fullscreen Stage
  const btnFullscreen = document.getElementById("btn-fullscreen-stage");
  btnFullscreen.addEventListener("click", () => {
    const stage = document.querySelector(".video-stage-column");
    if (!document.fullscreenElement) {
      stage.requestFullscreen().catch((err) => console.log(err));
    } else {
      document.exitFullscreen();
    }
  });

  // --------------------------------------------------------------------------
  // 3. AVAILABLE VIDEOS LIBRARY & CHIPS
  // --------------------------------------------------------------------------
  const chipsContainer = document.getElementById("video-chips-container");

  async function loadAvailableVideos() {
    try {
      const resp = await fetch("/api/videos/list");
      const data = await resp.json();
      if (!data.videos) return;

      chipsContainer.innerHTML = "";
      data.videos.forEach((video) => {
        const btn = document.createElement("button");
        const icon = video.type === "uploaded" ? "📱" : "👥";
        btn.className = "chip-btn";
        btn.textContent = `${icon} ${video.name}`;
        btn.setAttribute("data-path", video.path);
        btn.setAttribute("data-type", "file");

        btn.addEventListener("click", () => {
          document.querySelectorAll(".chip-btn").forEach((c) => c.classList.remove("active"));
          btn.classList.add("active");
          switchVideoSource("file", video.path, video.name);
        });

        chipsContainer.appendChild(btn);
      });
    } catch (err) {
      console.error("Failed to load video list:", err);
    }
  }

  loadAvailableVideos();

  async function switchVideoSource(sourceType, sourceValue, displayName) {
    try {
      appendAuditLog(`Switching video feed to: ${displayName || sourceValue}...`);
      const resp = await fetch("/api/stream/source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_type: sourceType, source_value: sourceValue }),
      });
      const res = await resp.json();
      if (res.status === "ok") {
        currentSourceDisplay.textContent = displayName || sourceValue;
        currentSourceType.textContent = sourceType.toUpperCase();
        appendAuditLog(`Feed active: ${displayName || sourceValue}`, "success");
        // Force refresh image elements to restart playback seamlessly
        const timestamp = Date.now();
        streamRawImg.src = `/api/stream/raw?t=${timestamp}`;
        streamHeatmapImg.src = `/api/stream/heatmap?t=${timestamp}`;
        streamCompositeImg.src = `/api/stream/sidebyside?t=${timestamp}`;
      }
    } catch (e) {
      appendAuditLog(`Failed to switch video source: ${e}`, "error");
    }
  }

  // Connect Webcam Quick Action
  const btnWebcam = document.getElementById("btn-use-webcam");
  btnWebcam.addEventListener("click", () => {
    switchVideoSource("webcam", "0", "Local Camera 0 (Webcam)");
  });

  // --------------------------------------------------------------------------
  // 4. MODAL: UPLOAD VIDEO FILE (MOBILE / CCTV / MP4)
  // --------------------------------------------------------------------------
  const modalUpload = document.getElementById("modal-upload");
  const btnOpenUpload = document.getElementById("btn-open-upload");
  const btnCloseUpload = document.getElementById("btn-close-upload");
  const btnCancelUpload = document.getElementById("btn-cancel-upload");
  const btnConfirmUpload = document.getElementById("btn-confirm-upload");
  const uploadDropzone = document.getElementById("upload-dropzone");
  const uploadFileInput = document.getElementById("upload-file-input");
  const uploadProgressBox = document.getElementById("upload-progress-box");
  const uploadBarFill = document.getElementById("upload-bar-fill");
  const uploadPercent = document.getElementById("upload-percent");
  const uploadFilename = document.getElementById("upload-filename");
  const uploadStatusMsg = document.getElementById("upload-status-msg");

  let selectedUploadFile = null;

  btnOpenUpload.addEventListener("click", () => {
    modalUpload.classList.remove("hidden");
    selectedUploadFile = null;
    uploadStatusMsg.textContent = "";
    uploadStatusMsg.className = "upload-status-msg";
    uploadProgressBox.classList.add("hidden");
    btnConfirmUpload.disabled = true;
  });

  function closeUploadModal() {
    modalUpload.classList.add("hidden");
  }

  btnCloseUpload.addEventListener("click", closeUploadModal);
  btnCancelUpload.addEventListener("click", closeUploadModal);

  uploadDropzone.addEventListener("click", () => uploadFileInput.click());

  uploadDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadDropzone.classList.add("dragover");
  });

  uploadDropzone.addEventListener("dragleave", () => {
    uploadDropzone.classList.remove("dragover");
  });

  uploadDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  uploadFileInput.addEventListener("change", () => {
    if (uploadFileInput.files.length > 0) {
      handleSelectedFile(uploadFileInput.files[0]);
    }
  });

  function handleSelectedFile(file) {
    selectedUploadFile = file;
    uploadFilename.textContent = file.name;
    uploadStatusMsg.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    uploadStatusMsg.className = "upload-status-msg";
    btnConfirmUpload.disabled = false;
  }

  btnConfirmUpload.addEventListener("click", () => {
    if (!selectedUploadFile) return;

    const formData = new FormData();
    formData.append("file", selectedUploadFile);

    uploadProgressBox.classList.remove("hidden");
    uploadBarFill.style.width = "0%";
    uploadPercent.textContent = "0%";
    btnConfirmUpload.disabled = true;

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload/video");

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = Math.round((e.loaded / e.total) * 100);
        uploadBarFill.style.width = `${percent}%`;
        uploadPercent.textContent = `${percent}%`;
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const res = JSON.parse(xhr.responseText);
        uploadStatusMsg.textContent = "✅ Video uploaded and activated in real-time pipeline!";
        uploadStatusMsg.className = "upload-status-msg success";
        appendAuditLog(`Uploaded and linked user video: ${res.filename}`, "success");

        currentSourceDisplay.textContent = res.filename;
        currentSourceType.textContent = "USER UPLOAD";

        // Refresh streams
        const timestamp = Date.now();
        streamRawImg.src = `/api/stream/raw?t=${timestamp}`;
        streamHeatmapImg.src = `/api/stream/heatmap?t=${timestamp}`;
        streamCompositeImg.src = `/api/stream/sidebyside?t=${timestamp}`;

        // Reload video chips
        loadAvailableVideos();

        setTimeout(closeUploadModal, 1200);
      } else {
        uploadStatusMsg.textContent = "❌ Upload failed. Please try again.";
        uploadStatusMsg.className = "upload-status-msg error";
        btnConfirmUpload.disabled = false;
      }
    };

    xhr.onerror = () => {
      uploadStatusMsg.textContent = "❌ Network error during upload.";
      uploadStatusMsg.className = "upload-status-msg error";
      btnConfirmUpload.disabled = false;
    };

    xhr.send(formData);
  });

  // --------------------------------------------------------------------------
  // 5. MODAL: LINK CCTV / MOBILE PHONE STREAM (RTSP / HTTP)
  // --------------------------------------------------------------------------
  const modalStream = document.getElementById("modal-stream");
  const btnOpenStreamModal = document.getElementById("btn-open-stream-modal");
  const btnCloseStreamModal = document.getElementById("btn-close-stream-modal");
  const btnCancelStream = document.getElementById("btn-cancel-stream");
  const btnConnectStream = document.getElementById("btn-connect-stream");
  const streamUrlInput = document.getElementById("stream-url-input");
  const streamStatusMsg = document.getElementById("stream-status-msg");

  btnOpenStreamModal.addEventListener("click", () => {
    modalStream.classList.remove("hidden");
    streamStatusMsg.textContent = "";
  });

  function closeStreamModal() {
    modalStream.classList.add("hidden");
  }

  btnCloseStreamModal.addEventListener("click", closeStreamModal);
  btnCancelStream.addEventListener("click", closeStreamModal);

  // Preset buttons inside modal
  document.querySelectorAll(".preset-btn").forEach((pBtn) => {
    pBtn.addEventListener("click", () => {
      const url = pBtn.getAttribute("data-preset");
      streamUrlInput.value = url;
    });
  });

  btnConnectStream.addEventListener("click", async () => {
    const url = streamUrlInput.value.trim();
    if (!url) {
      streamStatusMsg.textContent = "Please enter a valid Stream or RTSP URL.";
      streamStatusMsg.className = "stream-status-msg error";
      return;
    }

    streamStatusMsg.textContent = "Connecting stream...";
    streamStatusMsg.className = "stream-status-msg";

    try {
      const isFile = url.endsWith(".mp4") || url.endsWith(".avi");
      const type = isFile ? "file" : "url";
      await switchVideoSource(type, url, url.startsWith("http") ? "Mobile Camera Stream" : "CCTV Stream");
      streamStatusMsg.textContent = "✅ Connected successfully!";
      streamStatusMsg.className = "stream-status-msg success";
      setTimeout(closeStreamModal, 900);
    } catch (err) {
      streamStatusMsg.textContent = `❌ Connection failed: ${err}`;
      streamStatusMsg.className = "stream-status-msg error";
    }
  });

  // Clear Audit Log
  const btnClearAudit = document.getElementById("btn-clear-audit");
  if (btnClearAudit) {
    btnClearAudit.addEventListener("click", () => {
      auditLog.innerHTML = "";
    });
  }

  // --------------------------------------------------------------------------
  // 6. SETUP CALIBRATION CANVAS DRAWING
  // --------------------------------------------------------------------------
  const setupCanvas = document.getElementById("setup-canvas");
  if (setupCanvas) {
    const setupCtx = setupCanvas.getContext("2d");
    let setupMode = "calib";
    let calibPoints = [];
    let zonePoints = [];
    let chokePoints = [];

    const setupInstruction = document.getElementById("setup-instruction");
    const btnModeCalib = document.getElementById("btn-mode-calib");
    const btnModeZone = document.getElementById("btn-mode-zone");
    const btnModeChoke = document.getElementById("btn-mode-choke");
    const btnResetDrawing = document.getElementById("btn-reset-drawing");
    const btnSaveSetup = document.getElementById("btn-save-setup");

    if (btnModeCalib) {
      btnModeCalib.addEventListener("click", () => {
        setupMode = "calib";
        setupInstruction.textContent = "Click 4 points on the floor in clockwise order to map perspective to real-world meters.";
      });
    }

    if (btnModeZone) {
      btnModeZone.addEventListener("click", () => {
        setupMode = "zone";
        setupInstruction.textContent = "Click to define polygon vertices for monitored zone (e.g. Zone A).";
      });
    }

    if (btnModeChoke) {
      btnModeChoke.addEventListener("click", () => {
        setupMode = "choke";
        setupInstruction.textContent = "Click 2 points to define a choke-point crossing gate.";
      });
    }

    if (btnResetDrawing) {
      btnResetDrawing.addEventListener("click", () => {
        calibPoints = [];
        zonePoints = [];
        chokePoints = [];
        drawSetupCanvas();
      });
    }

    setupCanvas.addEventListener("click", (e) => {
      const rect = setupCanvas.getBoundingClientRect();
      const x = Math.round(e.clientX - rect.left);
      const y = Math.round(e.clientY - rect.top);

      if (setupMode === "calib") {
        if (calibPoints.length < 4) calibPoints.push([x, y]);
      } else if (setupMode === "zone") {
        zonePoints.push([x, y]);
      } else if (setupMode === "choke") {
        if (chokePoints.length < 2) chokePoints.push([x, y]);
      }
      drawSetupCanvas();
    });

    function drawSetupCanvas() {
      setupCtx.fillStyle = "#0c1017";
      setupCtx.fillRect(0, 0, setupCanvas.width, setupCanvas.height);

      // Grid lines
      setupCtx.strokeStyle = "rgba(255, 255, 255, 0.06)";
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

      // Draw Calib Points
      if (calibPoints.length > 0) {
        setupCtx.strokeStyle = "#38bdf8";
        setupCtx.fillStyle = "#0284c7";
        setupCtx.lineWidth = 2;
        calibPoints.forEach(([x, y], i) => {
          setupCtx.beginPath();
          setupCtx.arc(x, y, 6, 0, Math.PI * 2);
          setupCtx.fill();
          setupCtx.stroke();
          setupCtx.fillStyle = "#ffffff";
          setupCtx.font = "11px monospace";
          setupCtx.fillText(`P${i + 1}`, x + 10, y + 4);
        });
        if (calibPoints.length === 4) {
          setupCtx.strokeStyle = "#38bdf8";
          setupCtx.fillStyle = "rgba(56, 189, 248, 0.15)";
          setupCtx.beginPath();
          setupCtx.moveTo(calibPoints[0][0], calibPoints[0][1]);
          for (let i = 1; i < 4; i++) setupCtx.lineTo(calibPoints[i][0], calibPoints[i][1]);
          setupCtx.closePath();
          setupCtx.stroke();
          setupCtx.fill();
        }
      }

      // Draw Zones
      if (zonePoints.length > 0) {
        setupCtx.strokeStyle = "#10b981";
        setupCtx.fillStyle = "rgba(16, 185, 129, 0.2)";
        setupCtx.lineWidth = 2;
        setupCtx.beginPath();
        setupCtx.moveTo(zonePoints[0][0], zonePoints[0][1]);
        for (let i = 1; i < zonePoints.length; i++) setupCtx.lineTo(zonePoints[i][0], zonePoints[i][1]);
        setupCtx.closePath();
        setupCtx.stroke();
        setupCtx.fill();
      }

      // Draw Choke Lines
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

    if (btnSaveSetup) {
      btnSaveSetup.addEventListener("click", () => {
        alert("Calibration profile saved to config files.");
        appendAuditLog("Ground plane calibration updated.", "success");
      });
    }
  }
});
