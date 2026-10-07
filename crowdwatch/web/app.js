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
  const headerAngleBadge = document.getElementById("header-angle-badge");
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
          const roundedTotal = Math.round(data.total_count);
          const groundCount = data.calibrated_ground_count !== undefined ? Math.round(data.calibrated_ground_count) : roundedTotal;
          countBadge.textContent = data.calibrated ? `${groundCount}` : `${roundedTotal}`;
          metricHeadcount.textContent = data.calibrated ? `${groundCount}` : `${roundedTotal}`;
          heatmapCountPill.textContent = data.calibrated ? `${groundCount} in zone` : `${roundedTotal} ppl`;
        }

        // Density display
        const metricSubDensity = document.getElementById("metric-sub-density");
        if (metricSubDensity && data.density_m2 !== undefined) {
          if (data.calibrated) {
            metricSubDensity.textContent = `${data.density_m2} ppl/m² in ${data.floor_area_m2 || 0}m² zone (${data.total_count || 0} total in frame)`;
          } else {
            metricSubDensity.textContent = `Uncalibrated (${data.red_hotspot_pct || 0}% hotspots)`;
          }
        }

        // Camera viewing angle telemetry update
        if (data.camera_angle && headerAngleBadge) {
          headerAngleBadge.textContent = data.camera_angle.label || `${data.camera_angle.view_type} (${data.camera_angle.pitch_deg}°)`;
          if (data.camera_angle.view_type === "TOP_DOWN") {
            headerAngleBadge.style.color = "#38bdf8";
            headerAngleBadge.style.borderColor = "rgba(56, 189, 248, 0.4)";
          } else if (data.camera_angle.view_type === "HIGH_OBLIQUE") {
            headerAngleBadge.style.color = "#a78bfa";
            headerAngleBadge.style.borderColor = "rgba(167, 139, 250, 0.4)";
          } else {
            headerAngleBadge.style.color = "#fbbf24";
            headerAngleBadge.style.borderColor = "rgba(251, 191, 36, 0.4)";
          }
        }

        // Risk Level styling & Master Stampede Early-Warning Banner
        if (data.risk_level) {
          const level = data.risk_level.toLowerCase();
          headerRiskBadge.className = `risk-badge badge-${level}`;
          headerRiskBadge.textContent = data.risk_level;

          metricRiskLabel.className = `metric-number color-${level}`;
          metricRiskLabel.textContent = data.risk_level;

          masterRiskCard.className = `master-alert-card banner-${level}`;
          masterRiskStatus.textContent = `STATUS: ${data.risk_title || ("SYSTEM " + data.risk_level)}`;
          if (data.risk_action) {
            masterRiskAction.textContent = data.risk_action;
          }

          const alertIconEmoji = document.getElementById("alert-icon-emoji");
          if (alertIconEmoji) {
            alertIconEmoji.textContent = data.risk_icon || (level === "normal" ? "🛡️" : level === "watch" ? "👀" : level === "warning" ? "⚠️" : "🚨");
          }

          const dwellText = masterRiskCard ? masterRiskCard.querySelector(".dwell-text") : null;
          if (dwellText) {
            dwellText.textContent = `Hotspots: ${data.red_hotspot_pct || 0}% | Mid-Zone: ${data.yellow_zone_pct || 0}% | Free: ${data.blue_free_pct || 0}%`;
          }

          const advisoryTag = masterRiskCard ? masterRiskCard.querySelector(".advisory-tag") : null;
          if (advisoryTag && data.risk_advisory) {
            advisoryTag.textContent = data.risk_advisory;
          }
        }

        // Monitored Safety Zones Live Update
        const badgeZoneA = document.getElementById("badge-zone-a");
        const fillZoneA = document.getElementById("fill-zone-a");
        const badgeZoneB = document.getElementById("badge-zone-b");
        const fillZoneB = document.getElementById("fill-zone-b");

        if (badgeZoneA && data.zone_a_density !== undefined) {
          badgeZoneA.textContent = `${data.zone_a_density} ppl/m²`;
          const zLevelA = data.zone_a_density < 1.8 ? "normal" : data.zone_a_density < 3.0 ? "watch" : data.zone_a_density < 4.5 ? "warning" : "critical";
          badgeZoneA.className = `badge badge-${zLevelA}`;
          if (fillZoneA) {
            fillZoneA.style.width = `${Math.min(100, Math.round((data.zone_a_density / 4.0) * 100))}%`;
            fillZoneA.className = `zone-progress-fill ${zLevelA === "warning" || zLevelA === "critical" ? "warning" : ""}`;
          }
        }

        if (badgeZoneB && data.zone_b_density !== undefined) {
          badgeZoneB.textContent = `${data.zone_b_density} ppl/m²`;
          const zLevelB = data.zone_b_density < 1.8 ? "normal" : data.zone_b_density < 3.0 ? "watch" : data.zone_b_density < 4.5 ? "warning" : "critical";
          badgeZoneB.className = `badge badge-${zLevelB}`;
          if (fillZoneB) {
            fillZoneB.style.width = `${Math.min(100, Math.round((data.zone_b_density / 4.0) * 100))}%`;
            fillZoneB.className = `zone-progress-fill ${zLevelB === "warning" || zLevelB === "critical" ? "warning" : ""}`;
          }
        }

        if (data.source_name && currentSourceDisplay) {
          currentSourceDisplay.textContent = data.source_name;
        }

        // Camera drift alert handling
        const driftBanner = document.getElementById("drift-alert-banner");
        if (driftBanner) {
          if (data.drift_detected) {
            driftBanner.classList.remove("hidden");
            const driftRatioPct = Math.round((data.drift_ratio || 0) * 100);
            const msgSpan = driftBanner.querySelector("span:nth-child(2)");
            if (msgSpan) {
              msgSpan.textContent = `CAMERA DRIFT WARNING: Camera movement detected (match ${driftRatioPct}%)! Ground plane calibration may be invalid. Please freeze a new frame and re-calibrate.`;
            }
          } else {
            driftBanner.classList.add("hidden");
          }
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

  // Refresh feeds helper (bust cache, clear previous frame, and reconnect cleanly)
  function refreshStreamViews() {
    const timestamp = Date.now();
    streamRawImg.src = "";
    streamHeatmapImg.src = "";
    streamCompositeImg.src = "";
    setTimeout(() => {
      streamRawImg.src = `/api/stream/raw?t=${timestamp}`;
      streamHeatmapImg.src = `/api/stream/heatmap?t=${timestamp}`;
      streamCompositeImg.src = `/api/stream/sidebyside?t=${timestamp}`;
    }, 250);
  }

  // Refresh feeds button (bust cache / force reconnect)
  const btnReloadStream = document.getElementById("btn-reload-stream");
  btnReloadStream.addEventListener("click", () => {
    refreshStreamViews();
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
        refreshStreamViews();
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
        refreshStreamViews();

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
  // 6. SETUP INTERACTIVE GROUND CALIBRATION WIZARD (Phase 1)
  // --------------------------------------------------------------------------
  const setupCanvas = document.getElementById("setup-canvas");
  if (setupCanvas) {
    const setupCtx = setupCanvas.getContext("2d");
    let setupMode = "calib";
    let calibPoints = [];
    let zonePoints = [];
    let chokePoints = [];
    let bgImage = null;
    let previewGridLines = [];
    let showGrid = true;

    const setupInstruction = document.getElementById("setup-instruction");
    const setupCanvasLoading = document.getElementById("setup-canvas-loading");
    const calibStatusText = document.getElementById("calib-status-text");
    const calibAngleText = document.getElementById("calib-angle-text");
    const calibFloorArea = document.getElementById("calib-floor-area");
    const calibReprojError = document.getElementById("calib-reproj-error");
    const calibWidthInput = document.getElementById("calib-width-m");
    const calibHeightInput = document.getElementById("calib-height-m");
    const calibSaveFeedback = document.getElementById("calib-save-feedback");

    const btnFreezeFrame = document.getElementById("btn-freeze-frame");
    const btnAutoAngle = document.getElementById("btn-auto-angle");
    const btnAutoAngleSetup = document.getElementById("btn-auto-angle-setup");
    const btnModeCalib = document.getElementById("btn-mode-calib");
    const btnModeZone = document.getElementById("btn-mode-zone");
    const btnModeChoke = document.getElementById("btn-mode-choke");
    const btnToggleGrid = document.getElementById("btn-toggle-grid");
    const btnResetDrawing = document.getElementById("btn-reset-drawing");
    const btnCalcCalib = document.getElementById("btn-calc-calib");
    const btnSaveSetup = document.getElementById("btn-save-setup");
    const btnDriftRecalib = document.getElementById("btn-drift-recalib");

    // Fetch and freeze currently active video frame
    async function freezeVideoFrame() {
      if (setupCanvasLoading) setupCanvasLoading.classList.remove("hidden");
      try {
        const timestamp = Date.now();
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
          bgImage = img;
          if (setupCanvasLoading) setupCanvasLoading.classList.add("hidden");
          drawSetupCanvas();
          appendAuditLog("Calibration frame captured from video stream.", "info");
        };
        img.onerror = () => {
          if (setupCanvasLoading) setupCanvasLoading.classList.add("hidden");
          appendAuditLog("Could not capture video frame for calibration.", "error");
        };
        img.src = `/api/calibration/frame?t=${timestamp}`;
      } catch (err) {
        if (setupCanvasLoading) setupCanvasLoading.classList.add("hidden");
        console.error("Frame freeze error:", err);
      }
    }

    // Load active calibration status from backend
    async function loadCalibrationStatus() {
      try {
        const resp = await fetch("/api/calibration/status");
        const status = await resp.json();
        if (status.calibrated) {
          if (calibStatusText) {
            calibStatusText.textContent = "CALIBRATED (Active)";
            calibStatusText.style.color = "#10b981";
          }
          if (status.camera_angle && calibAngleText) {
            calibAngleText.textContent = `${status.camera_angle.label} (${status.camera_angle.pitch_deg}°, ${status.camera_angle.scale_gradient}x)`;
          }
          if (calibFloorArea) calibFloorArea.textContent = `${status.floor_area_m2 || 0} m²`;
          if (calibReprojError) calibReprojError.textContent = `${status.reprojection_error_px || 0} px`;
          if (calibWidthInput && status.real_width_m) calibWidthInput.value = status.real_width_m;
          if (calibHeightInput && status.real_height_m) calibHeightInput.value = status.real_height_m;

          // Fetch preview grid
          loadPreviewGrid();
        } else {
          if (calibStatusText) {
            calibStatusText.textContent = "UNCALIBRATED";
            calibStatusText.style.color = "#f59e0b";
          }
          if (calibFloorArea) calibFloorArea.textContent = "0.0 m²";
          if (calibReprojError) calibReprojError.textContent = "--";
        }
      } catch (err) {
        console.error("Failed to load calibration status:", err);
      }
    }

    // Load projected 1m x 1m grid from backend
    async function loadPreviewGrid() {
      try {
        const resp = await fetch("/api/calibration/grid");
        const data = await resp.json();
        if (data.lines) {
          previewGridLines = data.lines;
          drawSetupCanvas();
        }
      } catch (err) {
        console.error("Failed to load preview grid:", err);
      }
    }

    // Calculate Homography via API
    async function calculateHomography() {
      if (calibPoints.length !== 4) {
        alert("Please mark exactly 4 ground-plane points on the frozen image first (in clockwise order).");
        return;
      }
      const realWidth = parseFloat(calibWidthInput.value) || 6.0;
      const realHeight = parseFloat(calibHeightInput.value) || 10.0;

      if (realWidth <= 0 || realHeight <= 0) {
        alert("Real-world ground width and length must be positive meters.");
        return;
      }

      try {
        appendAuditLog(`Calculating ground homography for ${realWidth}m × ${realHeight}m...`);
        const resp = await fetch("/api/calibration/points", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            image_points: calibPoints,
            real_width_m: realWidth,
            real_height_m: realHeight,
            name: "Operator Ground Calibration",
          }),
        });

        const res = await resp.json();
        if (resp.ok && res.status === "ok") {
          appendAuditLog(`Homography computed: Area ${res.area_m2}m², Reproj Error ${res.reprojection_error_px}px`, "success");
          if (calibStatusText) {
            calibStatusText.textContent = "COMPUTED (Unsaved)";
            calibStatusText.style.color = "#38bdf8";
          }
          if (calibFloorArea) calibFloorArea.textContent = `${res.area_m2} m²`;
          if (calibReprojError) calibReprojError.textContent = `${res.reprojection_error_px} px`;

          showGrid = true;
          await loadPreviewGrid();
          alert(`Ground calibration computed successfully!\nReal Area: ${res.area_m2} m²\nReprojection Error: ${res.reprojection_error_px} px\n\nVerify that the cyan 1m grid lines align with the floor perspective, then click 'Save Calibration to System'.`);
        } else {
          alert(`Calibration error: ${res.detail || "Invalid input coordinates"}`);
        }
      } catch (err) {
        appendAuditLog(`Calibration request failed: ${err}`, "error");
      }
    }

    // Save calibration to disk
    async function saveCalibrationToDisk() {
      try {
        const resp = await fetch("/api/calibration/save", { method: "POST" });
        const res = await resp.json();
        if (resp.ok && res.status === "ok") {
          if (calibStatusText) {
            calibStatusText.textContent = "CALIBRATED & SAVED";
            calibStatusText.style.color = "#10b981";
          }
          if (calibSaveFeedback) {
            calibSaveFeedback.style.display = "block";
            setTimeout(() => { calibSaveFeedback.style.display = "none"; }, 3500);
          }
          appendAuditLog("Ground calibration persisted to config/calibration.json.", "success");
          // Clear any camera drift alert if re-calibrated
          const driftBanner = document.getElementById("drift-alert-banner");
          if (driftBanner) driftBanner.classList.add("hidden");
        } else {
          alert(`Save failed: ${res.detail || "No valid calibration available"}`);
        }
      } catch (err) {
        appendAuditLog(`Failed to save calibration: ${err}`, "error");
      }
    }

    // Mode Toolbar Buttons
    if (btnFreezeFrame) btnFreezeFrame.addEventListener("click", freezeVideoFrame);
    if (btnCalcCalib) btnCalcCalib.addEventListener("click", calculateHomography);
    if (btnSaveSetup) btnSaveSetup.addEventListener("click", saveCalibrationToDisk);
    if (btnAutoAngle) btnAutoAngle.addEventListener("click", autoDetectAngleCalibration);
    if (btnAutoAngleSetup) btnAutoAngleSetup.addEventListener("click", autoDetectAngleCalibration);

    // Auto-calibrate via camera angle estimation
    async function autoDetectAngleCalibration() {
      try {
        appendAuditLog("Auto-detecting camera viewing angle and computing perspective calibration...");
        const resp = await fetch("/api/calibration/auto_angle", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ force: true }),
        });
        const res = await resp.json();
        if (resp.ok && res.status === "ok") {
          const angle = res.angle || res;
          const calib = res.calibration || {};
          const area = calib.floor_area_m2 || res.area_m2 || 0;
          const reproj = calib.reprojection_error_px || res.reprojection_error_px || 0;
          appendAuditLog(`Angle auto-configured: ${angle.view_type} (${angle.pitch_deg}° pitch, scale ${angle.scale_gradient}x, ${area}m²)`, "success");
          if (calibStatusText) {
            calibStatusText.textContent = "CALIBRATED (Auto-Angle)";
            calibStatusText.style.color = "#10b981";
          }
          if (calibAngleText) {
            calibAngleText.textContent = `${angle.view_type} (${angle.pitch_deg}°, ${angle.scale_gradient}x)`;
          }
          if (calibFloorArea) calibFloorArea.textContent = `${area} m²`;
          if (calibReprojError) calibReprojError.textContent = `${reproj} px`;
          if (calibWidthInput && calib.real_width_m) calibWidthInput.value = calib.real_width_m;
          if (calibHeightInput && calib.real_height_m) calibHeightInput.value = calib.real_height_m;

          showGrid = true;
          await loadPreviewGrid();
          await freezeVideoFrame();
          alert(`Camera Angle Auto-Configured Successfully!\n\nView Type: ${angle.view_type}\nEstimated Pitch: ${angle.pitch_deg}°\nScale Gradient: ${angle.scale_gradient}x\nGround Area: ${area} m²\n\nPerspective trapezoid and density kernel have been optimized for this camera angle.`);
        } else {
          alert(`Auto-detection failed: ${res.detail || "Could not analyze frame"}`);
        }
      } catch (err) {
        appendAuditLog(`Auto-angle calibration failed: ${err}`, "error");
      }
    }

    if (btnToggleGrid) {
      btnToggleGrid.addEventListener("click", () => {
        showGrid = !showGrid;
        btnToggleGrid.classList.toggle("active", showGrid);
        drawSetupCanvas();
      });
    }

    if (btnModeCalib) {
      btnModeCalib.addEventListener("click", () => {
        setupMode = "calib";
        [btnModeCalib, btnModeZone, btnModeChoke].forEach((b) => b && b.classList.remove("active"));
        btnModeCalib.classList.add("active");
        setupInstruction.textContent = "Click 4 points on the floor in clockwise order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.";
      });
    }

    if (btnModeZone) {
      btnModeZone.addEventListener("click", () => {
        setupMode = "zone";
        [btnModeCalib, btnModeZone, btnModeChoke].forEach((b) => b && b.classList.remove("active"));
        btnModeZone.classList.add("active");
        setupInstruction.textContent = "Click to define polygon vertices for monitored zone (e.g. Zone A). Double-click or reset to clear.";
      });
    }

    if (btnModeChoke) {
      btnModeChoke.addEventListener("click", () => {
        setupMode = "choke";
        [btnModeCalib, btnModeZone, btnModeChoke].forEach((b) => b && b.classList.remove("active"));
        btnModeChoke.classList.add("active");
        setupInstruction.textContent = "Click 2 points across a corridor or entrance to define a choke-point gate.";
      });
    }

    if (btnResetDrawing) {
      btnResetDrawing.addEventListener("click", () => {
        if (setupMode === "calib") {
          calibPoints = [];
          previewGridLines = [];
        } else if (setupMode === "zone") {
          zonePoints = [];
        } else if (setupMode === "choke") {
          chokePoints = [];
        }
        drawSetupCanvas();
      });
    }

    // Canvas click handling
    setupCanvas.addEventListener("click", (e) => {
      const rect = setupCanvas.getBoundingClientRect();
      const scaleX = setupCanvas.width / rect.width;
      const scaleY = setupCanvas.height / rect.height;
      const x = Math.round((e.clientX - rect.left) * scaleX);
      const y = Math.round((e.clientY - rect.top) * scaleY);

      if (setupMode === "calib") {
        if (calibPoints.length < 4) {
          calibPoints.push([x, y]);
          if (calibPoints.length === 4) {
            setupInstruction.textContent = "4 points marked! Click '⚡ Calculate Homography' to project scale.";
          }
        }
      } else if (setupMode === "zone") {
        zonePoints.push([x, y]);
      } else if (setupMode === "choke") {
        if (chokePoints.length < 2) chokePoints.push([x, y]);
      }
      drawSetupCanvas();
    });

    // Draw canvas scene (video frame + 1m grid + points + polygons)
    function drawSetupCanvas() {
      setupCtx.clearRect(0, 0, setupCanvas.width, setupCanvas.height);

      if (bgImage) {
        setupCtx.drawImage(bgImage, 0, 0, setupCanvas.width, setupCanvas.height);
      } else {
        // Dark placeholder background
        setupCtx.fillStyle = "#0c1017";
        setupCtx.fillRect(0, 0, setupCanvas.width, setupCanvas.height);

        // Faint placeholder grid
        setupCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
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
      }

      // Draw 1m x 1m Ground Perspective Grid
      if (showGrid && previewGridLines.length > 0) {
        setupCtx.strokeStyle = "rgba(56, 189, 248, 0.75)";
        setupCtx.lineWidth = 1.5;
        previewGridLines.forEach((seg) => {
          if (seg.start && seg.end) {
            setupCtx.beginPath();
            setupCtx.moveTo(seg.start[0], seg.start[1]);
            setupCtx.lineTo(seg.end[0], seg.end[1]);
            setupCtx.stroke();
          }
        });
      }

      // Draw Calib Points (Floor Homography Quad)
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
          setupCtx.font = "bold 12px Inter, monospace";
          setupCtx.fillText(`P${i + 1}`, x + 10, y + 4);
        });

        if (calibPoints.length === 4) {
          setupCtx.strokeStyle = "#38bdf8";
          setupCtx.fillStyle = "rgba(56, 189, 248, 0.18)";
          setupCtx.beginPath();
          setupCtx.moveTo(calibPoints[0][0], calibPoints[0][1]);
          for (let i = 1; i < 4; i++) setupCtx.lineTo(calibPoints[i][0], calibPoints[i][1]);
          setupCtx.closePath();
          setupCtx.stroke();
          setupCtx.fill();
        }
      }

      // Draw Zone Polygons
      if (zonePoints.length > 0) {
        setupCtx.strokeStyle = "#10b981";
        setupCtx.fillStyle = "rgba(16, 185, 129, 0.22)";
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

    // Auto-load status & frame when calibration tab is opened
    const tabBtnSetup = document.getElementById("tab-btn-setup");
    if (tabBtnSetup) {
      tabBtnSetup.addEventListener("click", () => {
        loadCalibrationStatus();
        if (!bgImage) freezeVideoFrame();
      });
    }

    if (btnDriftRecalib) {
      btnDriftRecalib.addEventListener("click", () => {
        if (tabBtnSetup) tabBtnSetup.click();
        freezeVideoFrame();
      });
    }

    // Initial check on load
    loadCalibrationStatus();
  }
});
