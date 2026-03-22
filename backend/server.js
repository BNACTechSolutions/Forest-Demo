// server.js
import express from "express";
import multer from "multer";
import axios from "axios";
import { v2 as cloudinary } from "cloudinary";
import { PassThrough } from "stream";
import dotenv from "dotenv";
import FormData from "form-data";
import cors from "cors";

dotenv.config();

// ====================
// Cloudinary Config
// ====================
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
});

// ====================
// Express App
// ====================
const app = express();
const upload = multer(); // memory buffer only
app.use(cors());

// ====================
// API / Protocol versioning
// ====================
const SUPPORTED_PROTOCOL_VERSIONS = (process.env.SUPPORTED_PROTOCOL_VERSIONS || "1.0")
  .split(",")
  .map((v) => v.trim().replace(/^v/i, ""))
  .filter(Boolean);
const DEFAULT_PROTOCOL_VERSION = String(
  process.env.DEFAULT_PROTOCOL_VERSION || SUPPORTED_PROTOCOL_VERSIONS[0] || "1.0"
)
  .trim()
  .replace(/^v/i, "");
const STRICT_PROTOCOL_VERSION =
  String(process.env.STRICT_PROTOCOL_VERSION || "false").toLowerCase() === "true";
const API_CURRENT_VERSION = "v1";

const normalizeProtocolVersion = (value) => {
  if (value == null) return null;
  const normalized = String(value).trim().replace(/^v/i, "");
  return normalized || null;
};

const resolveProtocolVersion = ({ req, metadataObj }) => {
  const requestedVersion =
    normalizeProtocolVersion(req.get("x-ct-protocol-version")) ||
    normalizeProtocolVersion(req.body?.protocolVersion) ||
    normalizeProtocolVersion(metadataObj?.protocolVersion) ||
    normalizeProtocolVersion(metadataObj?.protocol_version);

  if (!requestedVersion) {
    return {
      requestedVersion: null,
      effectiveVersion: DEFAULT_PROTOCOL_VERSION,
      isSupported: true,
      fallbackReason: "missing",
    };
  }

  const isSupported = SUPPORTED_PROTOCOL_VERSIONS.includes(requestedVersion);
  return {
    requestedVersion,
    effectiveVersion: isSupported ? requestedVersion : DEFAULT_PROTOCOL_VERSION,
    isSupported,
    fallbackReason: isSupported ? null : "unsupported",
  };
};

const attachVersionHeaders = (res, versionInfo) => {
  res.set("x-api-version", API_CURRENT_VERSION);
  res.set("x-ct-protocol-version", versionInfo.effectiveVersion);
  res.set("x-ct-supported-protocol-versions", SUPPORTED_PROTOCOL_VERSIONS.join(","));
};

// ====================
// Aggregation config
// ====================
const AGGREGATION_WINDOW_MS = 60 * 1000; // 1 minute
const MAX_IMAGES_PER_EVENT = 5;
const UPLOAD_ACK_IMMEDIATE = String(process.env.UPLOAD_ACK_IMMEDIATE || "false").toLowerCase() === "true";
const AI_PROCESS_RETRIES = Number(process.env.AI_PROCESS_RETRIES || process.env.UPLOAD_PROCESS_RETRIES || 2);
const UPLOAD_RETRY_DELAY_MS = Number(process.env.UPLOAD_RETRY_DELAY_MS || 5000);
const HUMAN_INTERVENTION_WARNING = 'HUMAN_INTERVENTION_REQUIRED: AI processing failed for this image set. Please review images manually and confirm priority.';
const aggregations = new Map(); // aggKey (trapId:clientId:triggerTime) -> { trapId, images: [], timer }
const inFlightUploads = new Map(); // uploadKey -> { trapId, clientId, captureTime, startedAt, stage }

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ====================
// Correlation ID & Logger
// ====================
const generateCorrelationId = () => {
  const ts = Date.now().toString(36).toUpperCase();
  const rand = Math.random().toString(36).substring(2, 6).toUpperCase();
  return `CT-${ts}-${rand}`;
};

// LOG_LEVEL env var controls verbosity:
//   'debug' — all logs including detailed per-step lines (default in development)
//   'info'  — flow-level logs only: received, found, complete, failed (default in production)
// Set LOG_LEVEL explicitly to override. Falls back to NODE_ENV if LOG_LEVEL is not set.
const _logLevel = process.env.LOG_LEVEL
  ? process.env.LOG_LEVEL.toLowerCase()
  : (process.env.NODE_ENV === 'production' ? 'info' : 'debug');
const _isDev = _logLevel === 'debug';

const createLogger = (correlationId, _deprecated = {}) => {
  const prefix = `[${correlationId}]`;
  const ts = () => new Date().toISOString();
  return {
    // debug — verbose detail lines, only printed in development / LOG_LEVEL=debug
    debug: (msg) => { if (_isDev) console.log(`${ts()} ${prefix} DEBUG ${msg}`); },
    // info — key flow checkpoints, always printed
    info:  (msg) => console.log(`${ts()} ${prefix} INFO  ${msg}`),
    warn:  (msg) => console.warn(`${ts()} ${prefix} WARN  ${msg}`),
    error: (msg) => console.error(`${ts()} ${prefix} ERROR ${msg}`),
  };
};

const lookupTrapClientInfo = async (trapId, { correlationId, log } = {}) => {
  log?.debug(`Looking for trap with id: ${trapId}`);
  const { data: clientInfo } = await axios.post(
    process.env.BFF_CLIENT_LOOKUP_URL,
    { trapId },
    { timeout: 8000, headers: correlationId ? { 'x-correlation-id': correlationId } : {} }
  );
  log?.info(`Trap found — id: ${trapId}, client: ${clientInfo?.clientId || 'unknown'}`);
  return clientInfo;
};

const finalizeAggregation = async (aggKey) => {
  const agg = aggregations.get(aggKey);
  if (!agg) return;
  const trapId = agg.trapId;
  clearTimeout(agg.timer);
  const correlationId = agg.images[0]?.correlationId || generateCorrelationId();
  const log = createLogger(correlationId, { trapId });
  log.info(`Finalising event — ${agg.images.length} image(s) collected for trap ${trapId}`);
  log.debug(`Aggregation key: ${aggKey}`);
  try {
    const images = agg.images;
    if (!images.length) return;

    const clientInfo = images[0].clientInfo || {};

    // Extract timing information (same for all images)
    const firstImage = images[0];
    const triggerTime = firstImage.deviceTimings?.triggerTime || null;
    const pppStartTime = firstImage.deviceTimings?.pppStartTime || null;
    const pppConnectedTime = firstImage.deviceTimings?.pppConnectedTime || null;

    // AI processing times (aggregate across all images)
    const aiStartTime = images[0].processingStartTime || images[0].processingTime;
    const aiEndTime = images[images.length - 1].processingTime;

    // Build species scoring across images
    const speciesStats = {}; // species -> { score, count, examples }

    images.forEach((img, idx) => {
      (img.detectionsRaw || []).forEach((det) => {
        const key = det.species || det.label || 'Unknown';
        const score = Number(det.species_confidence ?? det.detector_confidence ?? 0);
        if (!speciesStats[key]) speciesStats[key] = { score: 0, count: 0, examples: [] };
        speciesStats[key].score += score;
        speciesStats[key].count += 1;
        speciesStats[key].examples.push({ score, det, imageUrl: img.imageUrl, imageIndex: idx });
      });
    });

    const speciesEntries = Object.entries(speciesStats);
    speciesEntries.sort((a, b) => {
      // sort by total score then by count
      if (b[1].score !== a[1].score) return b[1].score - a[1].score;
      return b[1].count - a[1].count;
    });

    const aggregatedDetections = speciesEntries.slice(0, 3).map(([species, s]) => ({
      species,
      aggregatedScore: Number((s.score / s.count || 0).toFixed(4)),
      count: s.count,
    }));

    // Choose best detection example for bbox/mask if available
    const bestSpecies = speciesEntries[0]?.[0] || 'Unknown';
    const bestExample = speciesEntries[0]?.[1]?.examples?.[0] ?? null;
    const bestImageIndex = bestExample?.imageIndex ?? 0;

    const eventStart = images[0].serverReceiptTime || images[0].captureTime || images[0].processingTime;
    const eventEnd = images[images.length - 1].processingTime;

    // Calculate total processing delay from trigger (if valid) to final completion
    let totalProcessingDelaySeconds = null;
    const firstTrigger = triggerTime;
    
    if (firstTrigger && eventEnd) {
      try {
        const diffMs = new Date(eventEnd) - new Date(firstTrigger);
        // Only accept if reasonable (not 1970 causing huge gap)
        if (!isNaN(diffMs) && diffMs >= 0 && diffMs < 3600000) {
          totalProcessingDelaySeconds = Math.round(diffMs / 1000);
        }
      } catch (e) {}
    }
    
    // Fallback: use server receipt time if trigger time was invalid
    if (totalProcessingDelaySeconds === null && eventStart && eventEnd) {
      try {
        const diffMs = new Date(eventEnd) - new Date(eventStart);
        if (!isNaN(diffMs) && diffMs >= 0 && diffMs < 3600000) {
          totalProcessingDelaySeconds = Math.round(diffMs / 1000);
        }
      } catch (e) {}
    }

    const finalPayload = {
      trapId,
      correlationId,
      eventStartTime: eventStart,  // Server receipt time (clean, no device 1970)
      eventEndTime: eventEnd,
      triggerTime,
      pppStartTime,
      pppConnectedTime,
      aiProcessingStartTime: aiStartTime,
      aiProcessingEndTime: aiEndTime,
      processingDelaySeconds: totalProcessingDelaySeconds,
      totalImagesReceived: images.length,
      bestImageIndex,
      // Compatibility fields for older single-image payloads
      captureTime: eventStart,  // Use server receipt time for consistency
      processingTime: eventEnd,
      // All images with their individual capture times
      images: images.map((i, idx) => ({ 
        imageUrl: i.imageUrl, 
        publicId: i.publicId, 
        thumbnailUrl: i.thumbnailUrl,
        captureTime: i.captureTime,
        processingTime: i.processingTime,
        sequence: idx,
        isBest: idx === bestImageIndex
      })),
      clientId: clientInfo.clientId,
      clientName: clientInfo.clientName || null,
      location: clientInfo.location || null,
      project: clientInfo.project || null,
      temperatureValues: images.map((i) => i.temperature).filter((t) => t != null),
      totalImagesReceived: images.length,
      aggregatedDetections,
      bestSpecies: bestSpecies,
      bestExample,
      // compatibility: expose top-level detections-summary similar to single-upload
      totalDetections: aggregatedDetections.length,
      detections: aggregatedDetections.map(d => ({ species: d.species, confidence: d.aggregatedScore, count: d.count })),
      // top image for UI preview
      imageUrl: bestExample?.imageUrl || images[bestImageIndex]?.imageUrl || images[0].imageUrl,
      publicId: images[bestImageIndex]?.publicId || images[0].publicId,
      thumbnailUrl: images[bestImageIndex]?.thumbnailUrl || images[0].thumbnailUrl,
      warnings: images.flatMap((i) => i.warnings || []),
      metadataParseErrors: images.map((i) => i.metadataParseError).filter(Boolean),
      createdAt: new Date().toISOString(),
    };

    if (!process.env.BFF_STORE_URL) {
      log.warn('BFF_STORE_URL is not configured — event will not be stored to backend');
    } else {
      try {
        log.debug(`Sending event to backend — ${images.length} image(s), top species: ${finalPayload.bestSpecies}`);
        await axios.post(process.env.BFF_STORE_URL, finalPayload, {
          timeout: 15000,
          headers: { 'x-correlation-id': correlationId },
        });
        log.info(`Event stored to backend — trap: ${trapId}, images: ${images.length}, top species: ${finalPayload.bestSpecies}`);
      } catch (postErr) {
        log.error(`Backend rejected event — HTTP ${postErr.response?.status || 'unknown'}: ${postErr.message}`);
        log.debug(`Rejected payload: trapId=${finalPayload.trapId}, images=${finalPayload.totalImagesReceived}, detections=${finalPayload.totalDetections}, top species=${finalPayload.bestSpecies}`);
      }
    }
  } catch (err) {
    log.error(`Event finalisation failed: ${err?.message || err}`);
  } finally {
    aggregations.delete(aggKey);
    log.info(`Aggregation pipeline complete for trap ${trapId}`);
  }
};

const addImageToAggregation = (aggKey, trapId, clientInfo, perImage) => {
  let agg = aggregations.get(aggKey);
  const now = Date.now();

  const scheduleFinalization = (aggregation) => {
    const delayMs = Math.max((aggregation.finalizeAt || now) - Date.now(), 0);
    aggregation.timer = setTimeout(() => finalizeAggregation(aggKey), delayMs);
  };

  if (!agg) {
    agg = {
      trapId,
      images: [],
      timer: null,
      finalizeAt: now + AGGREGATION_WINDOW_MS,
    };
    aggregations.set(aggKey, agg);
    scheduleFinalization(agg);
  } else {
    // Reset finalization to 1 minute from the latest received image.
    agg.finalizeAt = Date.now() + AGGREGATION_WINDOW_MS;
    clearTimeout(agg.timer);
    scheduleFinalization(agg);
  }

  agg.images.push({ clientInfo, ...perImage });
  const secondsRemaining = Math.max(Math.round((agg.finalizeAt - Date.now()) / 1000), 0);
  const aggLog = createLogger(perImage.correlationId || aggKey, { trapId });
  aggLog.info(`Image ${agg.images.length} added to event group — will flush in ${secondsRemaining}s`);
  aggLog.debug(`Aggregation key: ${aggKey}`);
  if (agg.images.length >= MAX_IMAGES_PER_EVENT) {
    clearTimeout(agg.timer);
    // finalize asynchronously
    setImmediate(() => finalizeAggregation(aggKey));
  }
};

const getAggregationSnapshot = () => {
  const aggregationData = Array.from(aggregations.entries()).map(([aggKey, agg]) => ({
    aggKey,
    trapId: agg.trapId,
    count: agg.images.length,
  }));
  const inFlightData = Array.from(inFlightUploads.entries()).map(([uploadKey, info]) => ({
    uploadKey,
    ...info,
  }));
  return {
    activeAggregationCount: aggregationData.length,
    inFlightUploadCount: inFlightData.length,
    aggregations: aggregationData,
    inFlightUploads: inFlightData,
  };
};

// Debug endpoint to inspect current aggregations and in-flight uploads
app.get('/debug/aggregations', (req, res) => {
  res.json(getAggregationSnapshot());
});

// Alias endpoint for convenience
app.get('/aggregations', (req, res) => {
  res.json(getAggregationSnapshot());
});

// ====================
// Helper: Upload buffer to Cloudinary
// ====================
const uploadToCloudinary = (buffer, originalName, trapId) => {
  return new Promise((resolve, reject) => {
    const uploadStream = cloudinary.uploader.upload_stream(
      {
        folder: `camera-traps/${trapId}`,
        public_id: `${Date.now()}_${originalName.split(".")[0]}`,
        overwrite: true,
        resource_type: "image",
        tags: ["camera-trap", trapId],
        context: `trapId=${trapId}`,
        eager: [
          { width: 800, crop: "limit", fetch_format: "auto", quality: "auto" },
        ],
      },
      (error, result) => {
        if (error) reject(error);
        else resolve(result);
      }
    );

    const stream = new PassThrough();
    stream.end(buffer);
    stream.pipe(uploadStream);
  });
};

// ====================
// Main Route
// ====================
const processUploadImage = async ({ trapId, clientId: ctClientId, captureTime, temperature, metadata, fileBuffer, fileName, serverReceiptTime, validatedClientInfo, protocolVersion, correlationId }) => {
  const log = createLogger(correlationId, { trapId });
  log.info(`New image upload from trap ${trapId} — capture time: ${captureTime}`);
  log.debug(`Client ID from device: ${ctClientId}`);

  // 1. Parse temperature
  const tempValue = temperature ? parseFloat(temperature) : null;

  // 2. Parse detailed timing metadata
  let timing = {};
  let metadataParseError = null;

  if (metadata) {
    try {
      const parsed = JSON.parse(metadata);
      timing = {
        triggerTime: parsed.triggerTime || null,
        captureTimeDevice: parsed.captureTime || null,
        pppStartTime: parsed.pppStartTime || null,
        pppConnectedTime: parsed.pppConnectedTime || null,
      };
      log.debug(`Metadata parsed — trigger time: ${timing.triggerTime}, capture time: ${timing.captureTimeDevice}`);
    } catch (e) {
      log.warn(`Metadata could not be parsed: ${e.message}`);
      metadataParseError = e.message;
    }
  }

  // Step 1: Client lookup
  const clientInfo = validatedClientInfo || await lookupTrapClientInfo(trapId, { correlationId, log });

  if (!clientInfo?.clientId) {
    throw new Error("Client not found for trapId");
  }

  if (String(clientInfo.clientId) !== String(ctClientId)) {
    throw new Error("Provided clientId does not match trap assignment");
  }

  log.info(`Client verified — id: ${clientInfo.clientId}`);

  // Step 2: Cloudinary upload
  log.debug('Uploading image to Cloudinary...');
  const clResult = await uploadToCloudinary(fileBuffer, fileName, trapId);
  const cloudinaryUploadTime = new Date().toISOString();
  const imageUrl = clResult.secure_url;
  const publicId = clResult.public_id;
  const thumbnailUrl = cloudinary.url(publicId, {
    secure: true,
    transformation: { width: 800, height: 600, crop: "limit", fetch_format: "auto" },
  });
  log.info(`Cloudinary upload complete — public id: ${publicId}`);

  // Step 3: AI inference with retry loop (retries AI only, not Cloudinary upload)
  let aiResponse = null;
  let aiError = null;
  const maxAiAttempts = Math.max(1, AI_PROCESS_RETRIES + 1);
  const aiRequestSentAt = new Date().toISOString();
  let aiResponseReceivedAt = null;
  let aiTimings = null;

  for (let attempt = 1; attempt <= maxAiAttempts; attempt += 1) {
    const form = new FormData();
    form.append("file", fileBuffer, fileName || "capture.jpg");
    form.append("client_id", clientInfo.clientId);
    form.append("run_sam", "false");
    form.append("detector_threshold", "0.30");
    form.append("topk_species", "3");

    log.debug(`Sending image to AI server (attempt ${attempt} of ${maxAiAttempts})`);
    try {
      aiResponse = await axios.post(process.env.AI_SERVER_URL, form, {
        headers: { ...form.getHeaders(), 'x-correlation-id': correlationId },
        timeout: 90000,
      });
      aiResponseReceivedAt = new Date().toISOString();
      aiTimings = {
        requestSentAt: aiRequestSentAt,
        responseReceivedAt: aiResponseReceivedAt,
        durationMs: new Date(aiResponseReceivedAt) - new Date(aiRequestSentAt),
        status: 'ok',
        attempts: attempt,
      };
      log.debug(`AI server responded in ${aiTimings.durationMs}ms (attempt ${attempt})`);
      break;
    } catch (err) {
      aiError = err;
      const lastAttempt = attempt === maxAiAttempts;
      log.warn(`AI attempt ${attempt}/${maxAiAttempts} failed: ${err?.message || 'unknown error'}`);
      if (!lastAttempt) {
        await sleep(UPLOAD_RETRY_DELAY_MS * attempt);
      }
    }
  }

  if (!aiResponse) {
    aiResponseReceivedAt = new Date().toISOString();
    aiTimings = {
      requestSentAt: aiRequestSentAt,
      responseReceivedAt: aiResponseReceivedAt,
      durationMs: new Date(aiResponseReceivedAt) - new Date(aiRequestSentAt),
      status: 'error',
      attempts: maxAiAttempts,
      error: aiError?.response?.data || aiError?.message || 'AI error',
    };
    aiResponse = { data: { detections: [], warnings: [HUMAN_INTERVENTION_WARNING, String(aiTimings.error)] } };
  }

  const { detections = [], warnings = [] } = aiResponse.data;
  const processingTime = new Date().toISOString();
  if (aiTimings?.status === 'error') {
    log.warn(`AI analysis failed after all ${maxAiAttempts} attempt(s) — event will be stored without AI detections`);
  } else {
    log.info(`AI analysis complete — ${detections.length} detection(s) identified`);
  }

  const calculateDelayMs = (start, end) => {
    if (!start || !end) return null;
    try {
      const startDate = new Date(start);
      const endDate = new Date(end);
      if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) return null;
      const diffMs = endDate - startDate;
      if (diffMs < 0 || diffMs > 3600000) return null;
      return Math.round(diffMs / 1000 * 10) / 10;
    } catch {
      return null;
    }
  };

  const delays = {
    triggerToCapture: calculateDelayMs(timing.triggerTime, timing.captureTimeDevice),
    captureToPppStart: calculateDelayMs(timing.captureTimeDevice, timing.pppStartTime),
    pppStartToConnected: calculateDelayMs(timing.pppStartTime, timing.pppConnectedTime),
    pppConnectedToProcessing: calculateDelayMs(timing.pppConnectedTime, processingTime),
    totalDeviceToProcessing: calculateDelayMs(timing.triggerTime, processingTime),
  };

  // Queue image into per-trap+client+trigger aggregation for final event write.
  const effectiveClientId = ctClientId;
  const aggKey = `${trapId}:${effectiveClientId}:${timing.triggerTime}`;
  log.debug(`Queuing image for event grouping — key: ${aggKey}`);
  addImageToAggregation(aggKey, trapId, clientInfo, {
    correlationId,
    protocolVersion: protocolVersion || DEFAULT_PROTOCOL_VERSION,
    captureTime,
    processingTime,
    serverReceiptTime,
    processingStartTime: aiRequestSentAt,
    imageUrl,
    publicId,
    thumbnailUrl,
    detectionsRaw: detections,
    warnings,
    metadataParseError,
    temperature: tempValue,
    processingDelaySeconds: delays.totalDeviceToProcessing,
    deviceTimings: {
      triggerTime: timing.triggerTime,
      captureTimeDevice: timing.captureTimeDevice,
      pppStartTime: timing.pppStartTime,
      pppConnectedTime: timing.pppConnectedTime,
    },
    aiTimings: aiTimings || null,
    processingStages: {
      cloudinaryUploadTime: cloudinaryUploadTime || null,
      aiRequestSentAt: aiRequestSentAt || null,
      aiResponseReceivedAt: aiResponseReceivedAt || null,
    },
  });

  const totalDelay = delays.totalDeviceToProcessing ?? Math.round((aiTimings?.durationMs || 0) / 1000);
  log.info(`Image processing complete — ${detections.length} detection(s), total delay: ${totalDelay}s`);
  return {
    imageUrl,
    correlationId,
    detections: detections.length,
    processingDelaySeconds: totalDelay,
    timingFeedback: {
      totalDelay,
      pppConnectionTime: delays.pppStartToConnected,
    },
  };
};

const uploadHandler = async (req, res) => {
  let uploadKey = null;
  const correlationId = generateCorrelationId();
  const log = createLogger(correlationId);
  try {
    const serverReceiptTime = new Date().toISOString();
    const { trapId, captureTime, temperature, metadata, clientId } = req.body;
    const file = req.file;
    log.info(`Upload request received — trap: ${trapId}, client: ${clientId}`);

    if (!trapId || !clientId || !captureTime || !file) {
      return res.status(400).json({
        error: "Missing required fields: trapId, clientId, captureTime, image",
      });
    }

    if (!metadata) {
      return res.status(400).json({ error: "metadata is required and must include triggerTime" });
    }

    let parsedMetadata = null;
    try {
      parsedMetadata = JSON.parse(metadata);
    } catch {
      return res.status(400).json({ error: "metadata must be valid JSON" });
    }

    if (!parsedMetadata?.triggerTime) {
      return res.status(400).json({ error: "metadata.triggerTime is required for batch grouping" });
    }

    const versionInfo = resolveProtocolVersion({ req, metadataObj: parsedMetadata });
    attachVersionHeaders(res, versionInfo);

    if (!versionInfo.isSupported && STRICT_PROTOCOL_VERSION) {
      return res.status(426).json({
        error: `Unsupported protocolVersion '${versionInfo.requestedVersion}'`,
        supportedProtocolVersions: SUPPORTED_PROTOCOL_VERSIONS,
        defaultProtocolVersion: DEFAULT_PROTOCOL_VERSION,
      });
    }

    let validatedClientInfo = null;
    try {
      validatedClientInfo = await lookupTrapClientInfo(trapId, { correlationId, log });
    } catch (lookupErr) {
      const status = lookupErr?.response?.status || 500;
      return res.status(status).json({
        error: lookupErr?.response?.data?.error || "Failed to validate trap",
      });
    }

    if (!validatedClientInfo?.clientId) {
      return res.status(404).json({ error: "Trap not found" });
    }

    if (String(validatedClientInfo.clientId) !== String(clientId)) {
      return res.status(403).json({ error: "clientId does not match trap assignment" });
    }

    const job = {
      trapId,
      clientId,
      captureTime,
      temperature,
      metadata,
      protocolVersion: versionInfo.effectiveVersion,
      correlationId,
      fileBuffer: Buffer.from(file.buffer),
      fileName: file.originalname || "capture.jpg",
      serverReceiptTime,
      validatedClientInfo,
    };
    uploadKey = `${trapId}:${clientId}:${captureTime}:${Date.now()}`;
    inFlightUploads.set(uploadKey, {
      trapId,
      clientId,
      captureTime,
      correlationId,
      startedAt: serverReceiptTime,
      stage: 'accepted',
    });

    if (UPLOAD_ACK_IMMEDIATE) {
      res.set('x-correlation-id', correlationId);
      res.status(202).json({
        status: "accepted",
        message: "Image accepted for background processing",
        apiVersion: API_CURRENT_VERSION,
        protocolVersion: versionInfo.effectiveVersion,
        protocolWarning: versionInfo.fallbackReason,
        correlationId,
        trapId,
        captureTime,
      });

      setImmediate(async () => {
        try {
          const existing = inFlightUploads.get(uploadKey);
          if (existing) inFlightUploads.set(uploadKey, { ...existing, stage: 'processing' });
          const result = await processUploadImage(job);
          log.info(`Background processing finished — ${result.detections} detection(s)`);
        } catch (error) {
          log.error(`Background processing failed: ${error.message || error}`);
        } finally {
          inFlightUploads.delete(uploadKey);
        }
      });

      return;
    }

    const existing = inFlightUploads.get(uploadKey);
    if (existing) inFlightUploads.set(uploadKey, { ...existing, stage: 'processing' });
    const result = await processUploadImage(job);
    inFlightUploads.delete(uploadKey);
    res.set('x-correlation-id', correlationId);
    return res.json({
      status: "success",
      message: "Image processed and queued",
      apiVersion: API_CURRENT_VERSION,
      protocolVersion: versionInfo.effectiveVersion,
      protocolWarning: versionInfo.fallbackReason,
      correlationId,
      imageUrl: result.imageUrl,
      detections: result.detections,
      processingDelaySeconds: result.processingDelaySeconds,
      timingFeedback: result.timingFeedback,
    });
  } catch (error) {
    if (uploadKey) {
      inFlightUploads.delete(uploadKey);
    }
    log.error(`Upload handler error: ${error.message || error}`);
    return res.status(500).json({
      error: "Processing failed",
      correlationId,
      details: error.response?.data || error.message,
    });
  }
};

app.post(["/upload", "/api/v1/upload"], upload.single("image"), uploadHandler);

// ====================
// Device Status & Time Sync
// ====================

// POST /daily_status - accept daily status from camera trap and forward to BFF
const dailyStatusHandler = async (req, res) => {
  const correlationId = generateCorrelationId();
  const log = createLogger(correlationId);
  try {
    const versionInfo = resolveProtocolVersion({ req, metadataObj: null });
    attachVersionHeaders(res, versionInfo);
    res.set('x-correlation-id', correlationId);

    if (!versionInfo.isSupported && STRICT_PROTOCOL_VERSION) {
      return res.status(426).json({
        error: `Unsupported protocolVersion '${versionInfo.requestedVersion}'`,
        supportedProtocolVersions: SUPPORTED_PROTOCOL_VERSIONS,
        defaultProtocolVersion: DEFAULT_PROTOCOL_VERSION,
      });
    }

    const {
      trapId,
      sd_free,
      sd_used,
      battery_voltage,
      total_triggers_today,
      failed_uploads,
    } = req.body;

    if (!trapId) {
      return res.status(400).json({ error: "trapId required" });
    }

    log.info(`Daily status received from trap ${trapId}`);

    // Forward to internal backend
    if (process.env.BFF_DAILY_STATUS_URL) {
      try {
        const response = await axios.post(
          process.env.BFF_DAILY_STATUS_URL,
          req.body,
          { timeout: 10000, headers: { 'x-correlation-id': correlationId } }
        );
        log.info(`Daily status forwarded to backend successfully`);
        return res.status(201).json({ ...response.data, correlationId });
      } catch (fwdErr) {
        log.error(`Failed to forward daily status to backend: ${fwdErr.message}`);
        return res.status(500).json({
          error: "Failed to store daily status",
          correlationId,
          details: fwdErr.response?.data || fwdErr.message,
        });
      }
    } else {
      log.warn('BFF_DAILY_STATUS_URL is not configured — daily status will not be stored');
      return res.status(201).json({
        message: "Daily status received (BFF forwarding disabled)",
        apiVersion: API_CURRENT_VERSION,
        protocolVersion: versionInfo.effectiveVersion,
        protocolWarning: versionInfo.fallbackReason,
        correlationId,
        trapId,
      });
    }
  } catch (error) {
    log.error(`Daily status handler error: ${error.message}`);
    res.status(500).json({ error: "Internal server error", correlationId });
  }
};

app.post(["/daily_status", "/api/v1/daily_status"], express.json(), dailyStatusHandler);

// GET /time_sync - return current UTC timestamp for camera trap time synchronization
const timeSyncHandler = (req, res) => {
  const versionInfo = resolveProtocolVersion({ req, metadataObj: null });
  attachVersionHeaders(res, versionInfo);
  res.json({
    timestamp: new Date().toISOString(),
    apiVersion: API_CURRENT_VERSION,
    protocolVersion: versionInfo.effectiveVersion,
    protocolWarning: versionInfo.fallbackReason,
  });
};

app.get(["/time_sync", "/api/v1/time_sync"], timeSyncHandler);

// Health check
const healthHandler = (req, res) => {
  const versionInfo = resolveProtocolVersion({ req, metadataObj: null });
  attachVersionHeaders(res, versionInfo);
  res.json({
    status: "ok",
    timestamp: new Date().toISOString(),
    apiVersion: API_CURRENT_VERSION,
    protocolVersion: versionInfo.effectiveVersion,
    supportedProtocolVersions: SUPPORTED_PROTOCOL_VERSIONS,
  });
};

app.get(["/health", "/api/v1/health"], healthHandler);

app.get(["/version", "/api/version"], (req, res) => {
  res.json({
    apiVersion: API_CURRENT_VERSION,
    supportedProtocolVersions: SUPPORTED_PROTOCOL_VERSIONS,
    defaultProtocolVersion: DEFAULT_PROTOCOL_VERSION,
    strictProtocolVersion: STRICT_PROTOCOL_VERSION,
    endpoints: {
      versionedBase: `/api/${API_CURRENT_VERSION}`,
      legacyCompat: true,
    },
  });
});

const PORT = process.env.PORT || 3003;
app.listen(PORT, () => {
  console.log(`Camera Trap Backend (ESM) running on http://localhost:${PORT}`);
  console.log(`Ready to receive images`);
  console.log(`Upload ACK mode: ${UPLOAD_ACK_IMMEDIATE ? "async (202 + background)" : "sync (wait for AI)"}`);
  console.log(`API version: ${API_CURRENT_VERSION}`);
  console.log(`Supported protocol versions: ${SUPPORTED_PROTOCOL_VERSIONS.join(", ")}`);
  console.log(`Default protocol version: ${DEFAULT_PROTOCOL_VERSION}`);
  console.log(`Strict protocol version mode: ${STRICT_PROTOCOL_VERSION}`);
  console.log(`\nAvailable endpoints:`);
  console.log(`- POST /upload (image processing with aggregation)`);
  console.log(`- POST /api/v1/upload (versioned alias)`);
  console.log(`- POST /daily_status (device status forwarding)`);
  console.log(`- POST /api/v1/daily_status (versioned alias)`);
  console.log(`- GET /time_sync (UTC timestamp for CT time sync)`);
  console.log(`- GET /api/v1/time_sync (versioned alias)`);
  console.log(`- GET /health (health check)`);
  console.log(`- GET /api/v1/health (versioned alias)`);
  console.log(`- GET /version or /api/version (capabilities)`);
  console.log(`- GET /debug/aggregations (inspect active aggregations)`);
});
