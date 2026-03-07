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
// Aggregation config
// ====================
const AGGREGATION_WINDOW_MS = 60 * 1000; // 1 minute
const AGGREGATION_EXTEND_MS = 60 * 1000; // extend window by 1 minute per incoming image
const MAX_IMAGES_PER_EVENT = 5;
const aggregations = new Map(); // trapId -> { images: [], timer }

const finalizeAggregation = async (trapId) => {
  const agg = aggregations.get(trapId);
  if (!agg) return;
  clearTimeout(agg.timer);
  console.log(`finalizeAggregation: starting for trap ${trapId} (images=${agg.images.length})`);
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
      console.warn('BFF_STORE_URL not set; skipping remote store. Payload:', finalPayload);
    } else {
      try {
        await axios.post(process.env.BFF_STORE_URL, finalPayload, { timeout: 15000 });
        console.log(`Aggregated event stored for trap ${trapId} (images=${images.length})`);
      } catch (postErr) {
        console.error('BFF store rejected payload for', trapId, 'status=', postErr.response?.status, 'body=', postErr.response?.data);
        console.error('FinalPayload keys:', Object.keys(finalPayload));
        console.error('FinalPayload (summary):', {
          trapId: finalPayload.trapId,
          eventStartTime: finalPayload.eventStartTime,
          eventEndTime: finalPayload.eventEndTime,
          totalImagesReceived: finalPayload.totalImagesReceived,
          totalDetections: finalPayload.totalDetections,
          bestSpecies: finalPayload.bestSpecies,
        });
      }
    }
  } catch (err) {
    console.error('Failed to finalize aggregation for', trapId, err?.response?.data || err.message || err);
  } finally {
    aggregations.delete(trapId);
    console.log(`finalizeAggregation: finished for trap ${trapId}`);
  }
};

const addImageToAggregation = (trapId, clientInfo, perImage) => {
  let agg = aggregations.get(trapId);
  const now = Date.now();

  const scheduleFinalization = (aggregation) => {
    const delayMs = Math.max((aggregation.finalizeAt || now) - Date.now(), 0);
    aggregation.timer = setTimeout(() => finalizeAggregation(trapId), delayMs);
  };

  if (!agg) {
    agg = {
      images: [],
      timer: null,
      finalizeAt: now + AGGREGATION_WINDOW_MS,
    };
    aggregations.set(trapId, agg);
    scheduleFinalization(agg);
  } else {
    agg.finalizeAt = (agg.finalizeAt || now) + AGGREGATION_EXTEND_MS;
    clearTimeout(agg.timer);
    scheduleFinalization(agg);
  }

  agg.images.push({ clientInfo, ...perImage });
  const secondsRemaining = Math.max(Math.round((agg.finalizeAt - Date.now()) / 1000), 0);
  console.log(`addImageToAggregation: trap=${trapId} images=${agg.images.length} flushIn=${secondsRemaining}s`);
  if (agg.images.length >= MAX_IMAGES_PER_EVENT) {
    clearTimeout(agg.timer);
    // finalize asynchronously
    setImmediate(() => finalizeAggregation(trapId));
  }
};

// Debug endpoint to inspect current aggregations (counts only)
app.get('/debug/aggregations', (req, res) => {
  const data = Array.from(aggregations.entries()).map(([trapId, agg]) => ({ trapId, count: agg.images.length }));
  res.json({ count: data.length, aggregations: data });
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
app.post("/upload", upload.single("image"), async (req, res) => {
  try {
    const serverReceiptTime = new Date().toISOString(); // Capture when request arrives
    const { trapId, captureTime, temperature, metadata } = req.body;
    const file = req.file;

    if (!trapId || !captureTime || !file) {
      return res.status(400).json({
        error: "Missing required fields: trapId, captureTime, image",
      });
    }

    console.log('Timings for hardware: ', metadata);

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
          captureTimeDevice: parsed.captureTime || null,     // renamed for clarity
          pppStartTime: parsed.pppStartTime || null,
          pppConnectedTime: parsed.pppConnectedTime || null,
        };
      } catch (e) {
        console.warn("Failed to parse metadata JSON:", e.message);
        metadataParseError = e.message;
      }
    }

    // Step 1: Client lookup (unchanged)
    const { data: clientInfo } = await axios.post(
      process.env.BFF_CLIENT_LOOKUP_URL,
      { trapId },
      { timeout: 8000 }
    );

    if (!clientInfo?.clientId) {
      return res.status(404).json({ error: "Client not found for trapId" });
    }

    // Step 2: Cloudinary upload (record time)
    // Note: serverReceiptTime is when the request arrived at this server
    const clResult = await uploadToCloudinary(file.buffer, file.originalname, trapId);
    const cloudinaryUploadTime = new Date().toISOString();
    const imageUrl = clResult.secure_url;
    const publicId = clResult.public_id;
    const thumbnailUrl = cloudinary.url(publicId, {
      secure: true,
      transformation: { width: 800, height: 600, crop: "limit", fetch_format: "auto" },
    });

    // Step 3: AI inference (record timings)
    const form = new FormData();
    form.append("file", file.buffer, file.originalname || "capture.jpg");
    form.append("client_id", clientInfo.clientId);
    form.append("run_sam", "false");
    form.append("detector_threshold", "0.30");
    form.append("topk_species", "3");
    console.log(process.env.AI_SERVER_URL);
    let aiResponse;
    const aiRequestSentAt = new Date().toISOString();
    let aiResponseReceivedAt = null;
    let aiTimings = null;
    try {
      aiResponse = await axios.post(process.env.AI_SERVER_URL, form, {
        headers: form.getHeaders(),
        timeout: 90000,
      });
      aiResponseReceivedAt = new Date().toISOString();
      aiTimings = {
        requestSentAt: aiRequestSentAt,
        responseReceivedAt: aiResponseReceivedAt,
        durationMs: new Date(aiResponseReceivedAt) - new Date(aiRequestSentAt),
        status: 'ok'
      };
    } catch (aiErr) {
      aiResponseReceivedAt = new Date().toISOString();
      aiTimings = {
        requestSentAt: aiRequestSentAt,
        responseReceivedAt: aiResponseReceivedAt,
        durationMs: new Date(aiResponseReceivedAt) - new Date(aiRequestSentAt),
        status: 'error',
        error: aiErr.response?.data || aiErr.message
      };
      console.error('AI call failed:', aiErr?.response?.data || aiErr.message || aiErr);
      // Continue but set empty detections and add a warning
      aiResponse = { data: { detections: [], warnings: [(aiErr.response?.data || aiErr.message || 'AI error').toString()] } };
    }

    const { detections = [], warnings = [] } = aiResponse.data;

    // ── New: Calculate detailed delays ──────────────────────────────────────
    const now = new Date();
    const processingTime = now.toISOString();

    const calculateDelayMs = (start, end) => {
      if (!start || !end) return null;
      try {
        const startDate = new Date(start);
        const endDate = new Date(end);
        if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) return null;
        
        const diffMs = endDate - startDate;
        // Reject unrealistic delays (negative or > 1 hour)
        // This prevents 1970 timestamps from creating huge delays
        if (diffMs < 0 || diffMs > 3600000) return null;
        
        return Math.round(diffMs / 1000 * 10) / 10; // 0.1s precision
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

    // Step 4: Final payload
    const finalPayload = {
      trapId,
      captureTime,                    // the one sent in form field (filename based)
      processingTime,
      processingDelaySeconds: delays.totalDeviceToProcessing ?? null,

      // New timing information
      deviceTimings: {
        triggerTime: timing.triggerTime,
        captureTimeDevice: timing.captureTimeDevice,
        pppStartTime: timing.pppStartTime,
        pppConnectedTime: timing.pppConnectedTime,
      },
      calculatedDelaysSeconds: delays,
      aiTimings: aiTimings || null,
      processingStages: {
        cloudinaryUploadTime: cloudinaryUploadTime || null,
        aiRequestSentAt: aiRequestSentAt || null,
        aiResponseReceivedAt: aiResponseReceivedAt || null,
      },

      temperature: tempValue,

      // Client info
      clientId: clientInfo.clientId,
      clientName: clientInfo.clientName || null,
      location: clientInfo.location || null,
      project: clientInfo.project || null,

      // Image
      imageUrl,
      publicId,
      thumbnailUrl,

      // AI
      totalDetections: detections.length,
      detections: detections.map(det => ({
        species: det.species || "Unknown",
        confidence: det.species_confidence ? Number(det.species_confidence.toFixed(4)) : null,
        detectorConfidence: Number(det.detector_confidence?.toFixed(4)),
        bbox: det.bbox,
        label: det.label,
        maskUrl: det.mask_png_base64 || null,
        topk: det.extra?.topk || null,
      })),

      warnings,
      metadataParseError, // only present if parsing failed
    };

    // Queue this image's result into the per-trap aggregator. Final storage
    // will happen once the event window completes or max images reached.
    addImageToAggregation(trapId, clientInfo, {
      captureTime,
      processingTime,
      serverReceiptTime,  // When server received the request
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

    // Response to device (per-image)
    res.json({
      status: "success",
      message: "Image processed and queued",
      imageUrl,
      detections: finalPayload.totalDetections,
      processingDelaySeconds: delays.totalDeviceToProcessing ?? Math.round((aiTimings?.durationMs || 0) / 1000),
      timingFeedback: {
        totalDelay: delays.totalDeviceToProcessing ?? Math.round((aiTimings?.durationMs || 0) / 1000),
        pppConnectionTime: delays.pppStartToConnected
      }
    });
  } catch (error) {
    console.error("Upload processing failed:", error);
    res.status(500).json({
      error: "Processing failed",
      details: error.response?.data || error.message,
    });
  }
});

// ====================
// Device Status & Time Sync
// ====================

// POST /daily_status - accept daily status from camera trap and forward to BFF
app.post("/daily_status", express.json(), async (req, res) => {
  try {
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

    // Forward to internal backend
    if (process.env.BFF_DAILY_STATUS_URL) {
      try {
        const response = await axios.post(
          process.env.BFF_DAILY_STATUS_URL,
          req.body,
          { timeout: 10000 }
        );
        console.log(`[daily_status] Forwarded to BFF for trap ${trapId}`);
        return res.status(201).json(response.data);
      } catch (fwdErr) {
        console.error(
          `[daily_status] Failed to forward for trap ${trapId}:`,
          fwdErr.response?.data || fwdErr.message
        );
        return res.status(500).json({
          error: "Failed to store daily status",
          details: fwdErr.response?.data || fwdErr.message,
        });
      }
    } else {
      console.warn("[daily_status] BFF_DAILY_STATUS_URL not set; storing locally would go here");
      return res.status(201).json({
        message: "Daily status received (BFF forwarding disabled)",
        trapId,
      });
    }
  } catch (error) {
    console.error("[daily_status] Error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// GET /time_sync - return current UTC timestamp for camera trap time synchronization
app.get("/time_sync", (req, res) => {
  res.json({ timestamp: new Date().toISOString() });
});

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

const PORT = process.env.PORT || 3003;
app.listen(PORT, () => {
  console.log(`Camera Trap Backend (ESM) running on http://localhost:${PORT}`);
  console.log(`Ready to receive images`);
  console.log(`\nAvailable endpoints:`);
  console.log(`- POST /upload (image processing with aggregation)`);
  console.log(`- POST /daily_status (device status forwarding)`);
  console.log(`- GET /time_sync (UTC timestamp for CT time sync)`);
  console.log(`- GET /health (health check)`);
  console.log(`- GET /debug/aggregations (inspect active aggregations)`);
});
