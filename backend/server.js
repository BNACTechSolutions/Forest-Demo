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

    // Build species scoring across images
    const speciesStats = {}; // species -> { score, count, examples }

    images.forEach((img) => {
      (img.detectionsRaw || []).forEach((det) => {
        const key = det.species || det.label || 'Unknown';
        const score = Number(det.species_confidence ?? det.detector_confidence ?? 0);
        if (!speciesStats[key]) speciesStats[key] = { score: 0, count: 0, examples: [] };
        speciesStats[key].score += score;
        speciesStats[key].count += 1;
        speciesStats[key].examples.push({ score, det, imageUrl: img.imageUrl });
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

    const eventStart = images[0].captureTime || images[0].processingTime;
    const eventEnd = images[images.length - 1].processingTime;

    const finalPayload = {
      trapId,
      eventStartTime: eventStart,
      eventEndTime: eventEnd,
      // keep compatibility fields expected by older single-image payloads
      captureTime: eventStart,
      processingTime: eventEnd,
      images: images.map((i) => ({ imageUrl: i.imageUrl, publicId: i.publicId, thumbnailUrl: i.thumbnailUrl, deviceTimings: i.deviceTimings || null, aiTimings: i.aiTimings || null })),
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
      imageUrl: bestExample?.imageUrl || images[0].imageUrl,
      warnings: images.flatMap((i) => i.warnings || []),
      metadataParseErrors: images.map((i) => i.metadataParseError).filter(Boolean),
      deviceTimings: images[0]?.deviceTimings || null,
      aiTimings: images[0]?.aiTimings || null,
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
  if (!agg) {
    agg = { images: [], timer: null };
    agg.timer = setTimeout(() => finalizeAggregation(trapId), AGGREGATION_WINDOW_MS);
    aggregations.set(trapId, agg);
  }
  agg.images.push({ clientInfo, ...perImage });
  console.log(`addImageToAggregation: trap=${trapId} images=${agg.images.length}`);
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
    const { trapId, captureTime, temperature, metadata } = req.body;
    const file = req.file;

    if (!trapId || !captureTime || !file) {
      return res.status(400).json({
        error: "Missing required fields: trapId, captureTime, image",
      });
    }

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
    const uploadStart = new Date().toISOString();
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
        return Math.round((new Date(end) - new Date(start)) / 1000 * 10) / 10; // 0.1s precision
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
      processingDelaySeconds: delays.totalDeviceToProcessing ?? -1,
      timingFeedback: {
        totalDelay: delays.totalDeviceToProcessing,
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

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

const PORT = process.env.PORT || 3003;
app.listen(PORT, () => {
  console.log(`Camera Trap Backend (ESM) running on http://localhost:${PORT}`);
  console.log(`Ready to receive images`);
});
