// server.js
import express from "express";
import multer from "multer";
import axios from "axios";
import { v2 as cloudinary } from "cloudinary";
import { PassThrough } from "stream";
import dotenv from "dotenv";
import FormData from "form-data";

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
    const { trapId, captureTime, gps, temperature } = req.body;
    const file = req.file;

    if (!trapId || !captureTime || !file) {
      return res.status(400).json({
        error: "Missing required fields: trapId, captureTime, image",
      });
    }

    // Step 1: Get client details from BFF
    const { data: clientInfo } = await axios.post(
      process.env.BFF_CLIENT_LOOKUP_URL,
      { trapId },
      { timeout: 8000 }
    );

    if (!clientInfo || !clientInfo.clientId) {
      return res.status(404).json({ error: "Client not found for this trapId" });
    }
    else{
      console.log('Client Info:', clientInfo);
    }

    // Step 2: Upload image to Cloudinary
    const clResult = await uploadToCloudinary(
      file.buffer,
      file.originalname,
      trapId
    );

    const imageUrl = clResult.secure_url;
    const publicId = clResult.public_id;
    const thumbnailUrl = cloudinary.url(publicId, {
      secure: true,
      transformation: { width: 800, height: 600, crop: "limit", fetch_format: "auto" },
    });

    // Step 3: Send to BioCLIP AI server
    const form = new FormData();
    form.append("file", file.buffer, file.originalname);
    form.append("client_id", clientInfo.clientId);
    form.append("run_sam", "false");
    form.append("detector_threshold", "0.30");
    form.append("topk_species", "3");

    const aiResponse = await axios.post(process.env.AI_SERVER_URL, form, {
      headers: form.getHeaders(),
      timeout: 90_000,
    });

    console.log('AI Response:', aiResponse.data);

    const { detections = [], warnings = [] } = aiResponse.data;

    // Step 4: Final payload to BFF
    const processingTime = new Date().toISOString();
    const delaySeconds = Math.round(
      (new Date(processingTime) - new Date(captureTime)) / 1000
    );

    const finalPayload = {
      trapId,
      captureTime,
      processingTime,
      processingDelaySeconds: delaySeconds,

      // From BFF lookup
      clientId: clientInfo.clientId,
      clientName: clientInfo.clientName,
      location: clientInfo.location,
      project: clientInfo.project,

      // Image
      imageUrl,
      publicId,
      thumbnailUrl,

      // AI Results
      totalDetections: detections.length,
      detections: detections.map((det) => ({
        species: det.species || "Unknown",
        confidence: det.species_confidence ? Number(det.species_confidence.toFixed(4)) : null,
        detectorConfidence: Number(det.detector_confidence.toFixed(4)),
        bbox: det.bbox,
        label: det.label,
        maskUrl: det.mask_png_base64 || null,
        topk: det.extra?.topk || null,
      })),

      // Extra metadata
      gps: gps || null,
      temperature: temperature || null,
      warnings,
    };

    console.log('Final Payload:', finalPayload);

    // Step 5: Send to BFF for storage
    await axios.post(process.env.BFF_STORE_URL, finalPayload, {
      timeout: 15_000,
    });

    console.log('Data successfully sent to BFF for storage.');

    // Step 6: Respond to camera trap
    res.json({
      status: "success",
      message: "Image processed and saved",
      imageUrl,
      detections: finalPayload.totalDetections,
      processingDelaySeconds: delaySeconds,
    });
  } catch (error) {
    console.error("Upload failed:", error.response?.data || error.message);

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

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Camera Trap Backend (ESM) running on http://localhost:${PORT}`);
  console.log(`Ready to receive images`);
});