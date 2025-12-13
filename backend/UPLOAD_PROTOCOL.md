# Camera Trap Upload Protocol

This document describes the minimal HTTP protocol the Camera Trap hardware should use to upload images to the model-research backend and hit the `/upload` endpoint.

## Endpoint

- URL: `POST http://<MODEL_RESEARCH_HOST>:<PORT>/upload`
- Default port used by the server: `3000` (unless overridden by `PORT` env)

## Method & Content-Type

- Method: `POST`
- Content-Type: `multipart/form-data` (file + form fields)

## Required form fields

- `trapId` (string) — unique identifier assigned to the camera trap. The backend uses this to look up the owning client/project.
- `captureTime` (string) — ISO 8601 timestamp for when the photo was taken (e.g. `2025-11-18T12:34:56Z`). The server uses it to compute processing delay.
- `image` (file) — the binary image file. Field name must be `image` (the server uses `multer().single('image')`).

Accepted image formats: JPEG/PNG recommended. Keep files reasonably sized (e.g. < 8–10 MB) if possible.

## Success response

On success the server responds with JSON similar to:

```
{
  "status": "success",
  "message": "Image processed and saved",
  "imageUrl": "https://res.cloudinary.com/.../image.jpg",
  "detections": 2,
  "processingDelaySeconds": 5
}
```

`detections` is the total number of detections the AI returned for the image. The backend will forward AI results to the BFF and store them.

## Error responses

- 400 Bad Request: missing required fields (e.g. missing `trapId`, `captureTime`, or `image`).
- 404 Not Found: trapId not recognized by the BFF lookup service.
- 500 Internal Server Error: processing failed (AI, Cloudinary, or BFF store error). Response JSON contains `details` when available.

Example error body:

```
{
  "error": "Processing failed",
  "details": "..."
}
```

## Recommended client behaviour (hardware/firmware)

- Timeout: set a request timeout (e.g. 60s). The backend may take additional time while the AI runs.
- Retries: on transient network errors, retry up to 2 times with exponential backoff.
- Idempotency: avoid duplicate uploads for the same capture (e.g. keep a short cache of recent capture timestamps). The backend does not currently deduplicate uploads.
- Image size: prefer a compressed JPEG at a resolution appropriate to detection (e.g. 1024px on longest side) to reduce upload time and bandwidth.

## Health check

- `GET /health` returns JSON `{ status: "ok", timestamp: "..." }` and can be used to verify the model-research service is running.