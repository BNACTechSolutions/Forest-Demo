# CameraTrap System — How It Works

## The Big Picture

A camera trap is a weatherproof camera installed in the field.  When it detects
motion, it takes one or more photographs, uploads them to our cloud system, and
our system analyses them using AI to identify the species captured.  The results
are stored and surfaced in the client dashboard.

There are three separate software services involved:

| Service | What it does |
|---|---|
| **Edge Ingestion Server** | Receives raw images and metadata from camera-trap devices over the internet |
| **AI Identification Server** | Runs the computer-vision pipeline (MegaDetector + species classifier) |
| **Backend / BFF** | Stores detection events in the database and serves the client dashboard |

The Edge Ingestion Server is the only piece that devices talk to directly.
Everything else is internal.

---

## Workflow 1 — Image Upload (the main flow)

This is the most important flow in the system.  It covers everything that
happens from the moment an animal triggers the camera to the moment a detection
event appears in the dashboard.

### Flow at a glance

| Stage | System involved | What happens | Result |
|---|---|---|---|
| 1 | Camera Trap Device | Sends `POST /upload` with image, trap ID, client ID, timestamps, and metadata | Upload request reaches the Edge Ingestion Server |
| 2 | Edge Ingestion Server | Validates fields and verifies that the trap belongs to the client | Invalid or mismatched requests are rejected early |
| 3 | Edge Ingestion Server + Cloudinary | Uploads the raw image and creates a thumbnail | Image is safely stored in cloud image storage |
| 4 | Edge Ingestion Server + AI Server | Sends the image for animal/person detection and species classification | AI returns detections, species labels, and confidence scores |
| 5 | Edge Ingestion Server | Returns success or accepted response to the device, including timing feedback | Device knows the image was received |
| 6 | Edge Ingestion Server | Adds the image to a cohort grouped by `clientId + trapId + triggerTime` | Multiple related images are bundled into one event |
| 7 | Edge Ingestion Server | Waits until either 60 seconds pass with no new image or the cohort reaches 5 images | Cohort is ready to be finalised |
| 8 | Edge Ingestion Server + Backend / BFF | Aggregates all images, chooses the best one, prepares the final event, and stores it in the database | Dashboard receives one final detection event |

### Detailed explanation of each step

#### 1. Device sends an image

When motion is detected, the camera takes a photo and immediately sends it to
the Edge Ingestion Server.  Along with the image file it sends:

- **trapId** — the unique ID of that specific camera
- **clientId** — the ID of the project/client that owns the camera
- **captureTime** — the timestamp from the camera's clock when the photo was taken
- **temperature** — the ambient temperature reading (optional)
- **metadata** — a small JSON payload that includes:
  - `triggerTime` — when the motion sensor actually fired (slightly before capture)
  - `internetConnectionStartTime` / `internetConnectedTime` — when the camera started network connection and when it became connected (used for performance diagnostics)
  - `protocolVersion` — the protocol version the firmware uses

#### 2. Field validation

Before doing anything, the server checks that all required fields are present
and that `metadata` is well-formed JSON with a `triggerTime`.  If anything is
missing, it returns a clear error immediately.

#### 3. Client / trap verification

The server calls the internal backend to confirm that the (`clientId`, `trapId`)
pair matches a real, registered camera.  This prevents one client's device from
accidentally (or maliciously) posting data under another client's account.

#### 4. Image stored in Cloudinary

The raw image is uploaded to Cloudinary (our cloud image storage provider)
under the folder `camera-traps/{trapId}/`.  A smaller thumbnail version (800 ×
600 pixels) is automatically generated at the same time.  The Cloudinary upload
happens before AI analysis so the image is safely stored even if AI inference
later fails.

#### 5. AI analysis

The image is forwarded to the AI Identification Server.  This server runs two
models in sequence:

1. **MegaDetector** — a wildlife-specialist detector that finds animals, people,
   and vehicles in the image and draws bounding boxes around them.
2. **CLIP species classifier** — for each bounding box labelled "animal", CLIP
   compares the crop against a vocabulary of ~220 wildlife species to identify
   the most likely species, together with a confidence score.

The result comes back as a list of detections, each with a species name and
confidence score.

**Night / infrared images** are handled automatically: if the image is dark
(typical of night-time IR cameras), the AI applies image enhancement before
classification and uses a slightly lower confidence threshold, because IR images
are inherently noisier.

**AI failure handling:** if the AI server is unavailable or returns an error,
the server retries up to `AI_PROCESS_RETRIES` additional times (configurable,
default 2 extra attempts) with a delay between retries.  If all attempts fail:
- The image is **still stored** (Cloudinary upload already happened).
- The detection event is created with **no species detections**.
- A `HUMAN_INTERVENTION_REQUIRED` warning flag is set so that staff can
  review the image manually and confirm the priority/species.

#### 6. Image added to the aggregation cohort

After AI analysis the image is placed into a **cohort**.  A cohort is a group
of images that belong to the same single trigger event — because one motion
trigger can capture multiple photos in quick succession.

The cohort key is:  `clientId + trapId + triggerTime`

- **New trigger** → a new cohort is created and a 60-second countdown begins.
- **Same trigger, another image arrives** → the image is added to the existing
  cohort and the 60-second timer is **reset** (i.e. the countdown restarts from
  60 seconds).  The image is matched to the correct existing cohort, not saved
  as a new event.
- **Cohort is full (5 images)** → the cohort is flushed immediately without
  waiting for the timer.  After a database record reaches 5 images, it is
  treated as **closed/full**.  Any later upload with the same `triggerTime`
  starts a new cohort and becomes a separate record instead of extending the
  full one.

This means: if three images arrive for the same trigger at t=0, t=5s, and t=50s,
the cohort will flush at t=110s (50s arrival + 60s window).  All three images
are part of one detection event in the database.

#### 7. Response to the device

In the default (synchronous) mode the server responds `200 OK` after AI
analysis is complete but before the cohort flushes.  The response tells the
device:

- The Cloudinary image URL
- How many detections the AI found
- How many seconds elapsed from trigger to processing completion

There is also an **async mode** (`UPLOAD_ACK_IMMEDIATE=true`) where the server
responds `202 Accepted` immediately and then continues processing in the
background.  This makes the device faster at the cost of the device not knowing
the final outcome.

#### 8. Cohort flush — writing the detection event to the database

When the cohort window closes (60s timeout or 5-image cap), the server
aggregates results across all images in the cohort:

- **Species scoring** — detections from all images are pooled.  For each species
  that appeared, the AI confidence scores are averaged.  The top 3 species by
  average score are used.
- **Best image** — the image containing the highest-confidence detection is
  selected as the "best" image for display in the dashboard.
- **Timing summary** — trigger time, internet connection timing, AI start/end times,
  and total end-to-end delay are all included.
- **Temperature readings** — gathered from every image in the cohort.

This final aggregated event is sent to the Backend / BFF which stores it in
the database.  The dashboard then shows it to the client.

---

## Workflow 2 — Device Sends a Daily Status Report

Once per day each camera trap sends a short status report.  This does not
involve images or AI.

The device sends:

| Field | Meaning |
|---|---|
| `clientId` | Which client owns the camera |
| `trapId` | Which camera |
| `battery_voltage` | Current battery level |
| `sd_free` / `sd_used` | SD card space in MB |
| `total_triggers_today` | How many motion triggers fired today |
| `failed_uploads` | How many uploads failed to reach the server today |

The Edge Ingestion Server validates the (`clientId`, `trapId`) pair and then
forwards the report to the internal backend for storage.  If the forwarding fails, the
server returns an error to the device.

**Why is this useful?**  Battery and SD card data powers the health dashboard.
`failed_uploads` tells us whether a camera is having connectivity issues.

---

## Workflow 3 — Device Clock Synchronisation

Camera traps run on battery and have a real-time clock (RTC) that can drift or
reset after a power cut.  Accurate timestamps matter because `triggerTime` is
the key used to group multiple images into one event.

Before uploading (typically on boot), the device calls `GET /api/v1/time_sync`.
The server returns the current UTC time.  The device adjusts its RTC so that
all subsequent `triggerTime` and `captureTime` values are accurate.

---

## Scenario Guide — What Happens When…

### Three images arrive for the same trigger

The device takes three quick photos from one motion event.  All three carry the
same `triggerTime`.

1. Image 1 arrives → a new cohort is created, 60-second timer starts. Server responds `200 OK` to the device immediately.
2. Image 2 arrives (15 seconds later) → added to the cohort, timer resets to 60 seconds.
3. Image 3 arrives (50 seconds after image 2) → added to the cohort, timer resets to 60 seconds.
4. After 60 seconds of silence → cohort flushes, one detection event with all three images is written to the database.

**Outcome:** one record in the database containing all three images, the best species prediction from across all three, and the combined timing data.

### A late image arrives after the cohort has already been flushed

If image 4 arrives 90 seconds after image 3 (timer already fired and the
cohort is gone), it will start a **new cohort** with the same `triggerTime` key.
If the earlier database record already contains 5 images, it is considered
full and will **not** be updated.  A second detection event will be created in
the database instead.

This is a known edge case.  The timer is generously sized (60 seconds) to cover
normal cell-modem delivery delays.  Extremely delayed images (e.g. a retry
hours later) will appear as a separate event.  This hard cap prevents bad CT
retries from growing one record to impossible sizes such as 22 images.

### The AI server is temporarily down

1. Image is uploaded and stored in Cloudinary — safe.
2. Server attempts AI analysis.  It retries up to 3 total attempts (configurable).
3. All attempts fail.
4. The image is added to the cohort with **zero detections** and a
   `HUMAN_INTERVENTION_REQUIRED` warning.
5. When the cohort flushes, the event is saved to the database with the warning
   visible in the dashboard.
6. A human reviewer is expected to look at the images and manually confirm the
   species / priority.

The image data is **never lost** — Cloudinary already has the raw photo.

### The AI server returns low-confidence results

This is handled automatically by the AI pipeline:

- If the top prediction confidence is below the threshold (adaptive for
  night/IR images) **or** the gap between the first and second species is too
  small (a near-tie is unreliable), the species is recorded as `UNKNOWN` rather
  than guessing.
- This prevents the dashboard from showing confidently-wrong species labels.

### An animal is detected that is not on the client's allowed-species list

The AI classifies against a global vocabulary of ~220 species.  After
classification, the result is matched against the client's own species list.

- If the species is in the list → stored under that species name.
- If the species is not in the list but the AI is confident → stored as `UNKNOWN`.
- If the AI is not confident enough → stored as `None` (no species label).

### The camera sends a duplicate image (same trapId + captureTime)

Each upload creates a distinct entry in the in-flight tracking table using a
unique key (`clientId:trapId:captureTime:timestamp`).  Duplicate payloads —
rare in practice — will be processed independently and both images will appear
in Cloudinary.  They will be grouped into the same cohort if they carry the
same `triggerTime`.

### The cellular network drops mid-upload

The device will receive a TCP error (or timeout) and is expected to retry.  The
server does not maintain partial upload state; each `POST /upload` is
fully atomic.  A retry is treated as a fresh upload.  If the original upload
actually succeeded before the error reached the device, the retry will result
in a second image in the cohort (same `triggerTime`), which is harmless — the
cohort simply gains an extra copy to pick the best image from.

---

## Key Configuration Knobs (for operations teams)

| Setting | What it controls | Default |
|---|---|---|
| `AGGREGATION_WINDOW_MS` | How long (ms) to wait for more images before flushing a cohort | 60 000 (1 minute) |
| `MAX_IMAGES_PER_EVENT` | Maximum images before forcing an early flush | 5 |
| `AI_PROCESS_RETRIES` | Extra retry attempts when AI inference fails | 2 |
| `UPLOAD_RETRY_DELAY_MS` | Base delay (ms) between AI retry attempts | 5 000 |
| `UPLOAD_ACK_IMMEDIATE` | `true` → respond 202 instantly, process in background | false |
| `STRICT_PROTOCOL_VERSION` | `true` → reject devices with unknown protocol versions | false |
| `SUPPORTED_PROTOCOL_VERSIONS` | Comma-separated list of accepted device firmware versions | `1.0` |

---

## End-to-End Timing Window

Understanding the timing from trigger to dashboard is useful for setting
stakeholder expectations.

| Stage | Typical time |
|---|---|
| Camera triggers and captures image | 0–2 seconds |
| Cellular modem starts network connection | 2–5 seconds |
| Modem connects to network | 5–10 seconds |
| Image uploaded to Edge Server | 10–20 seconds |
| Image stored in Cloudinary | +1–3 seconds |
| AI analysis (MegaDetector + CLIP) | +3–15 seconds |
| Server acknowledges device | ≈ 15–35 seconds from trigger |
| Aggregation window closes | up to 60 seconds after last image |
| Event written to database | immediately after cohort flush |
| **Total: trigger → dashboard** | **≈ 30–90 seconds under normal conditions** |

Delays beyond 90 seconds typically indicate slow cell connectivity.  The
`processingDelaySeconds` field in the upload response and the timing fields
stored in each event allow this to be diagnosed per-device.

---

## Correlation IDs — How to Trace a Request

Every request to the Edge Ingestion Server is assigned a **Correlation ID** —
a short unique code (e.g. `CT-M8X3K2-A1B2`).  This ID:

- Is returned to the device in the response.
- Appears in every server log line related to that request.
- Is forwarded to the AI server and the backend on internal calls.
- Is stored in the database as part of the detection event.

If a customer reports a problem with a specific image, providing the Correlation
ID lets the engineering team find every log line related to that image in
seconds.
