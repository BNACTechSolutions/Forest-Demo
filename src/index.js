import express from "express";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import cors from "cors";

const app = express();
app.use(cors());
const upload = multer({ storage: multer.memoryStorage() });

app.post("/identify", upload.single("image"), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: "No image uploaded" });

  try {
    const form = new FormData();
    form.append("file", req.file.buffer, { filename: "image.jpg" });

    const pyResp = await axios.post("http://localhost:8000/identify", form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });

    console.log("Python service response:", pyResp.data);

    // Forward Python response to frontend
    res.json(pyResp.data);
  } catch (err) {
    console.error("Error calling Python service:", err.message);
    res.status(500).json({ error: "identify_failed", details: err.message });
  }
});

app.listen(3000, () => console.log("Node server listening on port 3000"));
