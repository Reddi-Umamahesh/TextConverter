import express, { Request, Response } from "express";
import multer from "multer";
import { exec } from "child_process";
import path from "path";
import cors from "cors";
const app = express();
const PORT = 3000;

// Enable CORS for frontend
app.use(cors());

// Setup multer for file uploads
const upload = multer({ dest: path.join(__dirname, "../uploads/") });

app.post("/upload", upload.single("image"), (req: Request, res: Response) => {
  const filePath = path.resolve(req.file!.path);
  console.log(`File uploaded to: ${filePath}`);
  res.json({ prediction: "తెలుగు భాష చాలా అందమైనది మరియు సంపన్నమైనది." });
  // Call Python script with filePath
  //   exec(`python predict.py ${filePath}`, (error, stdout, stderr) => {
  //     if (error) {
  //       console.error(`Error: ${error.message}`);
  //       return res.status(500).json({ error: "Prediction failed" });
  //     }
  //     if (stderr) {
  //       console.error(`Stderr: ${stderr}`);
  //       return res.status(500).json({ error: "Prediction error" });
  //     }

  //     console.log(`Prediction result: ${stdout}`);
  //     res.json({ prediction: stdout.trim() });
  //   });
});

app.listen(PORT, () => {
  console.log(`✅ Server is running on http://localhost:${PORT}`);
});
