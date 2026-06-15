---
name: screenshot-ocr
description: Capture full-screen screenshots and OCR image files when dependencies are available.
allowed-tools: capture_screenshot ocr_image capture_and_ocr
metadata:
  version: "1.0.0"
  tags:
    - screenshot
    - ocr
    - image
---
Use screenshot-ocr when the user wants help with visible screen content or text inside images.

Rules:
- Capturing a live screenshot requires explicit confirmation.
- OCR of an explicit file path is read-only and can run directly.
- If OCR dependencies are unavailable, report the setup requirement clearly.
- Do not infer sensitive text beyond what OCR actually returns.
