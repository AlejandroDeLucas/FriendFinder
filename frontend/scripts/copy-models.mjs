import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, '..');
const distDir = path.join(frontendRoot, 'dist');
const publicDir = path.join(frontendRoot, 'public');

const faceApiSrc = path.join(frontendRoot, 'node_modules', '@vladmandic', 'face-api', 'model');
const neededFaceFiles = [
  'ssd_mobilenetv1_model-weights_manifest.json',
  'ssd_mobilenetv1_model.bin',
  'face_landmark_68_tiny_model-weights_manifest.json',
  'face_landmark_68_tiny_model.bin',
  'face_recognition_model-weights_manifest.json',
  'face_recognition_model.bin',
  'face_expression_model-weights_manifest.json',
  'face_expression_model.bin',
];

const mpWasmSrc = path.join(frontendRoot, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const neededWasmFiles = [
  'vision_wasm_internal.js',
  'vision_wasm_internal.wasm',
];

function copyList(srcDir, destDir, files) {
  if (!fs.existsSync(srcDir)) return;
  fs.mkdirSync(destDir, { recursive: true });
  for (const file of files) {
    const src = path.join(srcDir, file);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, path.join(destDir, file));
    }
  }
}

const targetBase = process.argv.includes('--public') ? publicDir : distDir;
copyList(faceApiSrc, path.join(targetBase, 'models'), neededFaceFiles);
copyList(mpWasmSrc, path.join(targetBase, 'wasm'), neededWasmFiles);
console.log(`[copy-models] Copied face-api models & MediaPipe WASM to ${targetBase}`);
