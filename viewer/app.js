import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const canvas = document.getElementById("renderCanvas");
const status = document.getElementById("status");
const statusDot = document.getElementById("statusDot");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080b10);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.001, 10000);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x26313d, 2.0));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
keyLight.position.set(1, 2, 3);
scene.add(keyLight);

let currentMesh = null;
let objectUrl = null;

function setStatus(message, kind = "ready") {
  status.textContent = message;
  statusDot.className = `status-dot ${kind === "ready" ? "" : kind}`;
}

function disposeMesh() {
  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh.geometry.dispose();
    currentMesh.material.dispose();
  }
  currentMesh = null;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
}

function frameMesh(mesh) {
  const box = new THREE.Box3().setFromObject(mesh);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const distance = Math.max(size.x, size.y, size.z, 0.1) * 1.35;
  controls.target.copy(center);
  camera.position.set(center.x, center.y, center.z + distance);
  camera.near = Math.max(distance / 10000, 0.001);
  camera.far = distance * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

function loadMesh(source, label, isFile = false) {
  setStatus(`Loading mesh: ${label}`, "busy");
  disposeMesh();
  const url = isFile ? (objectUrl = URL.createObjectURL(source)) : source;
  new PLYLoader().load(url, geometry => {
    const hasColor = Boolean(geometry.getAttribute("color"));
    const material = new THREE.MeshBasicMaterial({
      color: hasColor ? 0xffffff : 0xd5d9de,
      vertexColors: hasColor,
      side: THREE.DoubleSide,
    });
    currentMesh = new THREE.Mesh(geometry, material);
    scene.add(currentMesh);
    frameMesh(currentMesh);
    setStatus(`Loaded RGB mesh: ${label}`);
  }, undefined, error => {
    setStatus(`Mesh failed: ${error.message || error}`, "error");
    console.error(error);
  });
}

document.getElementById("loadMeshUrl").addEventListener("click", () => {
  const url = document.getElementById("meshUrl").value.trim();
  if (url) loadMesh(url, url);
});
document.getElementById("meshFile").addEventListener("change", event => {
  const file = event.target.files[0];
  if (!file) return;
  loadMesh(file, file.name, true);
});
document.getElementById("frameAll").addEventListener("click", () => {
  if (currentMesh) frameMesh(currentMesh);
});

const meshParam = new URLSearchParams(location.search).get("mesh");
if (meshParam) {
  document.getElementById("meshUrl").value = meshParam;
  loadMesh(meshParam, meshParam);
}

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
