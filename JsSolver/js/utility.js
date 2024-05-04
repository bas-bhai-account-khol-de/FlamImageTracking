import * as THREE from 'three';
export function addCanvasOfSize(doc, w = 1000, h = 1000) {
    const canvas = doc.createElement('canvas');
    canvas.height = h;
    canvas.width = w;
    doc.body.appendChild(canvas);
    return canvas;
}
export function Create3DScene(canvas) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 0;
    const renderer = new THREE.WebGLRenderer({ canvas: canvas });
    return { camera, renderer, scene };
}
