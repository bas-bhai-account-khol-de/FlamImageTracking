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
    const fov = 40;
    const camera = new THREE.PerspectiveCamera(fov, 1, 0.1, 1000);
    camera.position.z = 0;
    const renderer = new THREE.WebGLRenderer({ canvas: canvas });
    return { camera, renderer, scene, fov };
}
export function generateNPointsNormalized(n = 10) {
    let arr = [];
    for (var i = 0; i < n; i++) {
        arr.push([Math.random(), Math.random()]);
    }
    return arr;
}
