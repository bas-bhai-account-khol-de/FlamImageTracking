import * as THREE from 'three';
export const globalPrecisionFactor = 1e3;
export function setPrecision(n) {
    return Math.floor(n * globalPrecisionFactor) / globalPrecisionFactor;
}
/**
 * @description Creates canvas of a given size and return it
 * @abstract Creates canvas of a given size and return it
 * @param doc document object of page
 * @param w
 * @param h
 * @returns  a cannvas of given size
 */
export function addCanvasOfSize(doc, w = 1000, h = 1000) {
    const canvas = doc.createElement('canvas');
    canvas.height = h;
    canvas.width = w;
    doc.body.appendChild(canvas);
    return canvas;
}
/**
 * @description creates a 3d envournment ant returns the enviournment variable
 * @param canvas
 * @returns
 */
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
        arr.push([Math.floor(globalPrecisionFactor * Math.random()), Math.floor(globalPrecisionFactor * Math.random())]);
    }
    return arr;
}
/**
 * @abstract sets a url as canvas, texture so we can draw later
 * @param cb call back function to be called aferwards
 */
export function addTextureOnCanvas(canvas, url, cb = null) {
    var image = new Image();
    image.height = canvas.height;
    image.width = canvas.width;
    image.src = url;
    image.onload = () => { var dr = canvas.getContext('2d'); dr === null || dr === void 0 ? void 0 : dr.drawImage(image, 0, 0, canvas.width, canvas.height); cb === null || cb === void 0 ? void 0 : cb(); };
}
/**
 *
 * @param env 3D environment where we want to project
 * @param points  the scren points in 2d coord in range (0,1) we want to project
 * @returns
 */
export function getReprojectedPointsAfterTrasnform(env, points) {
    //convert point at unit distance into 3d space
    var s = 2 * Math.atan((env.fov / 2) * Math.PI / 180);
    var ScreenPoints = [];
    for (var i = 0; i < points.length; i++) {
        ScreenPoints.push(new THREE.Vector3(setPrecision((points[i][0] / globalPrecisionFactor - 0.5) * s), setPrecision((0.5 - points[i][1] / globalPrecisionFactor) * s), 0));
    }
    return ScreenPoints;
}
/**
 * @description puts a sphere at desired location
 * @param env
 * @param rad
 * @param location
 */
export function putASphereInEnvironment(env, rad, location) {
    const spg = new THREE.SphereGeometry(rad, 16, 16);
    const material = new THREE.MeshBasicMaterial({ color: 0xff0000, wireframe: true });
    const sphere = new THREE.Mesh(spg, material);
    sphere.position.add(location);
    env.scene.add(sphere);
}
