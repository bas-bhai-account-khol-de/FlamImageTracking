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
    const camera = new THREE.PerspectiveCamera(fov, canvas.width / canvas.height, 0.1, 10);
    camera.position.z = 0;
    const renderer = new THREE.WebGLRenderer({ canvas: canvas });
    return { camera, renderer, scene, fov };
}
/**
 * @description generates n point between 0 and 1
 * @param n
 * @returns
 */
export function generateNPointsNormalized(n = 100) {
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
 * @description thi fucntion traform all point on screen  to a 3d transform
 * @param env 3D environment where we want to project
 * @param points  the scren points in 2d coord in range (0,1) we want to project
 * @param transform the tranform of image
 * @param oncreen want points on creen or not
 * @returns
 */
export function getReprojectedPointsAfterTrasnform(env, points, transform = new THREE.Matrix4(), oncreen = true) {
    //convert point at unit distance into 3d space
    var s = 2 * Math.atan((env.fov / 2) * Math.PI / 180);
    var ScreenPoints = [];
    for (var i = 0; i < points.length; i++) {
        ScreenPoints.push(new THREE.Vector3(setPrecision((points[i][0] / globalPrecisionFactor - 0.5) * s), setPrecision((0.5 - points[i][1] / globalPrecisionFactor) * s), 0));
    }
    ScreenPoints = ScreenPoints.map((vec) => {
        var point = vec.applyMatrix4(transform);
        return point;
    });
    if (oncreen) {
        ScreenPoints = ScreenPoints.map((vec) => {
            var customProj = new THREE.Matrix4().set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0);
            var vecPos = new THREE.Vector4(vec.x, vec.y, vec.z, 1);
            // vecPos.applyMatrix4(tempMatProj)
            vecPos.applyMatrix4(customProj);
            vecPos.divideScalar(vecPos.w);
            return new THREE.Vector3(vecPos.x, vecPos.y, vecPos.z);
        });
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
/**
 * @description renders the enviournment
 * @param env environment
 */
export function RenderEnvironment(env) {
    env.renderer.render(env.scene, env.camera);
}
/**
 *
 * @param imagePoints @description this convers image points such that we can input point to get projected point
 * @param width
 * @param height
 */
export function AdjustedPointFromImagePoints(imagePoints, width, height) {
    var ratio = width / height;
    var changeRatio = globalPrecisionFactor / height;
    var adjustedPoints = imagePoints.map((val) => { return [val[0] * changeRatio - (changeRatio * width / 2 - globalPrecisionFactor / 2), val[1] * changeRatio]; }); // all point in image of height global precision
    return adjustedPoints;
}
