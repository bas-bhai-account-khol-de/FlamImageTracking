var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
import * as util from '../utility';
import * as THREE from 'three';
console.log('TransfomrObjectToStickOnImage');
let textureCanvas = document.getElementById('orignalImageCanvas');
let renderCanvas = document.getElementById('trasnformedImage');
let controllerCanvs = document.getElementById('controller');
let env = util.Create3DScene(renderCanvas);
let controlEnv = util.Create3DScene(controllerCanvs);
env.renderer.setClearColor('blue'); // set clear color of canvs
env.camera.translateZ(1.0); // move camera one unit
controlEnv.renderer.setClearColor('blue'); // set clear color of canvs
controlEnv.camera.translateZ(1.0); // move camera one unit
let points = [];
// add a image as tedxture on canvas
util.addTextureOnCanvas(textureCanvas, 'https://storage.googleapis.com/avatar-system/test/image-noise.jpg', () => {
    var ctx = textureCanvas.getContext('2d');
    points = util.generateNPointsNormalized();
    var w = textureCanvas.width;
    var h = textureCanvas.height;
    for (var i = 0; i < points.length; i++) {
        ctx === null || ctx === void 0 ? void 0 : ctx.beginPath();
        ctx === null || ctx === void 0 ? void 0 : ctx.arc(w * points[i][0] / util.globalPrecisionFactor, h * points[i][1] / util.globalPrecisionFactor, 3, 0, Math.PI * 2);
        ctx.fillStyle = 'blue';
        ctx === null || ctx === void 0 ? void 0 : ctx.fill();
        ctx === null || ctx === void 0 ? void 0 : ctx.closePath();
    }
    var threeDpoints = util.getReprojectedPointsAfterTrasnform(env, points);
    threeDpoints.forEach((pos) => { util.putASphereInEnvironment(env, 0.005, pos); });
    console.log('displayed poinsts in 3d', threeDpoints);
});
/**
 * @abstract this function adds a perfectly fitting plane in scen if camera is 1 unit awaay
 * @param textureURL url of texture to add
 * @param env Enviournment type 3d enviournment
 * @returns plane
 */
function addimageToSceneWithTexture(textureURL, env) {
    return __awaiter(this, void 0, void 0, function* () {
        let side = 2 * Math.tan((env.fov / 2) * Math.PI / 180);
        const planeGeometry = new THREE.PlaneGeometry(side, side); // Width, height
        let texture = yield new THREE.TextureLoader().load(textureURL);
        texture.colorSpace = THREE.SRGBColorSpace;
        const planeMaterial = new THREE.MeshBasicMaterial({ map: texture, color: 0xFFFFFF, side: THREE.DoubleSide });
        const plane = new THREE.Mesh(planeGeometry, planeMaterial);
        env.scene.add(plane);
        return plane;
    });
}
let plane;
let plane2;
/**
 * @description preprocess prefor going into animation loop
 */
function PrequisiteAnimate() {
    return __awaiter(this, void 0, void 0, function* () {
        plane = yield addimageToSceneWithTexture('https://storage.googleapis.com/avatar-system/test/image-noise.jpg', env);
        plane2 = yield addimageToSceneWithTexture('https://storage.googleapis.com/avatar-system/test/image-noise.jpg', controlEnv);
        //generate points on image
        animate();
    });
}
/**
 * @description animation loop
 */
function animate() {
    return __awaiter(this, void 0, void 0, function* () {
        env.renderer.render(env.scene, env.camera);
        controlEnv.renderer.render(controlEnv.scene, controlEnv.camera);
        // plane?.rotateY(0.1)
        requestAnimationFrame(animate);
    });
}
PrequisiteAnimate();
