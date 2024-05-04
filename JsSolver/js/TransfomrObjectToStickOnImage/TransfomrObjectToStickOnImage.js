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
let renderCanvas = util.addCanvasOfSize(document);
let env = util.Create3DScene(renderCanvas);
env.renderer.setClearColor('blue'); // set clear color of canvs
env.camera.translateZ(5);
function addimageToSceneWithTexture(textureURL, env) {
    return __awaiter(this, void 0, void 0, function* () {
        const planeGeometry = new THREE.PlaneGeometry(5, 5); // Width, height
        let texture = yield new THREE.TextureLoader().load(textureURL);
        const planeMaterial = new THREE.MeshBasicMaterial({ map: texture, color: 0xcccccc, side: THREE.DoubleSide });
        const plane = new THREE.Mesh(planeGeometry, planeMaterial);
        env.scene.add(plane);
        return plane;
    });
}
let plane;
function PrequisiteAnimate() {
    return __awaiter(this, void 0, void 0, function* () {
        plane = yield addimageToSceneWithTexture('https://storage.googleapis.com/avatar-system/test/image-noise.jpg', env);
        animate();
    });
}
function animate() {
    return __awaiter(this, void 0, void 0, function* () {
        env.renderer.render(env.scene, env.camera);
        plane === null || plane === void 0 ? void 0 : plane.rotateY(0.1);
        requestAnimationFrame(animate);
    });
}
PrequisiteAnimate();
