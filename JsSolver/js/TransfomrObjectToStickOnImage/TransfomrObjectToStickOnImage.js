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
let textureCanvas = util.addCanvasOfSize(document);
let renderCanvas = util.addCanvasOfSize(document);
let env = util.Create3DScene(renderCanvas);
env.renderer.setClearColor('blue'); // set clear color of canvs
env.camera.translateZ(1);
function addTextureOnCanvas() {
    var image = new Image();
    image.height = renderCanvas.height;
    image.width = renderCanvas.width;
    image.src = 'https://storage.googleapis.com/avatar-system/test/image-noise.jpg';
    image.onload = () => { var dr = textureCanvas.getContext('2d'); dr === null || dr === void 0 ? void 0 : dr.drawImage(image, 0, 0, renderCanvas.width, renderCanvas.height); };
}
addTextureOnCanvas();
function addimageToSceneWithTexture(textureURL, env) {
    return __awaiter(this, void 0, void 0, function* () {
        let side = 2 * Math.tan((env.fov / 2) * Math.PI / 180);
        const planeGeometry = new THREE.PlaneGeometry(side, side); // Width, height
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
        //generate points on image
        let points = util.generateNPointsNormalized();
        //  draw the points on screen
        renderCanvas.addEventListener('touchend', () => {
            let ctx = renderCanvas.getContext('webgl2');
            for (let i = 0; i < points.length; i++) {
                console.log('loaded ' + ctx);
                // ctx?.beginPath();
                // ctx?.arc(points[i][0],points[i][1], 6, 0, Math.PI * 2);
                // ctx!.fillStyle = 'blue';
                // ctx?.fill();
                // ctx?.closePath();
            }
        });
        console.log(points);
        animate();
    });
}
function animate() {
    return __awaiter(this, void 0, void 0, function* () {
        env.renderer.render(env.scene, env.camera);
        // plane?.rotateY(0.1)
        requestAnimationFrame(animate);
    });
}
PrequisiteAnimate();
