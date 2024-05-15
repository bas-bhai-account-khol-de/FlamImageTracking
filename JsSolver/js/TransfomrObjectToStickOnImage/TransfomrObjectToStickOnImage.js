var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
import { getVec3PointsInArray } from '../mathUtils';
import * as util from '../utility';
import * as THREE from 'three';
import * as m from 'mathjs';
// const baseImage: string = 'https://storage.googleapis.com/avatar-system/test/white-square-background-d28l8p1i4p1xnysj.jpg'
const baseImage = 'https://storage.googleapis.com/avatar-system/test/guide.webp';
console.log('TransfomrObjectToStickOnImage');
let textureCanvas = document.getElementById('orignalImageCanvas');
let renderCanvas = document.getElementById('trasnformedImage');
let controllerCanvs = document.getElementById('controller');
let newImageCanvas = document.getElementById('newImageCanvas');
let newrePorjectedImage = document.getElementById('newrePorjectedImage');
let orignalIn3D = document.getElementById('orignalIn3D');
let env = util.Create3DScene(renderCanvas);
let controlEnv = util.Create3DScene(controllerCanvs);
let newrePorjectedImageEnv = util.Create3DScene(newrePorjectedImage);
let orignalIn3DEnv = util.Create3DScene(orignalIn3D);
//solving transform
var solvedCanvas = document.getElementById('solvedTrasnform');
var solvedCanvasEnv = util.Create3DScene(solvedCanvas);
env.renderer.setClearColor('red'); // set clear color of canvs
env.camera.translateZ(0.0); // move camera one unit
controlEnv.renderer.setClearColor('blue'); // set clear color of canvs
controlEnv.camera.translateZ(0.0); // move camera one unit
let points = [];
let controlTransformMat = new THREE.Matrix4();
let OrignalPoints = [];
let FinalPoints = [];
// add a image as tedxture on canvas
function setUpImages() {
    return __awaiter(this, void 0, void 0, function* () {
        controlTransformMat.makeRotationY(0.5);
        // controlTransformMat.makeRotationY(Math.random())
        // controlTransformMat.multiply(new THREE.Matrix4().makeRotationX(Math.random())).multiply(new THREE.Matrix4().makeRotationZ(Math.random())).multiply(new THREE.Matrix4().makeTranslation(Math.random()-0.5,Math.random()-0.5,-1*Math.random()));
        //with traslation
        // controlTransformMat.multiply(new THREE.Matrix4().makeRotationX(Math.random())).multiply(new THREE.Matrix4().makeRotationZ(Math.random())).multiply(new THREE.Matrix4().makeTranslation(2*Math.random()-1,2*Math.random()-1,-1*Math.random()))
        controlTransformMat.premultiply(new THREE.Matrix4().makeTranslation(0, 0, -1)); // needed to shift points by 1
        util.addTextureOnCanvas(textureCanvas, baseImage, () => {
            var ctx = textureCanvas.getContext('2d');
            points = util.generateNPointsNormalized();
            // document.getElementById('originalpoint')!.innerHTML= points.join('<br>')
            document.getElementById('originalpoint').innerHTML = points.join('|');
            document.getElementById('originalpoint').style.overflow = 'hidden';
            document.getElementById('originalpoint').style.textOverflow = 'ellipsis';
            document.getElementById('originalpoint').style.whiteSpace = 'nowrap';
            // document.getElementById('originalpoint')!.style.height='20';
            document.getElementById('originalpoint').style.width = '200px';
            var w = textureCanvas.width;
            var h = textureCanvas.height;
            for (var i = 0; i < points.length; i++) {
                ctx === null || ctx === void 0 ? void 0 : ctx.beginPath();
                ctx === null || ctx === void 0 ? void 0 : ctx.arc(w * points[i][0] / util.globalPrecisionFactor, h * points[i][1] / util.globalPrecisionFactor, 3, 0, Math.PI * 2);
                ctx.fillStyle = 'blue';
                ctx === null || ctx === void 0 ? void 0 : ctx.fill();
                ctx === null || ctx === void 0 ? void 0 : ctx.closePath();
            }
            var threeDpoints = util.getReprojectedPointsAfterTrasnform(env, points, controlTransformMat, false);
            threeDpoints.forEach((pos) => { util.putASphereInEnvironment(env, 0.01, pos); });
            console.log('3d point', threeDpoints);
            //draw original point
            plane3 === null || plane3 === void 0 ? void 0 : plane3.translateZ(-1);
            var threeDpointsOrignal = util.getReprojectedPointsAfterTrasnform(orignalIn3DEnv, points, new THREE.Matrix4(), false);
            threeDpointsOrignal.forEach((pos) => { pos.z = -1; util.putASphereInEnvironment(orignalIn3DEnv, 0.01, pos); });
            setTimeout(() => { UpdateImageAfterTranform(); }, 1000);
            console.log("Tranform [" + controlTransformMat.elements.join(',') + ']');
            OrignalPoints = getVec3PointsInArray(threeDpointsOrignal);
            console.log('orignal [' + OrignalPoints.join(',') + ']');
        });
    });
}
function UpdateImageAfterTranform() {
    controlEnv.renderer.render(controlEnv.scene, controlEnv.camera);
    util.addTextureOnCanvas(newImageCanvas, controlEnv.renderer.domElement.toDataURL(), () => __awaiter(this, void 0, void 0, function* () {
        var s = 2 * Math.atan((env.fov / 2) * Math.PI / 180);
        var threeDpoints = util.getReprojectedPointsAfterTrasnform(env, points, controlTransformMat, true);
        // convert points on image
        var pointsToScreen = [];
        threeDpoints.forEach((val) => {
            pointsToScreen.push([((val.x / s) + 0.5), (0.5 - (val.y / s))]);
        });
        var w = newImageCanvas.width;
        var h = newImageCanvas.height;
        var ctx = newImageCanvas.getContext('2d');
        for (var i = 0; i < pointsToScreen.length; i++) {
            try {
                ctx === null || ctx === void 0 ? void 0 : ctx.beginPath();
                ctx === null || ctx === void 0 ? void 0 : ctx.arc(h * pointsToScreen[i][0] - (h - w) / 2, h * pointsToScreen[i][1], 3, 0, Math.PI * 2);
                ctx.fillStyle = 'green';
                ctx === null || ctx === void 0 ? void 0 : ctx.fill();
                ctx === null || ctx === void 0 ? void 0 : ctx.closePath();
            }
            catch (_a) { }
        }
        controlEnv.renderer.render(controlEnv.scene, controlEnv.camera);
        var newReprojectedPlane = yield addimageToSceneWithTexture(controlEnv.renderer.domElement.toDataURL(), newrePorjectedImageEnv, 1);
        newReprojectedPlane === null || newReprojectedPlane === void 0 ? void 0 : newReprojectedPlane.translateZ(-1);
        newReprojectedPlane === null || newReprojectedPlane === void 0 ? void 0 : newReprojectedPlane.scale.set(controllerCanvs.width / controllerCanvs.height, 1, 1);
        pointsToScreen = pointsToScreen.map((val) => { return [Math.floor(h * val[0] - (h - w) / 2), Math.floor(h * val[1])]; });
        document.getElementById('transformedpoint').innerHTML = pointsToScreen.join('|');
        // document.getElementById('transformedpoint')!.innerHTML = pointsToScreen.join('<br>');
        document.getElementById('transformedpoint').style.overflow = 'hidden';
        document.getElementById('transformedpoint').style.textOverflow = 'ellipsis';
        document.getElementById('transformedpoint').style.whiteSpace = 'nowrap';
        // document.getElementById('transformedpoint')!.style.height='20';
        document.getElementById('transformedpoint').style.width = '200px';
        pointsToScreen = util.AdjustedPointFromImagePoints(pointsToScreen, newImageCanvas.width, newImageCanvas.height);
        var threeDpointsnew = util.getReprojectedPointsAfterTrasnform(newrePorjectedImageEnv, pointsToScreen, new THREE.Matrix4(), false);
        threeDpointsnew.forEach((pos) => { pos.z = -1; util.putASphereInEnvironment(newrePorjectedImageEnv, 0.01, pos); });
        FinalPoints = getVec3PointsInArray(threeDpointsnew);
        console.log('final [' + FinalPoints.join(',') + ']');
        //after we have final point we need to calculate the tranform again
        // var x   = m.transpose( m.matrix(OrignalPoints))
        // var y   = m.transpose( m.matrix(FinalPoints))
        // var x_inv = m.pinv(x)
        // var pt =m.multiply(y,x_inv)
        // pt =m.subset(pt,m.index(3,[0,1,2,3]),[0,0,0,1])
        // console.log('PT after last row zero',pt.size(),pt)
        // var element =[]
        // for(var i=0;i<16;i++){
        //     element.push(pt.get([Math.floor(i/4),i%4]))
        // }
        // var solvedMat  =  new THREE.Matrix4().fromArray(element).transpose();
        // console.log(solvedMat)
        // var solvedPlane =await addimageToSceneWithTexture(baseImage, solvedCanvasEnv)
        // solvedPlane?.translateZ(-1);
        // solvedPlane.applyMatrix4(solvedMat)
        //using angle and dixtace
        //angle first 
        var o = OrignalPoints;
        var f = FinalPoints;
        var mat_rows = [];
        var sol_rows = [];
        function x(n) { return o[n][0]; }
        function y(n) { return o[n][1]; }
        function X(n) { return f[n][0]; }
        function Y(n) { return f[n][1]; }
        //p2->p1  p2->p3
        mat_rows.push([0, Math.pow(X(2), 2) + Math.pow(Y(2), 2) + 1, 0, -1 * (X(2) * X(1) + Y(2) * Y(1) + 1), -1 * (X(2) * X(3) + Y(2) * Y(3) + 1), (X(1) * X(3) + Y(1) * Y(3) + 1)]);
        sol_rows.push([x(1) * x(3) + x(2) * x(2) - x(2) * x(1) - x(3) * x(2) + y(1) * y(3) + y(2) * y(2) - y(1) * y(2) - y(3) * y(2)]);
        //p3->p1  p3->p2
        mat_rows.push([0, 0, Math.pow(X(3), 2) + Math.pow(Y(3), 2) + 1, 1 * (1 + X(2) * X(1) + Y(2) * Y(1)), -1 * (1 + X(2) * X(3) + Y(2) * Y(3)), -1 * (1 + X(1) * X(3) + Y(1) * Y(3))]);
        sol_rows.push([-1 * x(1) * x(3) + x(3) * x(3) + x(2) * x(1) - x(3) * x(2) - y(1) * y(3) + y(3) * y(3) + y(1) * y(2) - y(3) * y(2)]);
        //p1->p2  p1->p3
        mat_rows.push([Math.pow(X(1), 2) + Math.pow(Y(1), 2) + 1, 0, 0, -1 * (X(2) * X(1) + Y(2) * Y(1) + 1), 1 * (X(2) * X(3) + Y(2) * Y(3) + 1), -1 * (X(1) * X(3) + Y(1) * Y(3) + 1)]);
        sol_rows.push([-1 * x(1) * x(3) + x(1) * x(1) - x(2) * x(1) + x(3) * x(2) - y(1) * y(3) + y(1) * y(1) - y(1) * y(2) + y(3) * y(2)]);
        //ditance 
        // p1 -> p2
        mat_rows.push([(Math.pow(X(1), 2) + Math.pow(Y(1), 2)) + 1, Math.pow(X(2), 2) + Math.pow(Y(2), 2) + 1, 0, -2 * (X(1) * X(2) + Y(1) * Y(2) + 1), 0, 0]);
        sol_rows.push([Math.pow(x(1), 2) + Math.pow(x(2), 2) + Math.pow(y(1), 2) + Math.pow(y(2), 2) - 2 * (x(1) * x(2) + y(1) * y(2))]);
        //p2->p3
        mat_rows.push([0, Math.pow(X(2), 2) + Math.pow(Y(2), 2) + 1, (Math.pow(X(3), 2) + Math.pow(Y(3), 2)) + 1, 0, -2 * (X(3) * X(2) + Y(3) * Y(2) + 1), 0]);
        sol_rows.push([Math.pow(x(3), 2) + Math.pow(x(2), 2) + Math.pow(y(3), 2) + Math.pow(y(2), 2) - 2 * (x(3) * x(2) + y(3) * y(2))]);
        //p3->p1
        mat_rows.push([(Math.pow(X(1), 2) + Math.pow(Y(1), 2)) + 1, 0, (Math.pow(X(3), 2) + Math.pow(Y(3), 2)) + 1, 0, 0, -2 * (X(3) * X(1) + Y(3) * Y(1) + 1)]);
        sol_rows.push([Math.pow(x(3), 2) + Math.pow(x(1), 2) + Math.pow(y(3), 2) + Math.pow(y(1), 2) - 2 * (x(3) * x(1) + y(3) * y(1))]);
        var A = m.matrix(mat_rows);
        var B = m.matrix(sol_rows);
        //AX=B
        console.log(A);
        console.log(B);
        var A_inv = m.pinv(A);
        var z = m.multiply(A_inv, B);
        console.log(z);
    }));
}
/**
 * @abstract this function adds a perfectly fitting plane in scen if camera is 1 unit awaay
 * @param textureURL url of texture to add
 * @param env Enviournment type 3d enviournment
 * @returns plane
 */
function addimageToSceneWithTexture(textureURL, env, opacity = 1) {
    return __awaiter(this, void 0, void 0, function* () {
        let side = 2 * Math.tan((env.fov / 2) * Math.PI / 180);
        const planeGeometry = new THREE.PlaneGeometry(side, side); // Width, height
        let texture = yield new THREE.TextureLoader().load(textureURL);
        texture.colorSpace = THREE.SRGBColorSpace;
        const planeMaterial = new THREE.MeshBasicMaterial({ map: texture, color: 0xFFFFFF, side: THREE.DoubleSide, transparent: true, opacity: opacity });
        const plane = new THREE.Mesh(planeGeometry, planeMaterial);
        env.scene.add(plane);
        return plane;
    });
}
let plane;
let plane2;
let plane3;
/**
 * @description preprocess prefor going into animation loop
 */
function PrequisiteAnimate() {
    return __awaiter(this, void 0, void 0, function* () {
        plane = yield addimageToSceneWithTexture(baseImage, env);
        plane2 = yield addimageToSceneWithTexture(baseImage, controlEnv, 1);
        plane3 = yield addimageToSceneWithTexture(baseImage, orignalIn3DEnv, 1);
        setUpImages();
        //get temporary transform
        plane2.applyMatrix4(controlTransformMat);
        // plane.visible=false
        plane === null || plane === void 0 ? void 0 : plane.applyMatrix4(controlTransformMat);
        animate();
    });
}
/**
 * @description animation loop
 */
function animate() {
    return __awaiter(this, void 0, void 0, function* () {
        util.RenderEnvironment(env);
        util.RenderEnvironment(controlEnv);
        util.RenderEnvironment(newrePorjectedImageEnv);
        util.RenderEnvironment(orignalIn3DEnv);
        util.RenderEnvironment(solvedCanvasEnv);
        requestAnimationFrame(animate);
    });
}
PrequisiteAnimate();
