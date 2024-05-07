import { Environment } from '../utilTypes'
import * as util from '../utility'
import * as THREE from 'three'

console.log('TransfomrObjectToStickOnImage')
let textureCanvas = document.getElementById('orignalImageCanvas') as HTMLCanvasElement
let renderCanvas = document.getElementById('trasnformedImage') as HTMLCanvasElement
let controllerCanvs = document.getElementById('controller') as HTMLCanvasElement
let env = util.Create3DScene(renderCanvas)
let controlEnv = util.Create3DScene(controllerCanvs)
env.renderer.setClearColor('blue') // set clear color of canvs
env.camera.translateZ(0.0) // move camera one unit
controlEnv.renderer.setClearColor('blue') // set clear color of canvs
controlEnv.camera.translateZ(0.0) // move camera one unit


let points = []
let controlTransformMat: THREE.Matrix4 = new THREE.Matrix4()
// add a image as tedxture on canvas
util.addTextureOnCanvas(textureCanvas, 'https://storage.googleapis.com/avatar-system/test/image-noise.jpg',

    () => { // add rando points on image
        var ctx = textureCanvas.getContext('2d')
        points = util.generateNPointsNormalized();
        var w = textureCanvas.width
        var h = textureCanvas.height
        for (var i = 0; i < points.length; i++) {
            ctx?.beginPath();
            ctx?.arc(w * points[i][0] / util.globalPrecisionFactor, h * points[i][1] / util.globalPrecisionFactor, 3, 0, Math.PI * 2);
            ctx!.fillStyle = 'blue';
            ctx?.fill();
            ctx?.closePath();
        }
        controlTransformMat.makeRotationY(0.5)
        
        controlTransformMat.multiply(new THREE.Matrix4().makeRotationX(0.5))
        controlTransformMat.premultiply(new THREE.Matrix4().makeTranslation(0,0,-1)) // needed to shift points by 1
        var threeDpoints = util.getReprojectedPointsAfterTrasnform(env, points,controlTransformMat)
        threeDpoints.forEach((pos) => { util.putASphereInEnvironment(env, 0.01, pos) });
        
    });



/**
 * @abstract this function adds a perfectly fitting plane in scen if camera is 1 unit awaay
 * @param textureURL url of texture to add 
 * @param env Enviournment type 3d enviournment
 * @returns plane 
 */
async function addimageToSceneWithTexture(textureURL: string, env: Environment): Promise<THREE.Mesh> {
    let side = 2 * Math.tan((env.fov / 2) * Math.PI / 180)
    const planeGeometry = new THREE.PlaneGeometry(side, side); // Width, height
    let texture = await new THREE.TextureLoader().load(textureURL);
    texture.colorSpace = THREE.SRGBColorSpace
    const planeMaterial = new THREE.MeshBasicMaterial({ map: texture, color: 0xFFFFFF, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    env.scene.add(plane)
    return plane
}









let plane: THREE.Mesh | null
let plane2: THREE.Mesh | null


/**
 * @description preprocess prefor going into animation loop
 */
async function PrequisiteAnimate() {
    plane = await addimageToSceneWithTexture('https://storage.googleapis.com/avatar-system/test/image-noise.jpg', env)
    plane2 = await addimageToSceneWithTexture('https://storage.googleapis.com/avatar-system/test/image-noise.jpg', controlEnv)

    //get temporary transform
    plane2.applyMatrix4(controlTransformMat)
    plane?.applyMatrix4(controlTransformMat)
    
    


    animate()
}





/**
 * @description animation loop 
 */
async function animate() {
    env.renderer.render(env.scene, env.camera);
    controlEnv.renderer.render(controlEnv.scene, controlEnv.camera);

    plane?.applyMatrix4(plane?.matrixWorld.invert())
    plane2?.applyMatrix4(plane2?.matrixWorld.invert())
    plane?.applyMatrix4(controlTransformMat)
    plane2?.applyMatrix4(controlTransformMat)
    requestAnimationFrame(animate)
}


PrequisiteAnimate()