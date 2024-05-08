import { Environment } from '../utilTypes'
import * as util from '../utility'
import * as THREE from 'three'


// const baseImage: string = 'https://storage.googleapis.com/avatar-system/test/white-square-background-d28l8p1i4p1xnysj.jpg'
const baseImage: string = 'https://storage.googleapis.com/avatar-system/test/guide.webp'
console.log('TransfomrObjectToStickOnImage')
let textureCanvas = document.getElementById('orignalImageCanvas') as HTMLCanvasElement
let renderCanvas = document.getElementById('trasnformedImage') as HTMLCanvasElement
let controllerCanvs = document.getElementById('controller') as HTMLCanvasElement
let newImageCanvas = document.getElementById('newImageCanvas') as HTMLCanvasElement
let newrePorjectedImage = document.getElementById('newrePorjectedImage') as HTMLCanvasElement
let orignalIn3D = document.getElementById('orignalIn3D') as HTMLCanvasElement
let env = util.Create3DScene(renderCanvas)
let controlEnv = util.Create3DScene(controllerCanvs)
let newrePorjectedImageEnv = util.Create3DScene(newrePorjectedImage)
let orignalIn3DEnv = util.Create3DScene(orignalIn3D)


env.renderer.setClearColor('red') // set clear color of canvs
env.camera.translateZ(0.0) // move camera one unit
controlEnv.renderer.setClearColor('blue') // set clear color of canvs
controlEnv.camera.translateZ(0.0) // move camera one unit


let points: number[][] = []
let controlTransformMat: THREE.Matrix4 = new THREE.Matrix4()
// add a image as tedxture on canvas
async function setUpImages() {
    controlTransformMat.makeRotationY(Math.random())

    controlTransformMat.multiply(new THREE.Matrix4().makeRotationX(Math.random())).multiply(new THREE.Matrix4().makeRotationZ(Math.random())).multiply(new THREE.Matrix4().makeTranslation(Math.random()-0.5,Math.random()-0.5,-1*Math.random()));
    //with traslation
    // controlTransformMat.multiply(new THREE.Matrix4().makeRotationX(Math.random())).multiply(new THREE.Matrix4().makeRotationZ(Math.random())).multiply(new THREE.Matrix4().makeTranslation(2*Math.random()-1,2*Math.random()-1,-1*Math.random()))
    controlTransformMat.premultiply(new THREE.Matrix4().makeTranslation(0, 0, -1)) // needed to shift points by 1
   
    util.addTextureOnCanvas(textureCanvas, baseImage,

        () => { // add rando points on image
            var ctx = textureCanvas.getContext('2d')
            points = util.generateNPointsNormalized();
            document.getElementById('originalpoint')!.innerHTML= points.join('<br>')
            var w = textureCanvas.width
            var h = textureCanvas.height
            for (var i = 0; i < points.length; i++) {
                ctx?.beginPath();
                ctx?.arc(w * points[i][0] / util.globalPrecisionFactor, h * points[i][1] / util.globalPrecisionFactor, 3, 0, Math.PI * 2);
                ctx!.fillStyle = 'blue';
                ctx?.fill();
                ctx?.closePath();
            }

            var threeDpoints = util.getReprojectedPointsAfterTrasnform(env, points, controlTransformMat, false)
            threeDpoints.forEach((pos) => { util.putASphereInEnvironment(env, 0.01, pos) });


            //draw original point
            plane3?.translateZ(-1)
            
            var threeDpointsOrignal = util.getReprojectedPointsAfterTrasnform(orignalIn3DEnv, points, new THREE.Matrix4(), false)
            threeDpointsOrignal.forEach((pos) => { pos.z = -1; util.putASphereInEnvironment(orignalIn3DEnv, 0.01, pos) });

            setTimeout(() => { UpdateImageAfterTranform() }, 1000)



        });


}
function UpdateImageAfterTranform() {
    controlEnv.renderer.render(controlEnv.scene, controlEnv.camera);

    util.addTextureOnCanvas(newImageCanvas, controlEnv.renderer.domElement.toDataURL(), async () => {


        var s = 2 * Math.atan((env.fov / 2) * Math.PI / 180)
        var threeDpoints = util.getReprojectedPointsAfterTrasnform(env, points, controlTransformMat, true)
        // convert points on screen
        var pointsToScreen: number[][] = []
        threeDpoints.forEach((val) => {
            pointsToScreen.push([((val.x / s) + 0.5), (0.5 - (val.y / s))])
        })

        
        
        var w = newImageCanvas.width
        var h = newImageCanvas.height
        var ctx = newImageCanvas.getContext('2d')
        for (var i = 0; i < pointsToScreen.length; i++) {
           try{ ctx?.beginPath();
            ctx?.arc( h* pointsToScreen[i][0] - (h-w)/2, h * pointsToScreen[i][1], 3, 0, Math.PI * 2);
            ctx!.fillStyle = 'blue';
            ctx?.fill();
            ctx?.closePath();}catch{}
        }

        controlEnv.renderer.render(controlEnv.scene, controlEnv.camera);
        var newReprojectedPlane = await addimageToSceneWithTexture(controlEnv.renderer.domElement.toDataURL(), newrePorjectedImageEnv, 1)
        newReprojectedPlane?.translateZ(-1);
        newReprojectedPlane?.scale.set(controllerCanvs.width/controllerCanvs.height,1,1)
        var truth:number[][]=[]
        truth = pointsToScreen
        truth = truth.map((val) => { return [Math.floor(util.globalPrecisionFactor * val[0]), Math.floor(util.globalPrecisionFactor * val[1])] })
        pointsToScreen = pointsToScreen.map((val) => { return [Math.floor( h* val[0] - (h-w)/2), Math.floor(h * val[1])] })
        document.getElementById('transformedpoint')!.innerHTML = pointsToScreen.join('<br>');
        pointsToScreen = util.AdjustedPointFromImagePoints(pointsToScreen,newImageCanvas.width,newImageCanvas.height)
        console.log(truth)
        var threeDpoints = util.getReprojectedPointsAfterTrasnform(newrePorjectedImageEnv, pointsToScreen, new THREE.Matrix4(), false)
        threeDpoints.forEach((pos) => { pos.z = -1; util.putASphereInEnvironment(newrePorjectedImageEnv, 0.01, pos) });


    })
}



/**
 * @abstract this function adds a perfectly fitting plane in scen if camera is 1 unit awaay
 * @param textureURL url of texture to add 
 * @param env Enviournment type 3d enviournment
 * @returns plane 
 */
async function addimageToSceneWithTexture(textureURL: string, env: Environment, opacity: number = 1): Promise<THREE.Mesh> {
    let side = 2 * Math.tan((env.fov / 2) * Math.PI / 180)
    const planeGeometry = new THREE.PlaneGeometry(side, side); // Width, height
    let texture = await new THREE.TextureLoader().load(textureURL);
    texture.colorSpace = THREE.SRGBColorSpace
    const planeMaterial = new THREE.MeshBasicMaterial({ map: texture, color: 0xFFFFFF, side: THREE.DoubleSide, transparent: true, opacity: opacity });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    env.scene.add(plane)
    return plane
}









let plane: THREE.Mesh | null
let plane2: THREE.Mesh | null
let plane3: THREE.Mesh | null


/**
 * @description preprocess prefor going into animation loop
 */
async function PrequisiteAnimate() {
    plane = await addimageToSceneWithTexture(baseImage, env)
    plane2 = await addimageToSceneWithTexture(baseImage, controlEnv, 1)
    plane3 = await addimageToSceneWithTexture(baseImage, orignalIn3DEnv, 1)


    setUpImages()
    //get temporary transform
    plane2.applyMatrix4(controlTransformMat)
    // plane.visible=false
    plane?.applyMatrix4(controlTransformMat)




    animate()
}





/**
 * @description animation loop 
 */
async function animate() {
    util.RenderEnvironment(env)
    util.RenderEnvironment(controlEnv)
    util.RenderEnvironment(newrePorjectedImageEnv)
    util.RenderEnvironment(orignalIn3DEnv)

    requestAnimationFrame(animate)


}


PrequisiteAnimate()