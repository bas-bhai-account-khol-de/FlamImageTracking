import { Enviroment } from '../utilTypes'
import * as util from '../utility'
import * as THREE from 'three'

console.log('TransfomrObjectToStickOnImage')
let renderCanvas =util.addCanvasOfSize(document)
let env  =  util.Create3DScene(renderCanvas)
env.renderer.setClearColor('blue') // set clear color of canvs
env.camera.translateZ(5)

async function addimageToSceneWithTexture(textureURL : string ,env:Enviroment) :Promise<THREE.Mesh>{
    const planeGeometry = new THREE.PlaneGeometry(5, 5); // Width, height
    let texture  = await new THREE.TextureLoader().load(textureURL);
    const planeMaterial = new THREE.MeshBasicMaterial({ map:texture,color: 0xcccccc, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    env.scene.add(plane)
    return plane
    
}


let plane : THREE.Mesh | null
async function PrequisiteAnimate() {
     plane = await addimageToSceneWithTexture('https://storage.googleapis.com/avatar-system/test/image-noise.jpg',env)
    animate()
}






async function animate (){
    env.renderer.render(env.scene,env.camera);
    plane?.rotateY(0.1)
    requestAnimationFrame(animate)
}


PrequisiteAnimate()