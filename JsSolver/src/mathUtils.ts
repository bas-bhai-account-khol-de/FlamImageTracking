import { Vector3 } from "three";

export function getVec3PointsInArray(points:Vector3[]):number[][]{

    var convertedArray :number[][]= [];
    for(var i=0;i<points.length;i++){
        convertedArray.push([points[i].x,points[i].y,points[i].z,1]);
    }
    return convertedArray;
}