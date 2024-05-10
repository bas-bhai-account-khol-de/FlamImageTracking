export function getVec3PointsInArray(points) {
    var convertedArray = [];
    for (var i = 0; i < points.length; i++) {
        convertedArray.push([points[i].x, points[i].y, points[i].z, 1]);
    }
    return convertedArray;
}
