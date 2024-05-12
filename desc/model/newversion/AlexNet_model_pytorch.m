Network torchvision.models.alexnet {
Layer Conv2d-1 {
Type: CONV
Stride { X: 4, Y: 4 }
Dimensions { K: 64, C: 3, R: 11, S: 11, Y: 224, X: 224 }
}
Layer Conv2d-2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 192, C: 64, R: 5, S: 5, Y: 27, X: 27 }
}
Layer Conv2d-3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 384, C: 192, R: 3, S: 3, Y: 13, X: 13 }
}
Layer Conv2d-4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 384, R: 3, S: 3, Y: 13, X: 13 }
}
Layer Conv2d-5 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 13, X: 13 }
}
Layer Linear-6 {
Type: CONV
Dimensions { K: 4096, C: 9216, R: 1, S: 1, Y: 1, X: 1 }
}
Layer Linear-7 {
Type: CONV
Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1, X: 1 }
}
Layer Linear-8 {
Type: CONV
Dimensions { K: 1000, C: 4096, R: 1, S: 1, Y: 1, X: 1 }
}
}