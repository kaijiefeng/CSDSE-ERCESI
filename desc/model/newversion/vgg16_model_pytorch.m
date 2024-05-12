Network torchvision.models.vgg {
Layer Conv2d-1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 3, R: 3, S: 3, Y: 224, X: 224 }
}
Layer Conv2d-2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 224, X: 224 }
}
Layer Conv2d-3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 64, R: 3, S: 3, Y: 112, X: 112 }
}
Layer Conv2d-4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 112, X: 112 }
}
Layer Conv2d-5 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 128, R: 3, S: 3, Y: 56, X: 56 }
}
Layer Conv2d-6 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 56, X: 56 }
}
Layer Conv2d-7 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 56, X: 56 }
}
Layer Conv2d-8 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 256, R: 3, S: 3, Y: 28, X: 28 }
}
Layer Conv2d-9 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 28, X: 28 }
}
Layer Conv2d-10 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 28, X: 28 }
}
Layer Conv2d-11 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-12 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-13 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Linear-14 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 4096, C: 25088, R: 1, S: 1, Y: 1, X: 1 }
}
Layer Linear-15 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 4096, C: 4096, R: 1, S: 1, Y: 1, X: 1 }
}
Layer Linear-16 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1000, C: 4096, R: 1, S: 1, Y: 1, X: 1 }
}
}