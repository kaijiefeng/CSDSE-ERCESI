Network torchvision.models.resnet {
Layer Conv2d-1 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 64, C: 3, R: 7, S: 7, Y: 224, X: 224 }
}
Layer Conv2d-2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 64, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 56, X: 56 }
}
Layer Conv2d-4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 64, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-5 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 64, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-6 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 256, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-7 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 56, X: 56 }
}
Layer Conv2d-8 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 64, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-9 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 256, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-10 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 64, C: 64, R: 3, S: 3, Y: 56, X: 56 }
}
Layer Conv2d-11 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 64, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-12 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 256, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-13 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 56, X: 56 }
}
Layer Conv2d-14 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 128, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-15 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 512, C: 256, R: 1, S: 1, Y: 56, X: 56 }
}
Layer Conv2d-16 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 512, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-17 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 28, X: 28 }
}
Layer Conv2d-18 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 128, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-19 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 512, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-20 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 28, X: 28 }
}
Layer Conv2d-21 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 128, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-22 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 512, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-23 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 128, C: 128, R: 3, S: 3, Y: 28, X: 28 }
}
Layer Conv2d-24 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 128, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-25 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 512, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-26 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 28, X: 28 }
}
Layer Conv2d-27 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-28 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 1024, C: 512, R: 1, S: 1, Y: 28, X: 28 }
}
Layer Conv2d-29 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-30 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-31 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-32 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-33 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-34 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-35 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-36 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-37 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-38 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-39 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-40 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-41 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-42 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-43 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-44 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-45 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-46 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-47 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-48 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-49 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-50 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-51 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-52 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-53 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-54 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-55 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-56 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-57 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-58 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-59 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-60 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-61 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-62 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-63 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-64 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-65 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-66 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-67 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-68 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-69 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-70 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-71 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-72 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-73 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-74 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-75 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-76 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-77 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-78 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-79 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-80 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-81 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-82 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-83 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-84 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-85 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-86 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-87 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-88 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-89 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-90 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-91 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-92 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-93 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 256, C: 256, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-94 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 1024, C: 256, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-95 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-96 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 14, X: 14 }
}
Layer Conv2d-97 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 2048, C: 512, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Conv2d-98 {
Type: CONV
Stride { X: 2, Y: 2 }
Dimensions { K: 2048, C: 1024, R: 1, S: 1, Y: 14, X: 14 }
}
Layer Conv2d-99 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 2048, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Conv2d-100 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 7, X: 7 }
}
Layer Conv2d-101 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 2048, C: 512, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Conv2d-102 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 2048, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Conv2d-103 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 512, C: 512, R: 3, S: 3, Y: 7, X: 7 }
}
Layer Conv2d-104 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 2048, C: 512, R: 1, S: 1, Y: 7, X: 7 }
}
Layer Linear-105 {
Type: CONV
Dimensions { K: 1000, C: 2048, R: 1, S: 1, Y: 1, X: 1 }
}
}