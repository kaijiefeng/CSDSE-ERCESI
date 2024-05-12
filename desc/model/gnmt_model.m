Network gnmt_gemm {
Layer GEMM0 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 2048, C: 4096, R: 1, S: 1, Y: 128, X: 1 }


}
Layer GEMM1 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 2048, C: 4096, R: 1, S: 1, Y: 128, X: 1 }


}
Layer GEMM2 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 3072, C: 4096, R: 1, S: 1, Y: 320, X: 1 }


}
Layer GEMM3 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 2048, C: 4096, R: 1, S: 1, Y: 128, X: 1 }


}
Layer GEMM4 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 2048, C: 4096, R: 1, S: 1, Y: 128, X: 1 }


}
Layer GEMM5 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 3072, C: 4096, R: 1, S: 1, Y: 320, X: 1 }


}
Layer GEMM6 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 3072, C: 4096, R: 1, S: 1, Y: 320, X: 1 }


}
Layer GEMM7 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 3072, C: 4096, R: 1, S: 1, Y: 320, X: 1 }


}
Layer GEMM8 {
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K: 3072, C: 4096, R: 1, S: 1, Y: 320, X: 1 }


}
}