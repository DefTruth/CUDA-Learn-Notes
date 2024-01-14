# CUDA高频面试题汇总/C++笔记/CUDA笔记  

个人笔记，不定期更新...

## C++笔记  
- [cpp](./cpp)
- [[C++][3W字]💡静态链接和静态库实践指北-原理篇](https://zhuanlan.zhihu.com/p/595527528) 

## CUDA高频面试题汇总  

前段时间参加了一些面试，大部分都要手撕CUDA，因此也整体复习了一遍CUDA优化相关的内容，整理了一些高频题的基本写法，保存在这里也便于日后自己复习，具体见[CUDA高频面试题汇总](./cuda-check/check.cu)。当然，有些代码不一定是最优化解，比如GEMM，想要在面试短短的30分钟内写一个好的GEMM Kernel，那实在是太难了，普通人能写个shared memory + block-tile + k-tile 的版本的很不错了。相关kernel如下：  

- sgemm naive, sgemm + block-tile + k-tile + vec4
- sgemv k32/k128/k16 kernel
- warp/block reduce sum/max, block all reduce + vec4
- dot product, dot product + vec4
- elementwise, elementwise + vec4
- histogram, histogram + vec4 
- softmax, softmax + vec4 (grid level memory fence)
- safe softmax, safe softmax + vec4
- sigmoid, sigmoid + vec4
- relu, relu + vec4
- layer_norm, layer_norm + vec4
- rms_norm, rms_norm + vec4
- ....  

不定期更新...
