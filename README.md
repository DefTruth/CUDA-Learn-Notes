![cuda-learn-note](https://github.com/DefTruth/CUDA-Learn-Note/assets/31974251/882271fe-ab60-4b0e-9440-2e0fa3c0fb6f)   

<div align='center'>
  <img src=https://img.shields.io/badge/Language-CUDA-brightgreen.svg >
  <img src=https://img.shields.io/github/watchers/DefTruth/cuda-learn-note?color=9cc >
  <img src=https://img.shields.io/github/forks/DefTruth/cuda-learn-note.svg?style=social >
  <img src=https://img.shields.io/github/stars/DefTruth/cuda-learn-note.svg?style=social >
  <img src=https://img.shields.io/badge/Release-v2.0-brightgreen.svg >
  <img src=https://img.shields.io/badge/License-GPLv3.0-turquoise.svg >
 </div>   

📖**CUDA-Learn-Notes**: 🎉CUDA/C++ 笔记 / 技术博客: **fp32、fp16/bf16、fp8/int8**、flash_attn、sgemm、sgemv、warp/block reduce、dot prod、elementwise、softmax、layernorm、rmsnorm、hist etc. 👉News: Most of my time now is focused on **LLM/VLM/Diffusion** Inference. Please check 📖[Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference)  ![](https://img.shields.io/github/stars/DefTruth/Awesome-LLM-Inference.svg?style=social), 📖[Awesome-SD-Inference](https://github.com/DefTruth/Awesome-SD-Inference)  ![](https://img.shields.io/github/stars/DefTruth/Awesome-SD-Inference.svg?style=social) and 📖[CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes)  ![](https://img.shields.io/github/stars/DefTruth/CUDA-Learn-Notes.svg?style=social) for more details.

## 0x00 📖 博客目录

<img width="1438" alt="image" src="https://github.com/user-attachments/assets/0c5e5125-586f-43fa-8e8b-e2c61c1afbbe">

### 📖 大模型|多模态|Diffusion|推理优化 (本人作者)

|📖 类型-标题|📖 作者|
|:---|:---| 
|[[VLM推理优化][InternVL系列]📖InternLM2/.../InternVL1.5系列笔记: 核心点解析](https://zhuanlan.zhihu.com/p/702481058)|@DefTruth|
|[[LLM推理优化][TensorRT-LLM][5w字]📖TensorRT-LLM部署调优-指北](https://zhuanlan.zhihu.com/p/699333691)|@DefTruth|
|[[LLM推理优化][KV Cache优化]📖GQA/YOCO/CLA/MLKV: 层内和层间KV Cache共享](https://zhuanlan.zhihu.com/p/697311739)|@DefTruth|
|[[LLM推理优化][Prefill优化]📖图解vLLM Prefix Prefill Triton Kernel](https://zhuanlan.zhihu.com/p/695799736)|@DefTruth|
|[[LLM推理优化][Prefill优化][万字]📖图解vLLM Automatic Prefix Caching: TTFT优化](https://zhuanlan.zhihu.com/p/693556044)|@DefTruth|
|[[LLM推理优化][Attention优化]📖图解:从Online-Softmax到FlashAttention V1/V2/V3](https://zhuanlan.zhihu.com/p/668888063)|@DefTruth|
|[[LLM推理优化][Decoding优化]📖原理&图解FlashDecoding/FlashDecoding++](https://zhuanlan.zhihu.com/p/696075602)|@DefTruth|
|[[VLM推理优化][LLaVA系列]📖CLIP/LLaVA/LLaVA1.5/VILA笔记: 核心点解析](https://zhuanlan.zhihu.com/p/683137074)|@DefTruth|
|[[LLM推理优化][Attention优化][万字]📖TensorRT MHA/Myelin vs FlashAttention-2](https://zhuanlan.zhihu.com/p/678873216)|@DefTruth|
|[[LLM推理优化][PTX汇编]📖CUDA 12 PTX汇编: PRMT指令详解-通用模式](https://zhuanlan.zhihu.com/p/660630414)|@DefTruth|
|[[LLM推理优化][PTX汇编]📖CUDA 12 PTX汇编: LOP3指令详解](https://zhuanlan.zhihu.com/p/659741469)|@DefTruth|
|[[LLM推理优化][CUDA][3w字]📖高频面试题汇总-大模型手撕CUDA](https://zhuanlan.zhihu.com/p/678903537)|@DefTruth|
|[[LLM推理优化][Weight Only]📖WINT8/4-(00): 通俗易懂讲解-快速反量化算法](https://zhuanlan.zhihu.com/p/657072856)|@DefTruth|
|[[LLM推理优化][Weight Only]📖WINT8/4-(01): PRMT指令详解及FT源码解析](https://zhuanlan.zhihu.com/p/657070837)|@DefTruth|
|[[LLM推理优化][Weight Only]📖WINT8/4-(02): 快速反量化之INT8转BF16](https://zhuanlan.zhihu.com/p/657073159)|@DefTruth|
|[[LLM推理优化][Weight Only]📖WINT8/4-(03): LOP3指令详解及INT4转FP16/BF16](https://zhuanlan.zhihu.com/p/657073857)|@DefTruth|
|[[LLM推理优化][LLM Infra整理]📖100+篇: 大模型推理各方向新发展整理](https://zhuanlan.zhihu.com/p/693680304)|@DefTruth|
|[[LLM推理优化][LLM Infra整理]📖30+篇: LLM推理论文集-500页PDF](https://zhuanlan.zhihu.com/p/669777159)|@DefTruth|
|[[LLM推理优化][LLM Infra整理]📖FlashDecoding++: 比FlashDecoding还要快！](https://zhuanlan.zhihu.com/p/665022589)|@DefTruth|
|[[LLM推理优化][LLM Infra整理]📖TensorRT-LLM开源，TensorRT 9.1也来了](https://zhuanlan.zhihu.com/p/662361469)|@DefTruth|
|[[LLM推理优化][LLM Infra整理]📖20+篇: LLM推理论文集-300页PDF](https://zhuanlan.zhihu.com/p/658091768)|@DefTruth|
|[[LLM推理优化][LLM Infra整理]📖PagedAttention论文新鲜出炉](https://zhuanlan.zhihu.com/p/617015570)|@DefTruth|


### 📖 CV推理部署|C++|算法|技术随笔 (本人作者)

|📖 类型-标题|📖 作者|
|:---|:---| 
| [[推理部署][CV/NLP]📖FastDeploy三行代码搞定150+ CV、NLP模型部署](https://zhuanlan.zhihu.com/p/581326442)|@DefTruth|  
| [[推理部署][CV]📖如何在lite.ai.toolkit(3.6k+ stars)中增加您的模型？](https://zhuanlan.zhihu.com/p/523876625)|@DefTruth|  
| [[推理部署][CV]📖美团 YOLOv6 ORT/MNN/TNN/NCNN C++推理部署](https://zhuanlan.zhihu.com/p/533643238)|@DefTruth|  
| [[推理部署][ONNX]📖ONNX推理加速技术文档-杂记](https://zhuanlan.zhihu.com/p/524023964)|@DefTruth|  
| [[推理部署][TensorFlow]📖Mac源码编译TensorFlow C++指北](https://zhuanlan.zhihu.com/p/524013615)|@DefTruth|  
| [[推理部署][CV]📖1Mb!头部姿态估计: FSANet，一个小而美的模型(C++)](https://zhuanlan.zhihu.com/p/447364201)|@DefTruth|  
| [[推理部署][CV]📖opencv+ffmpeg编译打包全解指南](https://zhuanlan.zhihu.com/p/472115312)|@DefTruth|  
| [[推理部署][CV]📖RobustVideoMatting视频抠图静态ONNX模型转换](https://zhuanlan.zhihu.com/p/459088407)|@DefTruth|  
| [[推理部署][CV]📖190Kb!SSRNet年龄检测详细解读（含C++工程）](https://zhuanlan.zhihu.com/p/462762797)|@DefTruth|  
| [[推理部署][CV]📖MGMatting(CVPR2021)人像抠图C++应用记录](https://zhuanlan.zhihu.com/p/464732042)|@DefTruth|  
| [[推理部署][CV]📖超准确人脸检测(带关键点)YOLO5Face C++工程详细记录](https://zhuanlan.zhihu.com/p/461878005)|@DefTruth|  
| [[推理部署][ORT]📖解决: ONNXRuntime(Python) GPU 部署配置记录](https://zhuanlan.zhihu.com/p/457484536)|@DefTruth|  
| [[推理部署][CV]📖记录SCRFD(CVPR2021)人脸检测C++工程化(含docker镜像)](https://zhuanlan.zhihu.com/p/455165568)|@DefTruth|  
| [[推理部署][NCNN]📖野路子：记录一个解决onnx转ncnn时op不支持的trick](https://zhuanlan.zhihu.com/p/451446147)|@DefTruth|  
| [[推理部署][CV]📖升级版轻量级NanoDet-Plus MNN/TNN/NCNN/ORT C++工程记录](https://zhuanlan.zhihu.com/p/450586647)|@DefTruth|  
| [[推理部署][CV]📖超轻量级NanoDet MNN/TNN/NCNN/ORT C++工程记录](https://zhuanlan.zhihu.com/p/443419387)|@DefTruth|  
| [[推理部署][CV]📖详细记录MGMatting之MNN、TNN和ORT C++移植](https://zhuanlan.zhihu.com/p/442949027)|@DefTruth|  
| [[推理部署][CV]📖YOLOX NCNN/MNN/TNN/ONNXRuntime C++工程简记](https://zhuanlan.zhihu.com/p/447364122)|@DefTruth|  
| [[推理部署][TNN]📖手动修改YoloX的tnnproto记录-TNN](https://zhuanlan.zhihu.com/p/425668734)|@DefTruth|  
| [[推理部署][ORT]📖全网最详细 ONNXRuntime C++/Java/Python 资料！](https://zhuanlan.zhihu.com/p/414317269)|@DefTruth|  
| [[推理部署][CV]📖RobustVideoMatting: C++工程化记录-实现篇](https://zhuanlan.zhihu.com/p/413280488)|@DefTruth|  
| [[推理部署][CV]📖RobustVideoMatting: C++工程化记录-应用篇](https://zhuanlan.zhihu.com/p/412491918)|@DefTruth|  
| [[推理部署][ORT]📖ONNXRuntime C++ CMake 工程分析及编译](https://zhuanlan.zhihu.com/p/411887386)|@DefTruth|  
| [[推理部署][ORT]📖如何使用ORT C++ API处理NCHW和NHWC输入？](https://zhuanlan.zhihu.com/p/524230808)|@DefTruth|  
| [[推理部署][TNN]📖tnn-convert搭建简记-YOLOP转TNN](https://zhuanlan.zhihu.com/p/431418709)|@DefTruth|  
| [[推理部署][CV]📖YOLOP ONNXRuntime C++工程化记录](https://zhuanlan.zhihu.com/p/411651933)|@DefTruth|  
| [[推理部署][NCNN]📖超有用NCNN参考资料整理](https://zhuanlan.zhihu.com/p/449765328)|@DefTruth|  
| [[推理部署][MNN]📖超有用MNN参考资料整理](https://zhuanlan.zhihu.com/p/449761992)|@DefTruth|  
| [[推理部署][TNN]📖超有用TNN参考资料整理](https://zhuanlan.zhihu.com/p/449769615)|@DefTruth|  
| [[推理部署][ONNX]📖超有用ONNX参考资料整理](https://zhuanlan.zhihu.com/p/449773663)|@DefTruth|  
| [[推理部署][ONNX]📖超有用ONNX模型结构参考资料整理](https://zhuanlan.zhihu.com/p/449775926)|@DefTruth|  
| [[推理部署][OpenCV-DNN]📖超有用OpenCV-DNN参考资料整理](https://zhuanlan.zhihu.com/p/449778377)|@DefTruth|  
| [[推理部署][Tensorflow]📖超有用Tensorflow C++工程化知识点](https://zhuanlan.zhihu.com/p/449788027)|@DefTruth|  
| [[推理部署][模型转换]📖深度学习模型转换资料整理](https://zhuanlan.zhihu.com/p/449759361)|@DefTruth|  
| [[技术随笔][C++][CMake]📖超有用CMake参考资料整理](https://zhuanlan.zhihu.com/p/449779892)|@DefTruth|  
| [[技术随笔][C++][3W字]📖静态链接和静态库实践指北-原理篇](https://zhuanlan.zhihu.com/p/595527528)|@DefTruth|  
| [[技术随笔][C++]📖Mac下C++内存检查指北(Valgrind VS Asan)](https://zhuanlan.zhihu.com/p/508470880)|@DefTruth|  
| [[技术随笔][CV]📖torchlm: 人脸关键点检测库](https://zhuanlan.zhihu.com/p/467211561)|@DefTruth|  
| [[技术随笔][ML]📖《统计学习方法-李航: 笔记-从原理到实现-基于R》](https://zhuanlan.zhihu.com/p/684885595)|@DefTruth|  
| [[技术随笔][Git]📖如何优雅地git clone和git submodule？](https://zhuanlan.zhihu.com/p/639136221)|@DefTruth|  
| [[技术随笔][3D]📖人脸重建3D参考资料整理](https://zhuanlan.zhihu.com/p/524034741)|@DefTruth|  
| [[技术随笔][3D]📖BlendShapes参考资料整理](https://zhuanlan.zhihu.com/p/524036145)|@DefTruth|  
| [[技术随笔][3D]📖从源码安装Pytorch3D详细记录及学习资料](https://zhuanlan.zhihu.com/p/512347464)|@DefTruth|  
| [[技术随笔][ML]📖200页:《统计学习方法：李航》笔记 -从原理到实现](https://zhuanlan.zhihu.com/p/461520847)|@DefTruth|  


### 📖 CUTLASS|CuTe|NCCL|CUDA|文章推荐 (其他作者)

|📖 类型-标题|📖 作者|
|:---|:---| 
| [[cute系列详解][入门]📖cutlass cute 101](https://zhuanlan.zhihu.com/p/660379052)|@朱小霖|
| [[cute系列详解][入门]📖CUTLASS 2.x & CUTLASS 3.x Intro 学习笔记](https://zhuanlan.zhihu.com/p/710516489)|@BBuf|
| [[cute系列详解][Layout]📖cute 之 Layout](https://zhuanlan.zhihu.com/p/661182311)|@reed|
| [[cute系列详解][Layout]📖cute Layout 的代数和几何解释](https://zhuanlan.zhihu.com/p/662089556)|@reed|
| [[cute系列详解][Tensor]📖cute 之 Tensor](https://zhuanlan.zhihu.com/p/663093816)|@reed|
| [[cute系列详解][MMA]📖cute 之 MMA抽象](https://zhuanlan.zhihu.com/p/663092747)|@reed|
| [[cute系列详解][Copy]📖cute 之 Copy抽象](https://zhuanlan.zhihu.com/p/666232173)|@reed|
| [[cute系列详解][Swizzle]📖cute 之 Swizzle](https://zhuanlan.zhihu.com/p/671419093)|@reed|
| [[cute系列详解][Swizzle]📖cute Swizzle细谈](https://zhuanlan.zhihu.com/p/684250988)|@进击的Killua|
| [[cute系列详解][Swizzle]📖cutlass swizzle机制解析（一）](https://zhuanlan.zhihu.com/p/710337546)|@Titus|
| [[cute系列详解][Swizzle]📖cutlass swizzle机制解析（二）](https://zhuanlan.zhihu.com/p/711398930)|@Titus|
| [[cute系列详解][GEMM]📖cute 之 简单GEMM实现](https://zhuanlan.zhihu.com/p/667521327)|@reed|
| [[cute系列详解][GEMM]📖cute 之 GEMM流水线](https://zhuanlan.zhihu.com/p/665082713)|@reed|
| [[cute系列详解][GEMM]📖cute 之 高效GEMM实现](https://zhuanlan.zhihu.com/p/675308830)|@reed|
| [[cute系列详解][GEMM]📖GEMM流水线: single-stage、multi-stage、pipelined](https://zhuanlan.zhihu.com/p/712451053)|@Titus|
| [[cute系列详解][GEMM]📖GEMM细节分析(一): ldmatrix的选择](https://zhuanlan.zhihu.com/p/702818267)|@Anonymous|
| [[cute系列详解][GEMM]📖GEMM细节分析(二): TiledCopy与cp.async](https://zhuanlan.zhihu.com/p/703560147)|@Anonymous|
| [[cute系列详解][GEMM]📖GEMM细节分析(三): Swizzle<B,M,S>参数取值](https://zhuanlan.zhihu.com/p/713713957)|@Anonymous|
| [[cute系列详解][实践]📖Hopper Mixed GEMM的CUTLASS实现笔记](https://zhuanlan.zhihu.com/p/714378343)|@BBuf|
| [[cute系列详解][实践]📖CUTLASS CuTe实战(一): 基础](https://zhuanlan.zhihu.com/p/690703999)|@进击的Killua|
| [[cute系列详解][实践]📖CUTLASS CuTe实战(二): 应用](https://zhuanlan.zhihu.com/p/692078624)|@进击的Killua|
| [[cute系列详解][实践]📖FlashAttention fp8实现（ada架构)](https://zhuanlan.zhihu.com/p/712314257)|@shengying.wei|
| [[cute系列详解][实践]📖FlashAttention 笔记: tiny-flash-attention解读](https://zhuanlan.zhihu.com/p/708867810)|@shengying.wei|
| [[cute系列详解][实践]📖使用cutlass cute复现flash attention](https://zhuanlan.zhihu.com/p/696323042)|@66RING|
| [[cutlass教程][入门]📖cutlass 基本认知](https://zhuanlan.zhihu.com/p/677616101)|@JoeNomad|
| [[cutlass教程][入门]📖cutlass 软件架构](https://zhuanlan.zhihu.com/p/678915618)|@JoeNomad|
| [[cutlass教程][入门]📖CUTLASS 基础介绍](https://zhuanlan.zhihu.com/p/671324125)|@进击的Killua|
| [[cutlass教程][入门]📖乱谈CUTLASS GTC2020 SLIDES](https://zhuanlan.zhihu.com/p/674693873)|@zzk again|
| [[cutlass教程][深入]📖cutlass block swizzle 和 tile iterator(@JoeNomad)](https://zhuanlan.zhihu.com/p/679929705)|@JoeNomad|
| [[cutlass教程][深入]📖cutlass bank conflict free 的shared memory layout](https://zhuanlan.zhihu.com/p/681966685)|@JoeNomad|
| [[cutlass教程][深入]📖cutlass 多级流水线](https://zhuanlan.zhihu.com/p/687397095)|@JoeNomad|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-前言](https://zhuanlan.zhihu.com/p/686198447)|@reed|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-寄存器](https://zhuanlan.zhihu.com/p/688616037)|@reed|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-Load和Cache](https://zhuanlan.zhihu.com/p/692445145)|@reed|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-浮点运算](https://zhuanlan.zhihu.com/p/695667044)|@reed|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-整数运算](https://zhuanlan.zhihu.com/p/700921948)|@reed|
| [[GPU指令集架构][精解]📖NVidia GPU指令集架构-比特和逻辑操作](https://zhuanlan.zhihu.com/p/712356884)|@reed|
| [[CUDA优化][入门]📖CUDA（一）：CUDA 编程基础](https://zhuanlan.zhihu.com/p/645330027)|@紫气东来|
| [[CUDA优化][入门]📖CUDA（二）：GPU的内存体系及其优化指南](https://zhuanlan.zhihu.com/p/654027980)|@紫气东来|
| [[CUDA优化][实践]📖CUDA（三）：通用矩阵乘法：从入门到熟练](https://zhuanlan.zhihu.com/p/657632577)|@紫气东来|
| [[CUDA优化][实践]📖ops(1)：LayerNorm 算子的 CUDA 实现与优化](https://zhuanlan.zhihu.com/p/694974164)|@紫气东来|
| [[CUDA优化][实践]📖ops(2)：SoftMax算子的 CUDA 实现](https://zhuanlan.zhihu.com/p/695307283)|@紫气东来|
| [[CUDA优化][实践]📖ops(3)：Cross Entropy 的 CUDA 实现](https://zhuanlan.zhihu.com/p/695594396)|@紫气东来|
| [[CUDA优化][实践]📖ops(4)：AdamW 优化器的 CUDA 实现](https://zhuanlan.zhihu.com/p/695611950)|@紫气东来|
| [[CUDA优化][实践]📖ops(5)：激活函数与残差连接的 CUDA 实现](https://zhuanlan.zhihu.com/p/695703671)|@紫气东来|
| [[CUDA优化][实践]📖ops(6)：embedding 层与 LM head 层的 CUDA 实现](https://zhuanlan.zhihu.com/p/695785781)|@紫气东来|
| [[CUDA优化][实践]📖ops(7)：self-attention 的 CUDA 实现及优化 (上)](https://zhuanlan.zhihu.com/p/695898274)|@紫气东来|
| [[CUDA优化][实践]📖ops(8)：self-attention 的 CUDA 实现及优化 (下)](https://zhuanlan.zhihu.com/p/696197013)|@紫气东来|
| [[CUDA优化][实践]📖CUDA（四）：使用 CUDA 实现 Transformer 结构](https://zhuanlan.zhihu.com/p/694416583)|@紫气东来|
| [[GPU通信架构][精解]📖NVIDIA GPGPU（四）- 通信架构](https://zhuanlan.zhihu.com/p/680262016)|@Bruce|

💡说明: 大佬们写的文章实在是太棒了，学到了很多东西。欢迎大家提PR推荐更多优秀的文章！

## 0x01 📖 CUDA Kernel目录 (面试常考题目)
<div id="kernellist"></div>  

- / = not supported now.  
- ✔️ = known work and already supported now.
- ❔ = in my plan, but not coming soon, maybe a few weeks later.
- **workflow**: custom **CUDA** kernel impl -> **Torch** python binding -> Run tests.

|📖 cuda kernel| 📖 elem dtype| 📖 acc dtype| 📖 docs |
|:---|:---|:---|:---| 
| ✔️ [sgemm_sliced_k_f32_kernel](./sgemm/sgemm.cu)|f32|f32|❔|
| ✔️ [sgemm_t_tile_sliced_k_f32x4_kernel](./sgemm/sgemm.cu)|f32|f32|❔|
| ❔ [hgemm_sliced_k_f16_f32_kernel](./sgemm/sgemm.cu)|f16|f32|❔|
| ❔ [hgemm_t_tile_sliced_k_f16x2_f32_kernel](./sgemm/sgemm.cu)|f16|f32|❔|
| ✔️ [sgemv_k32_f32_kernel](./sgemv/sgemv.cu)|f32|f32|❔|
| ✔️ [sgemv_k128_f32x4_kernel](./sgemv/sgemv.cu)|f32|f32|❔|
| ✔️ [sgemv_k16_f32_kernel](./sgemv/sgemv.cu)|f32|f32|❔|
| ❔ [hgemv_k32_f16_kernel](./sgemv/sgemv.cu)|f16|f16|❔|
| ❔ [hgemv_k128_f16x2_kernel](./sgemv/sgemv.cu)|f16|f16|❔|
| ❔ [hgemv_k16_f16_kernel](./sgemv/sgemv.cu)|f16|f16|❔|
| ✔️ [warp_reduce_f32/f16/bf16_kernel](./reduce/block_all_reduce.cu)|f16/bf16/f32|f16/bf16/f32|[link](./reduce/)|
| ✔️ [block_reduce_f32_kernel](./reduce/block_all_reduce.cu)|f32|f32|[link](./reduce/)|
| ✔️ [block_all_reduce_sum_f32_f32_kernel](./reduce/block_all_reduce.cu)|f32|f32|[link](./reduce/)|
| ✔️ [block_all_reduce_sum_f32x4_f32_kernel](./reduce/block_all_reduce.cu)|f32|f32|[link](./reduce/)|
| ✔️ [block_all_reduce_sum_f16_f16_kernel](./reduce/block_all_reduce.cu)|f16|f16|[link](./reduce/)|
| ✔️ [block_all_reduce_sum_f16_f32_kernel](./reduce/block_all_reduce.cu)|f16|f32|[link](./reduce/)|
| ✔️ [block_all_reduce_sum_f16x2_f16_kernel](./reduce/block_all_reduce.cu)|f16|f16|[link](./reduce/)|
| ✔️ [block_all_reduce_sum_f16x2_f32_kernel](./reduce/block_all_reduce.cu)|f16|f32|[link](./reduce/)|
| ✔️ [block_all_reduce_sum_bf16_bf16_kernel](./reduce/block_all_reduce.cu)|bf16|bf16|[link](./reduce/)|
| ✔️ [block_all_reduce_sum_bf16_f32_kernel](./reduce/block_all_reduce.cu)|bf16|f32|[link](./reduce/)|
| ✔️ [block_all_reduce_sum_bf16x2_bf16_kernel](./reduce/block_all_reduce.cu)|bf16|bf16|[link](./reduce/)|
| ✔️ [block_all_reduce_sum_bf16x2_f32_kernel](./reduce/block_all_reduce.cu)|bf16|f32|[link](./reduce/)|
| ✔️ [block_all_reduce_sum_fp8_e4m3_f16_kernel](./reduce/block_all_reduce.cu)|fp8_e4m3|f16|[link](./reduce/)|
| ❔ [block_all_reduce_sum_i8_i32_kernel](./reduce/block_all_reduce.cu)|i8|i32|[link](./reduce/)|
| ✔️ [dot_product_f32_kernel](./dot-product/dot_product.cu)|f32|f32|❔|
| ✔️ [dot_product_f32x4_kernel](./dot-product/dot_product.cu)|f32|f32|❔|
| ❔ [dot_product_f16_f16_kernel](./dot-product/dot_product.cu)|f16|f16|❔|
| ❔ [dot_product_f16x2_f16_kernel](./dot-product/dot_product.cu)|f16|f16|❔|
| ❔ [dot_product_f16_f32_kernel](./dot-product/dot_product.cu)|f16|f32|/|❔|
| ❔ [dot_product_f16x2_f32_kernel](./dot-product/dot_product.cu)|f16|f32|/|❔|
| ✔️ [elementwise_f32_kernel](./elementwise/elementwise.cu)|f32|/|/|❔|
| ✔️ [elementwise_f32x4_kernel](./elementwise/elementwise.cu)|f32|/|/|❔|
| ❔ [elementwise_f16_kernel](./elementwise/elementwise.cu)|f16|/|/|❔|
| ❔ [elementwise_f16x2_kernel](./elementwise/elementwise.cu)|f16|/|/|❔|
| ✔️ [histogram_i32_kernel](./histogram/histogram.cu)|i32|/|/|❔|
| ✔️ [histogram_i32x4_kernel](./histogram/histogram.cu)|i32|/|/|❔|
| ✔️ [softmax_f32_kernel (grid level memory fence)](./softmax/softmax.cu)|f32|f32|❔|
| ✔️ [softmax_f32x4_kernel (grid level memory fence)](./softmax/softmax.cu)|f32|f32|❔|
| ❔ [softmax_f32x4_kernel (per token)](./softmax/softmax.cu)|f32|f32|❔|
| ❔ [safe_softmax_f32x4_kernel (per token)](./softmax/softmax.cu)|f32|f32|❔|
| ✔️ [sigmoid_f32_kernel](./sigmoid/sigmoid.cu)|f32|/|❔|
| ✔️ [sigmoid_f32x4_kernel](./sigmoid/sigmoid.cu)|f32|/|❔|
| ✔️ [relu_f32_kernel](./relu/relu.cu)|f32|/|❔|
| ✔️ [relu_f32x4_kernel](./relu/relu.cu)|f32|/|❔|
| ❔ [relu_f16_kernel](./relu/relu.cu)|f16|/|❔|
| ❔ [relu_f16x2_kernel](./relu/relu.cu)|f16|/|❔|
| ✔️ [layer_norm_f32_kernel (per token)](./layer-norm/layer_norm.cu)|f32|f32|❔|
| ✔️ [layer_norm_f32x4_kernel (per token)](./layer-norm/layer_norm.cu)|f32|f32|❔|
| ❔ [layer_norm_f16_kernel (per token)](./layer-norm/layer_norm.cu)|f16|f16|❔|
| ❔ [layer_norm_f16x2_kernel (per token)](./layer-norm/layer_norm.cu)|f16|f16|❔|
| ✔️ [rms_norm_f32_kernel (per token)](./rms-norm/rms_norm.cu)|f32|f32|❔|
| ✔️ [rms_norm_f32x4_kernel (per token)](./rms-norm/rms_norm.cu)|f32|f32|❔|
| ❔ [rms_norm_f16_kernel (per token)](./rms-norm/rms_norm.cu)|f16|f16|❔|
| ❔ [rms_norm_f16x2_kernel (per token)](./rms-norm/rms_norm.cu)|f16|f16|❔|
| ✔️ [flash_attn_1_fwd_f32_kernel](./flash-attn/flash_attn_1_fwd_f32.cu)|f32|f32|[link](./flash-attn)|
| ❔ [flash_attn_2_fwd_f32_kernel](./flash-attn/flash_attn_2_fwd_f32.cu)|f32|f32|[link](./flash-attn)|
| ❔ [flash_attn_2_fwd_f16_kernel](./flash-attn/flash_attn_2_fwd_f32.cu)|f16|f32|[link](./flash-attn)|
| ❔ [flash_attn_2_fwd_bf16_kernel](./flash-attn/flash_attn_2_fwd_f32.cu)|bf16|f32|[link](./flash-attn)|
| ✔️ [hard_nms cpp only](./nms/nms.cc)|f32|/|❔|
| ✔️ [notes v1(deprecated)](./notes-v1.cu)|f32|f32|/|

## ©️License
GNU General Public License v3.0

## Contribute
如果觉得有用，不妨给个🌟👆🏻Star支持一下吧~

<div align='center'>
<a href="https://star-history.com/#DefTruth/CUDA-Learn-Notes&Date">
  <picture align='center'>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=DefTruth/CUDA-Learn-Notes&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=DefTruth/CUDA-Learn-Notese&type=Date" />
    <img width=450 height=300 alt="Star History Chart" src="https://api.star-history.com/svg?repos=DefTruth/CUDA-Learn-Notes&type=Date" />
  </picture>
</a>  
</div>

<details>
<summary>📖 References </summary>

## References  
- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal)
- [tiny-flash-attention](https://github.com/66RING/tiny-flash-attention)
- [cute-gemm](https://github.com/reed-lau/cute-gemm)
- [cutlass_flash_atten_fp8](https://github.com/weishengying/cutlass_flash_atten_fp8)
- [cuda_learning](https://github.com/ifromeast/cuda_learning)
  
</details>
