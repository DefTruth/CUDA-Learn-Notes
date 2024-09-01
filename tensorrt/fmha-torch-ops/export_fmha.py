import torch
import warnings
from fmha_pattern_match_ops import QKVAttentionMatchTensorRTfusedMHA


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    B, S, N, h = 8, 1024, 8, 64
    H = N * h # 8 * 64 = 512
    Q = torch.randn((B, S, H)).float().cuda()
    K = torch.randn((B, S, H)).float().cuda()
    V = torch.randn((B, S, H)).float().cuda()

    qkv_attn_match_trt_fmha = QKVAttentionMatchTensorRTfusedMHA(n_heads=N)
    qkv_attn_match_trt_fmha = qkv_attn_match_trt_fmha.cuda().eval()

    torch.onnx.export(qkv_attn_match_trt_fmha,
                      {'Q': Q, 'K': K, 'V': V},
                      "fmha.onnx",
                      input_names=['Q', 'K', 'V'],
                      output_names=['out'],
                      export_params=True, 
                      verbose=False, 
                      opset_version=17)
    # python3 export_fmha.py
    # trtexec --onnx=fmha.onnx --saveEngine=fmha.fp16.engine --fp16
    # nsys profile --stats=true -t cuda,osrt,nvtx -o fmha.onnx --force-overwrite true trtexec --loadEngine=fmha.fp16.engine
