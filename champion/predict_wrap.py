"""PyTorch2.6 weights_only 默认坑修复 + nnUNet predict 入口。
⚠️ 必须 __main__ 守卫: nnUNet 用 spawn 多进程, 子进程会重新 import 本模块, 没守卫会重复触发 predict 导致 spawn 崩。
用法: python predict_wrap.py <nnUNetv2_predict 的所有参数>"""
import torch

_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault('weights_only', False)
    return _orig_load(*a, **k)

if __name__ == '__main__':
    torch.load = _patched_load
    from nnunetv2.inference.predict_from_raw_data import predict_entry_point
    predict_entry_point()
