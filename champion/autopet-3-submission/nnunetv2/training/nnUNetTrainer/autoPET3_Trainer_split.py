"""
非点击分治微调 trainer：继承冠军 autoPET3_Trainer（organ 双头 + multitask data + Misalign2 + DC_CE smooth=0），
唯一改动 = override do_split() 按 tracer(fdg/psma) 过滤 + 降 epoch/lr 做微调（从冠军 final checkpoint 续）。

用法：
  nnUNetv2_train 999 3d_fullres <FOLD> -tr autoPET3_Trainer_FDG_FT200 \
      -p nnUNetResEncUNetLPlansMultiTalent -pretrained_weights <winner fold_X checkpoint_final.pth>
"""
from nnunetv2.training.nnUNetTrainer.autoPET3_Trainer import autoPET3_Trainer


class autoPET3_Trainer_FDG_FT200(autoPET3_Trainer):
    """FDG specialist：只在 FDG case 上微调。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 200          # 微调，非 1500 fresh
        self.initial_lr = 1e-4         # 从已收敛冠军权重续 → 中低 LR

    def do_split(self):
        tr_keys, val_keys = super().do_split()
        tr_keys = [k for k in tr_keys if k.startswith('fdg')]
        val_keys = [k for k in val_keys if k.startswith('fdg')]
        self.print_to_log_file(f"[FDG split] tr={len(tr_keys)} val={len(val_keys)}")
        return tr_keys, val_keys


class autoPET3_Trainer_PSMA_FT200(autoPET3_Trainer):
    """PSMA specialist：只在 PSMA case 上微调。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 200
        self.initial_lr = 1e-4

    def do_split(self):
        tr_keys, val_keys = super().do_split()
        tr_keys = [k for k in tr_keys if k.startswith('psma')]
        val_keys = [k for k in val_keys if k.startswith('psma')]
        self.print_to_log_file(f"[PSMA split] tr={len(tr_keys)} val={len(val_keys)}")
        return tr_keys, val_keys
