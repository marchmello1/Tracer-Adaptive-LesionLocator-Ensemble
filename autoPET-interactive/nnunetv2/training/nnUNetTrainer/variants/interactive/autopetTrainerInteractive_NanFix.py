import os

import cc3d
import torch
import numpy as np
from torch import autocast
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform

from nnunetv2.training.nnUNetTrainer.variants.interactive.autopetTrainerInteractive import (
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2
)
from nnunetv2.training.dataloading.data_loader_clicks import (
    nnUNetDataLoaderChampionErrorCurriculum,
    nnUNetDataLoaderChampionMaskResidual,
    nnUNetDataLoaderClickLocalEdit3,
)
from nnunetv2.training.dataloading.click_local_edit import (
    decide_local_edits,
    local_edit_objective,
)
from nnunetv2.training.dataloading.nnunet_dataset_multi import nnUNetDatasetMultiTask
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix(
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2
):
    """
    NanFix: adds nan_to_num before forward pass to handle GammaTransform NaN on negative CT HU.
    grad_clip reduced from 12 to 1.0 to suppress residual gradient spikes.
    """

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            # Dataset999 target is [B, 2, ...] (ch0=lesion, ch1=organ/seg_prev=empty).
            # Slice to lesion-only [B, 1, ...] so DC dice builds a proper [bg,lesion] one-hot;
            # without this the 2-ch target == 2-class output shape and dice is computed against
            # the empty organ channel -> no lesion supervision -> foreground collapse. Mirrors
            # parent autopetTrainer which slices i[:, :1]. (Confirmed via overfit probe 2026-06-16.)
            target = [i[:, :1].to(self.device, non_blocking=True) for i in target]
        else:
            target = target[:, :1].to(self.device, non_blocking=True)

        # Clamp NaN/Inf values from GammaTransform on negative CT HU in float16 AMP
        data = torch.nan_to_num(data, nan=0.0, posinf=6.0, neginf=-6.0)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            # Dataset999 target is [B, 2, ...] (ch0=lesion, ch1=organ/seg_prev=empty).
            # Slice to lesion-only [B, 1, ...] so DC dice builds a proper [bg,lesion] one-hot;
            # without this the 2-ch target == 2-class output shape and dice is computed against
            # the empty organ channel -> no lesion supervision -> foreground collapse. Mirrors
            # parent autopetTrainer which slices i[:, :1]. (Confirmed via overfit probe 2026-06-16.)
            target = [i[:, :1].to(self.device, non_blocking=True) for i in target]
        else:
            target = target[:, :1].to(self.device, non_blocking=True)

        # Symmetric with train_step: clamp NaN/Inf. Single-task (no organ head) since
        # Dataset999 has only the lesion label; avoids original validation_step taking the
        # non-existent organ channel target[:, 1:] which produced val_loss nan.
        data = torch.nan_to_num(data, nan=0.0, posinf=6.0, neginf=-6.0)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            l = self.loss(output, target)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Run legacy fixed-click validation only when its IV click cache exists.

        AutoPET V is evaluated separately with the official cumulative-scribble
        loop. Dataset999 intentionally has no IV ``clicks/*_clicks.json`` cache.
        """
        click_dir = os.path.join(self.preprocessed_dataset_folder_base, "clicks")
        if not os.path.isdir(click_dir):
            self.print_to_log_file(
                "Skipping legacy AutoPET IV fixed-click validation: "
                f"{click_dir} is absent. Use the official AutoPET V cumulative-scribble OOF evaluator.",
                also_print_to_console=True,
            )
            return
        return super().perform_actual_validation(save_probabilities)


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_Smoke2ep(
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix
):
    """Correctness-only harness; identical training with an externally set stop epoch."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        stop_epoch = os.environ.get("AUTOPET_SMOKE_STOP_EPOCH")
        if stop_epoch is None:
            raise RuntimeError("AUTOPET_SMOKE_STOP_EPOCH is required for the smoke trainer")
        self.num_epochs = int(stop_epoch)


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum(
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix
):
    """One unified model trained on cumulative prompts from champion iter0 FP/FN."""

    curriculum_dataloader_class = nnUNetDataLoaderChampionErrorCurriculum

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._curriculum_batch_logged = False
        self.curriculum_zero_click_probability = 0.0

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = nnUNetDatasetMultiTask

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        training_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        validation_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        loader_kwargs = dict(
            label_manager=self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            probabilistic_oversampling=self.probabilistic_oversampling,
            point_width=self.point_width,
        )
        loader_train = self.curriculum_dataloader_class(
            dataset_tr, self.batch_size, initial_patch_size, patch_size,
            transforms=training_transforms, validation_mode=False,
            zero_click_probability=self.curriculum_zero_click_probability,
            **loader_kwargs,
        )
        loader_validation = self.curriculum_dataloader_class(
            dataset_val, self.batch_size, patch_size, patch_size,
            transforms=validation_transforms, validation_mode=True,
            zero_click_probability=0.0,
            **loader_kwargs,
        )
        allowed_processes = get_allowed_n_proc_DA()
        pin_memory = self.device.type == 'cuda' and os.environ.get(
            "AUTOPET_DA_PIN_MEMORY", "1"
        ) == "1"
        if allowed_processes == 0:
            train_generator = SingleThreadedAugmenter(loader_train, None)
            validation_generator = SingleThreadedAugmenter(loader_validation, None)
        else:
            train_generator = NonDetMultiThreadedAugmenter(
                data_loader=loader_train,
                transform=None,
                num_processes=allowed_processes,
                num_cached=max(6, allowed_processes // 2),
                seeds=None,
                pin_memory=pin_memory,
                wait_time=0.002,
            )
            validation_generator = NonDetMultiThreadedAugmenter(
                data_loader=loader_validation,
                transform=None,
                num_processes=max(1, allowed_processes // 2),
                num_cached=max(3, allowed_processes // 4),
                seeds=None,
                pin_memory=pin_memory,
                wait_time=0.002,
            )
        _ = next(train_generator)
        _ = next(validation_generator)
        return train_generator, validation_generator

    def _audit_curriculum_batch(self, batch):
        data = batch['data']
        if data.ndim != 5 or data.shape[1] != 4:
            raise RuntimeError(f"curriculum expected [B,4,D,H,W], got {tuple(data.shape)}")
        if self._curriculum_batch_logged:
            return
        if not torch.isfinite(data).all():
            raise RuntimeError("curriculum batch contains NaN/Inf before NanFix")
        positive_voxels = int(torch.count_nonzero(data[:, 2]).item())
        negative_voxels = int(torch.count_nonzero(data[:, 3]).item())
        traces = batch.get('prompt_traces', [])
        requested = sum(int(item['requested_corrections']) for item in traces)
        selected = sum(
            sum(step.get('selected') != 'none' for step in item['trace'])
            for item in traces
        )
        # A random crop may contain neither champion FP nor FN. That is a valid
        # no-prompt training sample, but it cannot prove the smoke exercised
        # the curriculum. Keep auditing later batches until a real correction
        # is selected and represented in an EDT channel.
        if selected == 0:
            return
        if positive_voxels + negative_voxels == 0:
            raise RuntimeError("curriculum selected FP/FN prompts but both EDT channels are empty")
        self.print_to_log_file(
            "CHAMPION_FP_FN_BATCH "
            f"keys={batch.get('keys')} requested_turns={requested} selected_turns={selected} "
            f"positive_edt_voxels={positive_voxels} negative_edt_voxels={negative_voxels} "
            f"traces={traces}",
            also_print_to_console=True,
        )
        self._curriculum_batch_logged = True

    def train_step(self, batch: dict) -> dict:
        self._audit_curriculum_batch(batch)
        return super().train_step(batch)


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_Smoke2ep(
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum
):
    """Two-epoch gate for the exact champion FP/FN curriculum implementation."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        stop_epoch = os.environ.get("AUTOPET_SMOKE_STOP_EPOCH")
        if stop_epoch is None:
            raise RuntimeError("AUTOPET_SMOKE_STOP_EPOCH is required for the curriculum smoke trainer")
        self.num_epochs = int(stop_epoch)


def _filter_tracer_keys(keys, tracer):
    prefix = f"{str(tracer).lower()}_"
    filtered = [key for key in keys if str(key).lower().startswith(prefix)]
    if not filtered:
        raise RuntimeError(f"{str(tracer).upper()} specialist split produced no {prefix} cases")
    return filtered


class _ChampionFPFNCurriculumTracerSpecialistMixin:
    """Restrict the exact champion-error curriculum to one tracer domain.

    The only domain-specific optimization choice is foreground oversampling:
    PSMA uses 0.5 probabilistic sampling because its lesions are typically
    smaller; FDG retains nnU-Net's established 0.33 sampling. All spatial and
    intensity augmentation, prompt generation, loss, and initialization stay
    identical so the tracer split is the sole experimental change.
    """

    tracer = None
    specialist_stop_epoch = 550
    specialist_oversample_foreground_percent = 0.33

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.tracer not in ("fdg", "psma"):
            raise RuntimeError(f"invalid specialist tracer: {self.tracer!r}")
        self.num_epochs = self.specialist_stop_epoch
        self.oversample_foreground_percent = self.specialist_oversample_foreground_percent
        self.probabilistic_oversampling = self.tracer == "psma"

    def do_split(self):
        train_keys, validation_keys = super().do_split()
        train_domain = _filter_tracer_keys(train_keys, self.tracer)
        validation_domain = _filter_tracer_keys(validation_keys, self.tracer)
        self.print_to_log_file(
            "CHAMPION_FP_FN_DOMAIN_SPLIT "
            f"tracer={self.tracer} train={len(train_domain)}/{len(train_keys)} "
            f"validation={len(validation_domain)}/{len(validation_keys)} "
            f"oversample_fg={self.oversample_foreground_percent} "
            f"probabilistic_oversampling={self.probabilistic_oversampling} "
            f"stop_epoch={self.num_epochs}",
            also_print_to_console=True,
        )
        return train_domain, validation_domain


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_FDG550ep(
    _ChampionFPFNCurriculumTracerSpecialistMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum,
):
    """Five-fold FDG specialist using champion iter0 FP/FN corrections."""

    tracer = "fdg"
    specialist_oversample_foreground_percent = 0.33


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_PSMA550ep(
    _ChampionFPFNCurriculumTracerSpecialistMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum,
):
    """Five-fold PSMA specialist using champion iter0 FP/FN corrections."""

    tracer = "psma"
    specialist_oversample_foreground_percent = 0.5


def _filter_psma_keys(keys):
    psma_keys = [k for k in keys if str(k).lower().startswith("psma_")]
    if not psma_keys:
        raise RuntimeError("PSMA specialist split produced no psma_* cases")
    return psma_keys


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_PSMASpecialist550ep(
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix
):
    """
    PSMA specialist fine-tune.

    Keep the mature AutoPET/nnU-Net architecture and click pipeline unchanged, but
    train/validate only on psma_* cases and increase foreground patch sampling so
    tiny PSMA lesions are seen more often. Intended to continue from the unified
    NanFix checkpoint, not to train from scratch.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 550
        self.oversample_foreground_percent = 0.5
        self.probabilistic_oversampling = True

    def do_split(self):
        tr_keys, val_keys = super().do_split()
        tr_psma = _filter_psma_keys(tr_keys)
        val_psma = _filter_psma_keys(val_keys)
        self.print_to_log_file(
            f"PSMA specialist split: train {len(tr_psma)}/{len(tr_keys)} psma cases, "
            f"val {len(val_psma)}/{len(val_keys)} psma cases; "
            f"oversample_fg={self.oversample_foreground_percent}, "
            f"probabilistic_oversampling={self.probabilistic_oversampling}, "
            f"num_epochs={self.num_epochs}",
            also_print_to_console=True,
        )
        return tr_psma, val_psma


class _FastDAFineTuneMixin:
    """
    Late-stage fine-tuning speed path.

    The current training bottleneck is CPU-side 3D SpatialTransform on a large
    initial crop. For specialist fine-tuning from a mature checkpoint, keep the
    non-geometric intensity/noise/mirroring pipeline but skip the expensive
    rotation/scale resampling and clamp the initial crop to the final patch.
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, default_initial_patch_size, mirror_axes = (
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        initial_patch_size = np.array(self.configuration_manager.patch_size)
        self.print_to_log_file(
            "FastDA fine-tune: skip 3D SpatialTransform; "
            f"initial_patch_size {default_initial_patch_size.tolist()} -> {initial_patch_size.tolist()}",
            also_print_to_console=True,
        )
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    @staticmethod
    def get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=None,
            is_cascaded=False,
            foreground_labels=None,
            regions=None,
            ignore_label=None,
    ):
        transforms = []
        ignore_axes = (0,) if do_dummy_2d_data_aug else None

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(RemoveLabelTansform(-1, 0))
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(MoveSegAsOneHotToDataTransform(
                source_channel_idx=1,
                all_labels=foreground_labels,
                remove_channel_from_source=True
            ))
            transforms.append(RandomTransform(
                ApplyRandomBinaryOperatorTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    strel_size=(1, 8),
                    p_per_label=1
                ), apply_probability=0.4
            ))
            transforms.append(RandomTransform(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(foreground_labels), 0)),
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15,
                    p_per_label=1
                ), apply_probability=0.2
            ))

        if regions is not None:
            transforms.append(ConvertSegmentationToRegionsTransform(
                regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                channel_in_seg=0
            ))

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_PSMASpecialist550ep_FastDA(
    _FastDAFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_PSMASpecialist550ep,
):
    """PSMA specialist fine-tune with the CPU-heavy 3D spatial DA disabled."""
    pass


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_FDGSpecialist550ep_FastDA(
    _FastDAFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix,
):
    """FDG specialist fine-tune with the CPU-heavy 3D spatial DA disabled."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 550


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_FDGSpecialist550ep_FastDA_ClickDropoutFT(
    _FastDAFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix,
):
    """ClickDropoutFT FDG specialist fine-tune used for an alternate checkpoint."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 550


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_Smoke2ep_FastDA(
    _FastDAFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_Smoke2ep,
):
    """Exact curriculum smoke using the mature-checkpoint fine-tune DA path."""
    pass


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_FDG550ep_FastDA(
    _FastDAFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_FDG550ep,
):
    """FDG champion-error curriculum with fine-tune intensity/mirror DA."""
    pass


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_PSMA550ep_FastDA(
    _FastDAFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_PSMA550ep,
):
    """PSMA champion-error curriculum with fine-tune intensity/mirror DA."""
    pass


class _ChampionInitializedInteractionFineTuneMixin:
    """Conservative fine-tune settings for a strict AutoPET-III initialization.

    The source champion has already completed 1500 epochs. A fresh 150-epoch
    low-LR schedule is therefore used to learn the two newly zero-initialized
    interaction channels without replaying a full segmentation training run.
    Twenty percent zero-click batches anchor the champion's global behavior;
    the remaining batches sample official-style cumulative turns 1--5.
    """

    champion_finetune_epochs = 150
    champion_finetune_lr = 3e-4
    champion_zero_click_probability = 0.2

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.champion_finetune_epochs
        self.initial_lr = self.champion_finetune_lr
        self.curriculum_zero_click_probability = self.champion_zero_click_probability

    def get_dataloaders(self):
        self.print_to_log_file(
            "AUTOPET_CHAMPION_INITIALIZED_FINETUNE "
            f"epochs={self.num_epochs} initial_lr={self.initial_lr} "
            f"zero_click_probability={self.curriculum_zero_click_probability} "
            "grid_search=false",
            also_print_to_console=True,
        )
        return super().get_dataloaders()

    def on_train_epoch_start(self):
        self._champion_zero_click_samples = 0
        self._champion_prompted_samples = 0
        return super().on_train_epoch_start()

    def train_step(self, batch: dict) -> dict:
        for trace in batch.get("prompt_traces", []):
            if int(trace["requested_corrections"]) == 0:
                self._champion_zero_click_samples += 1
            else:
                self._champion_prompted_samples += 1
        return super().train_step(batch)

    def on_train_epoch_end(self, train_outputs):
        result = super().on_train_epoch_end(train_outputs)
        total = self._champion_zero_click_samples + self._champion_prompted_samples
        if total <= 0:
            raise RuntimeError("champion-initialized epoch produced no curriculum samples")
        self.print_to_log_file(
            "AUTOPET_CHAMPION_CLICK_MIX "
            f"zero_click_samples={self._champion_zero_click_samples} "
            f"prompted_samples={self._champion_prompted_samples} "
            f"observed_zero_fraction={self._champion_zero_click_samples / total:.4f}",
            also_print_to_console=True,
        )
        return result


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionInit_FDG150ep_FastDA(
    _FastDAFineTuneMixin,
    _ChampionInitializedInteractionFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_FDG550ep,
):
    """FDG specialist initialized strictly from the AutoPET-III champion."""
    pass


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionInit_PSMA150ep_FastDA(
    _FastDAFineTuneMixin,
    _ChampionInitializedInteractionFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_PSMA550ep,
):
    """PSMA specialist initialized strictly from the AutoPET-III champion."""
    pass


class autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionInit_FDGSmoke2ep_FastDA(
    _FastDAFineTuneMixin,
    _ChampionInitializedInteractionFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_FDG550ep,
):
    """Two-epoch correctness gate for strict champion initialization."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        stop_epoch = os.environ.get("AUTOPET_SMOKE_STOP_EPOCH")
        if stop_epoch is None:
            raise RuntimeError("AUTOPET_SMOKE_STOP_EPOCH is required for champion-init smoke")
        self.num_epochs = int(stop_epoch)


class _ChampionMaskResidual5chMixin:
    """Predict local ADD/REMOVE actions conditioned on immutable champion M0."""

    curriculum_dataloader_class = nnUNetDataLoaderChampionMaskResidual
    residual_finetune_epochs = 150
    residual_finetune_lr = 3e-4

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.residual_finetune_epochs
        self.initial_lr = self.residual_finetune_lr
        # Zero-click identity is a hard inference bypass. Training zero-click
        # samples only drives both action heads negative and creates branch
        # imbalance, so corrected residual training uses prompted samples only.
        self.curriculum_zero_click_probability = 0.0
        self._residual_batch_logged = False

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        if num_output_channels != 2:
            raise RuntimeError(
                f"ADD/REMOVE residual model requires exactly 2 heads, got {num_output_channels}"
            )
        architecture_class_name = "nnunetv2.architecture.ResidualEncoderUNetOrgan.ResidualEncoderUNetOrgan"
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels + 3,  # M0, positive EDT, negative EDT
            num_output_channels,
            enable_deep_supervision,
        )

    def _build_loss(self):
        # The legal action domain depends on M0, which the standard nnU-Net
        # loss does not receive. train_step calls _residual_loss explicitly.
        return None

    @staticmethod
    def _single_residual_loss(logits, target, champion):
        target = target.float()
        champion = champion > 0.5
        valid = torch.cat((~champion, champion), dim=1)
        losses = []
        for channel in range(2):
            channel_logits = logits[:, channel]
            channel_target = target[:, channel] > 0.5
            channel_valid = valid[:, channel]
            # A branch with no clicked target must be ignored, not treated as
            # an all-negative sample. At inference the corresponding branch is
            # hard-disabled when its click map is empty.
            active_samples = channel_target.flatten(1).any(1)
            channel_valid = channel_valid & active_samples.view(
                -1, *([1] * (channel_valid.ndim - 1))
            )
            positives = channel_valid & channel_target
            negatives = channel_valid & ~channel_target
            if torch.any(positives):
                positive_bce = torch.nn.functional.softplus(-channel_logits[positives]).mean()
                probabilities = torch.sigmoid(channel_logits) * channel_valid
                numerator = 2 * probabilities[channel_target].sum() + 1e-5
                denominator = probabilities.sum() + channel_target.sum() + 1e-5
                dice_loss = 1 - numerator / denominator
            else:
                positive_bce = channel_logits.sum() * 0
                dice_loss = channel_logits.sum() * 0
            negative_bce = (
                torch.nn.functional.softplus(channel_logits[negatives]).mean()
                if torch.any(negatives) else channel_logits.sum() * 0
            )
            losses.append(positive_bce + negative_bce + dice_loss)
        return torch.stack(losses).mean()

    def _residual_loss(self, output, target, champion):
        if not isinstance(output, (list, tuple)):
            return self._single_residual_loss(output, target, champion)
        weights = torch.tensor(
            [1 / (2 ** index) for index in range(len(output))],
            dtype=torch.float32,
            device=output[0].device,
        )
        weights[-1] = 1e-6 if self.is_ddp else 0
        weights = weights / weights.sum()
        total = output[0].sum() * 0
        for weight, level_output, level_target in zip(weights, output, target):
            level_champion = torch.nn.functional.interpolate(
                champion.float(), size=level_output.shape[2:], mode="nearest"
            )
            total = total + weight * self._single_residual_loss(
                level_output, level_target, level_champion
            )
        return total

    @staticmethod
    def reconstruct_from_actions(champion, logits, has_clicks):
        """Apply legal residual actions; hard-bypass samples without clicks."""
        champion = champion > 0.5
        add = (torch.sigmoid(logits[:, 0:1]) > 0.5) & ~champion
        remove = (torch.sigmoid(logits[:, 1:2]) > 0.5) & champion
        reconstructed = (champion | add) & ~remove
        return torch.where(has_clicks[:, None, None, None, None], reconstructed, champion)

    @staticmethod
    def _connected_action_gate(champion, logits, click_channels, threshold=0.5,
                               seed_threshold=None, click_core_threshold=0.999):
        """Grow legal ADD/REMOVE regions outward from the actual scribble core.

        ``click_channels > 0`` is the full EDT support ball, not the user's
        scribble. Treating that ball as an anchor can accept a prediction that
        merely passes near the click. The seed is therefore restricted to the
        EDT maximum (the original point/scribble voxels). Candidate action
        components are grown on the probability mask and retained only when a
        sufficiently confident seed lies inside them. With no valid seed the
        action fails closed and M0 remains unchanged.
        """
        if seed_threshold is None:
            seed_threshold = threshold
        if seed_threshold < threshold:
            raise ValueError(
                f"seed_threshold ({seed_threshold}) must be >= growth threshold ({threshold})"
            )
        champion = champion > 0.5
        legal = torch.cat((~champion, champion), dim=1)
        probabilities = torch.sigmoid(logits.float())
        # Strict '>' is intentional: a zero-initialized residual head has
        # probability exactly 0.5 and must produce an identity correction.
        raw = (probabilities > threshold) & legal
        seed = (
            (click_channels >= click_core_threshold)
            & (probabilities > seed_threshold)
            & legal
        )
        gated = torch.zeros_like(raw)
        raw_numpy = raw.detach().cpu().numpy()
        seed_numpy = seed.detach().cpu().numpy()
        for batch_index in range(raw.shape[0]):
            for channel in range(2):
                if not seed_numpy[batch_index, channel].any():
                    continue
                labels = cc3d.connected_components(
                    raw_numpy[batch_index, channel].astype(np.uint8, copy=False),
                    connectivity=26,
                )
                component_ids = np.unique(labels[seed_numpy[batch_index, channel]])
                component_ids = component_ids[component_ids != 0]
                if component_ids.size:
                    keep = np.isin(labels, component_ids)
                    gated[batch_index, channel] = torch.from_numpy(keep).to(gated.device)
        return gated

    def _prepare_residual_batch(self, batch):
        data = torch.nan_to_num(
            batch['data'].to(self.device, non_blocking=True),
            nan=0.0, posinf=6.0, neginf=-6.0,
        )
        if data.ndim != 5 or data.shape[1] != 5:
            raise RuntimeError(f"residual trainer expected [B,5,D,H,W], got {tuple(data.shape)}")
        target = batch['target']
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
            if any(item.shape[1] != 2 for item in target):
                raise RuntimeError("residual deep-supervision target must have ADD/REMOVE channels")
        else:
            target = target.to(self.device, non_blocking=True)
            if target.shape[1] != 2:
                raise RuntimeError("residual target must have ADD/REMOVE channels")
        if not self._residual_batch_logged:
            full_target = target[0] if isinstance(target, list) else target
            self.print_to_log_file(
                "RESIDUAL5CH_BATCH_VERIFIED "
                f"data_shape={tuple(data.shape)} target_shape={tuple(full_target.shape)} "
                f"m0_voxels={int(torch.count_nonzero(data[:, 2]).item())} "
                f"positive_edt_voxels={int(torch.count_nonzero(data[:, 3]).item())} "
                f"negative_edt_voxels={int(torch.count_nonzero(data[:, 4]).item())} "
                f"add_target_voxels={int(torch.count_nonzero(full_target[:, 0]).item())} "
                f"remove_target_voxels={int(torch.count_nonzero(full_target[:, 1]).item())}",
                also_print_to_console=True,
            )
            self._residual_batch_logged = True
        return data, target

    def train_step(self, batch: dict) -> dict:
        data, target = self._prepare_residual_batch(batch)
        champion = data[:, 2:3]
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            loss = self._residual_loss(output, target, champion)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
        return {'loss': loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data, target = self._prepare_residual_batch(batch)
        champion = data[:, 2:3]
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            loss = self._residual_loss(output, target, champion)
        if isinstance(output, (list, tuple)):
            output, target = output[0], target[0]
        action_prediction = self._connected_action_gate(
            champion, output, data[:, 3:5], threshold=0.5
        )
        action_target = target > 0.5
        axes = [0] + list(range(2, action_prediction.ndim))
        action_tp, action_fp, action_fn, _ = get_tp_fp_fn_tn(
            action_prediction.float(), action_target, axes=axes
        )

        ground_truth = batch.get('evaluation_ground_truth')
        if ground_truth is None:
            raise RuntimeError("residual validation requires evaluation_ground_truth")
        ground_truth = ground_truth.to(self.device, non_blocking=True) > 0
        add = action_prediction[:, 0:1]
        remove = action_prediction[:, 1:2]
        reconstructed = (champion.bool() | add) & ~remove
        final_tp, final_fp, final_fn, _ = get_tp_fp_fn_tn(
            reconstructed.float(), ground_truth, axes=axes
        )
        base_tp, base_fp, base_fn, _ = get_tp_fp_fn_tn(
            champion.float(), ground_truth, axes=axes
        )
        return {
            'loss': loss.detach().cpu().numpy(),
            # Parent checkpoint selection now tracks the final reconstructed
            # lesion mask, not unconstrained action logits.
            'tp_hard': final_tp.detach().cpu().numpy(),
            'fp_hard': final_fp.detach().cpu().numpy(),
            'fn_hard': final_fn.detach().cpu().numpy(),
            'base_tp': base_tp.detach().cpu().numpy(),
            'base_fp': base_fp.detach().cpu().numpy(),
            'base_fn': base_fn.detach().cpu().numpy(),
            'action_tp': action_tp.detach().cpu().numpy(),
            'action_fp': action_fp.detach().cpu().numpy(),
            'action_fn': action_fn.detach().cpu().numpy(),
        }

    def on_validation_epoch_end(self, val_outputs):
        result = super().on_validation_epoch_end(val_outputs)

        def summed(key):
            return np.sum(np.stack([item[key] for item in val_outputs]), axis=0)

        def dice(tp, fp, fn):
            denominator = 2 * tp + fp + fn
            return np.divide(
                2 * tp, denominator,
                out=np.full_like(tp, np.nan, dtype=np.float64),
                where=denominator > 0,
            )

        base_dice = dice(summed('base_tp'), summed('base_fp'), summed('base_fn'))
        final_dice = dice(summed('tp_hard'), summed('fp_hard'), summed('fn_hard'))
        action_dice = dice(summed('action_tp'), summed('action_fp'), summed('action_fn'))
        self.print_to_log_file(
            "RESIDUAL_RECONSTRUCTION_VALIDATION "
            f"base_dice={float(np.nanmean(base_dice)):.6f} "
            f"final_dice={float(np.nanmean(final_dice)):.6f} "
            f"delta={float(np.nanmean(final_dice) - np.nanmean(base_dice)):+.6f} "
            f"add_dice={float(action_dice[0]):.6f} "
            f"remove_dice={float(action_dice[1]):.6f}",
            also_print_to_console=True,
        )
        return result


class autopetTrainerChampionMaskResidual5ch_FDG150ep_FastDA(
    _ChampionMaskResidual5chMixin,
    _FastDAFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_FDG550ep,
):
    """CT/PET/M0/positive EDT/negative EDT -> local ADD/REMOVE actions."""
    pass


class autopetTrainerChampionMaskResidual5ch_FDGSmoke2ep_FastDA(
    autopetTrainerChampionMaskResidual5ch_FDG150ep_FastDA
):
    """Two-epoch correctness gate for the exact FDG production training path."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        stop_epoch = os.environ.get("AUTOPET_SMOKE_STOP_EPOCH")
        if stop_epoch is None:
            raise RuntimeError("AUTOPET_SMOKE_STOP_EPOCH is required")
        self.num_epochs = int(stop_epoch)
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 1


class autopetTrainerChampionMaskResidual5chCorrected_FDG150ep_FastDA(
    autopetTrainerChampionMaskResidual5ch_FDG150ep_FastDA
):
    """Corrected loss, click-connected gating, and final-mask checkpoint metric."""
    pass


class autopetTrainerChampionMaskResidual5chCorrected_FDGSmoke2ep_FastDA(
    autopetTrainerChampionMaskResidual5chCorrected_FDG150ep_FastDA
):
    """Short end-to-end gate for the corrected production training path."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 2


class autopetTrainerChampionMaskResidual5chCorrected_PSMA150ep_4090FastDA(
    _ChampionMaskResidual5chMixin,
    _FastDAFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_PSMA550ep,
):
    """Corrected PSMA residual model validated on a 24 GB RTX 4090.

    The exact planned batch size two completed the hardware smoke at about
    19.9 GB, so PSMA deliberately retains the same batch size, 250 iterations,
    and learning rate as FDG. Only the tracer-domain split differs.
    """


class autopetTrainerChampionMaskResidual5chCorrected_PSMASmoke2ep_4090FastDA(
    autopetTrainerChampionMaskResidual5chCorrected_PSMA150ep_4090FastDA
):
    """Short PSMA correctness and 24 GB memory gate."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 2


class _ClickLocalEdit3Mixin:
    """RITM-style M0 guidance with FocalClick-style bounded local merging."""

    curriculum_dataloader_class = nnUNetDataLoaderClickLocalEdit3
    local_edit_epochs = 150
    local_edit_lr = 3e-4

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = self.local_edit_epochs
        self.initial_lr = self.local_edit_lr
        self.curriculum_zero_click_probability = 0.0
        self.local_probability_threshold = float(
            os.environ.get("AUTOPET_LOCAL_EDIT_PROBABILITY_THRESHOLD", "0.8")
        )
        self.local_keep_margin = float(
            os.environ.get("AUTOPET_LOCAL_EDIT_KEEP_MARGIN", "0.1")
        )
        maximum = os.environ.get("AUTOPET_LOCAL_EDIT_MAX_ACTION_VOXELS")
        if maximum is None:
            raise RuntimeError("AUTOPET_LOCAL_EDIT_MAX_ACTION_VOXELS is required")
        self.local_max_action_voxels = int(maximum)
        if self.local_max_action_voxels < 1:
            raise ValueError("AUTOPET_LOCAL_EDIT_MAX_ACTION_VOXELS must be positive")
        self._local_edit_logged = False

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        architecture_class_name = "nnunetv2.architecture.ResidualEncoderUNetOrgan.ResidualEncoderUNetOrgan"
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels + 5,  # M0, positive/negative EDT, positive/negative support
            3,  # KEEP, ADD, REMOVE
            enable_deep_supervision,
        )

    def _build_loss(self):
        return None

    def _prepare_local_batch(self, batch):
        data = torch.nan_to_num(
            batch['data'].to(self.device, non_blocking=True),
            nan=0.0, posinf=6.0, neginf=-6.0,
        )
        if data.ndim != 5 or data.shape[1] != 7:
            raise RuntimeError(f"local edit model expected [B,7,D,H,W], got {tuple(data.shape)}")
        target = batch['target']
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)
        ground_truth = batch['evaluation_ground_truth'].to(
            self.device, non_blocking=True
        ).float()
        if not self._local_edit_logged:
            full_target = target[0] if isinstance(target, list) else target
            self.print_to_log_file(
                "CLICK_LOCAL_EDIT_BATCH_VERIFIED "
                f"data_shape={tuple(data.shape)} target_shape={tuple(full_target.shape)} "
                f"keep={int((full_target == 0).sum())} "
                f"add={int((full_target == 1).sum())} "
                f"remove={int((full_target == 2).sum())}",
                also_print_to_console=True,
            )
            self._local_edit_logged = True
        return data, target, ground_truth

    def _local_loss(self, output, target, data, ground_truth):
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        targets = list(target) if isinstance(target, (list, tuple)) else [target]
        weights = torch.tensor(
            [1 / (2 ** index) for index in range(len(outputs))],
            dtype=torch.float32, device=outputs[0].device,
        )
        if len(outputs) > 1:
            weights[-1] = 1e-6 if self.is_ddp else 0
        weights /= weights.sum()
        total = outputs[0].sum() * 0
        diagnostics = None
        for level, (weight, logits, level_target) in enumerate(zip(weights, outputs, targets)):
            shape = logits.shape[2:]
            champion = torch.nn.functional.interpolate(
                data[:, 2:3], size=shape, mode="nearest"
            )
            positive_support = torch.nn.functional.interpolate(
                data[:, 5:6], size=shape, mode="nearest"
            )
            negative_support = torch.nn.functional.interpolate(
                data[:, 6:7], size=shape, mode="nearest"
            )
            level_gt = torch.nn.functional.interpolate(
                ground_truth, size=shape, mode="nearest"
            )
            level_loss, level_diagnostics = local_edit_objective(
                logits, level_target, champion, level_gt,
                positive_support, negative_support,
            )
            total = total + weight * level_loss
            if level == 0:
                diagnostics = level_diagnostics
        return total, diagnostics

    def train_step(self, batch: dict) -> dict:
        data, target, ground_truth = self._prepare_local_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            loss, diagnostics = self._local_loss(output, target, data, ground_truth)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
        return {
            'loss': loss.detach().cpu().numpy(),
            **{f"local_{key}": value.cpu().numpy() for key, value in diagnostics.items()},
        }

    def validation_step(self, batch: dict) -> dict:
        data, target, ground_truth = self._prepare_local_batch(batch)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            loss, _ = self._local_loss(output, target, data, ground_truth)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        decisions = decide_local_edits(
            logits,
            data[:, 2:3],
            data[:, 3:5] >= 0.999,
            data[:, 5:7] > 0.5,
            probability_threshold=self.local_probability_threshold,
            keep_margin=self.local_keep_margin,
            max_action_voxels=self.local_max_action_voxels,
        )
        champion = data[:, 2:3] > 0.5
        reconstructed = champion.clone()
        rejected = 0
        for batch_index, decision in enumerate(decisions):
            add = torch.from_numpy(decision.add).to(self.device)
            remove = torch.from_numpy(decision.remove).to(self.device)
            reconstructed[batch_index, 0] = (
                (champion[batch_index, 0] | add) & ~remove
            )
            rejected += decision.rejected_oversize_components
        truth = ground_truth > 0.5
        axes = [0] + list(range(2, reconstructed.ndim))
        final_tp, final_fp, final_fn, _ = get_tp_fp_fn_tn(
            reconstructed.float(), truth, axes=axes
        )
        base_tp, base_fp, base_fn, _ = get_tp_fp_fn_tn(
            champion.float(), truth, axes=axes
        )

        def per_sample_dice(prediction):
            flat_prediction = prediction.float().flatten(1)
            flat_truth = truth.float().flatten(1)
            return (
                (2 * (flat_prediction * flat_truth).sum(1) + 1e-5)
                / (flat_prediction.sum(1) + flat_truth.sum(1) + 1e-5)
            )

        delta = per_sample_dice(reconstructed) - per_sample_dice(champion)
        return {
            'loss': loss.detach().cpu().numpy(),
            'tp_hard': final_tp.detach().cpu().numpy(),
            'fp_hard': final_fp.detach().cpu().numpy(),
            'fn_hard': final_fn.detach().cpu().numpy(),
            'base_tp': base_tp.detach().cpu().numpy(),
            'base_fp': base_fp.detach().cpu().numpy(),
            'base_fn': base_fn.detach().cpu().numpy(),
            'sample_delta': delta.detach().cpu().numpy(),
            'rejected_oversize': np.asarray(rejected, dtype=np.int64),
        }

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        deltas = np.concatenate([
            np.atleast_1d(item['sample_delta']) for item in val_outputs
        ])
        degradation_rate = float(np.mean(deltas < -1e-6))
        mean_negative_delta = float(np.mean(np.maximum(-deltas, 0)))
        final_dice = float(self.logger.my_fantastic_logging['mean_fg_dice'][-1])
        selection_score = final_dice - 2.0 * mean_negative_delta - 0.1 * degradation_rate
        epoch = self.current_epoch
        previous_ema = (
            self.logger.my_fantastic_logging['ema_fg_dice'][epoch - 1]
            if epoch > 0 else None
        )
        self.logger.my_fantastic_logging['mean_fg_dice'][-1] = selection_score
        self.logger.my_fantastic_logging['ema_fg_dice'][-1] = (
            selection_score if previous_ema is None
            else 0.9 * previous_ema + 0.1 * selection_score
        )
        rejected = sum(int(item['rejected_oversize']) for item in val_outputs)
        self.print_to_log_file(
            "CLICK_LOCAL_EDIT_VALIDATION "
            f"final_dice={final_dice:.6f} mean_delta={float(deltas.mean()):+.6f} "
            f"degradation_rate={degradation_rate:.6f} "
            f"mean_negative_delta={mean_negative_delta:.6f} "
            f"selection_score={selection_score:.6f} rejected_oversize={rejected}",
            also_print_to_console=True,
        )


class autopetTrainerClickLocalEdit3_FDG150ep_FastDA(
    _ClickLocalEdit3Mixin,
    _FastDAFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_FDG550ep,
):
    """FDG specialist for bounded local KEEP/ADD/REMOVE correction."""


class autopetTrainerClickLocalEdit3_PSMA150ep_4090FastDA(
    _ClickLocalEdit3Mixin,
    _FastDAFineTuneMixin,
    autopetTrainerInteractiveClickGen10ptsRatio80_20EDT2_NanFix_ChampionFPFNCurriculum_PSMA550ep,
):
    """PSMA specialist for bounded local KEEP/ADD/REMOVE correction."""


class autopetTrainerClickLocalEdit3_FDGSmoke2ep_FastDA(
    autopetTrainerClickLocalEdit3_FDG150ep_FastDA
):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 2
        self.num_val_iterations_per_epoch = 2
