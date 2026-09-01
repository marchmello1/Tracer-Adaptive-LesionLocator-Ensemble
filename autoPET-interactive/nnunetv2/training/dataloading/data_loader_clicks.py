import os
import time
import warnings
from typing import Union, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import ball
from threadpoolctl import threadpool_limits
from scipy.ndimage import gaussian_filter

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetBaseDataset
from nnunetv2.training.dataloading.utils import generated_sparse_to_dense_point_gauss, simulate_clicks, \
    sparse_to_dense_point_gauss, generated_sparse_to_dense_point_nnInteractive, simulate_clicks_advanced
from nnunetv2.training.dataloading.champion_error_curriculum import (
    STRATEGIES,
    build_cumulative_error_clicks,
    build_touched_residual_targets,
)
from nnunetv2.training.dataloading.click_local_edit import (
    build_local_edit_target,
    build_local_support,
)
from nnunetv2.utilities.label_handling.label_handling import LabelManager

class nnUNetDataLoaderClicks(nnUNetDataLoader):    
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties, click_json = self._data.load_case_with_clicks(i)
            shape = data.shape[1:]

            # PROMPT HANDLING
            # Sample a random number of clicks from the click json
            num_clicks = np.random.randint(0, len(click_json['points']) + 1)
            clicks = np.random.choice(click_json['points'], size=num_clicks, replace=False)

            # Initialize the click volumes
            # pos_clicks, neg_clicks = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
            # if num_clicks > 0:
            #     clicks = np.random.choice(click_json['points'], size=num_clicks, replace=False)
            #     for clck in clicks:
            #         coord = clck['point']
            #         label = clck['name']
            #         coord = self.preprocess_point(coord, properties, shape)
            #         if label == 'tumor':
            #             # put point at the coordinate (not place_point)
            #             pos_clicks[*coord] = 1.0
            #         elif label == 'background':
            #             neg_clicks[*coord] = 1.0 # self.place_point(coord, neg_clicks, n_clck + 1)
            #         else:
            #             raise ValueError(f"Unknown label {label} in click json")
            #     pos_clicks = gaussian_filter(pos_clicks, sigma=3)
            #     neg_clicks = gaussian_filter(neg_clicks, sigma=3)

            pos_clicks, neg_clicks = sparse_to_dense_point_gauss(clicks, shape, properties, sigma=3)
                    
            # import napari
            # viewer = napari.Viewer()
            # viewer.add_image(data[0], name='CT')
            # viewer.add_image(data[1], name='PET')
            # viewer.add_labels(seg[0], name='segmentation')
            # viewer.add_labels(seg_prev[0], name='segmentation_prev')
            # viewer.add_labels(pos_clicks.astype(np.uint8), name='positive clicks')
            # viewer.add_labels(neg_clicks.astype(np.uint8), name='negative clicks')
            # napari.run()
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

            # combine clicks
            pos_clicks_cropped = crop_and_pad_nd(pos_clicks[None], bbox, 0)
            neg_clicks_cropped = crop_and_pad_nd(neg_clicks[None], bbox, 0)
            clicks_cropped = np.vstack((pos_clicks_cropped, neg_clicks_cropped))
            clicks_all[j] = clicks_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]
            clicks_all = clicks_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    clicks_all = torch.from_numpy(clicks_all).float()
                    images = []
                    segs = []
                    clicks = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b], 'regression_target': clicks_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                        clicks.append(tmp['regression_target'])
                    data_all = torch.stack(images)
                    clicks_all = torch.stack(clicks)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(crop_and_pad_nd(data, bbox, 0)[0], name='CT original')
        # viewer.add_image(crop_and_pad_nd(data, bbox, 0)[1], name='PET original')
        # viewer.add_image(pos_clicks_cropped, name='positive clicks')
        # viewer.add_image(neg_clicks_cropped, name='negative clicks')
        # viewer.add_image(data_all[1][0].numpy(), name='CT')
        # viewer.add_image(data_all[1][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][1,0].numpy(), name='segmentation')
        # viewer.add_image(clicks_all[1][0].numpy(), name='positive clicks da')
        # viewer.add_image(clicks_all[1][1].numpy(), name='negative clicks da')
        # napari.run()
                    
        # Combine clicks and image
        data_all = torch.cat((data_all, clicks_all), dim=1)
        
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


class nnUNetDataLoaderClicksGenerated(nnUNetDataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None,
                 point_width: float = 1.5):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities, pad_sides,
                         probabilistic_oversampling, transforms)
        self.point_width = point_width    
        
    def generate_train_batch_full_img(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]

            # PROMPT HANDLING
            # Sample a random number of clicks from the click json
            num_clicks = np.random.randint(0, 10)
            
            clicks = simulate_clicks(seg[0], data[1], fg=True, bg=True, center_offset=3, edge_offset=3, click_budget=num_clicks, use_gpu=False)

            pos_clicks, neg_clicks = generated_sparse_to_dense_point_gauss(clicks, shape, sigma=self.point_width)
                    
            # import napari
            # viewer = napari.Viewer()
            # viewer.add_image(data[0], name='CT')
            # viewer.add_image(data[1], name='PET')
            # viewer.add_labels(seg[0], name='segmentation')
            # viewer.add_labels(seg_prev[0], name='segmentation_prev')
            # viewer.add_labels(pos_clicks.astype(np.uint8), name='positive clicks')
            # viewer.add_labels(neg_clicks.astype(np.uint8), name='negative clicks')
            # napari.run()
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

            # combine clicks
            pos_clicks_cropped = crop_and_pad_nd(pos_clicks[None], bbox, 0)
            neg_clicks_cropped = crop_and_pad_nd(neg_clicks[None], bbox, 0)
            clicks_cropped = np.vstack((pos_clicks_cropped, neg_clicks_cropped))
            clicks_all[j] = clicks_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]
            clicks_all = clicks_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    clicks_all = torch.from_numpy(clicks_all).float()
                    images = []
                    segs = []
                    clicks = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b], 'regression_target': clicks_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                        clicks.append(tmp['regression_target'])
                    data_all = torch.stack(images)
                    clicks_all = torch.stack(clicks)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images, clicks

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0], name='CT original')
        # viewer.add_image(data[1], name='PET original')
        # viewer.add_image(pos_clicks, name='positive clicks')
        # viewer.add_image(neg_clicks, name='negative clicks')
        # viewer.add_image(data_all[1][0].numpy(), name='CT')
        # viewer.add_image(data_all[1][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][1,0].numpy(), name='segmentation')
        # viewer.add_image(clicks_all[1][0].numpy(), name='positive clicks da')
        # viewer.add_image(clicks_all[1][1].numpy(), name='negative clicks da')
        # napari.run()
                    
        # Combine clicks and image
        data_all = torch.cat((data_all, clicks_all), dim=1)
        
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
    
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})

                        num_pos_clicks, num_neg_clicks = np.random.randint(0, 6), np.random.randint(0, 6)
                        pet = tmp['image'][1].numpy()
                        clicks = simulate_clicks(tmp['segmentation'][0][0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                        pos_clicks, neg_clicks = generated_sparse_to_dense_point_gauss(clicks, pet.shape, sigma=self.point_width)
                        clicks_all = np.concat((pos_clicks[None], neg_clicks[None]), axis=0)
                        #clicks_arr.append(torch.from_numpy(clicks_all).float())

                        images.append(torch.cat((tmp['image'], torch.from_numpy(clicks_all).float()), dim=0))
                        segs.append(tmp['segmentation'])

                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images # , clicks, clicks_all, pos_clicks, neg_clicks, tmp

                    # click_array = torch.zeros_like(data_all)
                    # for b in range(self.batch_size):
                    #     pet = data_all[b][1].numpy()
                    #     num_pos_clicks, num_neg_clicks = np.random.randint(0, 6), np.random.randint(0, 6)
                    #     clicks = simulate_clicks(seg_all[0][b,0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                    #     pos_clicks, neg_clicks = generated_sparse_to_dense_point_gauss(clicks, pet.shape, sigma=1.5)
                    #     clicks_all = np.concat((pos_clicks[None], neg_clicks[None]), axis=0)
                    #     click_array[b] = torch.from_numpy(clicks_all).float()
        
                    # data_all = torch.cat((data_all, click_array), dim=1)
                    # del clicks_all, pos_clicks, neg_clicks, clicks, click_array, pet

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0], name='CT original')
        # viewer.add_image(data[1], name='PET original')
        # viewer.add_image(data_all[1][0].numpy(), name='CT')
        # viewer.add_image(data_all[1][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][1,0].numpy(), name='segmentation')
        # viewer.add_image(data_all[1][2].numpy(), name='positive clicks da')
        # viewer.add_image(data_all[1][3].numpy(), name='negative clicks da')
        # napari.run()
                
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
    

class nnUNetDataLoaderClicksGeneratedEDT(nnUNetDataLoader):
    def __init__(self,
                 data: nnUNetBaseDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...]] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None,
                 point_width: float = 1.5):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities, pad_sides,
                         probabilistic_oversampling, transforms)
        self.point_width = point_width
    
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})

                        num_pos_clicks, num_neg_clicks = np.random.randint(0, 6), np.random.randint(0, 6)
                        pet = tmp['image'][1].numpy()
                        clicks = simulate_clicks(tmp['segmentation'][0][0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                        pos_clicks, neg_clicks = generated_sparse_to_dense_point_nnInteractive(clicks, pet.shape, sigma=self.point_width)
                        clicks_all = torch.cat((pos_clicks[None], neg_clicks[None]), axis=0).float()

                        images.append(torch.cat((tmp['image'], clicks_all), dim=0))
                        segs.append(tmp['segmentation'])

                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images # , clicks, clicks_all, pos_clicks, neg_clicks, tmp

                    # click_array = torch.zeros_like(data_all)
                    # for b in range(self.batch_size):
                    #     pet = data_all[b][1].numpy()
                    #     num_pos_clicks, num_neg_clicks = np.random.randint(0, 6), np.random.randint(0, 6)
                    #     clicks = simulate_clicks(seg_all[0][b,0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                    #     pos_clicks, neg_clicks = generated_sparse_to_dense_point_gauss(clicks, pet.shape, sigma=1.5)
                    #     clicks_all = np.concat((pos_clicks[None], neg_clicks[None]), axis=0)
                    #     click_array[b] = torch.from_numpy(clicks_all).float()
        
                    # data_all = torch.cat((data_all, click_array), dim=1)
                    # del clicks_all, pos_clicks, neg_clicks, clicks, click_array, pet

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0], name='CT original')
        # viewer.add_image(data[1], name='PET original')
        # viewer.add_image(data_all[1][0].numpy(), name='CT')
        # viewer.add_image(data_all[1][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][1,0].numpy(), name='segmentation')
        # viewer.add_image(data_all[1][2].numpy(), name='positive clicks da')
        # viewer.add_image(data_all[1][3].numpy(), name='negative clicks da')
        # napari.run()
                
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
    

class nnUNetDataLoaderClicksGenerated10ptsEDT(nnUNetDataLoaderClicksGeneratedEDT):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        point_sampling_probs = np.log(np.linspace(2,12,11))[::-1]
        point_sampling_probs /= point_sampling_probs.sum()  # Normalize to sum to 1

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})

                        num_pos_clicks, num_neg_clicks = np.random.choice(np.arange(11), p=point_sampling_probs), np.random.choice(np.arange(11), p=point_sampling_probs)
                        pet = tmp['image'][1].numpy()
                        clicks = simulate_clicks(tmp['segmentation'][0][0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                        pos_clicks, neg_clicks = generated_sparse_to_dense_point_nnInteractive(clicks, pet.shape, sigma=self.point_width)
                        clicks_all = torch.cat((pos_clicks[None], neg_clicks[None]), axis=0).float()

                        images.append(torch.cat((tmp['image'], clicks_all), dim=0))
                        segs.append(tmp['segmentation'])

                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images # , clicks, clicks_all, pos_clicks, neg_clicks, tmp

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0], name='CT original')
        # viewer.add_image(data[1], name='PET original')
        # viewer.add_image(data_all[1][0].numpy(), name='CT')
        # viewer.add_image(data_all[1][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][1,0].numpy(), name='segmentation')
        # viewer.add_image(data_all[1][2].numpy(), name='positive clicks da', colormap='green')
        # viewer.add_image(data_all[1][3].numpy(), name='negative clicks da', colormap='red')
        # napari.run()
                
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
    

class nnUNetDataLoaderClicksGenerated10ptsRatio80_20EDT(nnUNetDataLoaderClicksGeneratedEDT):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        clicks_all = np.zeros((self.data_shape[0], 2, *self.data_shape[2:]), dtype=np.float32)

        point_sampling_probs = np.log(np.linspace(2,12,11))[::-1]
        point_sampling_probs /= point_sampling_probs.sum()  # Normalize to sum to 1

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)
            shape = data.shape[1:]
            
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})

                        num_pos_clicks, num_neg_clicks = np.random.choice(np.arange(11), p=point_sampling_probs), np.random.choice(np.arange(11), p=point_sampling_probs)
                        pet = tmp['image'][1].numpy()
                        if np.random.rand() < 0.8:  # 80% chance to use the normal click simulation
                            clicks = simulate_clicks(tmp['segmentation'][0][0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                        else:  # 20% chance to use the advanced click simulation
                            clicks = simulate_clicks_advanced(tmp['segmentation'][0][0].numpy(), pet, fg=True, bg=True, center_offset=3, edge_offset=3, pos_click_budget=num_pos_clicks, neg_click_budget=num_neg_clicks, use_gpu=False)
                        pos_clicks, neg_clicks = generated_sparse_to_dense_point_nnInteractive(clicks, pet.shape, sigma=self.point_width)
                        clicks_all = torch.cat((pos_clicks[None], neg_clicks[None]), axis=0).float()

                        images.append(torch.cat((tmp['image'], clicks_all), dim=0))
                        segs.append(tmp['segmentation'])

                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images # , clicks, clicks_all, pos_clicks, neg_clicks, tmp

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0], name='CT original')
        # viewer.add_image(data[1], name='PET original')
        # viewer.add_image(data_all[0][0].numpy(), name='CT')
        # viewer.add_image(data_all[0][1].numpy(), name='PET')
        # viewer.add_labels(seg_all[0][0,0].numpy(), name='segmentation')
        # viewer.add_image(data_all[0][2].numpy(), name='positive clicks da', colormap='green')
        # viewer.add_image(data_all[0][3].numpy(), name='negative clicks da', colormap='red')
        # napari.run()
                
        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


class nnUNetDataLoaderChampionErrorCurriculum(nnUNetDataLoaderClicksGeneratedEDT):
    """Use aligned champion iter0 masks to generate official-style FP/FN prompts."""

    force_error_crop = False
    max_error_case_resamples = 32

    def __init__(self, *args, validation_mode: bool = False,
                 zero_click_probability: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        champion_dir = os.environ.get("AUTOPET_CHAMPION_PREPROCESSED_DIR")
        if not champion_dir:
            raise RuntimeError("AUTOPET_CHAMPION_PREPROCESSED_DIR is required")
        self.champion_dir = champion_dir
        self.validation_mode = validation_mode
        self.zero_click_probability = float(zero_click_probability)
        if not 0.0 <= self.zero_click_probability < 1.0:
            raise ValueError(
                f"zero_click_probability must be in [0, 1), got {self.zero_click_probability}"
            )
        axes = os.environ.get("AUTOPET_TRAINING_TO_OFFICIAL_AXES")
        if axes is None:
            raise RuntimeError("AUTOPET_TRAINING_TO_OFFICIAL_AXES is required; Dataset999 uses 2,1,0")
        self.training_to_official_axes = tuple(int(axis) for axis in axes.split(","))
        if sorted(self.training_to_official_axes) != [0, 1, 2]:
            raise RuntimeError(f"invalid AUTOPET_TRAINING_TO_OFFICIAL_AXES={axes}")

    def generate_train_batch(self):
        profile_curriculum = os.environ.get("AUTOPET_PROFILE_CURRICULUM") == "1"
        batch_started = time.perf_counter()
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        # The inferred Dataset999 class exposes one GT channel. For curriculum
        # augmentation we deliberately carry a second, temporary channel with
        # the aligned champion prediction. NanFix slices supervision back to
        # lesion-only after prompt generation.
        seg_all = np.zeros((self.seg_shape[0], 2, *self.seg_shape[2:]), dtype=np.int16)
        prompt_traces = []

        for batch_index, case in enumerate(selected_keys):
            # Residual learning has no useful identity/no-click objective: the
            # production path hard-bypasses the model when no scribble exists.
            # Therefore residual batches must be centred on a real M0 error,
            # rather than spending most iterations on random background crops.
            for attempt in range(self.max_error_case_resamples + 1):
                data, seg, _, properties = self._data.load_case(case)
                champion_path = os.path.join(self.champion_dir, f"{case}.npy")
                if not os.path.isfile(champion_path):
                    raise FileNotFoundError(f"missing champion iter0 cache for {case}: {champion_path}")
                champion = np.load(champion_path, mmap_mode="r")
                if champion.ndim == 3:
                    champion = champion[None]
                if tuple(champion.shape[1:]) != tuple(seg.shape[1:]):
                    raise RuntimeError(
                        f"champion/GT shape mismatch for {case}: {champion.shape} vs {seg.shape}"
                    )
                residual_error = (seg[0] > 0) != (champion[0] > 0)
                if not self.force_error_crop or np.any(residual_error):
                    break
                if attempt == self.max_error_case_resamples:
                    raise RuntimeError(
                        "could not sample a case with a champion residual error "
                        f"after {self.max_error_case_resamples + 1} attempts"
                    )
                case = str(np.random.choice(self.indices))
            selected_keys[batch_index] = case
            shape = data.shape[1:]
            if self.force_error_crop:
                error_coordinates = np.argwhere(residual_error)
                # get_bbox follows nnU-Net's class_locations convention and
                # expects a leading segmentation-channel column.
                error_locations = np.concatenate(
                    (
                        np.zeros((len(error_coordinates), 1), dtype=error_coordinates.dtype),
                        error_coordinates,
                    ),
                    axis=1,
                )
                bbox_lbs, bbox_ubs = self.get_bbox(
                    shape, True, {1: error_locations}, overwrite_class=1
                )
            else:
                force_fg = self.get_do_oversample(batch_index)
                bbox_lbs, bbox_ubs = self.get_bbox(
                    shape, force_fg, properties['class_locations']
                )
            bbox = [[lower, upper] for lower, upper in zip(bbox_lbs, bbox_ubs)]
            data_all[batch_index] = crop_and_pad_nd(data, bbox, 0)
            combined_seg = np.vstack((seg[:1], champion[:1]))
            seg_all[batch_index] = crop_and_pad_nd(combined_seg, bbox, -1)

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images, segmentations, evaluation_ground_truths = [], [], []
                    for batch_index, case in enumerate(selected_keys):
                        sample_started = time.perf_counter()
                        transformed = self.transforms(
                            **{'image': data_all[batch_index], 'segmentation': seg_all[batch_index]}
                        )
                        transformed_at = time.perf_counter()
                        full_resolution_target = (
                            transformed['segmentation'][0]
                            if isinstance(transformed['segmentation'], list)
                            else transformed['segmentation']
                        )
                        ground_truth = full_resolution_target[0].numpy() > 0
                        champion = full_resolution_target[1].numpy() > 0
                        if self.validation_mode:
                            corrections = 5
                            strategy_index = sum(case.encode("utf-8")) % len(STRATEGIES)
                            strategy = STRATEGIES[strategy_index]
                            seed = 42
                        else:
                            # Official written protocol: iter0 followed by five
                            # cumulative corrective-scribble steps (iter1--5).
                            # Champion-initialized fine-tuning also retains a
                            # small zero-click fraction to prevent global
                            # segmentation forgetting outside the prompted area.
                            corrections = (
                                0 if np.random.random() < self.zero_click_probability
                                else int(np.random.randint(1, 6))
                            )
                            strategy = str(np.random.choice(STRATEGIES))
                            seed = int(np.random.randint(0, 2 ** 31 - 1))
                        if corrections == 0:
                            clicks, trace = {"tumor": [], "background": []}, []
                        else:
                            clicks, trace = build_cumulative_error_clicks(
                                ground_truth, champion, corrections, strategy, seed,
                                training_to_official_axes=self.training_to_official_axes,
                            )
                        curriculum_at = time.perf_counter()
                        pos_clicks, neg_clicks = generated_sparse_to_dense_point_nnInteractive(
                            clicks, ground_truth.shape, sigma=self.point_width
                        )
                        edt_at = time.perf_counter()
                        click_channels = torch.stack((pos_clicks, neg_clicks)).float()
                        model_input, model_target = self._assemble_model_sample(
                            transformed, click_channels, clicks
                        )
                        images.append(model_input)
                        segmentations.append(model_target)
                        evaluation_ground_truths.append(full_resolution_target[0:1].to(torch.int16))
                        prompt_traces.append({
                            "case": case,
                            "strategy": strategy,
                            "requested_corrections": corrections,
                            "tumor_points": len(clicks["tumor"]),
                            "background_points": len(clicks["background"]),
                            "trace": trace,
                        })
                        if profile_curriculum:
                            print(
                                "CURRICULUM_PROFILE "
                                f"case={case} validation={self.validation_mode} "
                                f"transform_s={transformed_at - sample_started:.3f} "
                                f"curriculum_s={curriculum_at - transformed_at:.3f} "
                                f"edt_s={edt_at - curriculum_at:.3f} "
                                f"batch_elapsed_s={edt_at - batch_started:.3f}",
                                flush=True,
                            )
                    data_all = torch.stack(images)
                    if isinstance(segmentations[0], list):
                        seg_all = [
                            torch.stack([segmentation[level] for segmentation in segmentations])
                            for level in range(len(segmentations[0]))
                        ]
                    else:
                        seg_all = torch.stack(segmentations)
                    evaluation_ground_truths = torch.stack(evaluation_ground_truths)

        return {
            'data': data_all,
            'target': seg_all,
            'keys': selected_keys,
            'prompt_traces': prompt_traces,
            'evaluation_ground_truth': evaluation_ground_truths,
        }

    def _assemble_model_sample(self, transformed, click_channels, clicks):
        """Legacy 4-channel segmentation sample kept only for reproducibility."""
        return (
            torch.cat((transformed['image'], click_channels), dim=0),
            transformed['segmentation'],
        )


class nnUNetDataLoaderChampionMaskResidual(nnUNetDataLoaderChampionErrorCurriculum):
    """Five-channel M0-conditioned loader with local ADD/REMOVE supervision."""

    force_error_crop = True

    @staticmethod
    def _resize_residual_target(target: torch.Tensor, shape) -> torch.Tensor:
        if tuple(target.shape[1:]) == tuple(shape):
            return target
        return F.interpolate(
            target[None].float(), size=tuple(shape), mode="nearest"
        )[0].to(torch.int16)

    def _assemble_model_sample(self, transformed, click_channels, clicks):
        segmentation = transformed['segmentation']
        full_resolution = segmentation[0] if isinstance(segmentation, list) else segmentation
        ground_truth = full_resolution[0].numpy() > 0
        champion = full_resolution[1].numpy() > 0
        residual = torch.from_numpy(
            build_touched_residual_targets(ground_truth, champion, clicks)
        ).to(torch.int16)

        if isinstance(segmentation, list):
            targets = [
                self._resize_residual_target(residual, level.shape[1:])
                for level in segmentation
            ]
        else:
            targets = residual

        model_input = torch.cat(
            (
                transformed['image'],
                full_resolution[1:2].float(),
                click_channels,
            ),
            dim=0,
        )
        if model_input.shape[0] != 5:
            raise RuntimeError(f"residual loader expected 5 channels, got {tuple(model_input.shape)}")
        return model_input, targets


class nnUNetDataLoaderClickLocalEdit3(nnUNetDataLoaderChampionErrorCurriculum):
    """Seven-channel Focus-Crop sample with KEEP/ADD/REMOVE supervision."""

    force_error_crop = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        radius = os.environ.get("AUTOPET_LOCAL_EDIT_RADIUS_VOXELS")
        if radius is None:
            raise RuntimeError("AUTOPET_LOCAL_EDIT_RADIUS_VOXELS is required")
        self.local_edit_radius_voxels = float(radius)
        if self.local_edit_radius_voxels <= 0:
            raise ValueError("AUTOPET_LOCAL_EDIT_RADIUS_VOXELS must be positive")

    @staticmethod
    def _resize_class_target(target: torch.Tensor, shape) -> torch.Tensor:
        if tuple(target.shape[1:]) == tuple(shape):
            return target
        return F.interpolate(
            target[None].float(), size=tuple(shape), mode="nearest"
        )[0].to(torch.int16)

    def _assemble_model_sample(self, transformed, click_channels, clicks):
        segmentation = transformed['segmentation']
        full_resolution = segmentation[0] if isinstance(segmentation, list) else segmentation
        ground_truth = full_resolution[0].numpy() > 0
        champion = full_resolution[1].numpy() > 0
        touched_actions = build_touched_residual_targets(ground_truth, champion, clicks)
        click_cores = (click_channels.numpy() >= 0.999)
        local_support = build_local_support(
            click_cores, radius_voxels=self.local_edit_radius_voxels
        )
        target = torch.from_numpy(
            build_local_edit_target(
                ground_truth, champion, touched_actions, local_support
            )[None]
        ).to(torch.int16)
        support_channels = torch.from_numpy(local_support).float()
        model_input = torch.cat(
            (
                transformed['image'],
                full_resolution[1:2].float(),
                click_channels,
                support_channels,
            ),
            dim=0,
        )
        if model_input.shape[0] != 7:
            raise RuntimeError(
                f"local edit loader expected 7 channels, got {tuple(model_input.shape)}"
            )
        if isinstance(segmentation, list):
            targets = [
                self._resize_class_target(target, level.shape[1:])
                for level in segmentation
            ]
        else:
            targets = target
        return model_input, targets
