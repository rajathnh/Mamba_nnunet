import torch
from torch import nn
from training.trainer.trainer import nnUNetTrainer
from training.models.mamba_seg import MambaSeg2D
from utilities.helpers import dummy_context
import numpy as np

class MambaTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        
        # --- VRAM SAVER FOR RTX 4050 (6GB) ---
        # 1. Modify the raw dictionary BEFORE the base class builds its configuration manager
        plans['configurations'][configuration]['batch_size'] = 2
        plans['configurations'][configuration]['patch_size'] = [256,256] 
        
        # 2. Now initialize the base class with the updated plans!
        super().__init__(plans, configuration, fold, dataset_json, device)


    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: list,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        
        # We ignore the standard plans and force it to use MambaSeg2D!
        # Using the exact logic you suggested:
        print("Building MambaSeg2D...")
        return MambaSeg2D(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            deep_supervision=enable_deep_supervision
        )

    def _get_deep_supervision_scales(self):
        # We must include [1.0, 1.0] so the targets list aligns with MambaSeg2D's outputs.
        # This results in a target list of: [256, 128, 64, 32, 16]
        return [[1.0, 1.0], [0.5, 0.5], [0.25, 0.25], [0.125, 0.125], [0.0625, 0.0625]]

    def _build_loss(self):
        # Standard nnUNet loss scaling for our 5 levels
        loss = super()._build_loss()
        
        if self.enable_deep_supervision:
            # Re-wrap with correct weights for our 5 outputs
            # Weights: 1, 0.5, 0.25, 0.125, 0 (we ignore the very smallest resolution's contribution usually)
            # Actually nnUNet default weights: 1/1, 1/2, 1/4, 1/8, 1/16 etc normalized
            weights = np.array([1 / (2**i) for i in range(5)])
            weights[-1] = 0 # nnUNet convention: last one is often too small
            weights = weights / weights.sum()
            
            # The base class already wrapped it, but with wrong weights (calculated from plans)
            # We unwrap and re-wrap:
            loss = loss.loss # Get the inner DC_and_CE_loss
            from training.loss.deep_supervision import DeepSupervisionWrapper
            loss = DeepSupervisionWrapper(loss, weights)
            
        return loss

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Use the same autocast context as training
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            
            # --- MAMBA VALIDATION FIX ---
            # If the model returned a single tensor (eval mode), compare it only to target[0]
            if self.enable_deep_supervision and not isinstance(output, (list, tuple)):
                # Use the internal .loss (DC_and_CE_loss) to bypass the list-requirement of the wrapper
                l = self.loss.loss(output, target[0])
            else:
                l = self.loss(output, target)

        # we only need the output with the highest output resolution for Dice calculation
        if self.enable_deep_supervision:
            # If it's already a single tensor, don't index it! 
            # (Indexing a tensor picks the first batch element, which we don't want here)
            if isinstance(output, (list, tuple)):
                output = output[0]
            target = target[0]

        # The rest of the function (Dice calculation) can be handled by the base class logic
        # But we'll implement the core metrics here to be safe:
        # The proper nnUNet v2 way to calculate pseudo dice
        axes = [0] + list(range(2, output.ndim))
        
        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
            target_for_metrics = target
        else:
            # We MUST one-hot encode the predictions AND target so the metric calculator can separate the classes!
            output_seg = output.argmax(1)[:, None]
            # Changed dtype to float32 so the Dice math can subtract!
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            
            target_for_metrics = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            target_for_metrics.scatter_(1, target.long(), 1)

        from training.loss.dice import get_tp_fp_fn_tn
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_for_metrics, axes=axes)
        
        return {
            'loss': l.detach().cpu().numpy(), 
            'tp_hard': tp.detach().cpu().numpy()[1:],  # [1:] ignores background, keeping Pancreas!
            'fp_hard': fp.detach().cpu().numpy()[1:], 
            'fn_hard': fn.detach().cpu().numpy()[1:]
        }
