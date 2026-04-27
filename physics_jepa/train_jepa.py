import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf

from .train import Trainer
from .utils.hydra import compose

class JepaTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @staticmethod
    def embedding_stats(prefix, x):
        with torch.no_grad():
            if x.ndim == 4:
                flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
            elif x.ndim == 5:
                flat = x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[1])
            else:
                flat = x.reshape(-1, x.shape[-1])
            std = flat.float().std(dim=0, unbiased=False)
            return {
                f"{prefix}_std_mean": std.mean(),
                f"{prefix}_std_min": std.min(),
                f"{prefix}_std_max": std.max(),
            }

    def pred_fn(self, batch, model_components, loss_fn):
        encoder, predictor = model_components[:2]
        use_ema_target = len(model_components) > 2
        target_encoder = model_components[2] if use_ema_target else encoder
        ctx_embed = encoder(batch['context'])
        target = batch['target']

        if target.ndim != 6:
            if use_ema_target:
                with torch.no_grad():
                    tgt_embed = target_encoder(target)
            else:
                tgt_embed = target_encoder(target)
            pred = predictor(ctx_embed)

            if len(pred.shape) < 5:
                loss_dict = loss_fn(pred.unsqueeze(2), tgt_embed.unsqueeze(2))
            else:
                loss_dict = loss_fn(pred, tgt_embed)
            loss_dict.update(self.embedding_stats("pred_embed", pred))
            loss_dict.update(self.embedding_stats("target_embed", tgt_embed))
            return pred, loss_dict

        # Multi-offset targets with shape (B, K, C, T, H, W)
        offsets = self.cfg.dataset.get("target_offsets", None)
        if offsets is None:
            offsets = list(range(1, target.shape[1] + 1))
        offsets = sorted(set(int(offset) for offset in offsets))
        weights = self.train_cfg.get("target_offset_weights", None)
        if weights is None:
            weights = [1.0] * len(offsets)
        else:
            weights = [float(weight) for weight in weights]
            if len(weights) != len(offsets):
                raise ValueError(
                    f"target_offset_weights must have same length as target_offsets: "
                    f"got {len(weights)} weights for {len(offsets)} offsets"
                )
        weight_sum = sum(weights)
        if weight_sum <= 0:
            raise ValueError(f"target_offset_weights must sum to a positive value, got {weights}")

        bsz, num_offsets, channels, timesteps, height, width = target.shape
        if use_ema_target:
            with torch.no_grad():
                tgt_embed = target_encoder(target.reshape(bsz * num_offsets, channels, timesteps, height, width))
                tgt_embed = tgt_embed.reshape(bsz, num_offsets, *tgt_embed.shape[1:])
        else:
            tgt_embed = target_encoder(target.reshape(bsz * num_offsets, channels, timesteps, height, width))
            tgt_embed = tgt_embed.reshape(bsz, num_offsets, *tgt_embed.shape[1:])

        predictor_module = getattr(predictor, "module", predictor)
        if hasattr(predictor_module, "predict_by_offset"):
            pred_by_offset = predictor(ctx_embed, offsets)
        else:
            pred_by_offset = {}
            rollout = ctx_embed
            for step in range(1, max(offsets) + 1):
                rollout = predictor(rollout)
                if step in offsets:
                    pred_by_offset[step] = rollout

        preds = torch.stack([pred_by_offset[offset] for offset in offsets], dim=1)

        loss_sums = {}
        metric_values = {}
        for idx, _ in enumerate(offsets):
            pred_i = preds[:, idx]
            tgt_i = tgt_embed[:, idx]

            if len(pred_i.shape) < 5:
                offset_loss = loss_fn(pred_i.unsqueeze(2), tgt_i.unsqueeze(2))
            else:
                offset_loss = loss_fn(pred_i, tgt_i)

            for key, value in offset_loss.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + weights[idx] * value
            for key, value in self.embedding_stats(f"offset_{offsets[idx]}_pred_embed", pred_i).items():
                metric_values[key] = value
            for key, value in self.embedding_stats(f"offset_{offsets[idx]}_target_embed", tgt_i).items():
                metric_values[key] = value

        loss_dict = {key: value / weight_sum for key, value in loss_sums.items()}
        loss_dict.update(metric_values)
        return preds, loss_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default=f"{Path(__file__).parent.parent}/configs/train_grayscott.yml")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--predictor_path", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run
    # cfg.train.encoder_path = args.encoder_path
    # cfg.train.predictor_path = args.predictor_path
    
    cfg.model.objective = "jepa"

    print(OmegaConf.to_yaml(cfg, resolve=True))

    trainer = JepaTrainer(cfg)
    trainer.train()
