import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutput

from util import nethook
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment

from .cov_cache import GenRecEditCovarianceCache
from .hparams import GenRecEditHyperParams
from .io_utils import genrecedit_load_json
from .model_bundle import GenRecEditModelBundle

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


class GenRecEdit:
    def __init__(self, hparams: GenRecEditHyperParams):
        self.hparams = hparams
        self.cov_cache = GenRecEditCovarianceCache(hparams.covariance_cache_dir)
        self.z_cache = defaultdict(list)

    def genrecedit_build_encoder_batch(self, mt: GenRecEditModelBundle, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        historys = [t["history"] for t in targets]
        input_ids = torch.tensor(historys, dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        return {
            "input_ids": input_ids.to(mt.model.device),
            "attention_mask": attention_mask.to(mt.model.device),
        }

    def genrecedit_build_decoder_batch(self, mt: GenRecEditModelBundle, targets: List[Dict], position: int) -> torch.Tensor:
        batch_decoder_inputs = []
        for target in targets:
            decoder_ids = [mt.model.t5.config.decoder_start_token_id]
            full_target_sids = target["target_sids"]
            for j in range(position):
                decoder_ids.append(int(full_target_sids[j]))
            batch_decoder_inputs.append(torch.tensor(decoder_ids, device=mt.model.device))
        return torch.stack(batch_decoder_inputs)

    def genrecedit_target_token_ids(self, mt: GenRecEditModelBundle, targets: List[Dict], position: int) -> torch.Tensor:
        ids = [int(t["target_sids"][position]) for t in targets]
        return torch.tensor(ids, device=mt.model.device)

    def genrecedit_build_past_prefix(self, model, encoder_outputs, attention_mask, decoder_input_ids):
        if decoder_input_ids.size(1) < 1:
            return None
        with torch.no_grad():
            out = model.t5(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )
        return out.past_key_values

    def genrecedit_clip_delta_norm_(self, delta_tensor: torch.Tensor, target_init: torch.Tensor):
        if target_init is None:
            return
        with torch.no_grad():
            delta_norm = torch.norm(delta_tensor)
            max_norm = self.hparams.z_vector_max
            if delta_norm > max_norm:
                delta_tensor.mul_(max_norm / delta_norm)

    def genrecedit_build_decoder_context(
        self,
        mt: GenRecEditModelBundle,
        history: List[int],
        position: int,
        full_target_sids: List[str],
    ) -> torch.Tensor:
        model = mt.model
        decoder_start_token_id = model.t5.config.decoder_start_token_id
        decoder_input = torch.tensor([[decoder_start_token_id]], device=model.device)
        for i in range(position):
            sid_id = int(full_target_sids[i])
            decoder_input = torch.cat([decoder_input, torch.tensor([[sid_id]], device=model.device)], dim=1)
        return decoder_input

    def genrecedit_probe_z_vector(
        self,
        mt: GenRecEditModelBundle,
        edit_target: Dict,
        position: int,
        z_vector: torch.Tensor,
        layer: int,
    ) -> Tuple[bool, float]:
        model = mt.model
        history = edit_target["history"]
        full_target_sids = edit_target["target_sids"]
        target_token_id = int(full_target_sids[position])

        input_ids = torch.tensor(history, dtype=torch.long).unsqueeze(0).to(model.device)
        attention_mask = (input_ids != 0).long().to(model.device)
        decoder_input = self.genrecedit_build_decoder_context(mt, history, position, full_target_sids)

        def _tmp_edit(cur_out, cur_layer):
            layer_name = f"decoder.block.{layer}.layer.2.DenseReluDense"
            if cur_layer == layer_name:
                tensor_out = cur_out[0] if isinstance(cur_out, (list, tuple)) else cur_out
                if tensor_out.ndim == 3:
                    tensor_out[:, -1, :] = z_vector
                else:
                    tensor_out[-1, :] = z_vector
                if isinstance(cur_out, (list, tuple)):
                    cur_out = list(cur_out)
                    cur_out[0] = tensor_out
                    cur_out = type(cur_out)(cur_out)
                else:
                    cur_out = tensor_out
            return cur_out

        with nethook.TraceDict(
            module=model.t5,
            layers=[f"decoder.block.{layer}.layer.2.DenseReluDense"],
            retain_input=False,
            retain_output=True,
            edit_output=_tmp_edit,
        ):
            with torch.no_grad():
                outputs = model.t5(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input,
                )

        probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
        target_prob = probs[0, target_token_id].item()

        for i in range(256):
            if i != target_token_id:
                prob = probs[0, 256 * position + i + 1].item()
                if prob > target_prob:
                    return False, target_prob

        if self.hparams.use_prob_threshold and target_prob > self.hparams.prob_threshold:
            return True, target_prob
        return False, target_prob

    def genrecedit_try_cache_hits(
        self,
        mt: GenRecEditModelBundle,
        targets: List[Dict],
        position: int,
        target_layer: int,
        satisfied: List[bool],
    ) -> Dict[int, torch.Tensor]:
        hits = {}
        for i, t in tqdm(
            enumerate(targets),
            total=len(targets),
            desc=f"GenRecEdit: cache hits position {position}",
            leave=False,
        ):
            if satisfied[i]:
                continue
            target_sid = t["target_sids"][position]
            k = (target_layer, str(target_sid), position)
            if k not in self.z_cache:
                continue
            for z in self.z_cache[k]:
                ok, _ = self.genrecedit_probe_z_vector(mt, t, position, z, target_layer)
                if ok:
                    satisfied[i] = True
                    hits[i] = z.to(mt.model.device, non_blocking=True)
                    break
        return hits

    def genrecedit_optimize_z_vectors(
        self,
        mt: GenRecEditModelBundle,
        targets: List[Dict],
        target_layer: int,
        position: int,
        batch_size: int = 2048,
    ) -> Tuple[List[Optional[torch.Tensor]], List[int], List[Optional[torch.Tensor]]]:
        model = mt.model
        device = model.device

        total_N = len(targets)
        z_all = [None] * total_N
        delta_all = [None] * total_N

        target_layer_name = f"decoder.block.{target_layer}.layer.2.DenseReluDense.wo"

        num_batches = (total_N + batch_size - 1) // batch_size
        batch_starts = range(0, total_N, batch_size)
        for start in tqdm(
            batch_starts,
            total=num_batches,
            desc=f"GenRecEdit: z batches position {position}",
            leave=False,
        ):
            end = min(start + batch_size, total_N)
            targets_b = targets[start:end]
            B = len(targets_b)

            deltas = torch.zeros(B, model.t5.model_dim, requires_grad=True, device=device)
            target_inits: List[Optional[torch.Tensor]] = [None] * B
            satisfied = [False] * B

            optimizer = torch.optim.Adam([deltas], lr=self.hparams.v_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.v_num_grad_steps, eta_min=0.01
            )

            enc = self.genrecedit_build_encoder_batch(mt, targets_b)
            dec = self.genrecedit_build_decoder_batch(mt, targets_b, position)
            tgt_ids = self.genrecedit_target_token_ids(mt, targets_b, position)

            cache_hits = self.genrecedit_try_cache_hits(mt, targets_b, position, target_layer, satisfied)
            z_local = [None] * B
            delta_local = [None] * B
            for i, z in cache_hits.items():
                z_local[i] = z

            with torch.no_grad():
                encoder_outputs = model.t5.encoder(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                )
                cached_enc = encoder_outputs.last_hidden_state.detach()

            active = [i for i in range(B) if not satisfied[i]]

            step_bar = tqdm(
                range(self.hparams.v_num_grad_steps),
                total=self.hparams.v_num_grad_steps,
                desc=f"GenRecEdit: z steps pos {position} batch {start // batch_size + 1}/{num_batches}",
                leave=False,
            )
            for step in step_bar:
                if not active:
                    break
                step_bar.set_postfix(active=len(active), refresh=False)

                enc_out = BaseModelOutput(last_hidden_state=cached_enc)
                past = self.genrecedit_build_past_prefix(
                    model=model,
                    encoder_outputs=enc_out,
                    attention_mask=enc["attention_mask"],
                    decoder_input_ids=dec[:, :-1],
                )

                def _edit(cur_out, cur_layer):
                    if cur_layer == target_layer_name:
                        tensor_out = cur_out[0] if isinstance(cur_out, (list, tuple)) else cur_out
                        for i in active:
                            if target_inits[i] is None:
                                target_inits[i] = tensor_out[i, -1, :].clone()
                        modified = tensor_out.clone()
                        for i in active:
                            modified[i, -1, :] = target_inits[i] + deltas[i]
                        if isinstance(cur_out, (list, tuple)):
                            cur_out = list(cur_out)
                            cur_out[0] = modified
                            return type(cur_out)(cur_out)
                        return modified
                    return cur_out

                with nethook.TraceDict(
                    module=model.t5,
                    layers=[target_layer_name],
                    retain_input=False,
                    retain_output=True,
                    edit_output=_edit,
                ):
                    outputs = model.t5(
                        encoder_outputs=enc_out,
                        input_ids=None,
                        attention_mask=enc["attention_mask"],
                        decoder_input_ids=dec[:, -1:],
                        use_cache=True,
                        past_key_values=past,
                        output_hidden_states=False,
                        output_attentions=False,
                    )

                last = outputs.logits[:, -1, :]
                p = F.softmax(last, dim=-1)
                hard_q = torch.zeros_like(p)
                hard_q.scatter_(1, tgt_ids.view(-1, 1), 1.0)

                eps = 1e-12
                neg_log_p = -torch.log(p.clamp_min(eps))
                per_loss = (hard_q * neg_log_p).sum(dim=-1)
                active_losses = per_loss[active]

                optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=device)
                for j, i in enumerate(active):
                    loss_i = active_losses[j]
                    if target_inits[i] is not None:
                        tnorm = torch.norm(target_inits[i])
                        dnorm = torch.norm(deltas[i])
                        wd = self.hparams.v_weight_decay * (dnorm / (tnorm + 1e-8))
                        loss_i = loss_i + wd
                    total_loss = total_loss + loss_i

                if total_loss.item() > 0:
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()

                with torch.no_grad():
                    for i in active:
                        if target_inits[i] is not None:
                            self.genrecedit_clip_delta_norm_(deltas[i], target_inits[i])

                if step > 0 and (step % 10 == 0 or step > self.hparams.v_num_grad_steps - 10):
                    new_active = []
                    for j, i in enumerate(active):
                        if target_inits[i] is None:
                            new_active.append(i)
                            continue
                        pred = torch.argmax(last[i]).item()
                        tgt = tgt_ids[i].item()
                        if pred == tgt:
                            satisfied[i] = True
                            z_local[i] = (target_inits[i] + deltas[i]).detach().clone()
                            delta_local[i] = deltas[i].detach().clone()
                        else:
                            new_active.append(i)
                    active = new_active

            for i in range(B):
                z_all[start + i] = z_local[i]
                delta_all[start + i] = delta_local[i]

            del enc, dec, tgt_ids, outputs, last, deltas
            torch.cuda.empty_cache()

        failed = [i for i, z in enumerate(z_all) if z is None]
        return z_all, failed, delta_all

    def genrecedit_collect_activations_batch(
        self,
        model,
        layer_name: str,
        batch_data: List[Dict],
        mt: GenRecEditModelBundle,
    ) -> Dict[int, torch.Tensor]:
        acts_by_pos: Dict[int, torch.Tensor] = {}
        num_pos = len(batch_data[0]["target_sids"])
        enc = self.genrecedit_build_encoder_batch(mt, batch_data)

        for position in tqdm(
            range(num_pos),
            total=num_pos,
            desc="GenRecEdit: collect activations by position",
            leave=False,
        ):
            dec = self.genrecedit_build_decoder_batch(mt, batch_data, position)
            with torch.no_grad():
                with nethook.TraceDict(
                    module=model.t5,
                    layers=[layer_name],
                    retain_input=True,
                    retain_output=False,
                ) as traces:
                    _ = model.t5(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        decoder_input_ids=dec,
                    )

                layer_in = traces[layer_name].input
                if isinstance(layer_in, (tuple, list)):
                    act = layer_in[0][:, -1, :]
                else:
                    act = layer_in[:, -1, :]
                acts_by_pos[position] = act.detach().cpu()

        return acts_by_pos

    def genrecedit_collect_layer_stats(
        self,
        model,
        layer_name: str,
        cov_data: List[Dict],
        mt: GenRecEditModelBundle,
        sample_size: Optional[int] = None,
        precision: Optional[str] = None,
    ) -> Dict[int, CombinedStat]:
        if sample_size is not None and sample_size < len(cov_data):
            sampled_indices = torch.randperm(len(cov_data))[:sample_size].tolist()
            sampled_data = [cov_data[i] for i in sampled_indices]
        else:
            sampled_data = cov_data

        num_pos = len(sampled_data[0]["target_sids"])
        stats_by_pos: Dict[int, CombinedStat] = {}
        for pos in range(num_pos):
            stats_by_pos[pos] = CombinedStat(**{k: STAT_TYPES[k]() for k in ["mean", "mom2"]})

        batch_size = self.hparams.stats_batch_size
        num_batches = (len(sampled_data) + batch_size - 1) // batch_size

        for batch_idx in tqdm(
            range(num_batches),
            total=num_batches,
            desc=f"GenRecEdit: layer stats {layer_name}",
            leave=False,
        ):
            s = batch_idx * batch_size
            e = min(s + batch_size, len(sampled_data))
            batch_data = sampled_data[s:e]

            acts_by_pos = self.genrecedit_collect_activations_batch(
                model=model,
                layer_name=layer_name,
                batch_data=batch_data,
                mt=mt,
            )

            for pos, act in acts_by_pos.items():
                act = act.to(dtype=torch.float64)
                if torch.isnan(act).any() or torch.isinf(act).any():
                    raise ValueError(f"Bad activations found at batch={batch_idx}, pos={pos}")
                stats_by_pos[pos].add(act)

        return stats_by_pos

    def genrecedit_get_or_compute_cov(
        self,
        mt: GenRecEditModelBundle,
        layer_name: str,
        cov_data_file: str,
        position: int,
    ) -> torch.Tensor:
        model = mt.model
        if not os.path.exists(cov_data_file):
            raise FileNotFoundError(f"Covariance data file not found: {cov_data_file}")

        cov_data = genrecedit_load_json(cov_data_file)
        self.hparams.mom2_n_samples = min(self.hparams.mom2_n_samples, len(cov_data))

        cache_key = self.cov_cache.genrecedit_make_cache_key(
            model_name=self.hparams.model_name,
            module_name=layer_name,
            cov_data_file=cov_data_file,
            sample_size=self.hparams.mom2_n_samples,
            dtype=self.hparams.mom2_dtype,
            position=position,
        )

        cached = self.cov_cache.genrecedit_load_covariance(cache_key)
        if cached is not None:
            return cached.to(model.device)

        stats_by_pos = self.genrecedit_collect_layer_stats(
            model=model,
            layer_name=layer_name,
            cov_data=cov_data,
            mt=mt,
            sample_size=self.hparams.mom2_n_samples,
            precision=self.hparams.mom2_dtype,
        )

        for pos, stat in tqdm(
            stats_by_pos.items(),
            total=len(stats_by_pos),
            desc=f"GenRecEdit: save covariance {layer_name}",
            leave=False,
        ):
            cov = stat.mom2.moment().float().to("cpu")
            metadata = {
                "model_name": self.hparams.model_name,
                "layer_name": layer_name,
                "data_file": cov_data_file,
                "sample_size": self.hparams.mom2_n_samples,
                "dtype": self.hparams.mom2_dtype,
                "num_cov_data": len(cov_data),
            }
            ck = self.cov_cache.genrecedit_make_cache_key(
                model_name=self.hparams.model_name,
                module_name=layer_name,
                cov_data_file=cov_data_file,
                sample_size=self.hparams.mom2_n_samples,
                dtype=self.hparams.mom2_dtype,
                position=pos,
            )
            self.cov_cache.genrecedit_save_covariance(cov, ck, metadata)

        return stats_by_pos[position].mom2.moment().float().to("cpu")

    def genrecedit_extract_keys(
        self,
        mt: GenRecEditModelBundle,
        targets: List[Dict],
        layer_idx: int,
        position: int,
        batch_size: int = 2048,
    ) -> List[torch.Tensor]:
        model = mt.model
        layer_name = f"decoder.block.{layer_idx}.layer.2.DenseReluDense.wo"
        keys: List[torch.Tensor] = []

        num_batches = (len(targets) + batch_size - 1) // batch_size
        batch_starts = range(0, len(targets), batch_size)
        for start in tqdm(
            batch_starts,
            total=num_batches,
            desc=f"GenRecEdit: extract key batches pos {position} layer {layer_idx}",
            leave=False,
        ):
            end = min(start + batch_size, len(targets))
            targets_b = targets[start:end]
            enc = self.genrecedit_build_encoder_batch(mt, targets_b)
            dec = self.genrecedit_build_decoder_batch(mt, targets_b, position)

            with torch.no_grad():
                with nethook.TraceDict(
                    module=model.t5,
                    layers=[layer_name],
                    retain_input=True,
                    retain_output=False,
                ) as traces:
                    _ = model.t5(
                        input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        decoder_input_ids=dec,
                    )

                layer_input = traces[layer_name].input
                if isinstance(layer_input, (tuple, list)):
                    key_vec = layer_input[0][:, -1, :]
                else:
                    key_vec = layer_input[:, -1, :]

                key_vec_cpu = key_vec.detach().to("cpu")
                keys.extend([v for v in key_vec_cpu])

            del enc, dec, traces, layer_input, key_vec, key_vec_cpu
            torch.cuda.empty_cache()

        return keys

    def genrecedit_pick_edit_layer(self, position: int) -> int:
        pos2layer = self.hparams.pos2layer
        if not pos2layer:
            raise ValueError("GenRecEdit pos2layer must contain at least one decoder layer index.")
        return pos2layer[position % len(pos2layer)]

    def genrecedit_solve_weight_delta(
        self,
        mt: GenRecEditModelBundle,
        layer_idx: int,
        z_vectors: List[torch.Tensor],
        key_vectors: List[torch.Tensor],
        valid_deltas: List[torch.Tensor],
        position: int,
    ) -> Dict[str, torch.Tensor]:
        model = mt.model
        device = model.device

        layer_name = f"decoder.block.{layer_idx}.layer.2.DenseReluDense.wo"
        wo_name = f"{layer_name}.weight"

        Z = torch.stack(z_vectors).to(device)
        K = torch.stack(key_vectors).to(device)
        delta = torch.stack(valid_deltas).to(device)

        K, Z, delta = K.T, Z.T, delta.T

        C = self.genrecedit_get_or_compute_cov(
            mt=mt,
            layer_name=layer_name,
            cov_data_file=self.hparams.covariance_data_file,
            position=position,
        ).to(device)

        _ = nethook.get_parameter(model.t5, wo_name)
        dW = (delta.double() @ K.T.double()) @ torch.linalg.inv(
            K.double() @ K.T.double() + int(self.hparams.cov_lambda) * C
        )
        return {wo_name: dW.float()}

    def genrecedit_run(
        self,
        mt: GenRecEditModelBundle,
        edit_targets: List[Dict],
        args,
        keep_original_weights: bool = True,
    ):
        output_dir = Path(args.output_dir) / str(args.category)
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = os.path.join(
            output_dir,
            f"deltaW_{self.hparams.edit_name}_{self.hparams.cov_lambda}_{self.hparams.number_knowledge}.pt",
        )
        if os.path.exists(save_path):
            print(f"Weight update file already exists at {save_path}, skipping.")
            return

        model = mt.model

        if keep_original_weights:
            self.original_weights = {}
            model_params = list(model.named_parameters())
            for name, param in tqdm(model_params, total=len(model_params), desc="GenRecEdit: save original weights"):
                if "DenseReluDense" in name:
                    self.original_weights[name] = param.data.clone()

        all_updates: Dict[str, torch.Tensor] = {}
        key_bank: Dict[int, Dict[int, List[torch.Tensor]]] = {}
        pos2layer = self.hparams.pos2layer
        if not pos2layer:
            raise ValueError("GenRecEdit pos2layer must contain at least one decoder layer index.")
        if len(pos2layer) > mt.tokenizer.n_digit:
            raise ValueError(
                f"GenRecEdit pos2layer maps {len(pos2layer)} positions, "
                f"but tokenizer.n_digit={mt.tokenizer.n_digit}. Do not include the EOS position."
            )
        edit_positions = range(len(pos2layer))
        key_layers = sorted(set(pos2layer))

        for position in tqdm(edit_positions, total=len(pos2layer), desc="GenRecEdit: extract keys by position"):
            key_bank[position] = {}
            for layer_idx in tqdm(
                key_layers,
                total=len(key_layers),
                desc=f"GenRecEdit: position {position} layers",
                leave=False,
            ):
                key_bank[position][layer_idx] = self.genrecedit_extract_keys(
                    mt=mt,
                    targets=edit_targets,
                    layer_idx=layer_idx,
                    position=position,
                )

        for position in tqdm(range(len(pos2layer)), total=len(pos2layer), desc="GenRecEdit: solve edits by position"):
            layer_idx = self.genrecedit_pick_edit_layer(position)

            z_vecs, _, deltas = self.genrecedit_optimize_z_vectors(
                mt=mt,
                targets=edit_targets,
                target_layer=layer_idx,
                position=position,
            )

            valid_idxs, valid_z, valid_d = [], [], []
            for i, z in tqdm(
                enumerate(z_vecs),
                total=len(z_vecs),
                desc=f"GenRecEdit: filter valid z position {position}",
                leave=False,
            ):
                if z is not None:
                    valid_idxs.append(i)
                    valid_z.append(z)
                    valid_d.append(deltas[i])

            if not valid_z:
                continue

            keys = []
            for idx in tqdm(
                valid_idxs,
                total=len(valid_idxs),
                desc=f"GenRecEdit: gather keys position {position}",
                leave=False,
            ):
                keys.append(key_bank[position][layer_idx][idx])

            upd = self.genrecedit_solve_weight_delta(
                mt=mt,
                layer_idx=layer_idx,
                z_vectors=valid_z,
                key_vectors=keys,
                valid_deltas=valid_d,
                position=position,
            )

            for name, deltaW in tqdm(
                upd.items(),
                total=len(upd),
                desc=f"GenRecEdit: accumulate deltaW position {position}",
                leave=False,
            ):
                all_updates[name] = all_updates.get(name, 0) + deltaW

        to_save = {
            k: v.detach().to("cpu", dtype=torch.float64)
            for k, v in tqdm(all_updates.items(), total=len(all_updates), desc="GenRecEdit: move deltaW to CPU")
        }
        torch.save(to_save, save_path)
        print("GenRecEdit deltaW saved to:", save_path)
