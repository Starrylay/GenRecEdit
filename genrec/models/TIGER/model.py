import ast

import torch
from transformers import T5Config, T5ForConditionalGeneration

from genrec.model import AbstractModel
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
from util import nethook
from util.classfiers import Logistic
import warnings


def _normalize_pos2layer(pos2layer):
    if pos2layer is None:
        return [0, 1, 2, 3]
    if isinstance(pos2layer, str):
        pos2layer = ast.literal_eval(pos2layer)
    if isinstance(pos2layer, tuple):
        pos2layer = list(pos2layer)
    if not isinstance(pos2layer, list) or not pos2layer:
        raise ValueError("TIGER config 'pos2layer' must be a non-empty list of decoder layer indices.")
    return [int(layer_idx) for layer_idx in pos2layer]


class TIGER(AbstractModel):
    """
    TIGER model from Rajput et al. "Recommender Systems with Generative Retrieval." NeurIPS 2023.

    Args:
        config (dict): Configuration parameters for the model.
        dataset (AbstractDataset): The dataset object.
        tokenizer (AbstractTokenizer): The tokenizer object.

    Attributes:
        t5 (T5ForConditionalGeneration): The T5 model for conditional generation.
    """
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super(TIGER, self).__init__(config, dataset, tokenizer)

        t5config = T5Config(
            num_layers=config['num_layers'], 
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            activation_function=config['activation_function'],
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.padding_token,
            eos_token_id=tokenizer.eos_token,
            decoder_start_token_id=0,
            feed_forward_proj=config['feed_forward_proj'],
            n_positions=tokenizer.max_token_seq_len,
        )
        self.deltaW_path = config.get('deltaW_path', None)
        #检测下
        try:
            _ = torch.load(self.deltaW_path, map_location="cpu")
        except Exception:
            self.deltaW_path = None  # 可选：避免后面引用未定义

        if self.deltaW_path is None:
            # raise NotImplementedError("Loading deltaW from a specified path is not implemented in this code snippet.")
            warnings.warn(
                "Loading deltaW from a specified path is not implemented in this code snippet.",
                category=UserWarning,
                stacklevel=2,
            )
        self.cls_model_path = config.get('cls_model_path', None) #
        if self.cls_model_path is None or ".pt" not in self.cls_model_path:
            warnings.warn(
                "Loading classifier model from a specified path is not implemented in this code snippet.",
                category=UserWarning,
                stacklevel=2,
            )

        self.t5 = T5ForConditionalGeneration(config=t5config)
        self.pos2layer = _normalize_pos2layer(config.get("pos2layer", [0, 1, 2, 3]))
        if len(self.pos2layer) > tokenizer.n_digit:
            raise ValueError(
                f"TIGER config 'pos2layer' maps {len(self.pos2layer)} positions, "
                f"but tokenizer.n_digit={tokenizer.n_digit}. Do not include the EOS position."
            )
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_parameters(self) -> str:
        """
        Calculates the number of trainable parameters in the model.

        Returns:
            str: A string containing the number of embedding parameters, non-embedding parameters, and total trainable parameters.
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.t5.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Forward pass of the model. Returns the output logits and the loss value.

        Args:
            batch (dict): A dictionary containing the input data for the model.

        Returns:
            outputs (ModelOutput): 
                The output of the model, which includes:
                - loss (torch.Tensor)
                - logits (torch.Tensor)
        """
        outputs = self.t5(**batch)
        return outputs

    def generate(self, batch: dict, n_return_sequences: int = 1, max_length: int=0) -> torch.Tensor:
        """
        Generates sequences using beam search algorithm.

        Args:
            batch (dict): A dictionary containing input_ids and attention_mask.
            n_return_sequences (int): The number of sequences to generate.

        Returns:
            torch.Tensor: The generated sequences.
        """
        if max_length == 0:
            n_digit = self.tokenizer.n_digit
        else:
            n_digit = max_length
        # import ipdb; ipdb.set_trace()
        if self.deltaW_path is None:
            outputs = self.beam_search_ori(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=n_digit+2,
                num_beams=self.config['num_beams'],
                num_return_sequences=n_return_sequences,
                return_score=False
            )
        else:
            outputs = self.beam_search(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=n_digit+2,
                num_beams=self.config['num_beams'],
                num_return_sequences=n_return_sequences,
                return_score=False
            )
        # import ipdb; ipdb.set_trace()
        outputs = outputs[:, 1:1+n_digit].reshape(-1, n_return_sequences, n_digit)
        return outputs
   
    def beam_search_ori(
        self,
        input_ids,
        attention_mask,
        max_length=6,
        num_beams=1,
        num_return_sequences=1,
        return_score=False
    ):
        batch_size = input_ids.shape[0]
        
        # Prepare beam search inputs
      
        input_ids, attention_mask, decoder_input_ids, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
                input_ids, attention_mask, batch_size, num_beams
            )
        # import ipdb; ipdb.set_trace()
        # Store encoder_outputs to prevent running full forward path repeatedly
        with torch.no_grad():
            encoder_outputs = self.t5.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        #==========================循环生成===============================
        while decoder_input_ids.shape[1] < max_length:
            with torch.no_grad():
                outputs = self.t5(encoder_outputs=encoder_outputs,
                                  attention_mask=attention_mask,
                                  decoder_input_ids=decoder_input_ids)
                
                decoder_input_ids, beam_scores = self.beam_search_step(
                    outputs.logits,
                    decoder_input_ids,
                    beam_scores,
                    beam_idx_offset,
                    batch_size,
                    num_beams
                )
        # (batch_size * num_beams, ) -> (batch_size * num_return_sequences, )
        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True

        if return_score:
            return decoder_input_ids[selection_mask.view(-1), :], \
                beam_scores[selection_mask.view(-1)] / (decoder_input_ids.shape[1] - 1)
        
        # import ipdb; ipdb.set_trace()
        return decoder_input_ids[selection_mask.view(-1), :]
 

    def beam_search(
        self,
        input_ids,
        attention_mask,
        max_length=6,
        num_beams=1,
        num_return_sequences=1,
        return_score=False
    ):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Perform beam search to generate sequences using the specified model. 

        *** This implementation does not include stopping conditions based on end-of-sequence (EOS) tokens. Instead, the
        sequence generation is controlled solely by the `max_length` parameter. ***

        Note: In scenarios where the generation should explicitly detect and respond to EOS tokens 
        to terminate the sequence early, this function would need modifications. In the current setup,
        setting `max_length` to a suitable fixed value (e.g., 6) can serve the purpose by limiting
        the maximum sequence length.

        Parameters:
        - input_ids (torch.Tensor): Tensor of input ids.
        - attention_mask (torch.Tensor): Tensor representing the attention mask.
        - max_length (int): Maximum length of the sequence to be generated; controls when to stop extending the sequence.
        - num_beams (int): Number of beams for beam search.
        - num_return_sequences (int): Number of sequences to return.
        - return_score (bool): If True, returns a tuple of (sequences, scores) where 'scores' are the average log likelihood of the returned sequences.

        Returns:
        - torch.Tensor: The final decoder input ids from the beam search, or a tuple of (decoder_input_ids, scores) if 'return_score' is True.

        Example usage:
        # Assuming the model, input_ids, and attention_mask are predefined:
        sequences = beam_search(model, input_ids, attention_mask, max_length=6, num_beams=5, num_return_sequences=5)
        """

        batch_size = input_ids.shape[0]
        
        # Prepare beam search inputs
      
        input_ids, attention_mask, decoder_input_ids, beam_scores, beam_idx_offset = \
            self.prepare_beam_search_inputs(
                input_ids, attention_mask, batch_size, num_beams
            )
        # import ipdb; ipdb.set_trace()
        # Store encoder_outputs to prevent running full forward path repeatedly
        with torch.no_grad():
            encoder_outputs = self.t5.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )


        # =======================download deltaW=======================
        # =============================================================
        # save_path = "results/tiger-rq-edit-ckpts/4/deltaW.pt"
        all_weight_updates = torch.load(self.deltaW_path, map_location="cpu")
        # =======================存储 init W=======================
        # =============================================================
        all_weight_init = {}
        for param_name, _ in all_weight_updates.items():
            init_param = nethook.get_parameter(self.t5, param_name)
            all_weight_init[param_name] = init_param.clone()

        if self.cls_model_path is not None and ".pt" in self.cls_model_path:
            ckpt = torch.load(self.cls_model_path, map_location="cpu") # ckpt[position][layer_idx] -> nn.Module
            cls_models = {}
            for position in ckpt:
                cls_models[position] = {}
                for layer_idx in ckpt[position]:
                    cls = Logistic(self.config['d_ff'])
                    cls.load_state_dict(ckpt[position][layer_idx])
                    cls.to(self.device)
                    cls.eval()
                    cls_models[position][layer_idx] = cls
        #==========================循环生成===============================

        while decoder_input_ids.shape[1] < max_length:
            holder = {}
            position = decoder_input_ids.shape[1] - 1
            if position < len(self.pos2layer):
                layer_idx = self.pos2layer[position]
                layer_mod  = f"decoder.block.{layer_idx}.layer.2.DenseReluDense.wo"
                layer_w    = layer_mod + ".weight"   # deltaW 的 key 一般是 weight 名
                # import ipdb; ipdb.set_trace()
                
                # cls = Logistic(self.config['d_ff'])
                # cls.load_state_dict(ckpt[position][layer_idx])
                # cls.to(self.device)
                # cls.eval()
                # edit_output hook：logit<0 -> 加 delta_out；否则原样输出

                def smart_edit_fn(cur_out, cur_layer):
                    tr = holder["trace"][cur_layer]   # 这是 Trace 对象
                    h = tr.input
                    h_last = h[:, -1, :]          # 只对当前生成位置的 hiddenstate 判定
                    deltaW = all_weight_updates[layer_w].to(h_last.device, dtype=h_last.dtype)
                    delta_last = h_last @ deltaW.T   # (B, d_ff) (d_ff, d_model)
                    out = cur_out.clone()
                    if self.cls_model_path is not None and ".pt" in self.cls_model_path:
                        cls = cls_models[position][layer_idx]
                        with torch.no_grad():
                            logit = cls(h_last)           # (N,)
                            use_edit = (logit < 0)        # True -> 用 edited
                        if not use_edit.any():
                            return cur_out
                        out[:, -1, :][use_edit] += delta_last[use_edit]
                    else: # 不用 classifier，全部加
                        out[:, -1, :] += delta_last

                    return out
            else:
                layer_mod = None

        # ===============================END edit==========================================
            
            with torch.no_grad():
                with nethook.TraceDict(
                    module=self.t5,
                    layers=[layer_mod] if layer_mod is not None else [],
                    retain_input=True,          # 关键：要拿 hiddenstate
                    retain_output=True,
                    edit_output=smart_edit_fn if layer_mod is not None else None,
                ) as tr:
                    holder["trace"] = tr
                    outputs = self.t5(
                        encoder_outputs=encoder_outputs,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids
                    )
                decoder_input_ids, beam_scores = self.beam_search_step(
                    outputs.logits,
                    decoder_input_ids,
                    beam_scores,
                    beam_idx_offset,
                    batch_size,
                    num_beams
                )

            # # 恢复初始值
            # if layer_name is not None:
            #     with torch.no_grad(): 
            #         init_param = all_weight_init[layer_name].to(current_param.device, dtype=current_param.dtype)
            #         current_param.copy_(init_param)




        # (batch_size * num_beams, ) -> (batch_size * num_return_sequences, )
        selection_mask = torch.zeros(batch_size, num_beams, dtype=bool)
        selection_mask[:, :num_return_sequences] = True

        if return_score:
            return decoder_input_ids[selection_mask.view(-1), :], \
                beam_scores[selection_mask.view(-1)] / (decoder_input_ids.shape[1] - 1)
        
        # import ipdb; ipdb.set_trace()
        return decoder_input_ids[selection_mask.view(-1), :]

    def prepare_beam_search_inputs(self, input_ids, attention_mask, batch_size, num_beams):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Prepares and duplicates the input data for beam search decoding.

        This function initializes decoder input IDs and beam scores, creates an offset for beam indices, 
        and expands the input_ids and attention_mask tensors to accommodate the specified number of beams for each instance in the batch.

        Parameters:
        - input_ids (torch.Tensor): The input IDs tensor of shape (batch_size, sequence_length) used for the encoder part of the model.
        - attention_mask (torch.Tensor): The attention mask tensor of shape (batch_size, sequence_length) indicating to the model which tokens should be attended to.
        - batch_size (int): The number of instances per batch in the input data.
        - num_beams (int): The number of beams to use in beam search. This expands the input data and scores accordingly.

        Returns:
        - input_ids (torch.Tensor): The expanded input IDs tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
        - attention_mask (torch.Tensor): The expanded attention mask tensor to match the number of beams, shape (batch_size * num_beams, sequence_length).
        - initial_decoder_input_ids (torch.Tensor): The initialized decoder input IDs for each beam, shape (batch_size * num_beams, 1).
        - initial_beam_scores (torch.Tensor): The initialized scores for each beam, flattened to a single dimension, shape (batch_size * num_beams,).
        - beam_idx_offset (torch.Tensor): An offset for each beam index to assist in reordering beams during the search, shape (batch_size * num_beams,).

        Each input sequence is replicated 'num_beams' times to provide separate candidate paths in beam search. Beam scores are initialized with 0 for the first beam and a very low number (-1e9) for others to ensure the first token of each sequence is chosen from the first beam.
        """

        decoder_input_ids = torch.ones((batch_size * num_beams, 1), device=self.t5.device, dtype=torch.long)
        initial_decoder_input_ids = decoder_input_ids * self.t5.config.decoder_start_token_id

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9  # Set a low score for all but the first beam to ensure the first beam is selected initially
        initial_beam_scores = beam_scores.view((batch_size * num_beams,))

        beam_idx_offset = torch.arange(batch_size, device=self.t5.device).repeat_interleave(num_beams) * num_beams

        input_ids = input_ids.repeat_interleave(num_beams, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

        return input_ids, attention_mask, initial_decoder_input_ids, initial_beam_scores, beam_idx_offset


    def beam_search_step(self, logits, decoder_input_ids, beam_scores, beam_idx_offset, batch_size, num_beams):
        """
        Adpated from huggingface's implementation
        https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/generation/utils.py#L2823

        Executes one step of beam search, calculating the next set of input IDs based on logits from a model.

        This function expands the current beam, calculates scores for all possible next tokens, selects the top tokens for each beam, and prepares the input IDs for the next iteration of the model. It utilizes logits output by the model to determine the most likely next tokens and updates the beam scores.

        Parameters:
        - logits (torch.Tensor): Logits returned from the model, shape (batch_size * num_beams, sequence_length, vocab_size).
        - decoder_input_ids (torch.Tensor): Current decoder input IDs, shape (batch_size * num_beams, current_sequence_length).
        - beam_scores (torch.Tensor): Current scores for each beam, shape (batch_size * num_beams,).
        - beam_idx_offset (torch.Tensor): Index offsets for each beam to handle batches correctly, shape (batch_size * num_beams,).
        - batch_size (int): Number of sequences being processed in a batch.
        - num_beams (int): Number of beams used in the beam search.

        Returns:
        - decoder_input_ids (torch.Tensor): Updated decoder input IDs after adding the next tokens, shape (batch_size * num_beams, current_sequence_length + 1).
        - beam_scores (torch.Tensor): Updated scores for each beam, shape (batch_size * num_beams,).

        The function selects the top `2 * num_beams` tokens from the logits based on their scores, reshapes and adjusts them based on the existing beam scores, and determines the next tokens to add to each beam path. The updated paths are then returned for use in the next iteration of the beam search.
        """
        assert batch_size * num_beams == logits.shape[0]

        vocab_size = logits.shape[-1]
        next_token_logits = logits[:, -1, :]
        next_token_scores = torch.log_softmax(next_token_logits, dim=-1)  # Calculate log softmax over the last dimension([400, 1027])
        # import ipdb; ipdb.set_trace()
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores) #[400, 1027]  每50个 有一排0  # beam score 是[400,1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size) # [8, 51350]  每行是一个history 的所有beam的所有token scores(1027 x 50)
        next_token_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
        # import ipdb; ipdb.set_trace()
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size  # 关键 step: 将token id 还原回来

        beam_scores = next_token_scores[:, :num_beams].reshape(-1)
        beam_next_tokens = next_tokens[:, :num_beams].reshape(-1)
        beam_idx = next_indices[:, :num_beams].reshape(-1)

        # beam_idx_offset: beam_idx contains sequence indicies relative to each individual batch. We need to offset the indicies to retrieve the correct sequence in the corresponding batch
        # for example, when batch_size = 2, beam_size = 3, beam_idx_offset = [0, 0, 0, 3, 3, 3]
        decoder_input_ids = torch.cat([decoder_input_ids[beam_idx + beam_idx_offset, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        return decoder_input_ids, beam_scores
