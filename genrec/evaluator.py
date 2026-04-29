import torch
import tqdm

class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k,
            'iid_ratio': self.recall_at_k
        }

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])

    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            for j in range(self.maxk):
                cur_pred = preds[i, j].tolist()
                if cur_pred == cur_label:
                    # import ipdb; ipdb.set_trace()
                    pos_index[i, j] = True
                    break
        return pos_index
    
    def calculate_pos_index_isexist(self, preds, labels_all_list):
        preds = preds.detach().cpu()
        # labels = labels.detach().cpu()


        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            for j in range(self.maxk):
                cur_pred = preds[i, j].tolist()
                # if cur_pred in labels_all_list:
                if cur_pred[:3] in labels_all_list:
                    # import ipdb; ipdb.set_trace()
                    pos_index[i, j] = True
                    # break
        return pos_index
    
    def calculate_pos_index_fine(self, preds, labels, top_position): # =1234
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
            for j in range(self.maxk):
                cur_pred = preds[i, j].tolist()
                if cur_pred[:top_position] == cur_label[:top_position]:
                    pos_index[i, j] = True
                    break
        return pos_index
    
    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()
    # def 
    def calculate_metrics(self, preds, labels, labels_all_list):
        results = {}
        pos_index = self.calculate_pos_index_fine(preds, labels, top_position=3)
        for k in self.config['topk']:
            results[f"ndcg@{k}"] = self.ndcg_at_k(pos_index, k)

        if labels_all_list is None or len(labels_all_list)==0:
            return results
        
        pos_index_isexist = self.calculate_pos_index_isexist(preds, labels_all_list)
        # import ipdb; ipdb.set_trace()
        for k in self.config['topk']:
            results[f"iid_ratio@{k}"] = self.metric2func[f"iid_ratio"](pos_index_isexist, k)

        return results
