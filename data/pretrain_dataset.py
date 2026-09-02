import torch
from datasets import load_dataset
from torch.utils.data.dataset import Dataset


class PretrainDataset(Dataset):
    def __init__(self, file_path: str, tokenizer = None, max_length: int = 1024):
        super().__init__()
        self.samples = load_dataset(path='json', data_files=file_path, split='train')
        self.max_length = max_length

        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 拿到一条文本
        sample = self.samples[index]['text']
        # 至少要留两个位置给开头的bos和结尾的eos
        input_ids = self.tokenizer(sample, max_length = self.max_length - 2).input_ids
        input_ids = [self.bos_token_id] + input_ids + [self.eos_token_id]
        # 长度不够的要进行padding填充，保证长度一致
        input_ids += [self.pad_token_id] * (self.max_length - len(input_ids))
        input_ids = torch.tensor(input_ids)

        labels = input_ids.clone()
        # padding的地方不需要进行预测，把padding的位置的token_id标记为-100，这样就不会计算这些位置的损失
        labels[input_ids == self.pad_token_id] = -100

        return {
            "input_ids" : input_ids, 
            "labels": labels
        }



if __name__ == "__main__":
    dataset = PretrainDataset('./datasets/pretrain_t2t_mini.jsonl')