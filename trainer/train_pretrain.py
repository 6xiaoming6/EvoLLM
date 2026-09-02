import torch
import argparse
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

from models.evollm import EvoLLMForCausalLM
from configs.evollm_config import EvoLLMConfig
from data.pretrain_dataset import PretrainDataset


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Evollm Pretrain")
    parse.add_argument("--batch_size", type=int, default=2, help="batch size")
    parse.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parse.add_argument("--learning_rate","-lr", type=float, default=1e-5, help="学习率")
    parse.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW里面的weight_decay")
    parse.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="训练设备")
    parse.add_argument("--tokenizer", type=str, default='./models/tokenizer/minimind/', help="分词器")
    parse.add_argument("--data_path", type=str, default='./data/datasets/pretrain_t2t_mini.jsonl', help="训练数据的文件路径")
    

    args = parse.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    cfg = EvoLLMConfig()
    model = EvoLLMForCausalLM(cfg)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    dataset = PretrainDataset(file_path=args.data_path, tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    model.train()
    totla_loss, total_step = 0, 0
    for epoch in range(epochs):
        print(f'开始训练{epoch} / {epochs}')
        for step, batch in enumerate(dataloader):
            input_ids, labels = batch['input_ids'], batch['labels']


            output = model(input_token_ids = input_ids, labels = labels)
            loss = output['loss']
            totla_loss += loss.item()
            total_step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={step + 1} "
                    f"loss={totla_loss / total_step:.4f}"
                )
                totla_loss = 0
                total_step = 0
