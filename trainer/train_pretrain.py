import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer

from models.evollm import EvoLLMForCausalLM
from configs.evollm_config import EvoLLMConfig
from data.pretrain_dataset import PretrainDataset

epochs = 2
batch_size = 2
learning_rate = 1e-5
weight_decay = 1e-4
update_step = 10

cfg = EvoLLMConfig()
model = EvoLLMForCausalLM(cfg)
tokenizer = AutoTokenizer.from_pretrained('./models/tokenizer/minimind/')
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

dataset = PretrainDataset(file_path='./data/dataset/pretrain_t2t_mini.jsonl', tokenizer=tokenizer)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

model.train()
for epoch in range(epochs):
    print(f'开始训练{epoch} / {epochs}')
    for step, batch in enumerate(dataloader):
        input_ids, labels = batch['input_ids'], batch['labels']

        optimizer.zero_grad()

        output = model(input_token_ids = input_ids, labels = labels)
        loss = output['loss']

        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(
                f"epoch={epoch} "
                f"step={step} "
                f"loss={loss.item():.4f}"
            )

        