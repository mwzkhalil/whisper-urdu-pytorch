import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import torch.optim as optim
import re
import os
from liger_kernel.transformers import LigerLayerNorm
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

from transformers import AutoTokenizer
import jiwer
import numpy as np

import librosa
from jiwer import wer
from datasets import concatenate_datasets, load_dataset

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b") 
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

SOT = '<|startoftranscript|>'
EOT = '<|endoftranscript|>'
transcribe = '<|transcribe|>'
prev = '<|prev|>'
special_tokens_dict = {'additional_special_tokens': [SOT, EOT, transcribe, prev]}
tokenizer.add_special_tokens(special_tokens_dict)

# Hyper-parameters
epochs = 2
block_size = 64
batch_size = 128
tgt_vocab_size = len(tokenizer)
embeddings_dims = 512
attn_dropout = 0.1
no_of_heads = 4
dropout = 0.1
max_lr = 1.5e-3
no_of_decoder_layers = 6
weight_decay_optim = 0.1
log_mel_features = 80
kernel_size = 3
stride = (2, 10)
sr = 16000
device = 'cuda:0'
SAMPLING_RATE = 16000
N_MELS = 80
WINDOW_DURATION = 0.025
STRIDE_DURATION = 0.010
max_t = 500
n_channels = N_MELS
clip = 1.0
use_flash_attention = True
use_liger = True
use_torch_compile = False
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
eps = 1e-6
beta_1 = 0.9
beta_2 = 0.98
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)


cv_ds = load_dataset("mozilla-foundation/common_voice_11_0", "ur")
validation_data = cv_ds['validation']
combined = concatenate_datasets([cv_ds['train'], cv_ds['test']])
split_result = combined.train_test_split(test_size=2000, shuffle=True, seed=42)
train_data = split_result['train']
split_tmp = validation_data.train_test_split(test_size=0.5, shuffle=True, seed=42)
test_data = split_tmp['test']


MAX_DURATION_IN_SECONDS = 30
def is_audio_length_in_range(d): return d < MAX_DURATION_IN_SECONDS

for split, name in [(train_data, 'train'), (validation_data, 'val'), (test_data, 'test')]:
    dur_col = [librosa.get_duration(path=x['audio']['path']) for x in tqdm(split, desc=f'duration-{name}')]
    split = split.add_column("duration", dur_col)
    split = split.filter(is_audio_length_in_range, input_columns=["duration"])
    globals()[f'truncated_cv_{name}'] = split.remove_columns(["duration"])


def _save_snapshot(model, optimizer, scheduler, epoch, step):
    snapshot = {
        "MODEL_STATE": model.module.state_dict(),
        "OPTIMIZER_STATE": optimizer.state_dict(),
        "EPOCHS_RUN": epoch,
        "STEP_RUN": step
    }
    torch.save(snapshot, f"snapshot_{step}.pt")
    print(f"Epoch: {epoch} | Step: {step} | Snapshot saved.")

def _load_snapshot(snapshot_path, model, optimizer, scheduler):
    snapshot = torch.load(snapshot_path)
    model.load_state_dict(snapshot["MODEL_STATE"])
    optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
    epoch = snapshot["EPOCHS_RUN"]
    step = snapshot["STEP_RUN"]
    print(f"Resuming from Epoch {epoch}, Step {step}")
    return epoch, step


def tokenize_function(examples):
    return tokenizer(examples['sentence'], max_length=block_size,
                     padding='max_length', truncation=True, return_tensors='pt')


def prepare_dataset(split, device, batch_size):
    def collate_fn(batch):
        def pad_to_max_t(spec, max_t):
            n_mels, t = spec.shape
            if t < max_t:
                spec = np.pad(spec, ((0, 0), (0, max_t - t)), mode='constant')
            else:
                spec = spec[:, :max_t]
            return spec

        def clean(txt):
            return re.sub(r'<[^>]*>', '', txt)

        n_fft = int(SAMPLING_RATE * WINDOW_DURATION)
        hop_length = int(SAMPLING_RATE * STRIDE_DURATION)

        specs, ids, labels, texts = [], [], [], []

        for item in batch:
            spec = librosa.feature.melspectrogram(
                y=item['audio']['array'],
                sr=SAMPLING_RATE,
                n_mels=N_MELS,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                fmax=SAMPLING_RATE // 2
            )
            spec = librosa.power_to_db(spec, ref=np.max)
            spec = pad_to_max_t(spec, block_size)
            spec = torch.tensor(spec, dtype=torch.float32)

            text = clean(item['sentence']).strip()
            text = SOT + 'ur' + transcribe + text + EOT
            tok = tokenizer(text, truncation=True, padding='max_length',
                            max_length=block_size, return_tensors='pt')

            spec_min, spec_max = spec.min(), spec.max()
            spec = 2 * ((spec - spec_min) / (spec_max - spec_min + 1e-8)) - 1

            tok['labels'] = tok['input_ids'].clone()
            tok['labels'][:, :-1] = tok['input_ids'][:, 1:]
            tok['labels'][:, -1] = tokenizer.eos_token_id

            specs.append(spec)
            ids.append(tok['input_ids'].squeeze(0))
            labels.append(tok['labels'].squeeze(0))
            texts.append(clean(item['sentence']))

        return {
            "real_text": texts,
            'spectrogram': torch.stack(specs),
            'input_ids': torch.stack(ids),
            'labels': torch.stack(labels)
        }

    ds = {'train': truncated_cv_train, 'val': truncated_cv_val, 'test': truncated_cv_test}[split]
    return DataLoader(
        ds,
        batch_size=batch_size if split != 'test' else 1,
        sampler=DistributedSampler(ds, shuffle=(split == 'train')),
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=False,
        pin_memory=True
    )

class PositionEmbeddings(nn.Module):
    def __init__(self, embeddings_dims=embeddings_dims, block_size=block_size):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.randn(1, block_size, embeddings_dims, device=device), requires_grad=True)
    def forward(self, x): return self.position_embeddings

class TgtTextEmbeddings(nn.Module):
    def __init__(self, vocab_size=tgt_vocab_size, embeddings_dims=embeddings_dims):
        super().__init__()
        self.embeddings_table = nn.Embedding(vocab_size, embeddings_dims, device=device)
    def forward(self, x): return self.embeddings_table(x)

class LayerNormalization(nn.Module):
    def __init__(self, embeddings_dims=embeddings_dims):
        super().__init__()
        self.norm = LigerLayerNorm(embeddings_dims) if use_liger else nn.LayerNorm(embeddings_dims)
    def forward(self, x): return self.norm(x)

class MLPBlock(nn.Module):
    def __init__(self, dropout=dropout, embeddings_size=embeddings_dims):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embeddings_size, 4 * embeddings_dims, device=device),
            nn.GELU(),
            nn.Linear(4 * embeddings_dims, embeddings_size, device=device),
            nn.Dropout(dropout)
        )
    def forward(self, x): return self.mlp(x)

class MaskedAttentionHead(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        self.no_of_heads = no_of_heads
        if not use_flash_attention:
            self.query = nn.Linear(embeddings_dims, self.head_size, device=device, bias=False)
            self.keys = nn.Linear(embeddings_dims, self.head_size, device=device, bias=False)
            self.values = nn.Linear(embeddings_dims, self.head_size, device=device, bias=False)
        else:
            self.qkv_proj = nn.Linear(embeddings_dims, 3 * embeddings_dims, bias=False, device=device)
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, x):
        B, T, C = x.shape
        if not use_flash_attention:
            q = self.query(x); k = self.keys(x); v = self.values(x)
            mask = torch.tril(torch.ones(T, T, device=device))
            wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
            wei = wei.masked_fill(mask == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            return wei @ v
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, self.no_of_heads, self.head_size).transpose(1, 2)
            k = k.view(B, T, self.no_of_heads, self.head_size).transpose(1, 2)
            v = v.view(B, T, self.no_of_heads, self.head_size).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
            return out

class MaskedMHA(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.heads = nn.ModuleList([MaskedAttentionHead(attn_dropout, embeddings_dims, no_of_heads) for _ in range(no_of_heads)])
        self.linear = nn.Linear(no_of_heads * embeddings_dims, embeddings_dims, device=device, bias=False)
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, x): return self.dropout(self.linear(torch.cat([h(x) for h in self.heads], dim=-1)))

class CrossAttentionHead(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        if not use_flash_attention:
            self.query = nn.Linear(embeddings_dims, self.head_size, device=device, bias=False)
            self.keys = nn.Linear(embeddings_dims, self.head_size, device=device, bias=False)
            self.values = nn.Linear(embeddings_dims, self.head_size, device=device, bias=False)
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, query, key, value, mask=None):
        B, T, C = query.shape
        if not use_flash_attention:
            q = self.query(query); k = self.keys(key); v = self.values(value)
            wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
            if mask is not None:
                wei = wei.masked_fill(mask == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            return wei @ v
        else:
            q = query.view(B, T, no_of_heads, self.head_size).transpose(1, 2)
            k = key.view(B, key.shape[1], no_of_heads, self.head_size).transpose(1, 2)
            v = value.view(B, value.shape[1], no_of_heads, self.head_size).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
            return out

class CrossMHA(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.heads = nn.ModuleList([CrossAttentionHead(attn_dropout, embeddings_dims, no_of_heads) for _ in range(no_of_heads)])
        self.linear = nn.Linear(no_of_heads * embeddings_dims, embeddings_dims, device=device, bias=False)
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, value, key, query, mask=None):
        out = torch.cat([h(query, key, value, mask) for h in self.heads], dim=-1)
        return self.dropout(self.linear(out))

class TransformerDecoderBlock(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout):
        super().__init__()
        self.masked = MaskedMHA(attn_dropout, embeddings_dims, no_of_heads)
        self.cross = CrossMHA(attn_dropout, embeddings_dims, no_of_heads)
        self.ln1 = LayerNormalization(embeddings_dims)
        self.ln2 = LayerNormalization(embeddings_dims)
        self.ln3 = LayerNormalization(embeddings_dims)
        self.mlp = MLPBlock(dropout, embeddings_dims)
    def forward(self, key, value, x, mask=None):
        x = x + self.ln1(self.masked(x))
        x = x + self.ln2(self.cross(value, key, x, mask))
        x = x + self.ln3(self.mlp(x))
        return x

class DecoderModel(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, block_size=block_size, dropout=dropout, no_of_decoder_layers=no_of_decoder_layers):
        super().__init__()
        self.pos = PositionEmbeddings()
        self.layers = nn.ModuleList([TransformerDecoderBlock(attn_dropout, embeddings_dims, no_of_heads, dropout) for _ in range(no_of_decoder_layers)])
        self.ln = LayerNormalization(embeddings_dims)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.02); nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, key, value, x, mask):
        x = x + self.pos(x)[:, :x.shape[1], :]
        for layer in self.layers: x = layer(key, value, x, mask)
        return self.ln(x)

class FullAttentionHead(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.head_size = embeddings_dims // no_of_heads
        if not use_flash_attention:
            self.query = nn.Linear(embeddings_dims, self.head_size, device=device, bias=False)
            self.keys = nn.Linear(embeddings_dims, self.head_size, device=device, bias=False)
            self.values = nn.Linear(embeddings_dims, self.head_size, device=device, bias=False)
        else:
            self.qkv_proj = nn.Linear(embeddings_dims, 3 * embeddings_dims, bias=False, device=device)
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        if not use_flash_attention:
            q = self.query(x); k = self.keys(x); v = self.values(x)
            wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)
            if mask is not None:
                wei = wei.masked_fill(mask == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            return self.dropout(wei @ v)
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, no_of_heads, self.head_size).transpose(1, 2)
            k = k.view(B, T, no_of_heads, self.head_size).transpose(1, 2)
            v = v.view(B, T, no_of_heads, self.head_size).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
            out = out.transpose(1, 2).contiguous().view(B, T, -1)
            return out

class FullMHA(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads):
        super().__init__()
        self.heads = nn.ModuleList([FullAttentionHead(attn_dropout, embeddings_dims, no_of_heads) for _ in range(no_of_heads)])
        self.linear = nn.Linear(no_of_heads * embeddings_dims, embeddings_dims, device=device, bias=False)
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, x, mask=None):
        out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        return self.dropout(self.linear(out))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads, dropout=dropout):
        super().__init__()
        self.mha = FullMHA(attn_dropout, embeddings_dims, no_of_heads)
        self.ln1 = LayerNormalization(embeddings_dims)
        self.ln2 = LayerNormalization(embeddings_dims)
        self.mlp = MLPBlock(dropout, embeddings_dims)
    def forward(self, x, mask=None):
        x = x + self.ln1(self.mha(x, mask))
        x = x + self.ln2(self.mlp(x))
        return x

class EncoderModel(nn.Module):
    def __init__(self, attn_dropout=attn_dropout, embeddings_dims=embeddings_dims, no_of_heads=no_of_heads,
                 block_size=block_size, dropout=dropout, no_of_decoder_layers=no_of_decoder_layers):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, embeddings_dims, kernel_size=kernel_size, device=device, padding=1)
        self.conv2 = nn.Conv1d(embeddings_dims, embeddings_dims, kernel_size=kernel_size, device=device, padding=1)
        self.pos = PositionEmbeddings()
        self.layers = nn.ModuleList([TransformerEncoderBlock(attn_dropout, embeddings_dims, no_of_heads, dropout) for _ in range(no_of_decoder_layers)])
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.02); nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, x, mask):
        x = self.conv1(x); x = F.gelu(x)
        x = self.conv2(x); x = F.gelu(x)
        x = x.permute(0, 2, 1) + self.pos(x.permute(0, 2, 1))
        for layer in self.layers: x = layer(x, mask)
        return x

class Whisper(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()
        self.tgt_text_embds = TgtTextEmbeddings(tgt_vocab_size, embeddings_dims)
        self.linear = nn.Linear(embeddings_dims, tgt_vocab_size, device=device, bias=False)
        self.le_loss = LigerFusedLinearCrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(device)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, actual_labels=None, inference=False):
        x = self.encoder(src, src_mask)
        y = self.tgt_text_embds(tgt)
        y = self.decoder(x, x, y, tgt_mask)
        if inference:
            return self.linear(y)
        if use_liger:
            y = y.contiguous().view(-1, embeddings_dims)
            labels = actual_labels.contiguous().view(-1)
            return self.le_loss(self.linear.weight, y, labels)
        else:
            logits = self.linear(y)
            return logits


def topk_sampling(model, batch, max_length=300, top_k=50, temperature=1.0, device='cuda'):
    spec = batch['spectrogram'].to(device)
    prompt = "<|startoftranscript|>ur<|transcribe|>"
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    generated = input_ids.clone()
    model.eval()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for _ in range(max_length):
            logits = model(spec, generated, inference=True)[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            next_token = torch.multinomial(top_probs, 1)
            next_token = top_indices.gather(-1, next_token)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id: break
    transcript = tokenizer.decode(generated[0], skip_special_tokens=True)
    return batch['real_text'][0], transcript


save_chechpoint_iter = 50
total_iters = 20000
eval_iters = 50
eval_check = 50
warmup_iters = 700
min_lr = 3e-6
lr_decay_iters = 20000
total_batch_size = 32768
micro_batch_size = batch_size
world_size = torch.cuda.device_count()
gradient_accumulation_steps = total_batch_size // (micro_batch_size * block_size * world_size)

def get_lr(it):
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def setup():
    init_process_group("nccl")

def cleanup():
    destroy_process_group()

def train():
    setup()
    device_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(device_id)
    if device_id == 0:
        wandb.init(project="CV-Urdu-DDP")

    model = Whisper().to(device_id)
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, betas=(beta_1, beta_2),
                            weight_decay=weight_decay_optim, eps=eps, fused=True)
    if use_torch_compile:
        model = torch.compile(model)
    model = DDP(model, device_ids=[device_id])

    def compute_wer(ref, hyp):
        return jiwer.wer(ref, hyp)

    train_loader = prepare_dataset('train', device_id, batch_size)
    val_loader = prepare_dataset('val', device_id, batch_size)
    test_loader = prepare_dataset('test', device_id, 1)

    train_it = iter(train_loader)
    val_it = iter(val_loader)
    test_it = iter(test_loader)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    for step in tqdm(range(total_iters)):
        # evaluationn
        if (step % eval_iters == 0 and step != 0) or step == total_iters - 1:
            model.eval()
            total_loss = 0; n = 0
            for _ in range(eval_check):
                try:
                    b = next(val_it)
                except StopIteration:
                    val_it = iter(val_loader); b = next(val_it)
                idx = b['input_ids'].to(device_id)
                tgt = b['labels'].to(device_id)
                spec = b['spectrogram'].to(device_id)
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = model(spec, idx, actual_labels=tgt)
                total_loss += loss.item(); n += 1
            val_loss = total_loss / n
            torch.distributed.reduce(torch.tensor([val_loss]).to(device_id), dst=0, op=torch.distributed.ReduceOp.SUM)
            val_loss = val_loss / world_size
            if device_id == 0:
                print(f"step {step}: val loss {val_loss:.4f}")
                wandb.log({"val_loss": val_loss})
            model.train()


        if step % save_chechpoint_iter == 0 and device_id == 0 and step != 0:
            _save_snapshot(model, optimizer, None, None, step)

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0
        for micro in range(gradient_accumulation_steps):
            try:
                b = next(train_it)
            except StopIteration:
                train_it = iter(train_loader); b = next(train_it)
            idx = b['input_ids'].to(device_id)
            tgt = b['labels'].to(device_id)
            spec = b['spectrogram'].to(device_id)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = model(spec, idx, actual_labels=tgt) / gradient_accumulation_steps
            accum_loss += loss.detach()
            model.require_backward_grad_sync = (micro == gradient_accumulation_steps - 1)
            scaler.scale(loss).backward()

        lr = get_lr(step)
        for g in optimizer.param_groups: g['lr'] = lr
        scaler.step(optimizer)
        scaler.update()

        torch.distributed.reduce(accum_loss, dst=0, op=torch.distributed.ReduceOp.SUM)
        if device_id == 0:
            wandb.log({"train_loss": accum_loss.item() / world_size, "lr": lr})

        if device_id == 0 and step % 20 == 0:
            try:
                b = next(test_it)
            except StopIteration:
                test_it = iter(test_loader); b = next(test_it)
            ref, hyp = topk_sampling(model, b, max_length=50, top_k=50, temperature=1.0, device=device_id)
            wer = compute_wer(ref, hyp)
            print(f"step {step} | WER: {wer:.3f} | ref: {ref} | hyp: {hyp}")
            wandb.log({"sample_wer": wer})

    if device_id == 0:
        wandb.finish()
    cleanup()

if __name__ == "__main__":
    train()
