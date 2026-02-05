# GPU Optimization Suggestions for Transformer Training in Colab

import torch as t
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

# ============================================================================
# 1. MIXED PRECISION TRAINING (AMP) - ~2-3x speedup
# ============================================================================

class OptimizedTransformerTrainer:
    """
    Optimized version of TransformerTrainer with AMP support
    """
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.args = args
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        self.scaler = GradScaler()  # For mixed precision
        self.step = 0

    def training_step(self, batch):
        """Training step with mixed precision"""
        tokens = batch["tokens"].to(device)
        
        # Use autocast for mixed precision
        with autocast():
            logits = self.model(tokens)
            loss = -get_log_probs(logits, tokens).mean()
        
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        self.step += 1
        wandb.log({"train_loss": loss.item()}, step=self.step)
        return loss


# ============================================================================
# 2. EFFICIENT ATTENTION with Flash Attention / SDPA
# ============================================================================

class OptimizedAttention(nn.Module):
    """
    Use PyTorch's scaled_dot_product_attention (available in PyTorch 2.0+)
    This is much faster than manual attention computation
    """
    def forward(self, normalized_resid_pre):
        # Calculate query, key, value
        q = einops.einsum(
            normalized_resid_pre, self.W_Q,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
        ) + self.b_Q
        
        k = einops.einsum(
            normalized_resid_pre, self.W_K,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
        ) + self.b_K
        
        v = einops.einsum(
            normalized_resid_pre, self.W_V,
            "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
        ) + self.b_V

        # Rearrange for SDPA: needs (batch, n_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)
        
        # Use efficient fused attention kernel
        z = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,  # Automatically applies causal mask
            dropout_p=0.0
        )
        
        # Rearrange back
        z = z.transpose(1, 2)  # (batch, seq, n_heads, d_head)
        
        # Output projection
        attn_out = einops.einsum(
            z, self.W_O,
            "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
        ) + self.b_O
        
        return attn_out


# ============================================================================
# 3. TORCH.COMPILE for PyTorch 2.0+ - ~20-30% speedup
# ============================================================================

def optimize_model_with_compile(model):
    """
    Compile the model for faster execution (PyTorch 2.0+)
    """
    if hasattr(t, 'compile'):
        # Use reduce-overhead mode for better performance on transformers
        model = t.compile(model, mode='reduce-overhead')
        print("✓ Model compiled with torch.compile")
    else:
        print("⚠ torch.compile not available (requires PyTorch 2.0+)")
    return model


# ============================================================================
# 4. GRADIENT ACCUMULATION - Simulate larger batch sizes
# ============================================================================

class GradientAccumulationTrainer:
    """
    Use gradient accumulation to simulate larger batch sizes
    without running out of memory
    """
    def __init__(self, args, model, accumulation_steps=4):
        self.model = model
        self.args = args
        self.accumulation_steps = accumulation_steps
        self.optimizer = t.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scaler = GradScaler()
        self.step = 0

    def training_step(self, batch, batch_idx):
        tokens = batch["tokens"].to(device)
        
        with autocast():
            logits = self.model(tokens)
            loss = -get_log_probs(logits, tokens).mean()
            # Scale loss by accumulation steps
            loss = loss / self.accumulation_steps
        
        self.scaler.scale(loss).backward()
        
        # Only update weights every accumulation_steps
        if (batch_idx + 1) % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            self.step += 1
            wandb.log({"train_loss": loss.item() * self.accumulation_steps}, step=self.step)
        
        return loss


# ============================================================================
# 5. DATALOADER OPTIMIZATIONS
# ============================================================================

def create_optimized_dataloader(dataset, batch_size, is_train=True):
    """
    Optimized DataLoader settings for GPU training
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,           # Already good
        pin_memory=True,         # Already good
        persistent_workers=True, # NEW: Keep workers alive between epochs
        prefetch_factor=2,       # NEW: Prefetch batches
    )


# ============================================================================
# 6. REMOVE UNNECESSARY MEMORY OPERATIONS
# ============================================================================

# In beam search, REMOVE this line (it's often counterproductive):
# t.cuda.empty_cache()  

# Only use empty_cache() if you're actually running out of memory,
# as it can slow things down by forcing synchronization


# ============================================================================
# 7. ENABLE TF32 for even faster matmuls on Ampere+ GPUs (T4, A100, etc)
# ============================================================================

def enable_tf32():
    """
    Enable TF32 on Ampere GPUs for ~2x speedup on matmuls
    (Colab often has T4 GPUs which support this)
    """
    if t.cuda.is_available():
        t.backends.cuda.matmul.allow_tf32 = True
        t.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for faster matrix multiplications")


# ============================================================================
# COMPLETE EXAMPLE: Putting it all together
# ============================================================================

def setup_optimized_training():
    """
    Complete setup for optimized training
    """
    # Enable TF32
    enable_tf32()
    
    # Create model
    model = DemoTransformer(model_cfg).to(device)
    
    # Compile model (PyTorch 2.0+)
    model = optimize_model_with_compile(model)
    
    # Create optimized dataloaders
    train_loader = create_optimized_dataloader(
        dataset_dict["train"], 
        batch_size=args.batch_size,
        is_train=True
    )
    
    # Use gradient accumulation + AMP trainer
    trainer = GradientAccumulationTrainer(
        args, 
        model,
        accumulation_steps=4  # Effectively 4x batch size
    )
    
    return model, trainer, train_loader


# ============================================================================
# QUICK WINS SUMMARY
# ============================================================================

"""
PRIORITY 1 - Mixed Precision (AMP): 
    - Add GradScaler and autocast to training loop
    - Expected speedup: 2-3x
    - Memory savings: ~40%

PRIORITY 2 - Use F.scaled_dot_product_attention:
    - Replace manual attention computation
    - Expected speedup: 1.5-2x for attention
    - Also handles causal masking automatically

PRIORITY 3 - torch.compile (PyTorch 2.0+):
    - One line: model = torch.compile(model)
    - Expected speedup: 20-30%

PRIORITY 4 - Enable TF32:
    - Two lines to enable
    - Expected speedup: ~2x on T4/A100 GPUs (no speedup on older GPUs)

PRIORITY 5 - Gradient Accumulation:
    - Simulate larger batch sizes
    - Better training stability and potentially faster convergence

Total potential speedup: 5-10x faster training!
"""
