# run on 2xGPU  
# make sure your current working directory is the root of the project
# Using QuantizedRL with DYNAMIC INT8 quantization (FlashRL approach)
# Model: Qwen2.5-7B (same as FlashRL's sglang_patch testing)
#
# Requirements:
#   1. FlashRL installed: pip install flash-rl
#   2. Profile file at /root/profile.7b.pt (for weight reload quantization)
#   3. Pre-quantized model for fast initial load
#   4. Set load_format=quantized_rl (see line 82 below)
#
# How it works:
#   - VERL Actor: Trains with Qwen/Qwen2.5-7B-Instruct (BF16)
#   - SGLang Initial Load: Uses pre-quantized model (INT8, fast startup)
#   - SGLang Weight Reloads: Quantizes BF16 weights to INT8 using profile
#
# PyTorch Profiling:
#   - Enable with SGLANG_ENABLE_PROFILER=1
#   - Profile first N steps with SGLANG_PROFILE_STEPS=5
#   - Output directory: SGLANG_PROFILER_DIR=/root/sglang_profiles
#   - View traces in Chrome: chrome://tracing
#   - Or use NSight Systems: nsys-ui

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

function now() {
    date '+%d-%H-%M'
}

EXPERIMENT_NAME="qwen2.5-7b_quantized_rl_$(now)"
export CUDA_VISIBLE_DEVICES=2,3

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.88 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.over_sample_rate=0.1 \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='quantized_rl' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=disable \
    actor_rollout_ref.rollout.load_format=quantized_rl \
    +actor_rollout_ref.rollout.quantized_rl_model=/root/.cache/huggingface/hub/models--RedHatAI--Qwen2.5-7B-Instruct-quantized.w8a8 \
    +actor_rollout_ref.rollout.quant_profile_path=/root/profile.7b.pt \
    trainer.total_epochs=15 $@


