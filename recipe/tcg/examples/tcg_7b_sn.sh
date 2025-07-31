set -x

# conda_cmd="source activate && conda activate python310_torch25_cuda"
# echo ${conda_cmd}
# eval ${conda_cmd}
# echo "CURRENT RANK: $RANK"
# XFLAGS --disable flash_attn

# PATHS
VERL_PATH=/mnt/zj-gpfs/home/wwu/verl
export PYTHONPATH=${VERL_PATH}:${VERL_PATH}/verl
MODEL_PATH=${MODEL_PATH:-"/mnt/zj-gpfs/home/wwu/models/Qwen2.5-7B"}
TRAIN_FILE=${TRAIN_FILE:-"/mnt/zj-gpfs/home/liyuanyang/tcg/train.parquet"}
TEST_FILE=${TEST_FILE:-"/mnt/zj-gpfs/home/liyuanyang/tcg/test.parquet"}

# ---------------------------
# 环境变量（可提前注入或由调度系统自动设置）
# ---------------------------
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
export WG_BACKEND='ray'
NNODES=${NNODES:-1}
MASTER_PORT=${MASTER_PORT:-6379}
DASHBOARD_PORT=${DASHBOARD_PORT:-8265}
CHECK_DIR=${CHECK_DIR:-"${VERL_PATH}/grpo_ray_check_dir"}
mkdir -p ${CHECK_DIR}

# Wandb
proj_name='TCG'
export WANDB_API_KEY=${WANDB_API_KEY:-"5229f53ade5d746040311fe9429131a5c249ad9f"}

# DAPO Settings
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 4))
max_start_length=${max_prompt_length}
max_obs_length=2048
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 1))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 3))
train_prompt_mini_bsz=16
n_resp_per_prompt=8

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Performance Related Parameter
sp_size=2
use_dynamic_bsz=True
infer_micro_batch_size=null
train_micro_batch_size=null
offload=True
gen_tp=4

# Redis Configuration
export SEARCH_REDIS_HOST='10.200.99.220'
export SEARCH_REDIS_PORT=32132

# TCG Configuration
do_search=True
do_code=True
coder_url="http://10.200.99.220:31773/run"

# ---------------------------
# Head 节点逻辑
# ---------------------------
if [ "$RANK" -eq "0" ]; then
    echo "[HEAD $RANK] Starting Ray head node..."

    # 获取本机 IP（确保是内网 IP，可连接）
    NODE_IP=$(hostname -I | awk '{print $1}')
    echo "$NODE_IP" > ${CHECK_DIR}/head_ip.txt
    echo "[HEAD] IP is $NODE_IP, written to ${CHECK_DIR}/head_ip.txt"

    ray start --head \
        --node-ip-address=${NODE_IP} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${DASHBOARD_PORT} \
        --port=${MASTER_PORT} \
        --include-dashboard=True \
        --num-gpus=8

    touch ${CHECK_DIR}/RANK_${RANK}.ready
    echo "[HEAD] Waiting for all ${NNODES} workers to become ready..."

    while [ $(ls ${CHECK_DIR}/RANK_*.ready 2>/dev/null | wc -l) -lt $NNODES ]; do
        sleep 1
    done

    echo "[HEAD] All workers are ready. Waiting for cluster resources..."
    sleep 20

    echo "[HEAD] All workers are ready. Showing cluster status:"
    ray status

    echo "[HEAD] Launching training script..."

    rm -f ${CHECK_DIR}/RANK_*.ready
    rm -f ${CHECK_DIR}/head_ip.txt

    exp_name="tcg_7b_$(date +%Y%m%d_%H%M%S)"
    CKPTS_DIR=${CKPTS_DIR:-"/mnt/zj-gpfs/home/wwu/ckpts/${proj_name}/${exp_name}"} # 修改这个dir可以控制resume
    mkdir -p ${CKPTS_DIR}

    python3 -m recipe.tcg.src.main_tcg \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.prompt_key=prompt \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.max_start_length=${max_start_length} \
        data.max_obs_length=${max_obs_length} \
        data.gen_batch_size=${gen_prompt_bsz} \
        data.train_batch_size=${train_prompt_bsz} \
        data.truncation='left' \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.n_agent=${n_resp_per_prompt} \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        algorithm.adv_estimator=${adv_estimator} \
        algorithm.use_kl_in_reward=${use_kl_in_reward} \
        algorithm.kl_ctrl.kl_coef=${kl_coef} \
        algorithm.filter_groups.enable=${enable_filter_groups} \
        algorithm.filter_groups.metric=${filter_groups_metric} \
        algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        +actor_rollout_ref.model.override_config.attention_dropout=0. \
        +actor_rollout_ref.model.override_config.embd_pdrop=0. \
        +actor_rollout_ref.model.override_config.resid_pdrop=0. \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        actor_rollout_ref.actor.optim.weight_decay=0.1 \
        actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        actor_rollout_ref.actor.ppo_micro_batch_size=${train_micro_batch_size} \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.grad_clip=1.0 \
        actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=${infer_micro_batch_size} \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
        actor_rollout_ref.rollout.enable_chunked_prefill=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k="${top_k}" \
        actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
        actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
        actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size=${infer_micro_batch_size} \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        reward_model.reward_manager=dapo \
        reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
        reward_model.overlong_buffer.len=${overlong_buffer_len} \
        reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
        trainer.logger=['console','wandb'] \
        trainer.project_name="${proj_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.max_actor_ckpt_to_keep=3 \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes="${NNODES}" \
        trainer.val_before_train=True \
        trainer.test_freq=5 \
        trainer.save_freq=25 \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.total_epochs=50 \
        retriever.topk=1 \
        do_search=${do_search} \
        do_code=${do_code} \
        coder.url="${coder_url}" \
        redis.host="${SEARCH_REDIS_HOST}" \
        redis.port="${SEARCH_REDIS_PORT}" $@ | tee "${CKPTS_DIR}/training_output.log"
    
    echo "[HEAD] Training completed. Cleaning up..."
    ray stop


# ---------------------------
# Worker 节点逻辑
# ---------------------------
else
    echo "[WORKER $RANK] Waiting for head IP to appear at ${CHECK_DIR}/head_ip.txt..."
    sleep 10
    while [ ! -f ${CHECK_DIR}/head_ip.txt ]; do
        sleep 1
    done
    sleep 20

    MASTER_ADDR=$(cat ${CHECK_DIR}/head_ip.txt)
    echo "[WORKER $RANK] Read MASTER_ADDR=${MASTER_ADDR}, connecting to Ray head..."

    ray start --address=${MASTER_ADDR}:${MASTER_PORT} --num-gpus=8
    touch ${CHECK_DIR}/RANK_${RANK}.ready

    echo "[WORKER $RANK] Joined cluster. Sleeping to keep process alive..."
    sleep infinity
fi