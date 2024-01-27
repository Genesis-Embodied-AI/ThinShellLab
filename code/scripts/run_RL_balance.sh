python training_env.py \
  --num_env 1 \
  --num_eval_envs 1 \
  --tot_step 50 \
  --env balance \
  --task_name balance_RL \
  --Kb 100.0 \
  --mu 5.0 \
  --load_dir ../data/balance_state \
  --model SAC