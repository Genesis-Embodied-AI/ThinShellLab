python training/training_env.py \
  --num_env 1 \
  --num_eval_envs 1 \
  --tot_step 50 \
  --env interact \
  --task_name following \
  --Kb 100.0 \
  --mu 5.0 \
  --reward_name compute_reward_1 \
  --model RecurrentPPO