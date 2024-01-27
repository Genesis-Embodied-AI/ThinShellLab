python training_env.py \
  --num_env 1 \
  --num_eval_envs 1 \
  --tot_step 50 \
  --env interact \
  --task_name interact_sep_soft \
  --Kb 0.1 \
  --mu 5.0 \
  --reward_name compute_reward \
  --model SAC
