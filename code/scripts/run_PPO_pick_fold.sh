python training_env.py \
  --num_env 1 \
  --num_eval_envs 1 \
  --tot_step 50 \
  --env pick \
  --task_name pick_fold_RL \
  --Kb 100.0 \
  --mu 5.0 \
  --reward_name compute_reward_pick_fold \
  --model RecurrentPPO