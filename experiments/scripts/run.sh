# nohup bash run_FrontierScience.sh \
#     --smoke \
#     --method experiments/configs/methods/noagent_emptymemory.yaml \
#     > 2026_3_2_test_frontierscience_noagent_emptymemory.log 2>&1 &
nohup bash run_HLE.sh \
    --debug \
    > 2026_3_11_test_hle_single_agent_emptymemory.log 2>&1 &
