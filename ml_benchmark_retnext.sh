#for seed in 0 1 42; do
#	aidsorb-lit fit -c configs/multitask_learning.yaml --seed_everything=${seed}
#done

for seed in 0; do
	aidsorb-lit fit -c configs/final_all_multitask_learning.yaml --seed_everything=${seed}
done
