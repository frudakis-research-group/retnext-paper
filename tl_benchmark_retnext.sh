#for target_task in 'CarbonDioxide_1_298K_0.5bar_mol/kg' \
#	'Methane_1_298K_0.05bar_mol/kg' \
#	'Hydrogen_1_77K_2bar_g/l' \
#	'Xenon_0.2_273K_1bar_mol/kg' \
#	'Krypton_0.8_273K_5bar_mol/kg' \
#	'Krypton_0.8_273K_10bar_mol/kg'; do
#
#	echo -e "\033[31;1mTarget task: ${target_task}\033[0m\n"
#
#	for ckpt_path in 'scratch'; do
#
#		python transfer_learning.py -c configs/transfer_learning.yaml \
#			--ckpt_path=${ckpt_path} \
#			--target="${target_task}" \
#			--voxels_path='/home/asarikas/databases/MOFXDB/hMOF/voxels_data_GS32_CB30' \
#			--labels_path='/home/asarikas/databases/MOFXDB/hMOF/hMOF.csv' \
#			--index_col='name' \
#			--lineval=False
#	done
#done

#for target_task in 'CO2_uptake_P0.15bar_T298K [mmol/g]' \
#	'CO2_uptake_P0.10bar_T363K [mmol/g]' \
#	'working_capacity_vacuum_swing [mmol/g]' \
#	'working_capacity_temperature_swing [mmol/g]' \
#	'CO2_binary_uptake_P0.15bar_T298K [mmol/g]' \
#	'N2_binary_uptake_P0.85bar_T298K [mmol/g]' \
#	'logSelectivity' \
#	'excess_CO2_uptake_P0.10bar_T363K [mmol/g]' \
#	'excess_CO2_binary_uptake_P0.15bar_T298K [mmol/g]'; do
#
#
#	echo -e "\033[31;1mTarget task: ${target_task}\033[0m\n"
#
##	# For lineval use extracted embeddings since it is faster!
#	#for ckpt_path in 'scratch' \
#	#	'ml_experiments/augmentation_cubic_boltzmann/final_all/lightning_logs/version_0/checkpoints/best.ckpt'; do
#
#		python transfer_learning.py -c configs/transfer_learning.yaml \
#			--ckpt_path=${ckpt_path} \
#			--target="${target_task}" \
#			--voxels_path='/home/asarikas/databases/UO/extracted_data/voxels_data_GS32_CB30' \
#			--labels_path='/home/asarikas/databases/UO/extracted_data/csv/logSelectivity_all_MOFs_screening_data.csv' \
#			--index_col='MOFname' \
#			--lineval=False
#	done
#done
#
##for target_task in 'Hydrogen_1_130K_100bar_g/l' \
#	#'Methane_1_298K_65bar_cm3(STP)/cm3' \
#	#'Hydrogen_1_160K_5bar_g/l' \
#	#'Hydrogen_1_77K_100bar_g/l' \
#for target_task in 'Xenon_0.2_298K_5bar_mmol/g' \
#	'Hydrogen_1_77K_6bar_g/l' \
#	'Hydrogen_1_200K_100bar_g/l' \
#	'Hydrogen_1_243K_100bar_g/l' \
#	'Xenon_0.2_298K_1bar_mmol/g' \
#	'Krypton_0.8_298K_1bar_mmol/g' \
#	'Krypton_0.8_298K_5bar_mmol/g' \
#	'Methane_1_298K_6bar_kj/mol' \
#	'Methane_1_298K_100bar_cm3(STP)/cm3' \
#	'Methane_1_298K_6bar_cm3(STP)/cm3'; do
#
#	echo -e "\033[31;1mTarget task: ${target_task}\033[0m\n"
#
#	# For lineval use extracted embeddings since it is faster!
#	for ckpt_path in 'scratch' \
#		'ml_experiments/augmentation_cubic_boltzmann/final_all/lightning_logs/version_0/checkpoints/best.ckpt'; do
#
#		python transfer_learning.py -c configs/transfer_learning.yaml \
#			--ckpt_path=${ckpt_path} \
#			--target="${target_task}" \
#			--voxels_path='/home/asarikas/databases/MOFXDB/Tobacco/voxels_data_GS32_CB30' \
#			--labels_path='/home/asarikas/databases/MOFXDB/Tobacco/float_Tobacco.csv' \
#			--index_col='name' \
#			--trainer.default_root_dir='Tobacco_tl_experiments/' \
#			--bias_init=True \
#			--lineval=False
#
#	done
#done
