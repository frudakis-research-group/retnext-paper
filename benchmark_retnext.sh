train_sizes='[Null]'
n_runs=3

for labels in "['Hydrogen_1_77K_2bar_g/l']" \
	"['Hydrogen_1_77K_100bar_g/l']" \
	"['CarbonDioxide_1_298K_0.05bar_mol/kg']" \
	"['CarbonDioxide_1_298K_0.5bar_mol/kg']" \
	"['CarbonDioxide_1_298K_2.5bar_mol/kg']" \
	"['Methane_1_298K_0.05bar_mol/kg']" \
	"['Methane_1_298K_0.5bar_mol/kg']" \
	"['Methane_1_298K_2.5bar_mol/kg']"; do

	echo -e "\033[31;1mRunning experiments for ${labels}\033[0m\n"

	#for config in configs/{boltzmann,cubic_boltzmann,augmentation_cubic_boltzmann}.yaml; do
	#for config in configs/{clip,cubic_clip,augmentation_cubic_clip}.yaml; do
	#for config in configs/{zero_clip,cubic_zero_clip,augmentation_cubic_zero_clip}.yaml; do
	for config in configs/{cubic_zero_clip,augmentation_cubic_zero_clip}.yaml; do

		python benchmark.py -c ${config} \
			--data.labels="${labels}" \
			--train_sizes="${train_sizes}" \
			--n_runs="${n_runs}"
	done
done
