from sklearn.metrics import accuracy_score, v_measure_score
import numpy as np

simulated_labels = snakemake.input['simulated_labels']
estimated_labels = snakemake.input['estimated_labels']
output_file = snakemake.output['fname']

true_nodes = np.loadtxt(simulated_labels, delimiter=',', dtype='str')
estimated_nodes = np.loadtxt(estimated_labels, delimiter=',', dtype='str')

true_clones = np.array([n[0] for n in true_nodes])
estimated_clones = np.array([n[0] for n in estimated_nodes])

node_accuracy = accuracy_score(estimated_nodes, true_nodes)
node_ari = v_measure_score(estimated_nodes, true_nodes)

clone_accuracy = accuracy_score(estimated_clones, true_clones)
clone_ari = v_measure_score(estimated_clones, true_clones)

labels = 'clone_accuracy,clone_ari,node_accuracy,node_ari'
vals = ','.join([str(clone_accuracy), str(clone_ari), str(node_accuracy), str(node_ari)])

with open(output_file, 'w') as file:
    file.write(labels + '\n')
    file.write(vals)
