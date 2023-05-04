#!/bin/bash


folder_name="$1"
cluster_algorithm="$2"
num_clusters=-1
if [[ $cluster_algorithm -ne "2" ]]; then
	num_clusters="$3"
fi
summary_length=655

if [[ -d "$folder_name" ]]; then
	echo -n "Calculating tf-idf scores..."
	python3 calculate_idf.py "$folder_name" > /dev/null
	echo "completed"
	echo -n "Creating Clusters..."
	python3 clustering.py "$folder_name" "$cluster_algorithm" "$num_clusters"> /dev/null
	echo "completed"
	echo -n "Producing Summary..."
	python3 summarizer.py "$folder_name"
	echo "completed"
	rm idf.out clusters.out indexForCluster.out
	rm *.pyc
	if [[ $cluster_algorithm -eq "2" ]]; then
		rm edges.p test.gpickle
	fi
else
	echo "Directory not found: $folder_name"
fi
