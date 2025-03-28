# Guidelines-Topology-GNNs4RecSys

This is the official repository for the paper "_Practical Guidelines to Reproduce, Benchmark, and Understand Your GNN-based Recommender System: a Topological Perspective_", currently under review.

### Create the environment

First, create the proper environment by:

```
$ pip install -r requirements.txt
```

### Reproducibility datasets
We used Gowalla, Yelp 2018, and Amazon Book datasets. The original links may be found here, where the train/test splitting has already been provided:

- Gowalla: https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data/gowalla
- Yelp 2018: https://github.com/kuandeng/LightGCN/tree/master/Data/yelp2018
- Amazon Book: https://github.com/xiangwang1223/neural_graph_collaborative_filtering/tree/master/Data/amazon-book

After downloading, create three folders ```./data/{dataset_name}```, one for each dataset. Then, run the script ```./map_dataset.py```, by changing the name of the dataset within the script itself. It will generate the train/test files for each dataset in a format compatible for Elliot (i.e., tsv file with three columns referring to user/item).

In case, we also provide the final tsv files for all the datasets in this repo.

### Additional datasets
We directly provide the train/validation/test splittings for Allrecipes and BookCrossing in this repo. As already stated for Gowalla, Yelp 2018, and Amazon Book, create one folder for each dataset in ```./data/{dataset_name}```.

### Replication of prior results
To reproduce the results, run the following:

```
$ CUBLAS_WORKSPACE_CONFIG=:4096:8 python -u start_experiments.py \
$ --dataset {dataset_name} \
$ --model {model_name} 
```
Note that ```CUBLAS_WORKSPACE_CONFIG=:4096:8``` (which may change depending on your configuration) is needed to ensure the complete reproducibility of the experiments (otherwise, PyTorch may run some operations in their non-deterministic version).

### Benchmarking graph CF approaches using alternative baselines
In addition to the graph-based models from above, we train and test four classic (and strong) CF baselines. You can use the exact same script as above to run them.

### Datasets generation
First, create the three folders under `./data/` with the names "yelp2018", "amazon-book", "gowalla". Second, download the train and test files from the following links, and place them in each folder accordingly. 

- Yelp-2018: https://github.com/tanatosuu/svd_gcn/tree/main/datasets/yelp
- Amazon-Book: https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network/tree/main/data/amazon-book
- Gowalla: https://github.com/kuandeng/LightGCN/tree/master/Data/gowalla

Then, create a single "dataset.tsv" file for each dataset, by running the script `create_dataset.py`. You will need to change the dataset name and the train/test names within the script.

After that, you can generate the sub-datasets by running the following script (with arguments):

```
$ python graph_sampling.py \
--dataset <dataset_name> \
--filename <filename_of_the_dataset> \
--sampling_strategies <list_of_sampling_strategies> \
--num_samplings <number_of_sampling> \
--random_seed <random_seed_for_reproducibility>
```

This will create the sub-folders ```./data/<dataset>/node-dropout/``` and/or ```./data/<dataset>/edge-dropout/``` with the tsv files for each sub-dataset within the corresponding sampling strategy. 

Moreover, you will find a file named ```sampling-stats.tsv``` for each of the three datasets, which reports basic statistics about the generated sub-datasets. You will need it for the training and evaluation of the models (see later).

### Models training and evaluation
Now you are all set to train and evaluate the graph-based recommender systems. To do so, you should run the following script (with arguments):

```
$ python start_experiments_topology.py \
--dataset <dataset_name> \
--gpu <gpu_id>
```

If you are curious about the hyper-parameter settings for the models, you may refer to the ```./config_files/``` folder, where all configuration files are stored.

Depending on your workstation, the training and evaluation could take very long time. Remember that you are training and evaluating four graph-based recommender systems on 1,800 datasets!

After the training and evaluation are done, you will find all performance files in the folder ```./results/<dataset_name>/performance/```. Do not worry about them, because there is a script that will collect all results and join them to the dataset characteristics (see later).

### Characteristics calculation and regression

To calculate the dataset characteristics, you should run the following script (with arguments):

```
$ python generate_characteristics.py \
--dataset <dataset_name> \
--start <start_dataset_id> \
--end <end_dataset_id> \
--characteristics <list_of_characteristics> \
--metric <list_of_metrics> \
--splitting <list_of_sampling_strategies> \
--proc <number_of_cores_for_multiprocessing> \
-mp <if_set_it_will_run_in_multiprocessing_for_speed_up>
```

This will produce the tsv file "characteristics_0_600.tsv" under each dataset folder.

After that, you may want to run the linear regression model on the generated datasets of characteristics/performance metrics. To do so, you should run the following script (with arguments):

```
$ python regression_first.py \
--dataset <dataset_name> \
--start_id <start_dataset_id> \
--end_id <end_dataset_id> \
--characteristics <list_of_characteristics>
```
This will produce the tsv files "regression_\<metric\>_0_600.tsv", one for each metric, under the dataset folder.

We also provide a script to generate the latex tables (only the results parts of the tables, without row and column headers) starting from the results. To do so, you should run the script:

```
$ python generate_table_first.py
```

This will produce a tsv file "table_first.tsv" in the folder ```./data/```, as it is unique for all datasets.

### Node and Edge Dropout Analysis

You should run the script 

```
$ python check_scale_free.py
```

to fit the power-law and exponential functions on the node degree distribution of gowalla. The script will display the plot, and generate the latex code for the plot (used for the paper).

Finally, to reproduce the results for the second table, you should run the scripts (set the alpha variable within the script accordingly):

```
$ python regression_second.py \
--dataset <dataset_name> \
--start_id <start_dataset_id> \
--end_id <end_dataset_id> \
--characteristics <list_of_characteristics>
--alpha <alpha_value>
```
This will produce the tsv files "regression_\<alpha_value\>_\<metric\>_0_600.tsv", one for each metric, under the dataset folder. Then:

```
$ python generate_table_second.py
```

that, similarly to above, will produce one tsv file "table_\<alpha\>_second.tsv" for each alpha value in the folder ```./data/```, as it is unique for all datasets. Again, the latex code contains only result cells, but no row and column headers.