# Support the Underground: Characteristics of Beyond-Mainstream Music Listeners

This reposity includes python scripts, ipython-notebooks and data necessary for reproducing the publication:

TODO


## Files
To reproduce our results, the python scripts and ipython-notebooks must be executed in following order:

1. Identification of BeyMS and MS.ipynb: Identifies BeyMS and MS based on mainstreaminess. 
2. Identification and Analysis of Track Clusters.ipynb: Clustering and analysis of tracks listened by BeyMS. Additional statistics on track clusters.
3. Identification and Analysis of User groups.ipynb: Assign users in BeyMS to track clusters. Additional statistics of user groups.
4. Rating Dataset Generation.ipynb: Create dataset used for recommendation experiments. Includes both, BeyMS and MS.
5. Recommendations.py: Run several recommendation algorithms and evaluate them groupwise.
6. Visualization of Recommendation Performance.ipynb: Visualize results of recommendation experiments.

## Requirements
* Python 3
* numpy
* matplotlib
* pandas
* seaborn
* ast
* sklearn
* scipy
* pycountry
* umap
* hdbscan
* surprise
* statsmodels

## Contributors
* Peter Müllner, Know-Center GmbH, pmuellner [AT] know [MINUS] center [DOT] at (Contact)
* Dominik Kowald, Know-Center GmBH
* Markus Schedl, JKU Linz
* Christine Bauer, JKU Linz
* Eva Zangerle, University of Innsbruck
* Elisabeth Lex, Graz University of Technology