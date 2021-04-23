# cgpunet
### Neuroevolution of CNN architectures for medical image segmentation

This repository is currently under construction and is built on top of a clone of the implementation by Suganuma: 
https://github.com/sg-nm/cgp-cnn-PyTorch

You may wish to refer to the original paper for the genotypic representation:
https://arxiv.org/abs/1704.00764

Some key differences in this implementation compared to the original:

<ol>
<li>This is a Keras implementation, translated from the original PyTorch implementation.</li>
<li>The CNNs are evolved for the task of image segmentation, rather than recognition as in the original work.</li>
<li>We have expanded the function set to incorporate a larger selection of functions, including transpose convolutions.</li>
<li>We include the implementation of a networkx DAG phenotype which allows for the use graph traversal algorithms. This is intended to allow for extension of the current work for the realisation of semantic-aware GP crossover operators.</li>
<li>We implement measures of structural diversity for the characterisation of local optima, which can subsequently be exploited using stochastic local search algorithms. We assert that this will result in a more efficient neural architecture search.</li>
</ol>

The full documentation will be available in the coming months. For now, the file cgpunet_main.py illustrates the use of the algorithm. If you would like to run the algorithm on your own datasets, please feel free to reach out to me if you have any questions.
