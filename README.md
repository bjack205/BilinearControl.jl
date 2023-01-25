# Data-Efficient Model Learning for Control with Jacobian-Regularized Dynamic Mode Decomposition

This contains the code used to generate the figures and examples in the published CoRL 2022 paper.

## Citing

If you use BilinearControl.jl and the JDMD algorithm as part of your research, teaching, or other activities, we would be grateful if you could cite our work:

[1] B. E. Jackson, J. H. Lee, K. Tracy, and Z. Manchester, “Data-Efficient Model Learning for Control with Jacobian-Regularized Dynamic Mode Decomposition,” presented at the Conference on Robot Learning (CoRL), Aukland, Dec. 2022. doi: 10.48550/ARXIV.2212.07885.

```
@inproceedings{jackson_data-efficient_2022,
	address = {Aukland},
	title = {Data-{Efficient} {Model} {Learning} for {Control} with {Jacobian}-{Regularized} {Dynamic} {Mode} {Decomposition}},
	copyright = {arXiv.org perpetual, non-exclusive license},
	url = {https://arxiv.org/abs/2212.07885},
	doi = {10.48550/ARXIV.2212.07885},
	language = {en},
	author = {Jackson, Brian E and Lee, Jeong Hun and Tracy, Kevin and Manchester, Zachary},
	month = dec,
	year = {2022},
}
```

## Running the Examples
To run the examples, run the `<xxx>_main.jl` files in the `examples/` folder, which are 
organized by system. Each file should activate the project environment and download all
required packages.

Note that the visualization code will probably not work, since the models used for the 
visuals in the paper were too lage to be included in the supplementary materials.
