# Online Alternating Minimization
Pytorch code for paper Choromanska et al. -- Beyond Backprop: Online Alternating Minimization with Auxiliary Variables -- http://proceedings.mlr.press/v97/choromanska19a.html

### Requirements
* Python 3.5 or above
* PyTorch 1.1.0
* Torchvision 0.4
* Pandas 0.25 (for Higgs dataset dataloader)

These can be installed using `pip` by running:

```bash
pip install -r requirements.txt
```

### Usage
Run the simulations by executing the script `run_experiments.sh`:

```bash
bash run_experiments.sh
```

Once all simulations are done, plot the results by running the python script `plot_results.py`:

```bash
python plot_results.py
```

## Note
* This implementation of the feedforward model includes BatchNorm, which generally slightly improves performance upon what reported in the paper. Hyperparameters were tuned for the models with BatchNorm.

## Citation
> Anna Choromanska, Benjamin Cowen, Sadhana Kumaravel, Ronny Luss, Mattia Rigotti, Irina Rish, Paolo Diachille, Viatcheslav Gurev, Brian Kingsbury, Ravi Tejwani, and Djallel Bouneffouf. Beyond Backprop: Online Alternating Minimization with Auxiliary Variables. Proceedings of the 36th International Conference on Machine Learning, PMLR 97:1193-1202, 2019 [[PDF](http://proceedings.mlr.press/v97/choromanska19a/choromanska19a.pdf)]

For citations use the following Bibtex entry:
```
@InProceedings{pmlr-v97-choromanska19a,
  title = {Beyond Backprop: Online Alternating Minimization with Auxiliary Variables},
  author = {Choromanska, Anna and Cowen, Benjamin and Kumaravel, Sadhana and Luss, Ronny and Rigotti, Mattia and Rish, Irina and Diachille, Paolo and Gurev, Viatcheslav and Kingsbury, Brian and Tejwani, Ravi and Bouneffouf, Djallel},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  pages = {1193--1202},
  year = {2019},
  editor = {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = {97},
  series = {Proceedings of Machine Learning Research},
  address = {Long Beach, California, USA},
  month = {09--15 Jun},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v97/choromanska19a/choromanska19a.pdf},
  url = {http://proceedings.mlr.press/v97/choromanska19a.html},
  note= {Code available at: https://github.com/IBM/online-alt-min}
}
```
