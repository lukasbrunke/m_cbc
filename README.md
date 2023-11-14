# Data-Driven Synthesis of Safety Controllers via Multiple Control Barrier Certificates
Implementation of Nejathi, A., Zamani, M., 'Data-Driven Synthesis of Safety Controllers via Multiple Control Barrier Certificates', IEEE L-CSS, 2023.

## Results

Running `main.py` with `use_paper_results = false`

```
eta = 0.00015577864747909922
gamma = [-3640.68775296 -3640.68775287]
lambda = [-3640.68790851 -3640.68790853]

# polynomial coefficients
q = [[-4.00633575e-03 4.92801026e-03 -2.65324309e-03 3.46095771e-03 -8.26277124e-04 -3.64068992e+03]
    [1.34025476e-02 -3.83705692e-03 3.25828336e-03 3.03551712e-03 -3.11868990e-05 -3.64068998e+03]]
```
![alt text](https://github.com/lukasbrunke/m_cbc/blob/main/figures/jet_state_trajectories.png "Results using implementation")

----

Running `main.py` with `use_paper_results = true` 

```
eta = -0.0126
gamma = [0.5704 0.5708]
lambda = [0.583  0.5812]
q = [[0.002 -0.0025 0.0037 0.4 -0.1515 0.4]
     [0.002 -0.0318 0.0507 0.4 -0.1356 0.3935]]
```
![alt text](https://github.com/lukasbrunke/m_cbc/blob/main/figures/jet_state_trajectories_paper.png "Results from original paper")
