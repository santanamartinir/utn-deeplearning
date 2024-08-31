# utn-deeplearning
Assignment: Generative HPO with Diffusers

### Pre-requisites

1. Download the source code of HPO-B Benchmark
    ```shell
    git clone https://github.com/machinelearningnuremberg/HPO-B.git
    ```
    Check the readme of HPO-B and install the required prerequisites mentioned there if needed.

2. Create a virtual env and install the required dependencies.
   
   e.g.
   ```shell
    virtualenv -p /usr/bin/python3.10 hpo_diffusors_env
    source hpo_diffusors_env/bin/activate
    ```
   and then install the requirements
   ```shell
      pip install -r requirements.txt
   ```

3. Problem solving for the plot generation, if needed:
   - Remove . for importing cd_diagram and hpob_handler:
   ```code
   from cd_diagram import draw_cd_diagram as draw
   from hpob_handler import HPOBHandler
   ```
   - Remove the path name (path_name=path+name+".png") from draw() in the draw_cd_diagram function in benchmark_plot.py.

   - Add if-clause around the last two lines of cd_diagram.py:
   ```code
   if __name__ == "__main__":
    df_perf = pd.read_csv('example.csv', index_col=False)

    draw_cd_diagram(df_perf=df_perf, title='Accuracy', labels=True)
    ```

Being able to run the following and generate plots validates the initial setup correctness
```shell
   cd HPO-B/
   python benchmark_plot.py
```
### To check results of Our Algorithm 

```python
   python main.py
```

# Notes for training
1. Update the surrogate path in HPO-B/hpob_handler.py:17 to:
surrogates_dir="HPO-B/saved-surrogates/"

2. To generate the results for our Continuous MyAlgorithm, the following changes are required
HPO-B/benchmark_plot.py:225
evaluate_continuous instead of evaluate

3. For visualising the loss in tensorboard:
```shell
tensorboard --logdir=runs
```