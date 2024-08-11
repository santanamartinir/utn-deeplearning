# utn-deeplearning
Assignment: Generative HPO with Diffusers

### Pre-requisites

1. Download the source code of HPO-B Benchmark
    ```shell
    git clone https://github.com/machinelearningnuremberg/HPO-B.git
    ```
    Check the readme of HPO-B and install the required prerequisites mentioned there.
    e.g download the following file, rename it as "cd_diagram" and place it inside the HPO-B repo root dir
    ```shell
      https://github.com/hfawaz/cd-diagram/blob/master/main.py
    ```
2. Download the meta dataset
   [meta-dataset](https://rewind.tf.uni-freiburg.de/index.php/s/xdrJQPCTNi2zbfL/download/hpob-data.zip).
   and merge it with the existing hpob-data of the downloaded repo
3. Copy HPO-B/results/GP.json into this_repo/results
4. create a virtual env and install dependencies
   
   e.g.
   ```shell
    virtualenv -p /usr/bin/python3.10 hpo_diffusors_env
    source hpo_diffusors_env/bin/activate
    ```
   and then install the requirements
   ```shell
      pip install -r requirements.txt
   ```
   
5. Problem solving for the plot generation:
   - Remove . for importing cd_diagram and hpob_handler:
   ```code
   from cd_diagram import draw_cd_diagram as draw
   from hpob_handler import HPOBHandler
   ```
   - remove the path name (path_name=path+name+".png") from draw() in the draw_cd_diagram function in benchmark_plot.py.
   - add if-clause around the last two lines of cd_diagram.py:
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
