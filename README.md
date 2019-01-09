# neptune_first_steps
First Steps with neptune
It is super easy.

Clone this repository

```bash
git clone https://github.com/neptune-ml/neptune-quick-start.git
```

and go to the project folder

```bash
cd neptune-quick-start
```

Install neptune by going

```bash
pip install neptune-cli
```

and login

```bash
neptune account login
```

Now you can run your experiment in the cloud with one simple command

```bash
neptune send --config neptune.yaml \
--worker xs-k80-preemptible \
--environment keras-2.2-gpu-py3 \
main.py
```

You can check example experiment [here](https://app.neptune.ml/-/dashboard/experiment/e70a461a-8e5a-4e76-b0d2-a81f551e611e/charts?getStartedState=folded)

![image](https://gist.githubusercontent.com/jakubczakon/f754769a39ea6b8fa9728ede49b9165c/raw/635a78f270e0c27347828e2761858e0762713a7a/neptune_quick_start.png)
