## raisim_env_anymal

### Dependencies
```raisimgym```

### Clone
Clone this repo into your ```raisim_ws``` (or the same directory where your other ```raisim``` projects are)

### Run
1. Compile raisimgym: 
The ```CMAKE_PREFIX_PATH``` arg may vary
```sh
python setup develop --CMAKE_PREFIX_PATH ~/raisim_ws/raisimLib/raisim/linux
```
2. run runner.py of the task (for anymal example): 
```sh
python raisimGymTorch/env/envs/rsg_anymal/runner.py
```

### Test policy
1. Compile raisimgym (see **Run** above)
2. run tester.py of the task with policy (for anymal example): 
```sh
python raisimGymTorch/env/envs/rsg_anymal/tester.py --weight data/roughTerrain/FOLDER_NAME/full_XXX.pt
```

### Retrain policy
1. run runner.py of the task with policy (for anymal example): 
```sh
python raisimGymTorch/env/envs/rsg_anymal/runner.py --mode retrain --weight data/roughTerrain/FOLDER_NAME/full_XXX.pt
```

### Debugging
1. Compile raisimgym with debug symbols: 
```
python setup develop --Debug
```
This compiles <YOUR_APP_NAME>_debug_app

2. Run it with Valgrind. I strongly recommend using Clion.
