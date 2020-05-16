import json
from pathlib import Path

models = ["Deep_Res50"]  # Options available: "UNet", "Deep_Res101", "ConvSame_3", "Deep_Res50", "Deep+_mobile", "ICNet"
start_lrs = [1e-02, 1e-02, 1e-02]
step_sizes = [10, 20, 25]
num_epochs = [100, 100, 100]
batch_sizes = [2, 2, 2]
config = {}
config_paths = []
models_name = []
for model in models:
    for i in range(len(start_lrs)):
        config["lr"] = start_lrs[i]
        config["model"] = model
        config["ID"] = str(i)
        config["batch_size"] = batch_sizes[i]
        config["num_epochs"] = num_epochs[i]
        config["scheduler_step_size"] = step_sizes[i]
        config["save_freq"] = 1
        config["save_path"] = "code/models/trained_models/Examples_Green/multiples"
        # save the config
        train_name = config["model"] + "_bs" + str(config["batch_size"]) + "_startLR" + format(config["lr"],
                                                                                               ".0e") + "Sched_Step_" + str(
            config["scheduler_step_size"]) + "ID" + config["ID"]  # sets name of model based on parameters
        model_save_path = Path.cwd() / Path(config["save_path"]) / train_name
        model_save_path.mkdir(parents=True, exist_ok=True)  # create folder to save results
        with open(str(model_save_path / "train_config.json"), "w") as js:  # save learn config
            json.dump(config, js)
        config_paths.append(str(model_save_path / "train_config.json"))
        models_name.append(model)

for i, cfg in enumerate(config_paths):
    from subprocess import call

    if "Deep_Res" in models_name[i]:
        recallParameter = 'qsub -N ' + "log_" + str(i) + models_name[i] + ' -l nv_mem_free=5G -v CFG=' + cfg + ' train_mixed.sge'
    else:
        recallParameter = 'qsub -N ' + "log_" + str(i) + models_name[
            i] + ' -l nv_mem_free=3.4G -v CFG=' + cfg + ' train_mixed.sge'

    call(recallParameter, shell=True)
