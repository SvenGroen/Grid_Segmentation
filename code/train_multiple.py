import json
from pathlib import Path

models = ["ICNet"]# Options available: "UNet", "Deep_Res101", "ConvSame_3", "Deep_Res50", "Deep+_mobile", "ICNet"
start_lrs = [1e-02]
step_sizes = [10]
num_epochs = [100]
batch_sizes = [2]
config = {}
config_paths = []
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

for cfg in config_paths:
    from subprocess import call

    recallParameter = 'qsub -v CFG=' + cfg + ' train_mixed.sge'
    call(recallParameter, shell=True)
