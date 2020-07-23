import json
from pathlib import Path

# models = ["Deep_mobile_lstmV3", "Deep_mobile_lstmV4", "Deep_mobile_lstmV5", "Deep_mobile_gruV3",
#           "Deep_mobile_gruV4", "Deep+_resnet50", "Deep_resnet50_lstmV2", "Deep_resnet50_lstmV3", "Deep_resnet50_lstmV4",
#           "Deep_resnet50_lstmV5"]
# models = ["Deep_mobile_lstmV3", "Deep_mobile_lstmV4", "Deep_mobile_lstmV5.1", "Deep_mobile_lstmV5.2"]
# start_lrs = [1e-3, 1e-3, 1e-3]
# step_sizes = [6, 6, 6]
# num_epochs = [30, 30, 30]
# batch_sizes = [8, 8, 8]
# loss = ["CrossEntropy", "SoftDice", "CrossDice"]  # / "SoftDice" / "Focal" / "CrossEntropy" / "Boundary" / "CrossDice"

models = ["Deep_mobile_gruV3", "Deep_mobile_gruV4", "Deep+_resnet50", "Deep_resnet50_lstmV2", "Deep_resnet50_lstmV3",
          "Deep_resnet50_lstmV4", "Deep_mobile_lstmV4", "Deep_mobile_lstmV5.1", "Deep_mobile_lstmV5.2"]

# models = ["Deep_mobile_lstmV4"]
start_lrs = [1e-3, 1e-4, 1e-5]
step_sizes = [6, 6, 6]
num_epochs = [30, 30, 30]
batch_sizes = [6, 6, 6]
loss = ["CrossDice", "CrossDice", "CrossDice"]  # / "SoftDice" / "Focal" / "CrossEntropy" / "Boundary" / "CrossDice"

config_paths = []
models_name = []
configs = []

for model in models:

    for i in range(len(start_lrs)):
        config = {}
        config["lr"] = start_lrs[i]
        config["model"] = model
        config["ID"] = str(i)
        config["batch_size"] = batch_sizes[i]
        config["num_epochs"] = num_epochs[i]
        config["scheduler_step_size"] = step_sizes[i]
        config["loss"] = loss[i]
        config["save_freq"] = 1
        config["save_path"] = "code/models/trained_models/minisV5"

        # print(config)
        configs.append(config)

for i, config in enumerate(configs):
    # print(config)
    from subprocess import call

    config["track_ID"] = i
    train_name = config["model"] + "_bs" + str(config["batch_size"]) \
                 + "_startLR" + format(config["lr"], ".0e") \
                 + "Sched_Step_" + str(config["scheduler_step_size"]) \
                 + "_" + config["loss"] + "_ID" + str(config["track_ID"])  # sets name of model based on parameters
    model_save_path = Path.cwd() / Path(config["save_path"]) / train_name
    model_save_path.mkdir(parents=True, exist_ok=True)  # create folder to save results
    with open(str(model_save_path / "train_config.json"), "w") as js:  # save learn config
        json.dump(config, js)

    vRam = "9G"
    recallParameter = 'qsub -N ' + "id" + str(i).zfill(2) + config["model"] \
                      + ' -l nv_mem_free=' + vRam + ' -v CFG=' + str(
        model_save_path / "train_config.json") + ' train_mixed.sge'
    # print(recallParameter)
    call(recallParameter, shell=True)

#
# for i, cfg in enumerate(config_paths):
#     from subprocess import call
#
#     cfg["track_ID"] = i
#     vRam = "9G"
#     recallParameter = 'qsub -N ' + "log_" + str(i).zfill(2)  + models_name[
#         i] + ' -l nv_mem_free=' + vRam + ' -v CFG=' + cfg + ' train_mixed.sge'
#     call(recallParameter, shell=True)

# model_save_path = Path.cwd() / Path(config["save_path"]) / train_name
# model_save_path.mkdir(parents=True, exist_ok=True)  # create folder to save results
# with open(str(model_save_path / "train_config.json"), "w") as js:  # save learn config
#     json.dump(config, js)
# config_paths.append(str(model_save_path / "train_config.json"))
# models_name.append(model)
# configs.append(config)
