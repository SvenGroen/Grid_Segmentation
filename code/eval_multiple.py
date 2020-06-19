import json
from pathlib import Path

model_path = Path("code/models/trained_models/LSTMs")
for i,model_name in enumerate(model_path.glob("*")):
    model = model_name.stem

    from subprocess import call

    recallParameter = 'qsub -N ' + "eval_" + str(i) + model[0:20] +' -l nv_mem_free=4.5G -v MDL=' + model + ' PTH=' +str(model_path)+' eval_model.sge'
    print(recallParameter)
    # call(recallParameter, shell=True)
