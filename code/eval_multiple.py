import json
from pathlib import Path

model_path = Path("code/models/trained_models/minisV4")
for i,model_name in enumerate(model_path.glob("*")):
    model = model_name.stem

    from subprocess import call

    recallParameter = 'qsub -N ' + "_eval_" + str(i) + model[0:20] +' -l nv_mem_free=4.1G -v MDL=' + model \
                      + ' -v MPTH=' +str(model_path)+' eval_model.sge'
    # print(recallParameter)
    call(recallParameter, shell=True)
