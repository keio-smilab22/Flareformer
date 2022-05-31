import csv

types = ["bss", "tss", "gmgs", "acc"]
head_to_details = {"id17a_": "(iii)ConvNext → Vanilla",
                   "id32_N3_": "(ii)N_trm = 3",
                   "id32_N2_": "(ii)N_trm = 2",
                   "id24_": "(i)w/o cRT",
                   "vanilla_trm_": "(iv)Informer → Vanilla Transformer"}

models_value = {}
valid_step = set([39, 42])
for t in types:
    with open(f"/home/initial/workspace/flare_transformer/output_csv/{t}.csv") as f:
        reader = csv.DictReader(f)
        for i, lines in enumerate(reader):
            step = int(lines["Step"])
            if step not in valid_step:
                continue
            for line, value in lines.items():
                if len(value) == 0:
                    continue
                invalid = "__" in line
                if invalid:
                    continue
                spl = line.split(" - ")
                if len(spl) < 2:
                    continue
                model = spl[0].replace(" ", "")
                models_value[(model, t)] = float(value)

# print(models_value)
print("condition,", ",".join(types))

seen = set()
for (model, t), v in models_value.items():
    if model in seen:
        continue
    _model = model
    for head, details in head_to_details.items():
        if model[:len(head)] == head:
            model = f"{details} {model[len(head):].replace('_ablation','')}"

    correct_year = None
    for year in range(2014, 2017 + 1):
        if str(year) in model:
            model = model.replace(str(year), "")
            correct_year = year

    print(f"{model},{correct_year},", end="")
    for t in types:
        v = models_value[(_model, t)]
        print(f"{v:.4f}", end=',')

    print("")
    seen.add(_model)
