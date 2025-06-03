import json
from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MMLU-Pro")
dataset=ds["test"]
physics_data=dataset.filter(lambda example: example["category"] == "biology")
print(len(physics_data), "original test data")

physics_data_splited=physics_data.train_test_split(test_size=(1/3))
print(physics_data_splited)

with open("./data/MAG/MMLUProBiol_1000.json", mode='w', encoding="utf-8") as fout:
    mags=[]
    for example in physics_data_splited["train"]:
        question=example["question"]
        qid=example["question_id"]
        options=example["options"]
        gold=example["answer"]
        q_o="%s" % question
        prefix=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for pr, o in zip(prefix, options):
            q_o+=(" %s) %s" % (pr, o))
        mag={"question": q_o, "gold_answer": gold, "MMLU_Pro_id": qid}
        mags.append(mag)
    json.dump(mags, fout, ensure_ascii=False, indent=2)
    print(len(mags))

with open("./data/test_data/MMLUProBiol_test.json", mode='w', encoding="utf-8") as fout:
    mags=[]
    for example in physics_data_splited["test"]:
        question=example["question"]
        qid=example["question_id"]
        options=example["options"]
        gold=example["answer"]
        q_o="%s" % question
        prefix=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for pr, o in zip(prefix, options):
            q_o+=(" %s) %s" % (pr, o))
        mag={"question": q_o, "answer": gold, "MMLU_Pro_id": qid}
        mags.append(mag)
    json.dump(mags, fout, ensure_ascii=False, indent=2)
    print(len(mags))
