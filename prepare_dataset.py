import json
import pickle


_SPLIT = 0.8


if __name__ == "__main__":
    ds_train = []
    ds_val = []
    with open('./data_finetuning/data', 'rb') as fp:
        data = pickle.load(fp)
        ds_len = len(data)
        for i, example in enumerate(data):
            if i < round(ds_len*_SPLIT):
                ds_train.append({"question": example[0], "answer": example[1]})
            else:
                ds_val.append({"question": example[0], "answer": example[1]})
        
        json_train= json.dumps(ds_train, indent=None)
        json_val= json.dumps(ds_val, indent=None)
        
        with open("./data_finetuning/train.json", "w") as outfile:
            outfile.write(json_train)
        with open("./data_finetuning/val.json", "w") as outfile:
            outfile.write(json_val)