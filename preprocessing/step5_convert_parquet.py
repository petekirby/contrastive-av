import os
import pickle
import pyarrow
import pyarrow.parquet as pq

dir_data = os.path.join("..", "data_preprocessed")
datasets = ["dict_author_fandom_doc_train", "dict_author_fandom_doc_val"]

for dataset in datasets:
    with open(os.path.join(dir_data, dataset), "rb") as f:
        author_fandom_dict = pickle.load(f)
    print("reading: ", dataset)

    authors, fandoms, samples = [], [], []

    for author_id, fandom_names in author_fandom_dict.items():
        for fandom_name, texts in fandom_names.items():
            for text in texts:
                authors.append(str(author_id))
                fandoms.append(str(fandom_name))
                samples.append(text)
    del author_fandom_dict

    table = pyarrow.table({"author_id": authors, "fandom_name": fandoms, "text": samples})
    del authors, fandoms, samples

    output_file = os.path.join(dir_data, dataset + ".parquet")
    pq.write_table(table, output_file, compression="snappy")
    print("wrote: ", output_file)
    del table
