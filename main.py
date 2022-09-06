
## PREPARE THE DATASET

import os
import pandas as pd
path = "/Users/maingo/Downloads/theses100/"
all_files = os.listdir(path + "docsutf8") # content file
all_keys = os.listdir(path + "keys") # file of keywords of the texts
# print(len(all_files), " files n", all_files, "n", all_keys)

all_documents = []
all_keys = []
all_files_names = []
for i, fname in enumerate(all_files):
    # print(fname)
    # print(path + 'docsutf8/' + fname)
    with open(path + 'docsutf8/' + fname) as f: # open each txt file
        lines = f.readlines() # get the content
    key_name = fname[:-4] # thesis holder's name
    # print(key_name)
    with open(path + 'keys/' + key_name + '.key') as f:
        k = f.readlines()
    all_text = ''.join(lines)
    keys = ''.join(k)
    # print(keys)
    all_documents.append(all_text) # add each thesis content to the all_documents list
    all_keys.append(keys.split("\n")) # split list of keywords by 'n'
    all_files_names.append(key_name)

dtf = pd.DataFrame({'goldkeys':all_keys, 'text': all_documents})
print(dtf.head())