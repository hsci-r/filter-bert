# BERT models for Finnic folk poetry

The trained model is contained in the subdirectory `filter-bert`. In
order to run it on a set of verses, execute:
```
python embed_and_find_sim.py -i INPUT_FILE -m filter-bert
```

The script will print line cosine similarities on standard output. You
can also save the BERT embeddings to a file using the option
`-e EMBEDDINGS_FILE`. For more options, run: 
`python embed_and_find_sim.py --help`.

## Installing dependencies

It is recommended to use Anaconda for managing dependencies. Use the file
`env.yaml` to create an environment:
```
conda env create --name filter-bert --file=env.yaml
```

## Training the model

The model can be trained with the following command:
```
python train.py -i INPUT_FILE -m BASE_MODEL -o OUTPUT_DIR
```

This will pull the `BASE_MODEL` (e.g. `TurkuNLP/bert-base-finnish-uncased-v1`)
from HuggingFace, fine-tune it using the supplied training data and save
the fine-tuned model in `OUTPUT_DIR`.

It is recommended to run the fine-tuning on a GPU-equipped machine.

## Input file format

The input file must be a CSV file containing one verse per row, with a
`text` column containing the verse text. The individual words in the
text should be separated by **underscores**. A sample of the input file
looks like this:
```
poem_id,pos,text
skvr01100010,1,vaan_se_on_vanha_väinämöinen
skvr01100010,2,lähtiäksensä_käkesi
skvr01100010,3,tullaksensa_toivotteli
skvr01100010,4,luotolah_lankohinsa
skvr01100010,5,väinöläh_sisärihinsä
```
