import argparse
import csv
from datasets import Dataset
import faiss
from transformers import AutoTokenizer, BertModel
import os.path
import pyarrow
import pyarrow.csv
import sys
import torch
import tqdm

BATCH_SIZE = 128
DIM = 768
MAX_WORDS = 20

def embed(strings, tokenizer, model):
    x = tokenizer(
            strings, is_split_into_words=True, padding=True,
            return_tensors='pt').to(model.device)
    y = model(**x)['last_hidden_state']
    return y


def compute_verse_embeddings(data, tokenizer, model, batch_size=32, normalize=True):
    v = torch.empty((len(data), DIM), device='cpu')
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(data) // batch_size + 1)):
            j, k = i*batch_size, min(len(data), (i+1)*batch_size)
            y = embed(data[j:k]['text'], tokenizer, model).cpu()
            v[j:k,] = y.detach().clone().sum(axis=1)
    if normalize:
        norms = (v ** 2).sum(axis = 1).sqrt()
        v = v / norms.reshape((norms.shape[0], 1))
    return v


# FIXME use vectorized operations for the indices!!!
def compute_similarities(v, index, k=10, threshold=0.7, query_size=100, print_progress=False):
    if print_progress:
        progressbar = tqdm.tqdm(total=v.shape[0])
    for i in range(0, v.shape[0], args.query_size):
        query = range(i, min(v.shape[0], i+args.query_size))
        D, I = index.search(v[query,], args.k)
        for j, q in enumerate(query):
            for m in range(k):
                if q != I[j,m] and D[j,m] > threshold:
                    yield (q, int(I[j,m]), float(D[j,m]))
        if print_progress:
            progressbar.update(D.shape[0])


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fine-tune BERT for folk poetry.')
    parser.add_argument(
        '-i', '--input-file', metavar='FILE',
        help='The input file (CSV) containing verses to process.')
    parser.add_argument(
        '-m', '--model', metavar='MODEL', help='The model name to use.')
    parser.add_argument(
        '-e', '--embeddings-file', metavar='FILE',
        help='The file to save the embeddings to.')
    parser.add_argument(
        '-g', '--faiss-use-gpu', action='store_true',
        help='Run FAISS on GPU if possible.')
    parser.add_argument(
        '-k', type=int, default=10,
        help='The number of nearest neighbors to find for each verse.')
    parser.add_argument(
        '-q', '--query-size', type=int, default=100,
        help='The number of verses to pass in a single query '
             '(doesn\'t affect the results, only performance)')
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.7,
        help='Minimum similarity to output.')
    parser.add_argument(
        '-p', '--print-progress', action='store_true',
        help='Print a progress bar.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    
    data = pyarrow.csv.read_csv(args.input_file)
    data_spl = Dataset(pyarrow.table(
        [pyarrow.compute.split_pattern(data['text'], '_')],
        names=['text']
    ))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = BertModel.from_pretrained(args.model).cuda()

    if args.embeddings_file is not None and os.path.isfile(args.embeddings_file):
        v = torch.load(args.embeddings_file)
    else:
        v = compute_verse_embeddings(data_spl, tokenizer, model, batch_size=BATCH_SIZE)
        if args.embeddings_file is not None:
            torch.save(v, args.embeddings_file)

    # create a FAISS index
    index = faiss.IndexFlatIP(DIM)
    if args.faiss_use_gpu:
        res = faiss.StandardGpuResources()
        if res is not None:
            index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(v)

    # compute the similarities
    sims = compute_similarities(
        v, index, k = args.k, threshold = args.threshold,
        query_size=args.query_size, print_progress=args.print_progress)
    fieldnames = [f'{c}_{suff}' for suff in [1,2] for c in data.column_names] + ['sim']
    writer = csv.DictWriter(sys.stdout, fieldnames)
    writer.writeheader()
    for i, j, sim in sims:
        row = {'sim': sim}
        for c in data.column_names:
            row[c + '_1'] = data[c][i]
            row[c + '_2'] = data[c][j]
        writer.writerow(row)

