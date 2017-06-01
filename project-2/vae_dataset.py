from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

from utils import H5PYDatasetCreator


TRAIN_PAIRS_FILE = 'storage/train_pairs_'
TEST_PAIRS_FILE = 'storage/test_pairs_'


def sample_word(words, probs):
    return words[np.argmax(probs + np.random.rand(*probs.shape))]


def frequency_sample(words_counter, total_count):
    words, frequency = zip(*[(a, b) for a, b in words_counter.items()
                            if 30 <= total_count[a] <= 3000])
    frequency = np.array(frequency)
    return sample_word(words, frequency / frequency.sum())


def create_train_test_set(verbs, nouns, verb_pairs):
    selected_pairs = []
    selected_nouns = []
    selected_verbs = []
    while len(selected_pairs) < 3000:
        print(len(selected_pairs), end='\r')
        verb_strings, counts = zip(*[(a, b) for a, b in verbs.items()
                                     if b <= 3000 and b >= 30])
        counts = np.array(counts)
        verb_prob = counts / counts.sum()

        selected_verb = sample_word(verb_strings, verb_prob)
        verb_N = verbs[selected_verb]

        verb_nouns = verb_pairs[selected_verb]
        try:
            selected_noun = frequency_sample(verb_nouns, nouns)
        except ValueError:
            continue

        noun_N = verb_nouns[selected_noun]

        if verb_N - noun_N < 30:
            continue

        if nouns[selected_noun] - noun_N < 30:
            continue

        del verb_nouns[selected_noun]
        nouns[selected_noun] -= noun_N
        verbs[selected_verb] -= verb_N
        selected_pairs.append((selected_verb, selected_noun))
        selected_nouns.append(selected_noun)
        selected_verbs.append(selected_verbs)

    print('create sets')
    test_set = selected_pairs
    train_set = []
    for v, all_n in verb_pairs.items():
        for n, c in all_n.items():
            train_set += [(v, n)] * c

    return train_set, test_set


def assert_set_correctness(train_set, test_set):
    train_verbs, train_nouns = [Counter(a) for a in zip(*train_set)]
    train_set = set(train_set)

    for p in test_set:
        assert p not in train_set
        v, n = p
        assert 30 <= train_verbs[v] <= 3000
        assert 30 <= train_nouns[n] <= 3000


def write_set(data, filename):
    with open(filename, 'w') as f:
        for v, n in data:
            f.write('{}||{}\n'.format(v, n))


def main(filename, glove_filename):
    verbs = []
    nouns = []
    verb_pairs = defaultdict(Counter)

    vectors = {}

    with open(glove_filename) as f:
        for i, line in enumerate(tqdm(f)):
            splits = line.split()
            word, values = splits[0], splits[1:]
            values = np.array([float(v) for v in values])
            vectors[word] = values

    with open(filename) as f:
        for i, line in enumerate(tqdm(f)):
            head, noun = line.strip().lower().split(' ', 1)
            verb, tag = head.split('_', 1)
            if verb not in vectors or noun not in vectors:
                continue
            verbs.append(verb)
            nouns.append(noun)
            verb_pairs[verb].update([noun])

    verbs = Counter(verbs)
    nouns = Counter(nouns)

    train_set, test_set = create_train_test_set(verbs, nouns, verb_pairs)
    assert_set_correctness(train_set, test_set)

    write_set(train_set, TRAIN_PAIRS_FILE)
    write_set(test_set, TEST_PAIRS_FILE)

    creator = H5PYDatasetCreator('/home/jorn/Desktop/outfile.test')

    creator.add_split('train', len(train_set))
    creator.add_split('test', len(test_set))
    creator.add_source('noun_vec', (300,), np.float32)
    creator.add_source('verb_vec', (300,), np.float32)

    print('Add train set')
    for v, n in tqdm(train_set):
        creator.add_row('train', verb_vec=vectors[v], noun_vec=vectors[n])
    print('Add test set')
    for v, n in tqdm(test_set):
        creator.add_row('test', verb_vec=vectors[v], noun_vec=vectors[n])


if __name__ == "__main__":
    main('data/all_pairs', '/home/jorn/Downloads/glove/glove.6B.300d.txt')
