from pathlib import Path
from typing import List

from nltk import Tree
from nltk.corpus import BracketParseCorpusReader

PTB_UNESCAPE_MAPPING = {
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‹": "'",
    "›": "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}


def ptb_unescape(words: List[str]) -> List[str]:
    cleaned_words = []

    for word in words:
        word = PTB_UNESCAPE_MAPPING.get(word, word)
        # This un-escaping for / and * was not yet added for the
        # parser version in https://arxiv.org/abs/1812.11760v1
        # and related model releases (e.g. benepar2_en2)
        word = word.replace("\\/", "/").replace("\\*", "*")
        # Mid-token punctuation occurs in biomedical text
        word = word.replace("-LSB-", "[").replace("-RSB-", "]")
        word = word.replace("-LRB-", "(").replace("-RRB-", ")")
        word = word.replace("-LCB-", "{").replace("-RCB-", "}")
        word = word.replace("``", '"').replace("`", "'").replace("''", '"')
        cleaned_words.append(word)

    return cleaned_words


def binarize(tree: Tree) -> Tree:
    tree: Tree = tree.copy(True)

    nodes = [tree]
    while nodes:
        node = nodes.pop()
        if isinstance(node, Tree):
            nodes.extend([child for child in node])
            if len(node) > 1:
                for index, child in enumerate(node):
                    if not isinstance(child[0], Tree):
                        node[index] = Tree(f"{node.label()}|<>", [child])

    tree.chomsky_normal_form('left', 0, 0)
    tree.collapse_unary()

    return tree


def factorize(tree: Tree, i: int = 0):
    label = tree.label()

    if len(tree) == 1 and not isinstance(tree[0], Tree):
        return (i + 1 if label is not None else i), []

    j, spans = i, []
    for child in tree:
        j, s = factorize(child, j)
        spans += s

    if label is not None and j > i:
        spans = [(i, j, label)] + spans

    return j, spans


def iter_ptb(path: Path, *, do_binarize: bool, do_factorize: bool):
    corpus_reader = BracketParseCorpusReader(str(path.resolve().parent), [path.name])

    for tree in corpus_reader.parsed_sents():
        words = ptb_unescape(tree.leaves())

        tree = tree[0]
        if do_binarize:
            tree = binarize(tree)
        if do_factorize:
            _, tree = factorize(tree)

        yield words, tree
