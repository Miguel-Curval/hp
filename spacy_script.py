import spacy
from spacy.language import Language
from pathlib import Path


model_map = {
    'uk' : 'en_core_web_',
    'us' : 'en_core_web_',
    'de' : 'de_core_news_'
}


@Language.component("custom_sentencizer")
def custom_sentencizer(doc):
    for token in doc[:-1]:
        continue
        if token.text in [',', ';', '…']:
            doc[token.i+1].is_sent_start = True
    return doc

def read_lines(infile):
    with open(infile) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        print('Read', len(lines), 'line(s) from', infile)
        return lines

def write_sentences(outfile, sentences):
    with open(outfile, 'w') as f:
        f.write('\n'.join(sentences))
        print('Wrote', len(sentences), 'sentence(s) to', outfile)

def separate_lang_sents(country_code, mode, model_size):
    nlp = spacy.load(model_map[country_code] + model_size)
    if mode == 'parser':
        nlp.disable_pipes(["tagger", "attribute_ruler", "lemmatizer", "ner"])
    elif mode == 'senter':
        nlp.disable_pipes(["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
        nlp.enable_pipe('senter')
    else:
        raise Exception("Invalid mode.")
    print('### Processing folder "' + country_code + '" using model ' + model_map[country_code], 'with the following pipeline:')
    print(nlp.pipe_names)
    #nlp.add_pipe("custom_sentencizer", before="parser")  # Insert before the parser
    p = Path(country_code)
    for txtfile in sorted(p.rglob('*.txt')):
        lines = read_lines(txtfile)
        sents = []
        for line in lines:
            sents += [sent.text for sent in nlp(line).sents]
        #sents = [sent.text for sent in nlp(line).sents for line in lines]
        outfile = Path(mode + '_' + model_size + '_' + str(txtfile))
        outfile.parent.mkdir(parents=True, exist_ok=True)
        write_sentences(outfile, sents)
        break


if __name__ == "__main__":
    for country_code in ['uk', 'us', 'de']:
        for mode in ['parser', 'senter']:
            for model_size in ['sm', 'md', 'lg']:
                separate_lang_sents(country_code, mode, model_size)

