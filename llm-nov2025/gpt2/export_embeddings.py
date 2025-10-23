import tensorflow as tf
from tensorboard.plugins import projector
from tensorboard import notebook
from pathlib import Path
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


if __name__ == '__main__':

    t = GPT2TokenizerFast.from_pretrained("gpt2")
    m = GPT2LMHeadModel.from_pretrained("gpt2")

    logdir = Path('./logs')
    if not logdir.exists():
        Path.mkdir(logdir)

    word_embeddings = m.transformer.wte.weight
    print('word_embeddings.shape:', word_embeddings.shape)

    vocab_list = sorted(t.vocab.items(), key=lambda x: x[1])
    with open(logdir / Path('metadata.tsv'), 'w') as f:
        for word, idx in vocab_list:
            #f.write('{}\n'.format(str(word.encode(encoding='iso-8859-1', errors='replace'))))
            #f.write('{}\n'.format(word)) # Includes 'Ġ' (Unicode U+0120) as space
            f.write('{}\n'.format(word.replace('\u0120', ' '))) # Replace the 'Ġ'

    embeddings = tf.Variable(word_embeddings.detach().numpy())
    checkpoint = tf.train.Checkpoint(embedding=embeddings)
    checkpoint.save(logdir / Path('embedding.ckpt'))
