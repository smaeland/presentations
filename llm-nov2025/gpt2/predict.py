import re
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

if __name__ == '__main__':

    text = input('Input text: ')

    t = GPT2TokenizerFast.from_pretrained("gpt2")
    m = GPT2LMHeadModel.from_pretrained("gpt2")

    while True:

        text = text.strip()
        print(f'dbg text: "{text}"')

        encoded_text = t(text, return_tensors="pt")

        #1. step to get the logits of the next token
        with torch.inference_mode():
          outputs = m(**encoded_text)

        next_token_logits = outputs.logits[0, -1, :]
        #print(next_token_logits.shape)
        #print(next_token_logits)

        # 2. step to convert the logits to probabilities
        next_token_probs = torch.softmax(next_token_logits, -1)

        # 3. step to get the top 10
        topk_next_tokens, topk_next_token_indices = torch.topk(next_token_probs, 5)
        topk_next_logits = next_token_logits[topk_next_token_indices]

        #print('tokens:')
        #print(topk_next_tokens)
        #print(topk_next_logits)
        #print(torch.softmax(topk_next_logits, -1))
        #print(next_token_probs[topk_next_token_indices])

        #putting it together
        #print(*[(t.decode(idx), prob.numpy()) for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values)], sep="\n")

        for idx, prob in zip(topk_next_token_indices, topk_next_tokens):
            print('{:10} {:6.2f}% '.format(t.decode(idx), prob.numpy().item()*100))

        print()

        word = input(text + ' ')
        
        if word == '':
            break

        # Don't insert space if punctuation
        if not re.fullmatch(r'[,;.!?:"\'-]', word):
            text += ' '
        
        text += word
        
