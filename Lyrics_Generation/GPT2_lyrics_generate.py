import argparse
import tensorflow as tf
import gpt_2_simple as gpt2

def generate(checkpoint_path, prefix, truncate, nsamples, temperature, top_p, top_k, 
             length, return_as_list, include_prefix):

    sess = None

    tf.compat.v1.reset_default_graph()
    if not sess:
        sess = gpt2.start_tf_sess()
    else:
        sess = gpt2.reset_session(sess)

    gpt2.load_gpt2(sess, run_name=checkpoint_path)

    lyrics_results = gpt2.generate(sess,
                            prefix=prefix,
                            truncate=truncate,
                            nsamples=nsamples,
                            temperature=temperature, # higher temperature the model gives more random text generations (default(0.7))
                            top_p=top_p, # cumulative probability of guesses
                            top_k=top_k, # top k guesses (default(0); 0 ~= disabled)
                            length=length, # number of tokens to generate (e.g. max: default(1023))
                            return_as_list=return_as_list,
                            include_prefix=include_prefix,
                            run_name=checkpoint_path)
    
    for i, lyric in enumerate(lyrics_results):
        print('-'*20 + f'Lyric {i}:' + '-'*20 + '\n')
        print(lyric)

    with open('generated_lyrics.txt', 'w') as f:
        for lyric in lyrics_results:
            lyric = '\n<song>\n' + lyric.replace('<|startoftext|>', '').strip()
            try:
                f.write(f"{lyric}\n")
            except:
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str,
                        help="path to the finetuned model checkpoint folder")
    parser.add_argument("--prefix", type=str, nargs='?', default="<|startoftext|>")
    parser.add_argument("--truncate", type=str, nargs='?', default="<|endoftext|>")
    parser.add_argument("--nsamples", type=int, nargs='?', default=5)
    parser.add_argument("--temperature", type=int, nargs='?', default=0.85)
    parser.add_argument("--top_p", type=int, nargs='?', default=0.9)
    parser.add_argument("--top_k", type=int, nargs='?', default=0)
    parser.add_argument("--length", type=int, nargs='?', default=200)
    parser.add_argument("--return_as_list", type=bool, nargs='?', default=True)
    parser.add_argument("--include_prefix", type=bool, nargs='?', default=True)

    args = parser.parse_args()

    generate(checkpoint_path=args.checkpoint_path,
            prefix=args.prefix,
            truncate=args.truncate,
            nsamples=args.nsamples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            length=args.length,
            return_as_list=args.return_as_list,
            include_prefix=args.include_prefix,
    )