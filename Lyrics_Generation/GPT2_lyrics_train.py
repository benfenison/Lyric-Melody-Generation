import argparse
import tensorflow as tf
import gpt_2_simple as gpt2

def train(dataset_path, learning_rate, optimizer, batch_size, model_name, steps, 
          sample_every, sample_length, save_every, print_every, restore_from, run_name):

    sess = None

    tf.compat.v1.reset_default_graph()
    if not sess:
        sess = gpt2.start_tf_sess()
    else:
        sess = gpt2.reset_session(sess)

    gpt2.finetune(sess,
                dataset_path,
                optimizer=optimizer,
                model_name=model_name,
                learning_rate=learning_rate,
                batch_size=batch_size,
                steps=steps,
                sample_every=sample_every,
                sample_length=sample_length,
                save_every=save_every,
                print_every=print_every,
                restore_from=restore_from,
                run_name=run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str,
                        help="path to the csv file with a single column of lyrics")
    parser.add_argument("--learning_rate", type=float, nargs='?', default=1e-4)
    parser.add_argument("--optimizer", type=str, nargs='?', default='adam')
    parser.add_argument("--batch_size", type=int, nargs='?', default=1)
    parser.add_argument("--model_name", type=str, nargs='?', default="124M")
    parser.add_argument("--steps", type=int, nargs='?', default=1000)
    parser.add_argument("--sample_every", type=int, nargs='?', default=1000)
    parser.add_argument("--sample_length", type=int, nargs='?', default=200)
    parser.add_argument("--save_every", type=int, nargs='?', default=200)
    parser.add_argument("--print_every", type=int, nargs='?', default=250)
    parser.add_argument("--restore_from", type=str, nargs='?', default='fresh')
    parser.add_argument("--run_name", type=str, nargs='?', default='finetune_run')

    args = parser.parse_args()

    train(dataset_path=args.dataset_path,
          learning_rate=args.learning_rate,
          optimizer=args.optimizer,
          batch_size=args.batch_size,
          model_name=args.model_name,
          steps=args.steps,
          sample_every=args.sample_every,
          sample_length=args.sample_length,
          save_every=args.save_every,
          print_every=args.print_every,
          restore_from=args.restore_from,
          run_name=args.run_name,
    )