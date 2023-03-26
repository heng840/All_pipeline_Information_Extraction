import argparse


def get_hparams_entity():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='ACE')

    parser.add_argument('--data_dir', type=str,
                        default='../ace2005/',
                        help="path to the preprocessed dataset")
    parser.add_argument('--output_dir', type=str, default='entity_output_ace',
                        help="output directory of the entity model")

    parser.add_argument('--max_span_length', type=int, default=8,
                        help="spans w/ length up to max_span_length are considered as candidates")
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help="batch size during training")
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help="batch size during inference")
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help="learning rate for the BERT encoder")
    parser.add_argument('--task_learning_rate', type=float, default=5e-4,
                        help="learning rate for task-specific parameters, i.e., classification head")
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help="the ratio of the warmup steps to the total steps")
    parser.add_argument('--num_epoch', type=int, default=100,
                        help="number of the training epochs")
    parser.add_argument('--print_loss_step', type=int, default=100,
                        help="how often logging the loss value during training")
    parser.add_argument('--eval_per_epoch', type=int, default=1,
                        help="how often evaluating the trained model on dev set during training")
    parser.add_argument("--bertadam", action="store_true", help="If bertadam, then set correct_bias = False")

    parser.add_argument('--do_train', action='store_false', default=True,
                        help="whether to run training")
    parser.add_argument('--train_shuffle', action='store_false', default=True,
                        help="whether to train with randomly shuffled data")
    parser.add_argument('--do_eval', action='store_false', default=True,
                        help="whether to run evaluation")
    parser.add_argument('--eval_test', action='store_true', default=False,
                        help="whether to evaluate on test set")
    parser.add_argument('--dev_pred_filename', type=str, default="ent_pred_dev.json",
                        help="the prediction filename for the dev set")
    parser.add_argument('--test_pred_filename', type=str, default="ent_pred_test.json",
                        help="the prediction filename for the test set")

    parser.add_argument('--use_albert', action='store_true',
                        help="whether to use ALBERT model")
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help="the base model name (a huggingface model)")
    parser.add_argument('--bert_model_dir', type=str, default=None,
                        help="the base model directory")

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--context_window', type=int, default=300,
                        help="the context window size W for the entity model")

    args = parser.parse_args()
    return args
