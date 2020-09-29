from parlai.scripts.eval_model import eval_model
from parlai.scripts.eval_model import setup_args as parlai_setupargs


def setup_args():
    parser = parlai_setupargs()
    parser.set_defaults(
        model_file='zoo:blender/blender_90M/model',
        eval_type='convai2',
        metrics='token_acc,ppl,loss,c_scores,f1',
        alpha=2,
        beta=0.5
    )
    return parser


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    eval_model(opt)
