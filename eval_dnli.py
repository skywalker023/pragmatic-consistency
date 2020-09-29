from parlai.scripts.eval_model import eval_model
from parlai.scripts.eval_model import setup_args as parlai_setupargs


def setup_args():
    parser = parlai_setupargs()
    parser.set_defaults(
        model_file='zoo:blender/blender_90M/model',
        eval_type='dnli',
        metrics='contradict@1,entail@1,neutral@1',
        alpha=8,
        beta=1,
        use_dnli=False
    )
    return parser


if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    eval_model(opt)
