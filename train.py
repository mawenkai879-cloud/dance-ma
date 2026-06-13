from args import parse_train_opt
from EDGE import EDGE


def train(opt):
    hloss = None
    if opt.hierarchical_loss_frames or opt.hierarchical_loss_joints:
        frames = None
        joints = None
        if opt.hierarchical_loss_frames:
            frames = tuple(int(x) for x in opt.hierarchical_loss_frames.split(","))
        if opt.hierarchical_loss_joints:
            joints = [int(x) for x in opt.hierarchical_loss_joints.split(",")]
        hloss = {
            "frames": frames,
            "joints": joints,
            "weight": opt.hierarchical_loss_weight,
        }
    model = EDGE(
        opt.feature_type,
        checkpoint_path=opt.checkpoint,
        use_ccl=not opt.no_ccl,
        use_hierarchical=opt.use_hierarchical,
        hierarchical_loss=hloss,
    )
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
