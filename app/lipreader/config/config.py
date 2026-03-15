import argparse


def load_arg():
    parse = argparse.ArgumentParser()

    # * checkpoint path
    parse.add_argument(
        "--path_c3d",
        type=str,
        default="D:/PycharmProjects/LipReaderClient/app/lipreader/checkpoint/checkpoint_c3d.pth",
    )
    parse.add_argument(
        "--path_wpelip_hz",
        type=str,
        default="D:/PycharmProjects/LipReaderClient/app/lipreader/checkpoint/checkpoint_wpelip_hz.pth",
    )
    parse.add_argument(
        "--path_wpelip_py",
        type=str,
        default="D:/PycharmProjects/LipReaderClient/app/lipreader/checkpoint/checkpoint_wpelip_py.pth",
    )
    parse.add_argument(
        "--path_auto_kws",
        type=str,
        default="D:/PycharmProjects/LipReaderClient/app/lipreader/checkpoint/checkpoint_autoKws.pth",
    )

    args = parse.parse_args()
    return args


args = load_arg()
