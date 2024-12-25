import argparse


def pretty_print_line(m: str = "", sep: str = "-", width: int = 130):
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    print(pretty_line)


def swizzle_permuted_j(i: int, j: int, col_stride: int = 64, step: int = 8):
    # i: row index; j: col index. 
    return ((int(j / step) ^ int(i / 4)) % int(col_stride / step)) * step


def print_swizzle_layout(rows: int = 16, col_stride: int = 64, step: int = 8):
    str_len = 0
    for i in range(rows):
        layout = tuple(swizzle_permuted_j(i, j, col_stride, step) 
                       for j in range(0, col_stride, step))
        layout_str = (f"| row {i:<2} | {layout} |")
        str_len = len(layout_str)
        if (i == 0):
            print("-" * str_len)
            pretty_print_line(f"swizzle layout", width=str_len)
            pretty_print_line(f"col 0~{col_stride}, step {step}", width=str_len)
            print("-" * str_len)
        print(layout_str)
        if ((i + 1) % 4 == 0 and i != (rows - 1)):
            print("-" * str_len)
    print("-" * str_len)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--col_stride", "--col", type=int, default=64)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--rows", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print_swizzle_layout(args.rows, args.col_stride, args.step)

