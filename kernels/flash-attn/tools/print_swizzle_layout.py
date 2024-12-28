import argparse


def pretty_print_line(m: str = "", 
                      sep: str = "-", 
                      width: int = 130, 
                      return_str: bool = False):
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    if not return_str:
        print(pretty_line)
    else:
        return pretty_line


PERMUTED_DOCS_STRING = \
"""----------------------------------------------------------------
[INFO] Assert smem store layout col_stride <= 16, prefer 16.   |
[INFO] For logical_col_stride > 16, we have to permute the     |
[INFO] smem store layout using col major ZigZag method:        |
[INFO] e.g, --> Q smem logical layout [Br][64].                |
[INFO]      --> col major ZigZag permuted -->                  |
[INFO]      --> Q smem store layout [4][Br][16].               |
----------------------------------------------------------------"""

def swizzle_permuted_j(i: int, 
                       j: int, 
                       col_stride: int = 16, 
                       num_elems_per_128b: int = 8):
    # i: row index; j: col index. col_stride <= 16.
    # assert col_stride <= 16, f"col_stride must <= 16, but got {col_stride}"
    # for col_stride > 16, we have to permute it using col major ZigZag order.
    # e.g, Q smem logical layout [Br,d]=[Br,64] -> store layout [4][Br][16].
    return (
        (int(j / num_elems_per_128b) ^ int(i / 4)) % 
        (int(col_stride / num_elems_per_128b))
    ) * num_elems_per_128b


def print_smem_swizzle_layout(rows: int = 16, 
                              logical_col_stride: int = 16, 
                              num_elems_per_128b: int = 8, 
                              smem_pading: int = 0,
                              show_logical_col_id: bool = False,
                              use_logical_col_stride: bool = False):
    # ----------------------------------------------------------------
    # [INFO] Assert smem store layout col_stride <= 16, prefer 16.   |
    # [INFO] For logical_col_stride > 16, we have to permute the     |
    # [INFO] smem store layout using col major ZigZag method:        |
    # [INFO] e.g, --> Q smem logical layout [Br][64].                |
    # [INFO]      --> col major ZigZag permuted -->                  |
    # [INFO]      --> Q smem store layout [4][Br][16].               |
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # -------------------------swizzle layout-------------------------
    # --------------------logical col 0~64, step 8--------------------
    # ---------------------smem col 0~16, step 8----------------------
    # ----------------------------------------------------------------
    # |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
    # |row 0 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
    # |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
    # |row 1 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
    # |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
    # |row 2 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
    # |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
    # |row 3 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
    # ----------------------------------------------------------------
    # |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
    # |row 4 | 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
    # |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
    # |row 5 | 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
    # |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
    # |row 6 | 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
    # |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
    # |row 7 | 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
    # ----------------------------------------------------------------
    # |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
    # |row 8 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
    # |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
    # |row 9 | 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
    # |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
    # |row 10| 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
    # |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
    # |row 11| 0:0  | 8:8  |16:0  |24:8  |32:0  |40:8  |48:0  |56:8  |
    # ----------------------------------------------------------------
    # |bank  |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |b 0~3 |b 4~7 |
    # |row 12| 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
    # |bank  |b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|b 8~11|b12~15|
    # |row 13| 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
    # |bank  |b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|b16~19|b20~23|
    # |row 14| 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
    # |bank  |b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|b24~27|b28~31|
    # |row 15| 0:8  | 8:0  |16:8  |24:0  |32:8  |40:0  |48:8  |56:0  |
    # ----------------------------------------------------------------
    str_len = 0
    total_banks = 0
    assert smem_pading == 0 or smem_pading == 8, "smem_pading must be 0 or 8"
    # 4 bytes per bank
    banks_per_col = int((16 * 2) / 4) if logical_col_stride >= 16 else 4 
    if use_logical_col_stride:
        banks_per_col = int((logical_col_stride * 2) / 4)
        if logical_col_stride > 16:
            print(f"[WARN] col_stride must <= 16, but got {logical_col_stride}") 
    if smem_pading == 8:
        banks_per_col += 4
        print(f"[INFO] smem padding 8 half values, 4 banks, banks_per_col: {banks_per_col}")

    banks_per_num_elems_per_128b = int((num_elems_per_128b * 2) / 4)
    for i in range(rows):
        layout_str_len = 0
        banks_str_len = 0

        # bank_layout_str
        banks_start = total_banks % 32 # 32 banks in total
        banks_end = (banks_start + banks_per_col)
        bank_layout_str = f"|bank  |"
        max_bank_str_len = 0
        if logical_col_stride >= 16 and (not use_logical_col_stride):
            for k in range(int(logical_col_stride / 16)):
                for j in range(banks_start, banks_end, banks_per_num_elems_per_128b):      
                    curr_bank_str = f"b{j:>2}~{j + banks_per_num_elems_per_128b - 1:<2}|" 
                    max_bank_str_len = max(max_bank_str_len, len(curr_bank_str))       
                    bank_layout_str += curr_bank_str
        else:
            for j in range(banks_start, banks_end, banks_per_num_elems_per_128b):      
                curr_bank_str = f"b{j:>2}~{j + banks_per_num_elems_per_128b - 1:<2}|" 
                max_bank_str_len = max(max_bank_str_len, len(curr_bank_str))       
                bank_layout_str += curr_bank_str

        # smem_layout_str
        logical_col_ids = []
        smem_layout_col_ids = []
        if logical_col_stride >= 16 and (not use_logical_col_stride):
            for k in range(int(logical_col_stride / 16)):
                for j in range(0, 16, num_elems_per_128b):
                    layout_j = swizzle_permuted_j(i, j, 16, 
                                                  num_elems_per_128b)
                    logical_col_ids.append(k * 16 + j)
                    smem_layout_col_ids.append(layout_j)
        else:
            for j in range(0, logical_col_stride, num_elems_per_128b):
                layout_j = swizzle_permuted_j(i, j, logical_col_stride, 
                                              num_elems_per_128b)
                logical_col_ids.append(j)
                smem_layout_col_ids.append(layout_j)

        smem_layout_str = f"|row {i:<2}|"

        r = 0
        for c, l in zip(logical_col_ids, smem_layout_col_ids):
            smem_layout_str += pretty_print_line(
                (f"{c:>2}:{l:<2}" if show_logical_col_id else f"{l:<2}"), 
                sep=" ",
                width=(max_bank_str_len-1), 
                return_str=True
            ) + "|"
            r += 1
            if logical_col_stride >= 16:
                if smem_pading == 8 and (r > 1 and r % 2 == 0):
                    smem_layout_str += pretty_print_line(
                        (f"pad"), 
                        sep=" ", width=max_bank_str_len-1, 
                        return_str=True
                    ) + "|"
            else:
                if smem_pading == 8:
                    smem_layout_str += pretty_print_line(
                        (f"pad"), 
                        sep=" ", width=max_bank_str_len-1, 
                        return_str=True
                    ) + "|"

        layout_str_len = len(smem_layout_str)
        str_len = max(layout_str_len, banks_str_len)

        # print banks and smem layout
        if (i == 0):
            print("-" * str_len)
            pretty_print_line(f"swizzle layout", width=str_len)
            pretty_print_line(f"logical col 0~{logical_col_stride}, "
                              f"step {num_elems_per_128b}", 
                              width=str_len)
            pretty_print_line(f"smem col 0~16, step {num_elems_per_128b}" 
                              if logical_col_stride >= 16 
                              else f"smem col 0~8, step {num_elems_per_128b}", 
                              width=str_len)
            print("-" * str_len)
        print(bank_layout_str)
        print(smem_layout_str)
        if ((i + 1) % 4 == 0 and i != (rows - 1)):
            print("-" * str_len)
        total_banks += banks_per_col
    print("-" * str_len)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--smem-padding", "--pad", type=int, default=0)
    parser.add_argument("--num-elems-per-128b", "--num-elems", type=int, default=8)
    parser.add_argument("--logical-col-stride", "--logical-col", "--col", type=int, default=64)
    parser.add_argument("--use-logical-col-stride", "--use-logical-col", action="store_true")
    parser.add_argument("--show-logical-col-id", "--show-logical-col", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    print(PERMUTED_DOCS_STRING)
    print_smem_swizzle_layout(rows=args.rows, 
                              logical_col_stride=args.logical_col_stride, 
                              num_elems_per_128b=args.num_elems_per_128b, 
                              smem_pading=args.smem_padding,
                              show_logical_col_id=args.show_logical_col_id,
                              use_logical_col_stride=args.use_logical_col_stride)

