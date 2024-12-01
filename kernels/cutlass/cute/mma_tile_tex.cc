/* makefile
copy from: https://gist.github.com/66RING/2e188b73fdf703e9f9dfc7371814dd15#file-mma_tile_tex-cpp

render: build
	./build/main > mma_tile.tex
	xelatex --cnf-line=main_memory=12000000 --halt-on-error mma_tile.tex && rm -rf *.aux *.log *.out

build:
	cmake -B build
	cmake --build build

.PHONY: build


*/


#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"


void print_header() {
  const char* latex_header =
    "\\documentclass{article}\n"
    "\\usepackage[a4paper, margin=0.5cm]{geometry}\n"
    "\\usepackage{adjustbox}\n"
    "\\usepackage{graphicx}\n"
    "\\usepackage{lipsum}\n"
    "\\usepackage{tikz}\n"
    "\n"
    "\\begin{document}\n";
  printf("%s", latex_header);
}


void print_footer() {
  const char* latex_footer = "\\end{document}\n";
  printf("%s", latex_footer);
}


// Copy from mma_atom.hpp
//
// Modified to remove printing header and footder, hence allows printing
// multiple MMAs per TEX file for easier comparisons.
template <class AtomLayoutMNK,
          class ValLayoutMNK,
          class PermutationsMNK,
          class LayoutC, class ThrIDC,
          class LayoutA, class ThrIDA,
          class LayoutB, class ThrIDB>
void
print_mma(const char* name,
          const AtomLayoutMNK& atom_layout_mnk,
          const ValLayoutMNK& val_layout_mnk,
          const PermutationsMNK& permutations_mnk,
          LayoutC const& C, ThrIDC const& TC,    // (m,n) -> (tid,vid)  and  tid -> thr_idx
          LayoutA const& A, ThrIDA const& TA,    // (m,k) -> (tid,vid)  and  tid -> thr_idx
          LayoutB const& B, ThrIDB const& TB) {  // (n,k) -> (tid,vid)  and  tid -> thr_idx
  using namespace cute;

  printf("\\begin{verbatim}\n");
  printf("\n%s\n\n", name);

  printf("  AtomLayoutMNK: "); print(atom_layout_mnk);   printf("\n");
  printf("   ValLayoutMNK: "); print(val_layout_mnk);    printf("\n");
  printf("PermutationsMNK: "); print(permutations_mnk); printf("\n\n");

  printf("LayoutC: "); print(C);  printf("\n");
  printf(" ThrIDC: "); print(TC); printf("\n");
  printf("LayoutA: "); print(A);  printf("\n");
  printf(" ThrIDA: "); print(TA); printf("\n");
  printf("LayoutB: "); print(B);  printf("\n");
  printf(" ThrIDB: "); print(TB); printf("\n");
  printf("\\end{verbatim}\n");

  // printf("\\begin{adjustbox}{max height=0.7\\textheight,max width=\\textwidth}%");
  printf("\\begin{adjustbox}{max height=0.7\\textheight,max width=\\textwidth}\n");
  printf("\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
         ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n");
  char const* color_map[8] = {"{rgb,255:red,175;green,175;blue,255}",
                              "{rgb,255:red,175;green,255;blue,175}",
                              "{rgb,255:red,255;green,255;blue,175}",
                              "{rgb,255:red,255;green,175;blue,175}",
                              "{rgb,255:red,210;green,210;blue,255}",
                              "{rgb,255:red,210;green,255;blue,210}",
                              "{rgb,255:red,255;green,255;blue,210}",
                              "{rgb,255:red,255;green,210;blue,210}"};

  // C starting at 0,0
  for (int m = 0; m < size<0>(C); ++m) {
    for (int n = 0; n < size<1>(C); ++n) {
      int thrid   = C(m,n) % size(TC);
      int val_idx = C(m,n) / size(TC);
      int thr_idx = TC(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             m, n,
             thr_idx, val_idx);
    }
  }

  // A starting at 0,-size<1>(A)-1
  for (int m = 0; m < size<0>(A); ++m) {
    for (int k = 0; k < size<1>(A); ++k) {
      int thrid   = A(m,k) % size(TA);
      int val_idx = A(m,k) / size(TA);
      int thr_idx = TA(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             m, k-1-size<1>(A),
             thr_idx, val_idx);
    }
  }

  // B starting at -size<1>(B)-1,0
  for (int n = 0; n < size<0>(B); ++n) {
    for (int k = 0; k < size<1>(B); ++k) {
      int thrid   = B(n,k) % size(TB);
      int val_idx = B(n,k) / size(TB);
      int thr_idx = TB(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             k-1-size<1>(B), n,
             thr_idx, val_idx);
    }
  }

  // A labels
  for (int m = 0, k = -1; m < size<0>(A); ++m) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k-1-size<1>(A), m);
  }
  for (int k = 0, m = -1; k < size<1>(A); ++k) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k-1-size<1>(A), k);
  }
  // B labels
  for (int n = 0, k = -1; n < size<0>(B); ++n) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k-1-size<1>(B), n, n);
  }
  for (int k = 0, n = -1; k < size<1>(B); ++k) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k-1-size<1>(B), n, k);
  }

  printf("\\end{tikzpicture}\n\\end{adjustbox}%\n");
}


template <class TiledCopy,
          class LayoutS, class ThrIDS,
          class LayoutD, class ThrIDD>
void
print_copy(const char* name,
           TiledCopy& copy,
           LayoutS const& S, ThrIDS const& TS,   // (m,n) -> (tid,vid)  and  tid -> thr_idx
           LayoutD const& D, ThrIDD const& TD) { // (m,n) -> (tid,vid)  and  tid -> thr_idx
  using namespace cute;

  CUTE_STATIC_ASSERT_V(rank(S) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<2>{});

  assert(size<0>(S) == size<0>(D));
  assert(size<1>(S) == size<1>(D));

  printf("\\begin{verbatim}\n");
  printf("\n%s\n\n", name);
  printf("LayoutCopy_TV: "); print(typename TiledCopy::TiledLayout_TV{});  printf("\n");
  printf(" ShapeTile_MN: "); print(typename TiledCopy::Tiler_MN{});      printf("\n\n");

  printf("      LayoutS: "); print(S);  printf("\n");
  printf("       ThrIDS: "); print(TS); printf("\n");
  printf("      LayoutD: "); print(D);  printf("\n");
  printf("       ThrIDD: "); print(TD); printf("\n");
  printf("\\end{verbatim}\n");

  printf("\\begin{adjustbox}{max height=0.7\\textheight,max width=\\textwidth}\n");
  printf("\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
         ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n");
  char const* color_map[8] = {"{rgb,255:red,175;green,175;blue,255}",
                              "{rgb,255:red,175;green,255;blue,175}",
                              "{rgb,255:red,255;green,255;blue,175}",
                              "{rgb,255:red,255;green,175;blue,175}",
                              "{rgb,255:red,210;green,210;blue,255}",
                              "{rgb,255:red,210;green,255;blue,210}",
                              "{rgb,255:red,255;green,255;blue,210}",
                              "{rgb,255:red,255;green,210;blue,210}"};

  // S starting at 0,0
  for (int i = 0; i < size<0>(S); ++i) {
    for (int j = 0; j < size<1>(S); ++j) {
      int thrid   = S(i,j) % size(TS);
      int val_idx = S(i,j) / size(TS);
      int thr_idx = TS(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             i, j,
             thr_idx, val_idx);
    }
  }

  // D starting at 0,size<1>(S)+3
  for (int i = 0; i < size<0>(D); ++i) {
    for (int j = 0; j < size<1>(D); ++j) {
      int thrid   = D(i,j) % size(TD);
      int val_idx = D(i,j) / size(TD);
      int thr_idx = TD(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             i + size<0>(S) + 3, j,
             thr_idx, val_idx);
    }
  }

  // S Labels
  for (int i = 0, j = -1; i < size<0>(S); ++i) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
  }
  for (int j = 0, i = -1; j < size<1>(S); ++j) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
  }
  // D Labels
  for (int i = 0, j = -1; i < size<0>(S); ++i) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i + size<0>(S) + 3, j, i);
  }
  for (int j = 0, i = -1; j < size<1>(D); ++j) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i + size<0>(S) + 3, j, j);
  }

  printf("\\end{tikzpicture}\n\\end{adjustbox}%\n");
}


template <class MMA_Atom_Arch,
          class AtomLayoutMNK,
          class ValLayoutMNK,
          class PermutationsMNK>
void print_mma_content(
  const char* name,
  cute::TiledMMA<MMA_Atom_Arch, AtomLayoutMNK, ValLayoutMNK, PermutationsMNK> const& mma) {
  printf("\n\\newpage\n");

  auto layout_and_thrid_C = mma.get_layoutC_MN();
  auto layoutC_MN = cute::get<0>(layout_and_thrid_C);
  auto thrID_C    = cute::get<1>(layout_and_thrid_C);

  auto layout_and_thrid_A = mma.get_layoutA_MK();
  auto layoutA_MK = cute::get<0>(layout_and_thrid_A);
  auto thrID_A    = cute::get<1>(layout_and_thrid_A);

  auto layout_and_thrid_B = mma.get_layoutB_NK();
  auto layoutB_NK = cute::get<0>(layout_and_thrid_B);
  auto thrID_B    = cute::get<1>(layout_and_thrid_B);

  print_mma(name,
            AtomLayoutMNK{},
            ValLayoutMNK{},
            PermutationsMNK{},
            layoutC_MN, thrID_C,
            layoutA_MK, thrID_A,
            layoutB_NK, thrID_B);
}


template <class TiledCopy>
void print_copy_content(const char* name, TiledCopy& copy) {
  printf("\n\\newpage\n");

  auto [layoutS_MN, thrID_S] = copy.get_layoutS_MN();
  auto [layoutD_MN, thrID_D] = copy.get_layoutD_MN();

  print_copy(name, copy,
             layoutS_MN, thrID_S,
             layoutD_MN, thrID_D);
}


void print_layouts_for_mma() {
  using namespace cute;
  using _X = cute::Underscore;

  {
    auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                    Layout<Shape<_4,_1, _1>>{},  // AtomLayoutMNK
                                    Layout<Shape<_1,_2, _1>>{}   // ValLayoutMNK
    );
    print_mma_content("flash2: SM80_16x8x16_F32F16F16F32_TN", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                    Layout<Shape<_1,_1, _1>>{},  // AtomLayoutMNK
                                    Layout<Shape<_1,_2, _1>>{}   // ValLayoutMNK
    );
    print_mma_content("flash2: SM80_16x8x16_F32F16F16F32_TN", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                    Layout<Shape<_2,_1, _1>>{},  // AtomLayoutMNK
                                    Layout<Shape<_1,_2, _1>>{}   // ValLayoutMNK
    );
    print_mma_content("flash2: SM80_16x8x16_F32F16F16F32_TN", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                    Layout<Shape<_4,_1, _1>>{},  // AtomLayoutMNK
                                    Layout<Shape<_1,_1, _1>>{}   // ValLayoutMNK
    );
    print_mma_content("flash2: SM80_16x8x16_F32F16F16F32_TN", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                    Layout<Shape<_1,_1, _1>>{},  // AtomLayoutMNK
                                    Layout<Shape<_1,_1, _1>>{}   // ValLayoutMNK
    );
    print_mma_content("flash2: SM80_16x8x16_F32F16F16F32_TN", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                    Layout<Shape<_1,_1, _1>>{},  // AtomLayoutMNK
                                    Layout<Shape<_1,_1, _1>>{}   // ValLayoutMNK
    );
    print_mma_content("flash2: SM75_16x8x8_F32F16F16F32_TN", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                    Layout<Shape<_1,_1, _1>>{},  // AtomLayoutMNK
                                    Layout<Shape<_1,_2, _1>>{}   // ValLayoutMNK
    );
    print_mma_content("flash2: SM75_16x8x8_F32F16F16F32_TN", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{},
                                    Layout<Shape<_4,_1, _1>>{},  // AtomLayoutMNK
                                    Layout<Shape<_1,_2, _2>>{}   // ValLayoutMNK
    );
    print_mma_content("flash2: SM75_16x8x8_F32F16F16F32_TN", tiled_mma);
  }
}

int main() {
  print_header();

  print_layouts_for_mma();

  print_footer();
  return 0;
}

