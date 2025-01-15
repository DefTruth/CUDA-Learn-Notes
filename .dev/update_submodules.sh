# update submodules
set -x
git submodule init
# git submodule update --remote # update all submodule
git submodule update --remote ffpa-attn-mma # only update ffpa-attn-mma
git add .
git commit -m "Automated submodule update"
set +x