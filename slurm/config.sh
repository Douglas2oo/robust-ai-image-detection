# 所有脚本共用的路径：在项目根目录执行 `source slurm/config.sh`。
# 公共目录 group11 只放数据集；其余产物都在各自的个人目录下。
export ACCOUNT=mscaispod2026
export PROJ=/project/mscaispod2026
export MY_PROJ=$PROJ/$USER                            # 个人目录

export CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # 本仓库
export DATA=$PROJ/group11                             # 公共：只放数据集（genimage/、ufd/、raw/）
export SPLITS=$MY_PROJ/splits/v1                      # 数据清单（内容与仓库 manifests/ 相同）
export FEATS=$MY_PROJ/scratch/clip_feats              # CLIP 特征缓存（可重新生成）
export RUNS=$MY_PROJ/clip_runs                        # 训练好的线性头
export RESULTS=$MY_PROJ/clip_results                  # 指标与逐图预测
export HF_HOME=$MY_PROJ/hf_cache                      # CLIP 权重缓存
export CONDA_ENV=aigi

# 传给 python 入口的路径参数
export PATH_ARGS="--data_root $DATA --manifest_dir $SPLITS --cache_dir $FEATS --checkpoint_dir $RUNS --output_dir $RESULTS"

# 数据集目录（相对于 $DATA；同一生成器的多个目录用逗号分隔）
export GEN_SD14=genimage/sd14
export GEN_SD15=genimage/sd15
export GEN_ADM=genimage/adm
export GEN_MJ=genimage/midjourney
export GEN_BIGGAN=genimage/biggan
export EXT_PROGAN=ufd/test/progan
export EXT_LDM200=ufd/diffusion_datasets/ldm_200:ufd/diffusion_datasets/laion

activate_env() {
  source "$HOME/miniforge3/etc/profile.d/conda.sh" 2>/dev/null || true
  local had_u=0; [[ $- == *u* ]] && had_u=1
  set +u; conda activate "$CONDA_ENV"; local rc=$?
  [ $had_u = 1 ] && set -u
  return $rc
}
