#! /bin/bash
#SBATCH --job-name=train_with_xlm_ft_ud
#SBATCH --nodes=1
#SBATCH -p publicgpu
#SBATCH -A lilpa
#SBATCH --time=00:59:00
#SBATCH --error=erreur_train_with_xlm_ft_ud.txt
#SBATCH --output=output_train_with_xlm_ft_ud.txt

module load python/python-3.11.3
source /home2020/home/lilpa/dbernhar/experiences/dagur/env_pos_transformers/bin/activate

#languages=("af ar be bg ca cs cu cy da de el en es et eu fa fi fo fr fro ga gd gl got grc he hi hr hu hy hyw id is it ja ko la lt lv lzh mr mt nl no orv pcm pl pt ro ru sa sk sl sme sr sv ta te tr ug uk ur vi wo zh")
#languages=("ug fo la eu tr is ro fi te grc")
#languages=("ug")
#languages=("fo la eu tr is ro fi te grc")
#languages=("fo la eu tr is ro fi te grc ug") #3 epochs
#languages=("te grc ug")
languages=("pl")
epochs=3
#languages=("none") # 10 epochs
seeds=(15 41 74 53 99)

for l in $languages
do
  for s in ${seeds[@]}
  do
    echo "Language: $l"
    echo "Seed: $s"
    python train_with_xlm_ft_ud.py --language $l --num_epochs $epochs --seed $s
  done
  echo "DONE WITH LANGUAGE $l"
done