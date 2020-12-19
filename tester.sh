OUT="out"
IN="in"

N=$1
mpicxx -o multi-astar-pancake multi-astar-pancake.cpp -O3

for t in "$IN"/*.in ; do
  if [ -f "$t" ]; then
    curtest=$(basename $t)
    mpirun -n "$N" multi-astar-pancake < "$IN"/"$curtest" 
  fi
done
