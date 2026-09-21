# Run on the client VM. Keeps the load in the foreground shell's job table so
# it cannot be orphaned, and profiles the server from the middle of it.
cd ~/aex2
streams=${1:-16}
store=${2:-mem-noisy.zarr}
pattern=${3:-none}
reps=${4:-60}
target/release/aexbench http://192.168.100.207:50391 "$store" --bytes $((4<<30)) \
  --streams "$streams" --reps "$reps" --prefault --pattern "$pattern" --label prof > ~/load.txt 2>&1 &
load=$!
sleep 6
ssh -o BatchMode=yes 192.168.100.207 \
  "pid=\$(pgrep -x aex-server)
   read -r _ _ _ _ _ _ _ _ _ _ _ _ _ u0 s0 _ < /proc/\$pid/stat
   sudo perf record -a -e cpu-clock -F 499 -g -o ~/p-$streams.data -- sleep 15 > /dev/null 2>&1
   read -r _ _ _ _ _ _ _ _ _ _ _ _ _ u1 s1 _ < /proc/\$pid/stat
   echo \"server used \$(( (u1-u0+s1-s0)/100 )) CPU-seconds in the 15 s window\""
kill $load 2>/dev/null
wait $load 2>/dev/null
tail -1 ~/load.txt
