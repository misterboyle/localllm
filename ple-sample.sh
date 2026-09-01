#!/usr/bin/env bash
# Sample memory posture for PLE on-direct vs lazy-mmap A/B.
# Usage: ./ple-sample.sh <label> >> ~/localllm/samples-<label>.csv &   (stop with kill)
label="${1:-run}"
echo "ts,label,used_gb,active_gb,inactive_gb,free_gb,wired_gb,swap_used_mb,server_rss_gb,shard_resident_gb"
while true; do
  read -r used act inact free wired <<< "$(vm_stat | awk -F': ' '
    function n(s){gsub(/\./,"",s); return s+0}
    /Pages wired down/{w=n($2)}
    /^Pages active/{a=n($2)}
    /^Pages inactive/{i=n($2)}
    /^Pages free/{f=n($2)}
    /occupied by compressor/{c=n($2)}
    END{printf "%.1f %.1f %.1f %.1f %.1f", (w+a+c)*16384/2^30, a*16384/2^30, i*16384/2^30, f*16384/2^30, w*16384/2^30}')"
  sw=$(sysctl -n vm.swapusage | awk '{for(j=1;j<=NF;j++) if($j=="used") print $(j+2)}' | tr -d M)
  pid=$(pgrep -f 'llama-server.*30085' | head -1); rss=NA; res=NA
  if [ -n "$pid" ]; then
    rss=$(( $(ps -o rss= -p "$pid") / 1048576 ))
    res=$(vmmap "$pid" 2>/dev/null | awk '
      function cv(v){u=substr(v,length(v),1); s=substr(v,1,length(v)-1)+0
        return s*((u=="K")?1024:(u=="M")?1048576:(u=="G")?1073741824:1)}
      /SHM/ && /Qwen3.8-Flash-Next/{for(j=NF;j>1;j--) if($j=="["){t+=cv($(j+2)); break}}
      END{printf "%.1f", t/2^30}')
  fi
  echo "$(date +%s),$label,$used,$act,$inact,$free,$wired,$sw,$rss,$res"
  sleep 30
done
