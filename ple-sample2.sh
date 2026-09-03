#!/usr/bin/env bash
# Sample memory posture v2: adds the queues v1 is blind to (purgeable, speculative,
# internal/external split) so tool formulas can be reproduced from one snapshot.
# Usage: ./ple-sample2.sh <label> >> ~/localllm/samples2-<label>.csv &
label="${1:-run2}"
echo "ts,label,used_gb,active_gb,inactive_gb,free_gb,wired_gb,purgeable_gb,spec_gb,internal_gb,external_gb,swap_used_mb,server_rss_gb,shard_resident_gb"
while true; do
  read -r used act inact free wired purg spec <<< "$(vm_stat | awk -F': ' '
    function n(s){gsub(/[^0-9]/,"",s); return s+0}
    /Pages wired down/{w=n($2)}
    /^Pages active/{a=n($2)}
    /^Pages inactive/{i=n($2)}
    /^Pages free/{f=n($2)}
    /^Pages speculative/{s=n($2)}
    /purgeable/{p=n($2)}
    /occupied by compressor/{c=n($2)}
    END{printf "%.1f %.1f %.1f %.1f %.1f %.1f %.1f", (w+a+c)*16384/2^30, a*16384/2^30, i*16384/2^30, f*16384/2^30, w*16384/2^30, p*16384/2^30, s*16384/2^30}')"
  read -r internal external <<< "$(sysctl -n vm.page_pageable_internal_count vm.page_pageable_external_count | \
    awk '{v[NR]=$1} END{printf "%.1f %.1f", v[1]*16384/2^30, v[2]*16384/2^30}')"
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
  echo "$(date +%s),$label,$used,$act,$inact,$free,$wired,$purg,$spec,$internal,$external,$sw,$rss,$res"
  sleep 30
done
