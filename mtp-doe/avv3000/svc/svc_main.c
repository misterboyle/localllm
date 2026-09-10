/* svc_main.c — telemetry service skin: wire FIFO (stdin) -> ring -> CSV
 * console (stdout). The golden rungs (t_svc_burst, t_svc_log) exercise
 * telem_svc.c through the API; this main is syntax/compile-gated only.
 *
 * Model: records arrive as canonical wire lines with the service clock
 * read off the ts column (svc_ms is driven by the data, no wall clocks —
 * same determinism rule as the tests). Malformed lines are counted and
 * reported at exit; the service never stalls the stream for them.
 */
#include <stdio.h>
#include <string.h>

#include "telem_svc.h"

static void sink_stdout(void *ctx, const char *line)
{
    (void)ctx;
    fputs(line, stdout);
    fputc('\n', stdout);
}

int main(void)
{
    svc_t svc;
    svc_stats_t st;
    char line[256];
    uint32_t bad = 0;

    svc_init(&svc, sink_stdout, 0);
    while (fgets(line, (int)sizeof line, stdin) != 0) {
        svc_rec_t rec;
        int rc = svc_parse_line(line, &rec);
        if (rc == SVC_IGNORE) {
            continue;
        }
        if (rc == SVC_EBAD) {
            bad++;
            continue;
        }
        svc_ms(&svc, rec.ts_ms);
        svc_ingest(&svc, &rec);
    }
    svc_flush(&svc);

    svc_stats(&svc, &st);
    fprintf(stderr,
            "# svc: win_in=%u st_in=%u st_rejected=%u st_evicted=%u "
            "win_overwrite=%u logged=%u st_logged=%u bad=%u\n",
            st.win_in, st.st_in, st.st_rejected, st.st_evicted,
            st.win_overwrite, st.logged, st.st_logged, bad);
    return (st.win_overwrite != 0u || bad != 0u) ? 1 : 0;
}
