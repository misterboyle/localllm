/* sh.h — ARM semihosting console/exit for the qemu fw runner */
#ifndef SH_H
#define SH_H

void sh_putc(int c);
void sh_puts(const char *s);
void sh_putu(unsigned v);
void sh_exit(int rc); /* never returns */

#endif /* SH_H */
