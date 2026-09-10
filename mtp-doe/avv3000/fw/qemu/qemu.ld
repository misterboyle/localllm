/* qemu.ld — cortex-m3 (qemu -M lm3s6965evb): code in flash@0, data in RAM */
ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 256K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 64K
}

_stack_top = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
    .isr_vector : { KEEP(*(.isr_vector)) } >FLASH

    .text : {
        *(.text .text*)
        *(.rodata .rodata*)
        *(.gnu.linkonce.t.*)
        . = ALIGN(4);
    } >FLASH

    _sidata = LOADADDR(.data);

    .data : {
        . = ALIGN(4);
        _sdata = .;
        *(.data .data*)
        *(.gnu.linkonce.d.*)
        . = ALIGN(4);
        _edata = .;
    } >RAM AT>FLASH

    .bss (NOLOAD) : {
        . = ALIGN(4);
        _sbss = .;
        *(.bss .bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } >RAM

    /DISCARD/ : { *(.ARM.exidx*) *(.ARM.attributes*) }
}
