// ads8851_pkg.sv — ADS-8851 register contract (datasheet Rev B §6)
// SystemVerilog package shared by the RTL and documented in
// payload/ADS-8851-datasheet.md §6 (Table 6-1) and §7.2.
// All numbers here trace to the datasheet; keep in sync.
`ifndef ADS8851_PKG_SV
`define ADS8851_PKG_SV
package ads8851_pkg;

  localparam int ADDR_W = 12;   // AXI address width (64-register space)
  localparam int DATA_W = 32;

  // ---- offsets (Table 6-1) ----
  localparam logic [ADDR_W-1:0] ADDR_CTRL        = 12'h000;
  localparam logic [ADDR_W-1:0] ADDR_CFG         = 12'h004;
  localparam logic [ADDR_W-1:0] ADDR_STAT        = 12'h008;
  localparam logic [ADDR_W-1:0] ADDR_ALARM_CFG   = 12'h00C;
  localparam logic [ADDR_W-1:0] ADDR_ALARM_STATUS = 12'h010;
  localparam logic [ADDR_W-1:0] ADDR_WAVE_CFG    = 12'h014;
  localparam logic [ADDR_W-1:0] ADDR_WAVE_DATA   = 12'h018;
  localparam logic [ADDR_W-1:0] ADDR_WAVE_STATUS = 12'h01C;
  localparam logic [ADDR_W-1:0] ADDR_ERR_STAT    = 12'h020;
  localparam logic [ADDR_W-1:0] ADDR_IRQ_MASK    = 12'h024;
  localparam logic [ADDR_W-1:0] ADDR_TH_RATE     = 12'h028;
  localparam logic [ADDR_W-1:0] ADDR_SAFETY_GATE = 12'h02C;
  localparam logic [ADDR_W-1:0] ADDR_DATA_BASE   = 12'h040; // CH0..CH7 at +4*ch

  // ---- CTRL (6.7) ----
  localparam logic [DATA_W-1:0] CTRL_ADC_EN     = 32'h0000_0001;
  localparam logic [DATA_W-1:0] CTRL_WF_EN      = 32'h0000_0002;
  localparam logic [DATA_W-1:0] CTRL_SAFELIGHT_MODE = 32'h0000_000C; // [3:2]
  localparam logic [DATA_W-1:0] CTRL_LPM        = 32'h0000_0010;
  localparam logic [DATA_W-1:0] CTRL_ALARM_CLR  = 32'h0000_0020; // W1C
  localparam logic [DATA_W-1:0] CTRL_SWRST      = 32'h0000_0040; // self-clearing

  // ---- CFG (6.8, Rev B) ----
  localparam logic [DATA_W-1:0] CFG_SDR_WINDOW  = 32'h0000_0003; // [1:0]
  localparam logic [DATA_W-1:0] CFG_OVR_SAMPLE  = 32'h0000_001C; // [4:2]
  localparam logic [DATA_W-1:0] CFG_AVG_MODE    = 32'h0000_0020; // [5]  (Rev A: bit 7)
  localparam logic [DATA_W-1:0] CFG_FILT_EN     = 32'h0000_0040; // [6]
  localparam logic [DATA_W-1:0] CFG_RSVD0       = 32'h0000_0080; // [7]  Rev-B reserved 0
  localparam logic [DATA_W-1:0] CFG_TH_POL      = 32'h0000_0300; // [9:8]

  // ---- STAT (6.9) + ES-2.1.3 ----
  localparam logic [DATA_W-1:0] STAT_DRDY       = 32'h0000_0001;
  localparam logic [DATA_W-1:0] STAT_BUSY       = 32'h0000_0002;
  localparam logic [DATA_W-1:0] STAT_OCCL_DET   = 32'h0000_0004;
  localparam logic [DATA_W-1:0] STAT_TH_FAULT   = 32'h0000_0008;
  localparam logic [DATA_W-1:0] STAT_ERR_CNT    = 32'h0000_00F0; // [7:4] RC
  localparam logic [DATA_W-1:0] STAT_SEQ        = 32'h0000_FF00; // [15:8]

  // ---- ALARM_STATUS (6.11) ----
  localparam logic [DATA_W-1:0] ALM_OCCL        = 32'h0000_0001;
  localparam logic [DATA_W-1:0] ALM_HTEMP       = 32'h0000_0002;
  localparam logic [DATA_W-1:0] ALM_LTEMP       = 32'h0000_0004;
  localparam logic [DATA_W-1:0] ALM_LINE        = 32'h0000_0008;
  localparam logic [DATA_W-1:0] ALM_COMMS       = 32'h0000_0010;
  localparam logic [DATA_W-1:0] ALM_LATCHED     = 32'h0000_0020;

  // ---- ERR_STAT (6.16) ----
  localparam logic [DATA_W-1:0] ERR_BRESP       = 32'h0000_0001;
  localparam logic [DATA_W-1:0] ERR_STAT_VIOL   = 32'h0000_0002;
  localparam logic [DATA_W-1:0] ERR_FIFO_OVF    = 32'h0000_0004;
  localparam logic [DATA_W-1:0] ERR_CRC         = 32'h0000_0008;
  localparam logic [DATA_W-1:0] ERR_W_VIOL_MASK = 32'hFFFF_FF00; // [31:8] W_VIOLATIONS

endpackage
`endif