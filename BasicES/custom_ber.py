#!/usr/bin/env python
##################################################
# Gnuradio Python Flow Graph
# Title: Custom Ber
# Generated: Thu Dec  5 19:01:36 2013
##################################################

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes

class custom_ber(gr.hier_block2):

    def __init__(self, n_bits=1000, bits_per_symbol=3):
        gr.hier_block2.__init__(
            self, "Custom Ber",
            gr.io_signaturev(2, 2, [gr.sizeof_char*1, gr.sizeof_char*1]),
            gr.io_signature(1, 1, gr.sizeof_float*1),
        )

        ##################################################
        # Parameters
        ##################################################
        self.n_bits = n_bits
        self.bits_per_symbol = bits_per_symbol

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 8e6

        ##################################################
        # Blocks
        ##################################################
        self.blocks_xor_xx_0 = blocks.xor_bb()
        self.blocks_unpack_k_bits_bb_0 = blocks.unpack_k_bits_bb(bits_per_symbol)
        self.blocks_uchar_to_float_0 = blocks.uchar_to_float()
        self.blocks_moving_average_xx_0 = blocks.moving_average_ff(n_bits, 1/float(n_bits), n_bits*4)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_moving_average_xx_0, 0), (self, 0))
        self.connect((self.blocks_uchar_to_float_0, 0), (self.blocks_moving_average_xx_0, 0))
        self.connect((self.blocks_unpack_k_bits_bb_0, 0), (self.blocks_uchar_to_float_0, 0))
        self.connect((self.blocks_xor_xx_0, 0), (self.blocks_unpack_k_bits_bb_0, 0))
        self.connect((self, 1), (self.blocks_xor_xx_0, 1))
        self.connect((self, 0), (self.blocks_xor_xx_0, 0))


# QT sink close method reimplementation

    def get_n_bits(self):
        return self.n_bits

    def set_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.blocks_moving_average_xx_0.set_length_and_scale(self.n_bits, 1/float(self.n_bits))

    def get_bits_per_symbol(self):
        return self.bits_per_symbol

    def set_bits_per_symbol(self, bits_per_symbol):
        self.bits_per_symbol = bits_per_symbol

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate


