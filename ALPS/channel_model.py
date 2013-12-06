#!/usr/bin/env python
##################################################
# Gnuradio Python Flow Graph
# Title: Channel Model
# Generated: Thu Dec  5 19:01:14 2013
##################################################

execfile("./custom_ber.py")
from gnuradio import blocks
from gnuradio import channels
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import numpy

class channel_model(gr.top_block):

    def __init__(self, constellation=[-1, 1]):
        gr.top_block.__init__(self, "Channel Model")

        ##################################################
        # Parameters
        ##################################################
        self.constellation = constellation

        ##################################################
        # Variables
        ##################################################
        self.const_points = const_points = constellation
        self.taps = taps = [1.0, 0.25-0.25j, 0.50 + 0.10j, 0.3 + 0.2j]
        self.samp_rate = samp_rate = 8e6
        self.noise_level = noise_level = 0.3
        self.frequency = frequency = 2000000
        self.const_type = const_type = 1
        self.const_dist = const_dist = digital.constellation_calcdist(const_points,[],0,1)
        self.const = const = digital.constellation_8psk().bits_per_symbol()

        ##################################################
        # Blocks
        ##################################################
        self.digital_constellation_decoder_cb_0_0 = digital.constellation_decoder_cb(const_dist.base())
        self.digital_constellation_decoder_cb_0 = digital.constellation_decoder_cb(const_dist.base())
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc((const_dist.points()), 1)
        self.custom_ber_0 = custom_ber(
            n_bits=10000,
        )
        self.channels_channel_model_0_0 = channels.channel_model(
        	noise_voltage=0,
        	frequency_offset=0,
        	epsilon=1,
        	taps=([1,0,0,0]),
        	noise_seed=0,
        	block_tags=False
        )
        self.channels_channel_model_0 = channels.channel_model(
        	noise_voltage=noise_level,
        	frequency_offset=0,
        	epsilon=1,
        	taps=(taps),
        	noise_seed=0,
        	block_tags=False
        )
        self.blocks_vector_sink_x_1 = blocks.vector_sink_f(1)
        self.analog_random_source_x_0 = blocks.vector_source_b(map(int, numpy.random.randint(0, len(const_points), 10000)), False)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.custom_ber_0, 0), (self.blocks_vector_sink_x_1, 0))
        self.connect((self.channels_channel_model_0, 0), (self.digital_constellation_decoder_cb_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.channels_channel_model_0, 0))
        self.connect((self.analog_random_source_x_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0_0, 0), (self.custom_ber_0, 0))
        self.connect((self.digital_constellation_decoder_cb_0, 0), (self.custom_ber_0, 1))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.channels_channel_model_0_0, 0))
        self.connect((self.channels_channel_model_0_0, 0), (self.digital_constellation_decoder_cb_0_0, 0))


# QT sink close method reimplementation

    def get_constellation(self):
        return self.constellation

    def set_constellation(self, constellation):
        self.constellation = constellation
        self.set_const_points(self.constellation)

    def get_const_points(self):
        return self.const_points

    def set_const_points(self, const_points):
        self.const_points = const_points
        self.set_const_dist(digital.constellation_calcdist(self.const_points,[],0,1))

    def get_taps(self):
        return self.taps

    def set_taps(self, taps):
        self.taps = taps
        self.channels_channel_model_0.set_taps((self.taps))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate

    def get_noise_level(self):
        return self.noise_level

    def set_noise_level(self, noise_level):
        self.noise_level = noise_level
        self.channels_channel_model_0.set_noise_voltage(self.noise_level)

    def get_frequency(self):
        return self.frequency

    def set_frequency(self, frequency):
        self.frequency = frequency

    def get_const_type(self):
        return self.const_type

    def set_const_type(self, const_type):
        self.const_type = const_type

    def get_const_dist(self):
        return self.const_dist

    def set_const_dist(self, const_dist):
        self.const_dist = const_dist

    def get_const(self):
        return self.const

    def set_const(self, const):
        self.const = const

if __name__ == '__main__':
    parser = OptionParser(option_class=eng_option, usage="%prog: [options]")
    (options, args) = parser.parse_args()
    tb = channel_model()
    tb.start()
    tb.wait()

