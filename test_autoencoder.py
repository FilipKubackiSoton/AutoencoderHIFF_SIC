import pytest
from autoencoder import Autoencoder, suplement_layers_params
from SINDY import sindy_library_tf
from HIFF import generate_training_sat
import matplotlib.pyplot as plt
from typing import List, Optional
import tensorflow as tf

# still to add thorough testing of autoencoder createion 
# to consider:  shape, layers params
# NOTE: layer params are approximates so 1.0 -> 1.00000000000756

def test_suplement_layers_params():
    assert 0 == 0


