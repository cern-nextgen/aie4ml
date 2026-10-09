import hls4ml
import numpy as np
import tensorflow as tf
from aie4ml.report import report
from qkeras import QActivation, QDense, quantized_bits
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

N_IN = 256
BATCH = 256
ITERS = 4
PLATFORM = 'xilinx_vek280_base_202520_1'

PROJECT_NAME = 'hardware_new'


def build_model():
    inp = tf.keras.Input(batch_size=BATCH, shape=(N_IN,), name='inp')
    x = QActivation(quantized_bits(8, 0), name='input_quant')(inp)
    x = QDense(128, kernel_quantizer=quantized_bits(8, 0, alpha=1), name='dense_0')(x)
    x = QActivation(quantized_bits(8, 0), name='quant_0')(x)
    out = QActivation(quantized_bits(8, 0), name='quant_out')(x)
    return Model(inp, out, name='single_dense_large')


model = build_model()
model.compile(optimizer=Adam(1e-3), loss='mse')
model.summary()


# ── HLS config ──
cfg = hls4ml.utils.config_from_keras_model(model, granularity='name')

for h in range(1):
    cfg['LayerName'][f'dense_{h}_linear']['Precision']['result'] = 'fixed<8,3,TRN,WRAP,0>'

print('\nLayer precision summary:')
for name, layer_cfg in cfg.get('LayerName', {}).items():
    print(f"  {name}: {layer_cfg.get('Precision', {})}")

# ── AIE conversion ──
# target='hardware' enables PL data mover + XRT host emission (default 'aie' = AIE-only).
aie_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=cfg,
    output_dir='proj_aie_' + PROJECT_NAME,
    backend='aie',
    project_name='proj_aie_' + PROJECT_NAME,
    batch_size=BATCH,
    iterations=ITERS,
    part=PLATFORM,
    optimize='resource',  # resource | throughput | latency
    target='hardware',  # hardware | aie
    pl_memory='uram',  # uram | bram
    enable_pl_timing=True,  # True | False
    pl_data_mover_mode='memory_stream',  # benchmark | memory_stream | external_stream
)

aie_model.compile()
# By default the simulation works for the hardware target. To build for hardware emulation,
# use aie_model.build(make_target='hw_emu')
aie_model.build()

# Simulation
x = np.random.random((BATCH, N_IN)).astype(np.float32)
y_aie = aie_model.predict(x, simulator='aie')[:BATCH]

print(report(aie_model))
